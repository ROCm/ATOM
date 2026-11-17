# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which V4.1 mono decode a deployment takes (``install``): gfx1250's single launch where it is enabled, else the first
of the gfx950 backends (``PRIORITY``) that serves it:

- ``tk``: the TileRT-structured decode step (``gfx950.tk``), whole decode steps behind ATOM's step override
  (``Hook.STEP``); a request it admitted cannot fall back, so neither can its install;
- ``hi``: the high-interactivity monokernels (``hi_gate``, ``hi_dispatch``), every decoder layer of a served step in one
  persistent kernel (``Hook.LAYER``), on top of k1k2 (``requires``): its other decode steps take k1k2, and an install
  the TP group refuses (``MonoUnsupported``) falls back to it (``falls_back``);
- ``k1k2``: ATOM's mono decode, two kernels a layer (``gfx950.k1k2.dispatch.MonoDecodeModel``, ``Hook.LAYER``).

``choose`` asks them in that order, each refusal logged or raised as ``ATOM_DSV41_MONO_HI`` (``Mode``) says: tk where
its host gate (``considered``) and its refusal (``serves``) pass, unless the mode forces hi; then k1k2's refusal, which
leaves the model unwrapped (an error when hi is forced); then hi where the mode and ``hi_gate`` let it serve; k1k2 last.
"""

from __future__ import annotations

import enum
import logging
from typing import Protocol, runtime_checkable

from atom.models.deepseek_v41.mono import hi_gate
from atom.models.deepseek_v41.mono.deployment import Deployment
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.step_override import StepOverrideModel

logger = logging.getLogger("atom")


class Hook(enum.Enum):
    """How ModelRunner reaches an installed backend."""

    LAYER = "layer"  # the installed model's forward routes decode steps (MonoDecodeModel, HiDecodeModel)
    STEP = "step"  # the installed model is a StepOverrideModel (ModelRunner.run_model offers it whole steps)


class Mode(enum.Enum):
    """``ATOM_DSV41_MONO_HI`` (``hi_gate.mode``: any other value raises ValueError)."""

    AUTO = "auto"
    ON = "1"
    OFF = "0"

    @classmethod
    def read(cls) -> Mode:
        return cls(hi_gate.mode())


@runtime_checkable
class MonoBackend(Protocol):
    name: str
    hook: Hook
    requires: tuple[
        str, ...
    ]  # the backends it runs on top of: considered only where they serve
    falls_back: bool  # MonoUnsupported at install takes the next candidate

    def considered(self, dep: Deployment) -> str | None:
        """Why this host does not ask it at all (no line logged), or None."""

    def serves(self, dep: Deployment) -> str | None:
        """Why it does not serve ``dep``, or None: reads only (no runtime import, allocation or collective)."""

    def install(self, model, dep: Deployment):
        """``model`` wrapped (the drafter's model too, its decode round handed to the builder); may raise
        ``MonoUnsupported``."""


class _Tk:
    name, hook, requires, falls_back = "tk", Hook.STEP, (), False

    def considered(self, dep):
        from atom.models.deepseek_v41.mono.gfx950.tk import gate

        return None if gate.enabled() else "not ATOM_MONO_ENABLE on gfx950"

    def serves(self, dep):
        from atom.models.deepseek_v41.mono.gfx950.tk.dispatch import install_refusal

        return install_refusal(dep.atom_config, dep.metadata_builder)

    def install(self, model, dep):
        from atom.models.deepseek_v41.mono.gfx950.tk.dispatch import install_tk_step

        return install_tk_step(
            model, dep.atom_config, dep.drafter_obj, dep.metadata_builder
        )


class _Hi:
    name, hook, requires, falls_back = "hi", Hook.LAYER, ("k1k2",), True

    def considered(self, dep):
        return None

    def serves(self, dep):
        cfg, drafter, builder = dep.atom_config, dep.drafter_obj, dep.metadata_builder
        return hi_gate.refusal(cfg, drafter, builder, dep.dp_attention_ranks) or None

    def install(self, model, dep):
        from atom.models.deepseek_v41.mono.hi_dispatch import install_hi

        cfg, drafter, builder = dep.atom_config, dep.drafter_obj, dep.metadata_builder
        return install_hi(model, cfg, drafter, builder, dep.dp_attention_ranks)


class _K1K2:
    name, hook, requires, falls_back = "k1k2", Hook.LAYER, (), False

    def considered(self, dep):
        return None

    def serves(self, dep):
        from atom.models.deepseek_v41.mono.gfx950.k1k2 import config

        return config.config_refusal(dep.atom_config)

    def install(self, model, dep):
        from atom.models.deepseek_v41.mono.gfx950.k1k2 import dispatch

        return dispatch.install(
            model, dep.atom_config, dep.drafter_obj, dep.dp_attention_ranks
        )


PRIORITY: tuple[MonoBackend, ...] = (_Tk(), _Hi(), _K1K2())


def choose(dep: Deployment) -> tuple[MonoBackend, ...]:
    """The backends ``install`` tries for ``dep``, in order (empty: the model stays unwrapped); each one passed over
    logs its line once, or raises where ``ATOM_DSV41_MONO_HI=1`` requires the hi path (module docstring).
    """
    tk, hi, k1k2 = PRIORITY
    if tk.considered(dep) is None:
        why = (
            "ATOM_DSV41_MONO_HI=1 keeps the hi path"
            if Mode.read() is Mode.ON
            else tk.serves(dep)
        )
        if why is None:
            return (tk,)
        hi_gate.log_once(logging.INFO, "V4.1 TileRT-path step off: %s", why)
    why = k1k2.serves(dep)  # (hi's requirement)
    if why is not None:
        if Mode.read() is Mode.ON:
            raise RuntimeError(
                f"ATOM_DSV41_MONO_HI=1, but V4.1 mono decode is off: {why}"
            )
        logger.info("V4.1 mono decode off: %s", why)
        return ()
    mode = Mode.read()
    if mode is Mode.OFF:
        hi_gate.log_once(logging.INFO, "V4.1 mono hi off: ATOM_DSV41_MONO_HI=0")
        return (k1k2,)
    why = hi.serves(dep)
    if why is None:
        return (hi, k1k2)
    if mode is Mode.ON:
        raise RuntimeError(
            f"ATOM_DSV41_MONO_HI=1, but the hi path does not serve this deployment: {why}"
        )
    hi_gate.log_once(logging.INFO, "V4.1 mono hi off: %s; ATOM's mono path", why)
    return (k1k2,)


def install(model, atom_config, drafter, metadata_builder=None):
    """``model`` wrapped by the mono decode that serves this deployment (``drafter``'s draft model too), or ``model``
    itself where none does."""
    from atom.models.deepseek_v41.mono import gfx1250

    if gfx1250.enabled():
        from atom.models.deepseek_v41.mono.gfx1250.dispatch import (
            install_single_step,
        )

        return install_single_step(model, atom_config, drafter)
    dep = Deployment.of(atom_config, drafter, metadata_builder)
    for backend in choose(dep):
        try:
            wrapped = backend.install(model, dep)
        except MonoUnsupported as why:
            # (hi, the one backend that falls back) a refusal only the loaded model shows, agreed over the TP group:
            # every rank falls back alike
            if not backend.falls_back or Mode.read() is Mode.ON:
                raise
            hi_gate.log_once(logging.INFO, "V4.1 mono hi off: %s", why)
            continue
        # ModelRunner tells the hooks apart by type (``StepOverrideModel``)
        assert isinstance(wrapped, StepOverrideModel) == (
            backend.hook is Hook.STEP
        ), backend.name
        return wrapped
    return model
