# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether V4.1 mono decode's layers run on AITER's routed decode-layer ops
(``ATOM_DSV41_MONO_BACKEND``), and how a call's route is checked.

- "atom" (the default): ATOM's in-tree K1 / K2 (``runner``), unchanged.
- "aiter": ``aiter_runner`` calls ``deepseek_v41_decode_attn_pre`` and
  ``deepseek_v41_decode_layer_post`` (AITER's v1 contracts) for every target
  layer, and the provider ``ATOM_DSV41_MONO_AITER_PROVIDER`` is enabled with
  ``aiter.plugin.configure()``. AITER dispatches each call to that provider or,
  when it declines, to the contracts' pure-torch native reference: correct,
  but orders of magnitude slower.

Every input of a decision here is the same on every TP rank: the environment,
ATOM's configuration, the installed AITER and its provider list, and (a
width's preflight, ``aiter_runner.prepare``) the provider's ``select()`` on
shapes and dtypes, agreed over the TP group. The provider's kernels wait on
every peer rank, so a rank routed alone would wait on the others.

``ATOM_DSV41_MONO_AITER_STRICT`` (default "1") makes every "not on the
provider" an error naming why: a startup refusal, a preflight decline, a call
whose recorded route (``aiter.plugin.last_decision()``) was not the provider.
"0" logs once and lets AITER fall back.

Reads AITER lazily: importing this module needs neither AITER nor a GPU.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass
from typing import Any, Callable

from atom.models.deepseek_v41.mono.aiter_args import ATTN_PRE, LAYER_POST
from atom.utils import envs

logger = logging.getLogger("atom")

BACKENDS = ("atom", "aiter")
OPS = (ATTN_PRE, LAYER_POST)
OP_VERSION = 1
# ``configure(source=...)``: the provenance AITER records on every route
SOURCE = "ATOM deepseek_v41 mono decode (ATOM_DSV41_MONO_BACKEND=aiter)"
ENTRY_POINT_GROUP = "aiter.plugins"


def backend() -> str:
    """``ATOM_DSV41_MONO_BACKEND``, validated."""
    value = str(envs.ATOM_DSV41_MONO_BACKEND).strip().lower()
    if value not in BACKENDS:
        raise ValueError(
            f"ATOM_DSV41_MONO_BACKEND={envs.ATOM_DSV41_MONO_BACKEND!r}: one of "
            + ", ".join(BACKENDS)
        )
    return value


def selected() -> bool:
    return backend() == "aiter"


def strict() -> bool:
    return bool(envs.ATOM_DSV41_MONO_AITER_STRICT)


def provider_name() -> str:
    return str(envs.ATOM_DSV41_MONO_AITER_PROVIDER).strip()


@dataclass(frozen=True)
class AiterOps:
    """AITER's two routed ops and its plugin module (``aiter.plugin``)."""

    attn_pre: Callable[..., None]
    layer_post: Callable[..., None]
    plugin: Any

    def call(self, op: str) -> Callable[..., None]:
        return self.attn_pre if op == ATTN_PRE else self.layer_post


class Unavailable(Exception):
    """This AITER cannot route the V4.1 decode-layer ops."""


def load_ops() -> AiterOps:
    """The installed AITER's ops, or ``Unavailable`` naming what is missing."""
    try:
        plugin = importlib.import_module("aiter.plugin")
    except ImportError as error:
        raise Unavailable(f"aiter has no aiter.plugin ({error})") from error
    try:
        ops = importlib.import_module("aiter.ops.deepseek_v41_decode")
    except ImportError as error:
        raise Unavailable(
            f"aiter does not route the V4.1 decode-layer ops ({error})"
        ) from error
    return AiterOps(
        ops.deepseek_v41_decode_attn_pre, ops.deepseek_v41_decode_layer_post, plugin
    )


def refusal(atom_config, dp_ranks: int, ops: AiterOps | None) -> str | None:
    """Why the "aiter" backend cannot serve this deployment, or None.

    On top of ``config.config_refusal`` (which already passed): the contracts
    are TP only (no DP attention, no expert parallel), and AITER has to route
    the ops with the provider installed and not switched off. ``ops`` None:
    ``load_ops`` failed (the caller names why)."""
    if dp_ranks:
        return f"DP attention over {dp_ranks} ranks (the contracts are TP only)"
    if getattr(atom_config, "enable_expert_parallel", False):
        return "expert parallel (the contracts are TP only)"
    if ops is None:
        return "aiter cannot route the ops"
    try:
        status = ops.plugin.status()
    except Exception as error:  # noqa: BLE001 - a broken plugin layer is a refusal
        return f"aiter.plugin.status() failed: {error}"
    if status.get("kill_switch"):
        return "AITER_PLUGINS=0 turns every provider off"
    name = provider_name()
    installed = [p.get("name") for p in status.get("providers", ())]
    if name not in installed:
        return (
            f"provider {name!r} is not installed (aiter sees "
            f"{installed or 'none'}; skipped: "
            f"{[s.get('entry_point') for s in status.get('skipped_providers', ())]})"
        )
    return None


def configure(ops: AiterOps) -> tuple[str, ...]:
    """Enable the provider through ``aiter.plugin.configure()``, keeping any
    provider already enabled ahead of it (``configure`` replaces the list).
    Returns the enable list in force."""
    name = provider_name()
    enabled = tuple(ops.plugin.status().get("enabled", ()))
    if name not in enabled:
        ops.plugin.configure(source=SOURCE, enable=[*enabled, name])
        enabled = tuple(ops.plugin.status().get("enabled", ()))
    return enabled


def provider_reason(ops: AiterOps, op: str, kwargs: dict) -> str | None:
    """The provider's own decline class (its ``fallback_reason``, when it has
    one: ``NO_SCHEMA`` / ``NO_WORKLOAD`` / ``OUT_OF_CONTRACT`` for the mono
    provider), found through its entry point. Never raises."""
    try:
        name = provider_name()
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if entry.name != name:
                continue
            loaded = entry.load()
            provider = loaded if hasattr(loaded, "select") else loaded()
            explain = getattr(provider, "fallback_reason", None)
            if explain is None:
                return None
            return explain(ops.plugin.Op(op, OP_VERSION), (), dict(kwargs))
    except Exception:  # noqa: BLE001 - a diagnostic only
        return None
    return None


def explain(ops: AiterOps, op: str, kwargs: dict) -> tuple[bool, str]:
    """``(served, why)``: whether AITER would hand this call to the provider
    (``aiter.plugin.explain``, which runs no kernel), and a line saying so."""
    name = provider_name()
    try:
        found = ops.plugin.explain(op, **kwargs)
    except Exception as error:  # noqa: BLE001 - a malformed call is a refusal
        return False, f"explain failed: {type(error).__name__}: {error}"
    decision = found.decision
    if decision is None:
        return False, "AITER_PLUGINS=0: native"
    if decision.route == "provider" and decision.provider == name:
        return True, f"{name} {decision.kernel_id}"
    detail = dict(found.declines).get(name, "not asked")
    fine = provider_reason(ops, op, kwargs)
    via = f" via {decision.provider}" if decision.provider else ""
    return False, (
        f"{decision.route}{via} ({decision.reason}); {name}: {detail}"
        + (f" [{fine}]" if fine else "")
    )


def routed(ops: AiterOps, op: str) -> tuple[bool, str]:
    """``(served, why)`` of the call to ``op`` that just ran, by AITER's record
    of it (``last_decision``): a ContextVar read, cheap enough a call."""
    name = provider_name()
    decision = ops.plugin.last_decision()
    if decision is None:
        return False, "no route recorded (no provider enabled, or AITER_PLUGINS=0)"
    if decision.op.name != op:
        return False, f"the last route recorded is {decision.op.name}'s"
    if decision.route == "provider" and decision.provider == name:
        return True, f"{name} {decision.kernel_id}"
    return False, f"{decision.route} ({decision.reason})"
