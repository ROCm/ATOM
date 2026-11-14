# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Enable out-of-tree aiter kernel providers (``aiter.plugin``) for ATOM.

Opt-in and default OFF (``--aiter-provider-enable`` / ``ATOM_AITER_PROVIDER_ENABLE``
-> ``Config.aiter_provider_enable``). ATOM never imports ``aiter.plugin`` at module
scope: an aiter without the provider interface, and every CPU ``import atom``, must
keep working. Not ``atom/plugin/`` (ATOM's vLLM / SGLang / rtp-llm frontend adapter).

Contract with aiter (stock-main ``aiter.plugin``, ``API_VERSION`` 1):

* ``configure(*, source, enable)`` is the only way to enable a provider; ``enable``
  is the ordered preference list. ATOM calls it once per worker process from
  ``ModelRunner.__init__`` -- after device / distributed setup, before the model is
  built, long before warm-up and CUDA-graph capture (aiter snapshots its config at
  the first dispatch and graph replay never re-decides).
* ``AITER_PLUGINS=0`` is aiter's only environment control and can only turn routing
  off; ATOM documents it, never sets it.
* A routed op registers when its aiter module is imported (``aiter.ops.qsa``) or its
  lazy ``aiter.<op>`` export is resolved: resolve before reading
  ``status()["routed_ops"]`` or calling ``explain`` (else ``KeyError``).
* The interface has no lifecycle hook: the framework that enables a provider owns its
  ``warmup()`` (this module calls it before capture, for the providers that declare
  the op).

Everything here is fail-open: a missing / broken provider, an interface version
ATOM was not written against or a routed op this aiter lacks is a log line, and the
caller keeps its non-provider path.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("atom.aiter_provider")

#: aiter.plugin.API_VERSION this module was written against.
SUPPORTED_PLUGIN_API = 1

#: How to make a routed op register: op name -> aiter module that routes it.
ROUTED_OP_MODULES = {"qsa_paged_mqa_logits": "aiter.ops.qsa"}


def atom_version() -> str:
    try:
        from atom._version import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - absent in source checkouts / CPU tests
        logger.debug("atom/_version.py is absent; using package metadata")
    try:
        from importlib.metadata import version

        return version("atom")
    except Exception:  # noqa: BLE001 - not installed, e.g. a bare checkout
        return "unknown"


def normalize_provider_names(value) -> tuple[str, ...]:
    """``None`` / empty -> ``()``; a comma-separated string (CLI / env) or an iterable
    of strings (``Config``) -> stripped, de-duplicated names in preference order."""
    if value is None:
        return ()
    raw = value.split(",") if isinstance(value, str) else list(value)
    names: list[str] = []
    for item in raw:
        if item is None:
            continue
        if not isinstance(item, str):
            raise TypeError(
                f"aiter provider names must be strings, got {type(item).__name__}"
            )
        name = item.strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


@dataclass
class ProviderState:
    """What this process configured (``None`` from :func:`configure_aiter_providers`
    while OFF)."""

    enabled: tuple[str, ...]
    configured: bool = False
    reason: str = ""
    routed_ops: tuple[str, ...] = ()
    #: provider name -> op names it declares (from ``status()["providers"]``)
    provider_ops: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)

    def serves(self, op: str) -> tuple[str, ...]:
        """Enabled providers that declare ``op``, in preference order (empty when
        not configured or the op is not routed in this aiter)."""
        if not self.configured or op not in self.routed_ops:
            return ()
        return tuple(n for n in self.enabled if op in self.provider_ops.get(n, ()))


_STATE: ProviderState | None = None


def state() -> ProviderState | None:
    return _STATE


def _status_dict(status) -> dict:
    try:
        return dict(status)
    except Exception:  # noqa: BLE001
        return {}


def configure_aiter_providers(config, rank_name: str = "") -> ProviderState | None:
    """``aiter.plugin.configure(source=..., enable=...)`` once per process.

    Returns ``None`` while OFF (nothing imported). Never raises."""
    global _STATE
    try:
        enable = normalize_provider_names(getattr(config, "aiter_provider_enable", ()))
    except (TypeError, ValueError) as exc:
        logger.warning("aiter providers: unusable enable list (%s); off", exc)
        return None
    if not enable:
        return None
    if _STATE is not None and _STATE.enabled == enable:
        return _STATE
    st = ProviderState(enabled=enable)
    try:
        plugin = importlib.import_module("aiter.plugin")
    except Exception as exc:  # noqa: BLE001 - an aiter without the interface
        st.reason = f"aiter.plugin unavailable: {exc!r}"
    else:
        api = getattr(plugin, "API_VERSION", None)
        if api != SUPPORTED_PLUGIN_API:
            st.reason = f"aiter.plugin API {api!r} != {SUPPORTED_PLUGIN_API}"
        else:
            for op, module in ROUTED_OP_MODULES.items():
                try:  # registers the route (see the module docstring)
                    importlib.import_module(module)
                except Exception:  # noqa: BLE001 - this aiter lacks the op
                    logger.debug("aiter providers: %s (%s) not importable", op, module)
            try:
                plugin.configure(source=f"atom {atom_version()}", enable=enable)
                status = _status_dict(plugin.status())
            except Exception as exc:  # noqa: BLE001
                st.reason = f"configure()/status() failed: {exc!r}"
            else:
                st.configured, st.status = True, status
                st.routed_ops = tuple(status.get("routed_ops") or ())
                for p in status.get("providers") or ():
                    p = _status_dict(p)
                    st.provider_ops[p.get("name")] = tuple(p.get("op_versions") or ())
    if st.configured:
        missing = [n for n in enable if n not in st.provider_ops]
        logger.info(
            "aiter providers [%s]: enabled=%s routed=%s providers=%s skipped=%s%s",
            rank_name or "-",
            enable,
            st.routed_ops,
            {k: list(v) for k, v in st.provider_ops.items()},
            st.status.get("skipped_providers"),
            f" NOT INSTALLED={missing}" if missing else "",
        )
    else:
        logger.warning(
            "aiter providers [%s]: requested %s but %s; non-provider path",
            rank_name or "-",
            enable,
            st.reason,
        )
    _STATE = st
    return st


def load_provider(name: str):
    """The installed ``aiter.plugins`` entry-point object for ``name`` (an instance;
    a zero-argument factory is called), or None."""
    try:
        from importlib.metadata import entry_points

        matches = [e for e in entry_points(group="aiter.plugins") if e.name == name]
        if not matches:
            return None
        obj = matches[0].load()
        if callable(obj) and not hasattr(obj, "select"):
            obj = obj()
        return obj
    except Exception:
        logger.warning("aiter providers: loading %r failed", name, exc_info=True)
        return None


def warmup_aiter_providers(op: str, payload, *, device=None, rank_name="") -> dict:
    """Call ``warmup(payload, device=...)`` of every enabled provider that declares
    ``op`` (framework-owned lifecycle; before CUDA-graph capture). Never raises.
    Returns ``{provider: bool}``."""
    out: dict = {}
    st = _STATE
    if st is None:
        return out
    for name in st.serves(op):
        provider = load_provider(name)
        warm = getattr(provider, "warmup", None)
        if not callable(warm):
            continue
        try:
            out[name] = bool(warm(payload, device=device))
        except Exception:
            logger.warning("aiter providers: %r warmup() failed", name, exc_info=True)
            out[name] = False
    return out


def _reset_for_testing() -> None:
    global _STATE
    _STATE = None
