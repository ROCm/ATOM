# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Once-per-process capability check of aiter's in-tree QSA index scorer op.

``aiter.ops.qsa_index_scorer`` (ROCm/aiter#7377) is a new coarse op that aiter
defines in-tree; ATOM calls it from the QSA indexer when
``Config.qsa_index_scorer.profile`` is set. This module decides ONCE PER
PROCESS whether that op can serve this worker at all:

1. aiter's module imports (it is absent on released aiter wheels);
2. ``API_VERSION`` major equals :data:`route.CONTRACT_API_MAJOR`;
3. the op's leading parameters are ATOM's scorer signature (no drift);
4. the named profile loads (``load_profile``; its ``kernel_sha256`` is the
   kernel the profile was GPU-checked against);
5. which API-1.1 entry points are usable (``launch`` with a cached decision,
   ``prewarm``, cache-scoped ``prewarm``), and whether aiter's ``Decision``
   carries ``path`` (aiter serves uncovered calls on its own native path).

The worker calls :func:`init_qsa_index_scorer` from ``ModelRunner.__init__``
after ``_setup_device_and_distributed()`` and before
``_build_and_load_model()`` (through :mod:`atom.aiter_fused_ops`), so the
result exists before any QSA layer is constructed; guest frameworks (no ATOM
``ModelRunner``) reach the same cached check lazily from ``build_route``.

Fail-open: every failure is a :class:`Capability` with ``ok=False`` and a
reason; QSA layers then keep ATOM's stock scorer. Nothing here raises, and
aiter is imported only inside :func:`check_capability`.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

from .route import (
    AITER_MODULE,
    AITER_UNAVAILABLE,
    API_VERSION_MISMATCH,
    CAPTURE_ONLY,
    CONTRACT_API_MAJOR,
    CONTRACT_REASONS,
    LAUNCH_API_VERSION,
    OK,
    SIGNATURE_DRIFT,
    api_version_tuple,
    launch_signature_drift,
    signature_drift,
)

logger = logging.getLogger("atom")

#: ATOM-side reasons added by the process check.
PROFILE_UNLOADABLE = "atom_profile_unloadable"  # load_profile raised
CAPABILITY_ERROR = "atom_capability_error"  # the check itself raised


@dataclasses.dataclass(frozen=True)
class Capability:
    """What aiter's QSA index scorer op can do in this process."""

    ok: bool
    reason: str
    detail: str = ""
    profile: str = ""
    api: Any = dataclasses.field(default=None, repr=False, compare=False)
    api_version: str = ""
    #: kernel sha256 the profile was GPU-checked against ("" if unknown).
    kernel_sha256: str = ""
    #: aiter ``launch`` (API >= 1.1, signature checked) or None (1.0 op).
    launch: Callable | None = dataclasses.field(default=None, repr=False)
    #: aiter ``prewarm`` (API >= 1.1) or None.
    prewarm: Callable | None = dataclasses.field(default=None, repr=False)
    #: ``prewarm`` accepts ``compressed_k_cache=`` (builds only the variants
    #: that cache's page size and dtype can select).
    prewarm_scoped: bool = False
    #: aiter ``Decision`` has ``path`` ("fused" | "native"): aiter serves
    #: uncovered calls itself, so ATOM needs no stock fallback for them.
    aiter_native_path: bool = False

    @property
    def fused_call(self) -> str:
        return "launch" if self.launch is not None else "op"


_CAPABILITIES: dict[str, Capability] = {}
_LAUNCH_DRIFT_LOGGED: set[str] = set()


def _decision_has_path(api) -> bool:
    cls = getattr(api, "Decision", None)
    if cls is None:
        return False
    if dataclasses.is_dataclass(cls):
        return any(f.name == "path" for f in dataclasses.fields(cls))
    return hasattr(cls, "path")


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _check(profile: str, api) -> Capability:
    if not profile:
        return Capability(False, CAPTURE_ONLY, "no profile configured", profile)
    if api is None:
        try:
            api = importlib.import_module(AITER_MODULE)
        except Exception as exc:  # noqa: BLE001 -- any failure => stock path
            return Capability(
                False, AITER_UNAVAILABLE, f"import {AITER_MODULE}: {exc!r}", profile
            )
    version = str(getattr(api, "API_VERSION", ""))
    if version.split(".")[0] != CONTRACT_API_MAJOR:
        return Capability(
            False,
            API_VERSION_MISMATCH,
            f"aiter API_VERSION {version!r}, ATOM needs {CONTRACT_API_MAJOR}.x",
            profile,
            api_version=version,
        )
    drift = signature_drift(api)
    if drift is not None:
        return Capability(False, SIGNATURE_DRIFT, drift, profile, api_version=version)
    kernel_sha = ""
    load_profile = getattr(api, "load_profile", None)
    if callable(load_profile):
        try:
            prof = load_profile(profile)
        except Exception as exc:  # noqa: BLE001 -- unloadable profile => stock
            reason = getattr(getattr(exc, "reason", None), "value", None)
            return Capability(
                False,
                reason if reason in CONTRACT_REASONS else PROFILE_UNLOADABLE,
                f"load_profile({profile!r}): {exc}",
                profile,
                api_version=version,
            )
        kernel_sha = str(getattr(prof, "kernel_sha256", "") or "")
    launch = prewarm = None
    if api_version_tuple(version) >= LAUNCH_API_VERSION:
        launch_drift = launch_signature_drift(api)
        if launch_drift is None:
            launch = api.launch
        elif launch_drift not in _LAUNCH_DRIFT_LOGGED:
            # The 1.0 op is still correct (it decides per call): use it.
            _LAUNCH_DRIFT_LOGGED.add(launch_drift)
            logger.warning(
                "QSA index scorer: aiter API_VERSION %r launch unusable (%s); "
                "fused calls use the 1.0 op",
                version,
                launch_drift,
            )
        candidate = getattr(api, "prewarm", None)
        if callable(candidate):
            prewarm = candidate
    return Capability(
        True,
        OK,
        "",
        profile,
        api=api,
        api_version=version,
        kernel_sha256=kernel_sha,
        launch=launch,
        prewarm=prewarm,
        prewarm_scoped=prewarm is not None and _accepts(prewarm, "compressed_k_cache"),
        aiter_native_path=launch is not None and _decision_has_path(api),
    )


def check_capability(profile: str, *, api=None) -> Capability:
    """The capability for ``profile``; never raises.

    With ``api=None`` (production) aiter's module is imported here and the
    result is cached for the process lifetime. An injected ``api`` (CPU
    tests) is checked every time and never cached.
    """
    profile = profile or ""
    if api is None:
        hit = _CAPABILITIES.get(profile)
        if hit is not None:
            return hit
    try:
        cap = _check(profile, api)
    except Exception as exc:  # noqa: BLE001 -- the check must never raise
        cap = Capability(False, CAPABILITY_ERROR, repr(exc), profile)
    if api is None:
        _CAPABILITIES[profile] = cap
    return cap


def index_page_size(config) -> int | None:
    """Index-cache page size the KV block size gives: ``block / ratio``."""
    block = getattr(config, "kv_cache_block_size", None)
    hf = getattr(config, "hf_config", None)
    ratio = getattr(hf, "indexer_compress_ratio", None) if hf is not None else None
    try:
        block, ratio = int(block), int(ratio or 4)
    except (TypeError, ValueError):
        return None
    if ratio <= 0 or block % ratio:
        return None
    return block // ratio


def startup_line(cap: Capability, config=None, rank_name: str = "") -> str:
    """The one startup provenance line a worker logs."""
    page = index_page_size(config) if config is not None else None
    block = getattr(config, "kv_cache_block_size", None)
    if cap.ok:
        uncovered = "aiter-native" if cap.aiter_native_path else "stock"
        prewarm = (
            "none"
            if cap.prewarm is None
            else ("cache-scoped" if cap.prewarm_scoped else "profile")
        )
        state = (
            f"enabled profile={cap.profile} aiter_api={cap.api_version} "
            f"kernel_sha={cap.kernel_sha256[:16] or 'unknown'} "
            f"fused_call={cap.fused_call} prewarm={prewarm} "
            f"uncovered={uncovered}"
        )
    else:
        state = (
            f"DISABLED (stock scorer) profile={cap.profile or '-'} "
            f"reason={cap.reason}"
            + (f" aiter_api={cap.api_version}" if cap.api_version else "")
            + (f": {cap.detail}" if cap.detail else "")
        )
    return (
        f"QSA index scorer [{rank_name or '-'}]: {state} "
        f"kv_block={block} index_page={page}"
    )


def init_qsa_index_scorer(config, rank_name: str = "") -> Capability | None:
    """Worker-process hook (``ModelRunner.__init__``): None while OFF.

    Runs the capability check once and logs one startup line; the result is
    what every QSA layer of this process binds to. Never raises.
    """
    cfg = getattr(config, "qsa_index_scorer", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    cap = check_capability(cfg.profile)
    try:
        line = startup_line(cap, config, rank_name)
    except Exception as exc:  # noqa: BLE001 -- logging must not break startup
        line = f"QSA index scorer [{rank_name}]: {cap.reason} ({exc!r})"
    (logger.info if cap.ok else logger.warning)(line)
    return cap


def _reset_for_testing() -> None:
    _CAPABILITIES.clear()
    _LAUNCH_DRIFT_LOGGED.clear()
