# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Model-runner entry points for GEMM fusion. Default OFF.

Called from ``ModelRunner`` (never from model files). With no profile set every
function is a no-op, so OFF behavior is unchanged.
"""

import logging
from typing import Optional

logger = logging.getLogger("atom.gemm_fusion")

_SITES: dict = {}
_RES = None  # FusionResourcesWrapper | None
_PROFILE: str = ""
DECISIONS: list = []  # bounded dispatch evidence, newest last
_MAX_DECISIONS = 4096


def profile_enabled(config=None) -> bool:
    if config is None:
        try:
            from atom.config import get_current_atom_config

            config = get_current_atom_config()
        except Exception:
            return False
    gf = getattr(config, "gemm_fusion", None)
    return bool(gf is not None and gf.profile)


def register_site(site) -> None:
    _SITES[site.name] = site


def sites() -> dict:
    return _SITES


def resources():
    return _RES


def world_or_1() -> int:
    return _RES.world if _RES is not None else 1


def record_decision(d) -> None:
    DECISIONS.append(d)
    if len(DECISIONS) > _MAX_DECISIONS:
        del DECISIONS[: len(DECISIONS) - _MAX_DECISIONS]


def init_gemm_fusion(config, sp_group, resources=None) -> None:
    """After distributed + SP group init, before model construction/KV alloc."""
    global _RES, _PROFILE
    if not profile_enabled(config):
        return
    from .resources import FusionResourcesWrapper
    from .stages import aiter_gf, sp_world

    gf = aiter_gf()
    _PROFILE = config.gemm_fusion.profile
    gf.load_profile(_PROFILE)  # fail fast on a missing/invalid profile
    _RES = resources or FusionResourcesWrapper(sp_group, sp_world())
    logger.info("gemm_fusion ON profile=%s world=%d", _PROFILE, _RES.world)


def bind_kv_cache(kv_cache_data) -> None:
    for s in _SITES.values():
        s.bind_caches(kv_cache_data)


def before_graph_capture(buckets) -> None:
    """Resolve every bucket on every rank BEFORE capture, then clean epoch."""
    if _RES is None:
        return
    for s in _SITES.values():
        for t in buckets:
            s.resolve(t, graph_mode="capture")
    _RES.quiesce_and_reset()


def after_graph_capture() -> None:
    if _RES is not None:
        _RES.quiesce_and_reset()


def recover() -> None:
    """After a poison/error raised by a fused op."""
    if _RES is not None:
        _RES.quiesce_and_reset()


def shutdown() -> None:
    global _RES
    if _RES is not None:
        _RES.release()
        _RES = None
    _SITES.clear()
    DECISIONS.clear()


def reset_for_tests(res: Optional[object] = None) -> None:
    global _RES, _PROFILE
    _SITES.clear()
    DECISIONS.clear()
    _RES = res
    _PROFILE = ""
