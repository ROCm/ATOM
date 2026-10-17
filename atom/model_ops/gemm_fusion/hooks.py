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


_KV = None  # bound kv_cache_data, needed to prepare plans before capture


def bind_kv_cache(kv_cache_data) -> None:
    global _KV
    _KV = kv_cache_data
    for s in _SITES.values():
        s.bind_caches(kv_cache_data)


def before_graph_capture(buckets, device=None, out_dtype=None) -> None:
    """Outside capture, on every rank, in this order:
    resolve every bucket (windows ``ensure``d, consensus) -> ``finalize`` ->
    ``plan.prepare`` for buckets the kernel can take -> clean epoch.

    ``buckets``: padded token counts, or ``(padded, real)`` pairs. The small-M
    kernel needs every row live, so only ``padded == real`` buckets are
    prepared; others still resolve (rank-uniform) and fall back at run time
    with reason ``pad_rows``.
    """
    if _RES is None:
        return
    pairs = [b if isinstance(b, tuple) else (b, b) for b in buckets]
    for s in _SITES.values():
        for padded, _ in pairs:
            s.resolve(padded, graph_mode="capture")
    _RES.finalize()
    if _KV is not None:
        if device is None:
            import torch

            device = torch.device("cuda", torch.cuda.current_device())
        if out_dtype is None:
            from atom.config import get_current_atom_config

            out_dtype = get_current_atom_config().torch_dtype
        for s in _SITES.values():
            for padded, real in pairs:
                if padded == real:
                    s.prepare_bucket(padded, device, _KV, out_dtype)
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
    global _RES, _PROFILE, _KV
    _SITES.clear()
    DECISIONS.clear()
    _KV = None
    _RES = res
    _PROFILE = ""
