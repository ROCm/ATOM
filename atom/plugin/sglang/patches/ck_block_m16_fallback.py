"""Skip tuned CK MoE rows whose tile the heuristic dispatcher rejects.

AITER's CK 2-stage heuristic dispatches ``block_m`` in {32, 64, 128, 256}.
The gfx942 tuned table for Qwen3.5-35B-A3B per-token FP8 (the PTPC
checkpoint, token bucket 256, 2048x512, 257 experts) selects
``moe_ck2stages_gemm1_256x16x64x256``. CUDA-graph capture then dies with
``Unsupported block_m value for moe heuristic dispatch: 16``.

Re-query that shape with ``AITER_BYPASS_TUNE_CONFIG`` so the untuned path
runs. Per-token FP8 with token>16 is the asm 1-stage kernel; the 2-stage
heuristic only returns 32/64/128. Other tuned rows are left alone.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("atom")

_INSTALL_FLAG = "_atom_ck_block_m16_fallback"


def _stage1_kernel_name(metadata: Any) -> str:
    stage1 = getattr(metadata, "stage1", None)
    keywords = getattr(stage1, "keywords", None)
    if isinstance(keywords, dict):
        return str(keywords.get("kernelName") or "")
    return ""


def _is_unsupported_ck_block_m16(metadata: Any) -> bool:
    """True for a tuned CK 2-stage row whose tile is block_m=16."""
    if getattr(metadata, "run_1stage", False):
        return False
    if getattr(metadata, "full_impl", None) is not None:
        return False
    try:
        block_m = int(metadata.block_m)
    except (TypeError, ValueError, AttributeError):
        return False
    if block_m != 16:
        return False
    name = _stage1_kernel_name(metadata)
    return "moe_ck2stages" in name and "x16x" in name


def _query_without_tuned_row(original, args, kwargs):
    # get_2stage_cfgs is lru-cached, and the cache key does not include
    # AITER_BYPASS_TUNE_CONFIG. Drop the cached block_m=16 row or the
    # re-query returns it again.
    cache_clear = getattr(original, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    previous = os.environ.get("AITER_BYPASS_TUNE_CONFIG")
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
    try:
        return original(*args, **kwargs)
    finally:
        if previous is None:
            os.environ.pop("AITER_BYPASS_TUNE_CONFIG", None)
        else:
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = previous


def bypass_unsupported_ck_block_m(original, *args, **kwargs):
    """Return ``original`` unless it picked the unsupported block_m=16 tile."""
    metadata = original(*args, **kwargs)
    if not _is_unsupported_ck_block_m16(metadata):
        return metadata
    logger.info(
        "CK MoE tuned row uses block_m=16 (%s); "
        "AITER heuristic dispatch has no 16-tile, using the untuned kernel",
        _stage1_kernel_name(metadata),
    )
    return _query_without_tuned_row(original, args, kwargs)


def apply_ck_block_m16_fallback() -> None:
    """Install the block_m=16 fallback on aiter's fused-MoE config lookup."""
    try:
        import aiter.fused_moe as fused_moe_mod
    except Exception:  # noqa: BLE001 - aiter is absent in CPU unit tests
        return
    if getattr(fused_moe_mod, _INSTALL_FLAG, False):
        return
    original = fused_moe_mod.get_2stage_cfgs

    def get_2stage_cfgs(*args, **kwargs):
        return bypass_unsupported_ck_block_m(original, *args, **kwargs)

    fused_moe_mod.get_2stage_cfgs = get_2stage_cfgs
    setattr(fused_moe_mod, _INSTALL_FLAG, True)
