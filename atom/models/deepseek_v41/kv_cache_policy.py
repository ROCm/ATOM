# SPDX-License-Identifier: MIT
"""Report V4.1 KV performance guidance without changing the requested format."""

import logging

logger = logging.getLogger("atom")


def warn_fp4_performance(*, num_heads, attention_tp_size):
    """Advise gfx950 H64/H128 users while preserving their FP4 memory choice.

    ``attention_tp_size`` is the width that shards attention, which is one
    under native DP attention. Called during FP4 configuration validation.
    """
    heads = num_heads // attention_tp_size
    if heads not in (64, 128):
        return

    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        return
    logger.warning(
        "DeepSeek-V4.1 gfx950 H%d: FP4 KV attention performance is still "
        "being optimized. Consider benchmarking --kv_cache_dtype fp8 "
        "(FP8 KV + A8W8) for your workload if memory allows. "
        "Keeping the requested FP4 KV cache and FP4 cache sizing.",
        heads,
    )
