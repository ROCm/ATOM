# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


def validate_runtime_config(config):
    """Refuse MiniMax-M3's MXFP4 index cache (``--index-cache-dtype fp4``) where it
    cannot run, at startup rather than in the first forward."""
    from atom.config import is_plugin_mode

    if config.index_cache_dtype != "fp4":
        return
    if is_plugin_mode():
        raise ValueError(
            "the MXFP4 index cache does not run under the vLLM/SGLang bridges"
        )
    block_size = config.kv_cache_block_size
    if block_size != 128:
        raise ValueError(
            f"MXFP4 index pages are 128 tokens; --block-size is {block_size}"
        )
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"the MXFP4 index cache needs gfx950, got {arch}")
