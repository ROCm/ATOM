# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


def validate_runtime_config(config):
    """Refuse MiniMax-M3's MXFP4 caches where they cannot run, at startup rather
    than in the first forward (``--kv-cache-dtype fp4``, held as ``kv_mxfp4``, and
    ``--index-cache-dtype fp4``)."""
    from atom.config import is_plugin_mode

    if not (config.kv_mxfp4 or config.index_cache_dtype == "fp4"):
        return
    if is_plugin_mode():
        raise ValueError(
            "MXFP4 KV / index caches do not run under the vLLM/SGLang bridges"
        )
    block_size = config.kv_cache_block_size
    if block_size != 128:
        raise ValueError(f"MXFP4 pages are 128 tokens; --block-size is {block_size}")
    text_cfg = getattr(config.hf_config, "text_config", config.hf_config)
    sparse_cfg = getattr(text_cfg, "sparse_attention_config", None) or {}
    sparse_local_block = sparse_cfg.get("sparse_local_block", 0)
    if config.kv_mxfp4 and sparse_local_block < 1:
        # The decode plan is refreshed once a step, which holds only while every
        # layer's rows have the same lengths -- i.e. the tail is always picked.
        raise ValueError(
            "--kv-cache-dtype fp4 needs sparse_local_block >= 1, "
            f"got {sparse_local_block}"
        )
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"MXFP4 KV / index caches need gfx950, got {arch}")
