# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


def dense_kv_cache_dtype(kv_cache_dtype: str) -> str:
    """The KV dtype of a layer that is not MXFP4: ``--kv-cache-dtype fp4`` makes only
    the indexed sparse layers' pages MXFP4, the rest stay fp8."""
    return "fp8" if kv_cache_dtype == "fp4" else kv_cache_dtype


def validate_runtime_config(config):
    """Refuse MiniMax-M3's MXFP4 caches where they cannot run, at startup rather
    than in the first forward (``--kv-cache-dtype fp4`` and
    ``--index-cache-dtype fp4``)."""
    import torch

    from atom.config import is_plugin_mode

    kv_fp4 = config.kv_cache_dtype == "fp4"
    if not (kv_fp4 or config.index_cache_dtype == "fp4"):
        return
    if is_plugin_mode():
        raise ValueError(
            "MXFP4 KV / index caches do not run under the vLLM/SGLang bridges"
        )
    if kv_fp4 and config.torch_dtype != torch.bfloat16:
        # aiter's MXFP4 pa_decode only writes bf16; the attention output takes the
        # model dtype, so an fp16 checkpoint would fail in its first forward.
        raise ValueError(
            "--kv-cache-dtype fp4 needs a bfloat16 model, " f"got {config.torch_dtype}"
        )
    block_size = config.kv_cache_block_size
    if block_size != 128:
        raise ValueError(f"MXFP4 pages are 128 tokens; --block-size is {block_size}")
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"MXFP4 KV / index caches need gfx950, got {arch}")
