# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


def mxfp4_kv_layout(config):
    """The ``MhaKvPool`` layout of every paged attention layer under
    ``--kv-cache-dtype fp4`` on MiniMax-M3, else None (the dtype-wide cache).

    Read from the TARGET's config, also for its draft: the draft's own hf_config
    does not name M3, but its KV follows the target's flag."""
    from atom.config import _is_minimax_m3_config

    if config.kv_cache_dtype != "fp4" or not _is_minimax_m3_config(
        getattr(config, "hf_config", None)
    ):
        return None
    from atom.model_ops.minimax_m3.mxfp4_pages import Mxfp4KvLayout

    return Mxfp4KvLayout()


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
    spec = config.speculative_config
    if kv_fp4 and spec is not None:
        # A draft keeps the target's fp4 (`Config.draft_config`) and has no fp8 pages to
        # fall back on, so a flavor whose MXFP4 path is not yet GPU-validated refuses
        # here. TEMPORARY: each branch goes when its path is validated.
        if spec.method == "eagle3":
            raise ValueError(
                "--kv-cache-dtype fp4 with EAGLE3 drafts remains disabled pending GPU "
                "validation; prefix caching also needs a safe draft-KV policy before "
                "enabling"
            )
        if spec.use_dspark():
            raise ValueError(
                "--kv-cache-dtype fp4 with DSpark drafts remains disabled pending GPU "
                "validation of the MXFP4 window reader"
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
