# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""

from atom.config import FP4_CACHE_DTYPES


def _mxfp4_kv_refusal(config) -> str | None:
    """Why ``--kv-cache-dtype fp4`` cannot run this MiniMax-M3 here, or None.

    The MXFP4 KV has a writer and a reader only for indexed sparse layers (M3.1: every
    layer), so nothing falls back to FP8 for a layer or page it cannot hold: refused.
    """
    import torch

    if config.torch_dtype != torch.bfloat16:
        # aiter's MXFP4 pa_decode only writes bf16; an fp16 model fails in its first forward.
        return f"it needs a bfloat16 model, got {config.torch_dtype}"
    if config.kv_transfer_config:
        # Offload / P-D transfer moves pages by a layout that does not know this format.
        return "it does not run with KV offload or --kv-transfer-config"
    if config.sequence_parallel_size > 1:
        # Ulysses leaves a top-k reusing layer's qkv without index columns; the fused
        # MXFP4 writer checks the row width and needs them.
        return "it does not run with --sequence-parallel-size > 1"
    spec = config.speculative_config
    if spec is not None and spec.draft_kv_dtype in FP4_CACHE_DTYPES:
        # An FP4 dtype is left only on a draft in the target's pool (MTP), which has no
        # FP4 attention impl to write its rows; an own-pool draft is bf16 and passes.
        return "the speculative draft would have FP4 KV, which it has no layout for"

    text = getattr(config.hf_config, "text_config", None) or config.hf_config
    layers = getattr(text, "num_hidden_layers", None)
    sparse_cfg = getattr(text, "sparse_attention_config", None) or {}
    freq = sparse_cfg.get("sparse_attention_freq")
    # Dense attention is a layer the model's freq marks 0 (M3's original checkpoints
    # have them). Read exactly: one literal 1 a layer, since the model treats any
    # non-zero as indexed sparse and another meaning must not pass as one.
    if (
        not sparse_cfg.get("use_sparse_attention")
        or not isinstance(freq, (list, tuple))
        or not isinstance(layers, int)
        or len(freq) != layers
        or any(type(v) is not int or v != 1 for v in freq)
    ):
        return (
            "it needs every one of num_hidden_layers attention layers to be an "
            "indexed sparse one (sparse_attention_config.use_sparse_attention and "
            "a sparse_attention_freq of exactly 1 per layer, as MiniMax-M3.1); "
            "dense attention layers have no MXFP4 writer or reader"
        )
    return None


def validate_runtime_config(config):
    """Refuse MiniMax-M3's FP4 caches (``--kv-cache-dtype fp4|nvfp4`` and
    ``--index-cache-dtype fp4|nvfp4``) where they cannot run, at startup rather
    than in the first forward. The rules, stated here and in the configuration guide
    and nowhere else; nothing falls back to FP8:

    - both caches: native ATOM (not the vLLM/SGLang bridges), ``--block-size 128``,
      gfx950, no decode or prefill context parallelism (the writer stores a rank's own tokens
      only and the sparse readers take global lengths: M3 has no DCP gather/merge);
    - an FP4 KV also: the same FP4 index cache (the default when omitted, an explicit other value is refused;
      MXFP4 and NVFP4 never pair),
      every attention layer an indexed sparse one (M3.1), a bf16 model, no sequence parallelism, no KV offload
      or ``--kv-transfer-config``, and no speculative draft left with FP4 KV.
    """
    from atom.config import is_plugin_mode

    kv, index = config.kv_cache_dtype, config.index_cache_dtype
    if kv not in FP4_CACHE_DTYPES and index not in FP4_CACHE_DTYPES:
        return
    if is_plugin_mode():
        raise ValueError(
            "FP4 (MXFP4 / NVFP4) KV / index caches do not run under the "
            "vLLM/SGLang bridges"
        )
    if config.decode_context_parallel_size > 1:
        raise ValueError(
            "FP4 KV / index caches do not run with --decode-context-parallel-size "
            f"{config.decode_context_parallel_size} (> 1): M3's sparse readers have "
            "no DCP gather/merge over the rank-local keys"
        )
    if config.prefill_context_parallel_size > 1:
        # a PCP rank's q rows are a shard of what the sparse metadata covers, which
        # `_metadata_rows` asserts against (with or without TBO)
        raise ValueError(
            "FP4 KV / index caches do not run with --prefill-context-parallel-size "
            f"{config.prefill_context_parallel_size} (> 1)"
        )
    if kv in FP4_CACHE_DTYPES and index != kv:
        # `sparse_pa_{mxfp4,nvfp4}` walk every row of q; only the matching FP4 index
        # scorer is sized to those rows, the other index caches' selectors can leave
        # some unwritten. An FP4 index under a different FP4 KV is no pairing either.
        raise ValueError(
            f"--kv-cache-dtype {kv} needs --index-cache-dtype {kv}, got {index}"
        )
    if index == "nvfp4" and kv not in ("nvfp4", "bf16", "fp8"):
        raise ValueError(f"--index-cache-dtype nvfp4 does not pair with KV {kv}")
    if kv in FP4_CACHE_DTYPES:
        why = _mxfp4_kv_refusal(config)
        if why is not None:
            raise ValueError(f"--kv-cache-dtype {kv}: {why}")
    block_size = config.kv_cache_block_size
    from atom.model_ops.minimax_m3.geometry import SPARSE_BLOCK_SIZE

    if block_size != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"FP4 pages are {SPARSE_BLOCK_SIZE} tokens; --block-size is {block_size}"
        )
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"FP4 KV / index caches need gfx950, got {arch}")
