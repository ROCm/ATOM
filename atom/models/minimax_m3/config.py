# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


# The FP4 cache dtypes: "fp4" is MXFP4 (e2m1 + e8m0 per 32), "nvfp4" is NVFP4
# (e2m1 + e4m3 per 16, single-buffer K/V pages). Both are one page layout for every
# attention layer, 128-token pages, gfx950, bf16, no offload: one set of refusals.
_FP4_DTYPES = ("fp4", "nvfp4")


def _mxfp4_kv_refusal(config) -> str | None:
    """Why ``--kv-cache-dtype fp4`` cannot run this MiniMax-M3 here, or None.

    The MXFP4 KV is one page layout for every attention layer of the model, and a
    writer and a reader that exist only for the indexed sparse layers (MiniMax-M3.1:
    every layer). Nothing quietly falls back to FP8 for a layer or an offloaded page
    it cannot hold, so each such case is refused here instead.
    """
    import torch

    from atom.config import kv_pages_leave_pool

    if config.torch_dtype != torch.bfloat16:
        # aiter's MXFP4 pa_decode only writes bf16; the attention output takes the
        # model dtype, so an fp16 checkpoint would fail in its first forward.
        return f"it needs a bfloat16 model, got {config.torch_dtype}"
    if kv_pages_leave_pool(config):
        # Offloaded pages (ATOM_KV_OFFLOAD, a P/D connector) are moved by a layout
        # that was never told this byte format, and its namespace keys no FP4.
        return "it does not run with KV offload or --kv-transfer-config"
    spec = config.speculative_config
    if spec is not None and spec.kv_cache_dtype in _FP4_DTYPES:
        # `resolve_draft_kv_dtype` leaves "fp4" only on a draft in the target's
        # pool (EAGLE3, MTP): one with no MXFP4 attention impl to write its rows.
        # A DSpark draft with its own pool is fp8 / bf16 and passes.
        return "the speculative draft would have FP4 KV, which it has no layout for"

    text = getattr(config.hf_config, "text_config", None) or config.hf_config
    layers = getattr(text, "num_hidden_layers", None)
    sparse_cfg = getattr(text, "sparse_attention_config", None) or {}
    freq = sparse_cfg.get("sparse_attention_freq")
    # Dense attention is a layer the model's own freq marks 0, and M3's original
    # checkpoints have them. Read off that list alone -- an MLP layout says nothing
    # about the attention -- and exactly: one entry a layer, each a literal 1,
    # because the model treats any non-zero as indexed sparse and a value of
    # another meaning must not pass as one.
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
    """Refuse MiniMax-M3's FP4 caches where they cannot run, at startup rather
    than in the first forward (``--kv-cache-dtype fp4|nvfp4`` and
    ``--index-cache-dtype fp4|nvfp4``)."""
    from atom.config import is_plugin_mode

    kv, index = config.kv_cache_dtype, config.index_cache_dtype
    if kv not in _FP4_DTYPES and index not in _FP4_DTYPES:
        return
    if is_plugin_mode():
        raise ValueError(
            "FP4 (MXFP4 / NVFP4) KV / index caches do not run under the "
            "vLLM/SGLang bridges"
        )
    if kv in _FP4_DTYPES and index != kv:
        # `sparse_pa_{mxfp4,nvfp4}` walk every row of q; only the matching FP4 index
        # scorer is sized to those rows, the other index caches' selectors can leave
        # some unwritten. An FP4 index under a different FP4 KV is no pairing either.
        raise ValueError(
            f"--kv-cache-dtype {kv} needs --index-cache-dtype {kv}, got {index}"
        )
    if index == "nvfp4" and kv not in ("nvfp4", "bf16", "fp8"):
        raise ValueError(f"--index-cache-dtype nvfp4 does not pair with KV {kv}")
    if kv in _FP4_DTYPES:
        why = _mxfp4_kv_refusal(config)
        if why is not None:
            raise ValueError(f"--kv-cache-dtype {kv}: {why}")
    block_size = config.kv_cache_block_size
    if block_size != 128:
        raise ValueError(f"FP4 pages are 128 tokens; --block-size is {block_size}")
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"FP4 KV / index caches need gfx950, got {arch}")
