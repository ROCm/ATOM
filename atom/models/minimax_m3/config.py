# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""


def _reuses_index_topk(text, layers: int) -> bool:
    """Whether any layer reuses an earlier layer's index top-k. The rule of
    ``models/minimax_m3/model.py:_should_skip_minimax_m3_index_topk`` for a model
    whose every layer is indexed sparse (a layer's ordinal is then its id), kept
    here so the startup gate imports no model, torch graph or AITER; a test holds
    the two together where the model imports."""
    if not getattr(text, "use_index_cache", False):
        return False
    pattern = getattr(text, "index_topk_pattern", None)
    if pattern is not None:
        return any(p == "S" for p in pattern[:layers])
    freq = int(getattr(text, "index_topk_freq", 1) or 1)
    if freq <= 0:
        raise ValueError("index_topk_freq must be a positive integer")
    offset = int(getattr(text, "index_skip_topk_offset", 0))
    return freq > 1 and any(max(i - offset, 0) % freq != 0 for i in range(layers))


def _mxfp4_kv_refusal(config) -> str | None:
    """Why ``--kv-cache-dtype fp4`` cannot run this MiniMax-M3 here, or None.

    The MXFP4 KV is one page layout for every attention layer of the model, and a
    writer and a reader that exist only for the indexed sparse layers (MiniMax-M3.1:
    every layer). Nothing quietly falls back to FP8 for a layer or an offloaded page
    it cannot hold, so each such case is refused here instead.
    """
    import torch

    from atom.config import kv_offload_requested

    if config.torch_dtype != torch.bfloat16:
        # aiter's MXFP4 pa_decode only writes bf16; the attention output takes the
        # model dtype, so an fp16 checkpoint would fail in its first forward.
        return f"it needs a bfloat16 model, got {config.torch_dtype}"
    if kv_offload_requested(config):
        # Offloaded pages (ATOM_KV_OFFLOAD, a P/D connector) are moved by a layout
        # that was never told this byte format, and its namespace keys no FP4.
        return "it does not run with KV offload or --kv-transfer-config"
    spec = config.speculative_config
    if spec is not None and getattr(spec, "kv_cache_dtype", None) == "fp4":
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
    # A layer that reuses another's top-k is written without an index branch, by a
    # Triton writer that only produces the SHUFFLE cache. Asked of the model's rule
    # (mirrored above), not of the `use_index_cache` flag: that alone skips nothing.
    if _reuses_index_topk(text, layers):
        return (
            "a sparse layer that reuses another's index top-k has no MXFP4 K/V writer"
        )
    return None


def validate_runtime_config(config):
    """Refuse MiniMax-M3's MXFP4 caches where they cannot run, at startup rather
    than in the first forward (``--kv-cache-dtype fp4`` and
    ``--index-cache-dtype fp4``)."""
    from atom.config import is_plugin_mode

    kv_fp4 = config.kv_cache_dtype == "fp4"
    if not (kv_fp4 or config.index_cache_dtype == "fp4"):
        return
    if is_plugin_mode():
        raise ValueError(
            "MXFP4 KV / index caches do not run under the vLLM/SGLang bridges"
        )
    if kv_fp4:
        why = _mxfp4_kv_refusal(config)
        if why is not None:
            raise ValueError(f"--kv-cache-dtype fp4: {why}")
    block_size = config.kv_cache_block_size
    if block_size != 128:
        raise ValueError(f"MXFP4 pages are 128 tokens; --block-size is {block_size}")
    # The live GPU, not `get_gfx()`, which follows GPU_ARCHS (a build setting).
    from aiter.jit.utils.chip_info import get_gfx_runtime

    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise ValueError(f"MXFP4 KV / index caches need gfx950, got {arch}")
