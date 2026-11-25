# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's startup checks on a ``Config``, run from ``Config.__post_init__``
the way DeepSeek-V4.1's are (``models/deepseek_v41/config.py``). GPU-free: it
imports neither the model nor AITER until the architecture check needs it."""

from atom.config import FP4_CACHE_DTYPES

# An NVFP4 page is self-contained (data and e4m3 scales inline, the per-layer global
# K/V scale a constant 1.0) and an NVFP4 index entry is one page: the in-process
# LMCache offload moves both as opaque per-block bytes.
NVFP4_PAGE_CONNECTORS = frozenset({"lmcache_offload"})

# A draft whose KV dtype differs from the target's has been verified (save and restore)
# only as a DSpark draft under these, whatever NVFP4_PAGE_CONNECTORS grows to take.
DRAFT_DTYPE_CONNECTORS = frozenset({"lmcache_offload"})


def _leaves_within(config, names) -> bool:
    """Whether every configured transport, ``multi`` expanded, is one of ``names``."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    try:
        leaves = KVConnectorFactory.leaf_connectors(config)
    except ValueError:  # an unknown connector: refused by the caller
        return False
    return bool(leaves) and all(name in names for name, _ in leaves)


def _connectors_move_nvfp4_pages(config) -> bool:
    """Whether every configured transport can move NVFP4 pages."""
    return _leaves_within(config, NVFP4_PAGE_CONNECTORS)


def nvfp4_pages_move(config) -> bool:
    """Whether this config's NVFP4 KV and index pages move under every transport it
    runs."""
    return config.kv_cache_dtype == "nvfp4" and _connectors_move_nvfp4_pages(config)


def draft_dtype_moves(config) -> bool:
    """Whether an own-pool draft may keep a KV dtype other than the target's under this
    config's transports: a DSpark draft beside an NVFP4 KV, every transport in
    ``DRAFT_DTYPE_CONNECTORS``."""
    spec = config.speculative_config
    return (
        config.kv_cache_dtype == "nvfp4"
        and spec is not None
        and spec.method == "dspark"
        and _leaves_within(config, DRAFT_DTYPE_CONNECTORS)
    )


def _fp4_kv_refusal(config) -> str | None:
    """Why ``--kv-cache-dtype fp4|nvfp4`` cannot run this MiniMax-M3 here, or None.

    The FP4 (MXFP4 / NVFP4) KV has a writer and a reader only for indexed sparse layers
    (M3.1: every layer), so nothing falls back to FP8 for a layer or page it cannot
    hold: refused.
    """
    import torch

    if config.torch_dtype != torch.bfloat16:
        # aiter's FP4 writers and sparse PA only handle bf16; an fp16 model fails in its
        # first forward.
        return f"it needs a bfloat16 model, got {config.torch_dtype}"
    if config.kv_transfer_config and not nvfp4_pages_move(config):
        # Not verified: MXFP4 pages under any transport, NVFP4 pages under one
        # outside NVFP4_PAGE_CONNECTORS.
        if config.kv_cache_dtype == "nvfp4":
            return (
                "it runs with KV offload or --kv-transfer-config only under "
                + ", ".join(sorted(NVFP4_PAGE_CONNECTORS))
            )
        return "it does not run with KV offload or --kv-transfer-config"
    if config.sequence_parallel_size > 1:
        # Ulysses leaves a top-k reusing layer's qkv without index columns; the fused
        # FP4 writer checks the row width and needs them.
        return "it does not run with --sequence-parallel-size > 1"
    spec = config.speculative_config
    if spec is not None and spec.draft_kv_dtype in FP4_CACHE_DTYPES:
        # An FP4 dtype (either) is left only on a draft in the target's pool (MTP),
        # which has no FP4 attention impl to write its rows; an own-pool draft is bf16
        # and passes.
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
            "dense attention layers have no FP4 writer or reader"
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
      or ``--kv-transfer-config`` (an NVFP4 KV excepted where every transport, ``multi`` expanded, is in
      ``NVFP4_PAGE_CONNECTORS``), and no speculative draft left with FP4 KV.
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
    if index == "nvfp4" and kv != "nvfp4":
        # the NVFP4 index feeds the NVFP4 sparse attention only: there is no emit of
        # the page-16 table from it, and no run of it beside another KV is wired
        raise ValueError(
            f"--index-cache-dtype nvfp4 needs --kv-cache-dtype nvfp4, got {kv}"
        )
    if kv in FP4_CACHE_DTYPES:
        why = _fp4_kv_refusal(config)
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
