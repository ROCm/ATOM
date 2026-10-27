# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM-side constructors for ``aiter.gemm_fusion`` spec objects.

aiter owns the schema (spec.py); this module only reads ATOM layer objects and
builds those descriptors. ``aiter.gemm_fusion`` is imported lazily so ATOM
still imports (and the OFF path still runs) without it.
"""

import importlib


def aiter_gf():
    """The ``aiter.gemm_fusion`` module."""
    return importlib.import_module("aiter.gemm_fusion")


def sp_world() -> int:
    from atom.distributed.ulysses_sp import get_sp_world_size

    return get_sp_world_size()


def qkv_fields(
    *,
    q: int,
    kv: int,
    head_dim: int,
    iq: int = 0,
    ik: int = 0,
    idx_head_dim: int = 0,
    world: int | None = None,
):
    """Column ranges of the (tp=1) fused QKV[+indexer] GEMM output.

    Shard counts follow ``ulysses_attention``: q scatters over the world; k/v
    (and iq) use ``min(world, width // head_dim)`` shards (GQA replication when
    fewer than ``world``); ik is replicated.
    """
    gf = aiter_gf()
    w = sp_world() if world is None else world
    kv_shards = min(w, kv // head_dim)
    fields = [
        gf.Field("q", 0, q, w),
        gf.Field("k", q, kv, kv_shards),
        gf.Field("v", q + kv, kv, kv_shards),
    ]
    if iq:
        iq_shards = min(w, iq // idx_head_dim)
        fields.append(gf.Field("iq", q + 2 * kv, iq, iq_shards))
        fields.append(gf.Field("ik", q + 2 * kv + iq, ik, 1))
    return tuple(fields)


def sp_exchange(fields, drop=(), world: int | None = None):
    gf = aiter_gf()
    return gf.SPExchange(
        world=sp_world() if world is None else world,
        fields=tuple(fields),
        drop=tuple(drop),
    )


def _kv_block_size() -> int:
    """The dense layers' real KV page size (config.kv_cache_block_size).

    No silent default: the page is part of the fusion signature, so a guessed value
    would either miss the profile or, worse, match a row for a different layout.
    """
    from atom.config import get_current_atom_config

    return int(get_current_atom_config().kv_cache_block_size)


def index_cache_signature(impl) -> str:
    """The ``HeadNormRopeCache.index`` value for THIS layer's real indexer-key cache.

    ``"fp8_unit_page128"`` is the one layout the profile's rows were built and validated for: an FP8
    cache addressed flat as ``[slot][128]``. The kernels write ``index_cache + slot * 128`` bytes, so
    a bf16 cache, another index head dim, or any other element size would be written at the wrong
    addresses with no error. Every other combination therefore gets a name no profile row carries:
    ``match_prefix`` finds no row and the layer stays native. Decided from the impl's configuration
    (the same on every rank); the bound tensor is checked again at bind time
    (``GemmFusionSite.cache_layout_reason``).
    """
    dtype = getattr(impl, "index_cache_dtype", None)
    dim = getattr(impl, "index_head_dim", None)
    if dtype == "fp8" and dim == 128:
        return "fp8_unit_page128"
    return f"{dtype}_d{dim}_unit_page128"


def head_norm_rope_cache_from_attention(attn, index: str | None = None):
    """``HeadNormRopeCache`` describing what ``attn.impl.rope_cache`` computes.

    ``numerics``/``kv_cache`` are the golden-order family names; they are part
    of the signature, so a mismatch with the profile means "no row" (native).
    """
    gf = aiter_gf()
    impl = attn.impl
    sparse_full = index is not None
    kv_dtype = getattr(impl, "kv_cache_dtype", "bf16")
    # Sparse impl views the page-128 pool as page-16 SHUFFLE; dense keeps its own
    # page. Both paths are SHUFFLE layout with per-token fp8 scales.
    page = 16 if hasattr(impl, "index_head_dim") else _kv_block_size()
    return gf.HeadNormRopeCache(
        head_dim=impl.head_dim,
        q_heads=impl.num_heads,
        kv_heads=impl.num_kv_heads,
        rotary_dim=impl.rotary_emb.rotary_dim,
        rope="neox",
        eps=float(impl.q_norm.variance_epsilon),
        norm="gemma_rms",
        numerics=(
            "aiter_fused_qknorm_idxrqknorm_v1"
            if sparse_full
            else "triton_norm_rope_cache_v1"
        ),
        kv_cache=f"shuffle_page{page}_{kv_dtype}_ptok",
        index=index,
    )


def native_fns(stages, attn):
    """Native callables (today's ATOM code) for the three entry depths.

    ``full`` runs everything natively; ``after_exchange`` continues from the
    exchanged rows; ``after_prologue`` continues after norm/rope/cache.
    """
    from . import stages_native as N

    return N.NativeFns(
        full=N.native_attention,
        after_exchange=N.attend_after_exchange,
        after_prologue=N.attend_after_prologue,
    )
