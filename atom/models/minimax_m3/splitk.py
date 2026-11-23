# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's split-K reduce left to the consumer: which GEMMs defer
(`LinearBase.splitk_consumer_max_m`) and which attention impls then claim the
partials. GPU-free, so the CPU job tests it without importing the model."""

from aiter.jit.utils.chip_info import get_gfx_runtime
from torch import nn

from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.paged_attention import Attention as AtomAttention
from atom.utils import envs


def _rope_cache_claims_splitk_partials(attn: nn.Module) -> bool:
    """Whether ``attn``'s impl claims a deferred QKV's split-K partials in its
    rope_cache (`PagedAttentionImpl.claims_splitk_partials`): ATOM's own
    attention layer only (the plugins' run their own), not under Ulysses SP (its
    head exchange reads qkv before the impl does)."""
    return (
        isinstance(attn, AtomAttention)
        and attn.sp_columns is None
        and isinstance(attn.impl, PagedAttentionImpl)
        and attn.impl.claims_splitk_partials
    )


def defer_splitk_reduce(linear: nn.Module, consumer_claims: bool) -> bool:
    """Let ``linear``'s per-token FP8 GEMM leave its split-K reduction to the one
    kernel that reads its output, which sums the partials as it reads them (the
    result is bit-identical), when that consumer claims them; returns whether it
    did. gfx1250 only (the live GPU, as M3's other gfx1250 paths read it), where
    the tuned M3 decode GEMMs split K: elsewhere, gfx950 included, M3 keeps the
    plain GEMM and SwiGLU-OAI and no impl claims. The GEMM then decides per call
    whether it splits K and defers (aiter.gemm_a8w8_bpreshuffle_splitk_deferred).
    Not checked here: that ``linear`` runs per-token FP8, which online
    quantization settles only at weight load; LinearBase takes the deferred GEMM
    only on that path, and a consumer of a GEMM that did not defer finds no
    partials and runs as usual.
    """
    max_m = envs.ATOM_SPLITK_CONSUMER_REDUCE_MAX_M
    if not (consumer_claims and max_m > 0 and get_gfx_runtime() == "gfx1250"):
        return False
    linear.splitk_consumer_max_m = max_m
    return True


def defer_qkv_splitk_reduce(qkv_proj: nn.Module, attn: nn.Module) -> None:
    """qkv's only reader is ``attn``'s rope_cache: defer ``qkv_proj``'s split-K
    reduce to it where it claims the partials, and tell it so."""
    if defer_splitk_reduce(qkv_proj, _rope_cache_claims_splitk_partials(attn)):
        attn.impl.qkv_splitk_deferred = True
