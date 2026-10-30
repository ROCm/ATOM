# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native (today's) code for each entry depth of a QKV+attention site.

Each function is a thin adapter over code that already exists, so the native
path is exactly the pre-fusion behavior:
  full            qkv_proj -> ulysses exchange -> impl.forward (rope_cache +
                  backend) -> gather_heads_fp8      (== minimax_m3_attention_fp8)
  after_exchange  impl.forward on exchanged rows -> gather_heads_fp8
  after_prologue  backend only (caches already written) -> gather_heads_fp8
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeFns:
    full: Callable
    after_exchange: Callable
    after_prologue: Callable


def _output_transform():
    from atom.model_ops.minimax_m3.attention_fp8 import gather_heads_fp8

    return gather_heads_fp8


def native_attention(site, x, x_scale, ctx):
    from atom.distributed.ulysses_sp import ulysses_attention

    qkv = site.linear(x, x_scale=x_scale)
    q, k, v = site.split_qkv(qkv)
    return ulysses_attention(
        ctx.attn,
        q,
        k,
        v,
        ctx.positions,
        None,
        qkv,
        output_transform=_output_transform(),
    )


def attend_after_exchange(site, packed, ctx):
    from atom.distributed.ulysses_sp import attend_packed

    impl = ctx.attn.impl
    return attend_packed(
        ctx.attn,
        packed,
        ctx.positions,
        None,
        impl.num_heads * impl.head_dim,
        impl.num_kv_heads * impl.head_dim,
        fused_qkv=True,
        output_transform=_output_transform(),
    )


def attend_after_prologue(site, outs, state, ctx):
    from atom.distributed.ulysses_sp import finish_attention

    real = ctx.real_tokens
    impl = ctx.attn.impl
    k = outs.get("k")
    v = outs.get("v")
    out = impl.forward_after_prologue(
        outs["q"][:real],
        None if k is None else k[:real],
        None if v is None else v[:real],
        state["k_cache"],
        state["v_cache"],
        state.get("k_scale_view", state.get("k_scale")),
        state.get("v_scale_view", state.get("v_scale")),
        ctx.fwd_ctx,
        index_q=(outs["index_q"][:real] if "index_q" in outs else None),
    )
    return finish_attention(out, ctx.padded_tokens, real, _output_transform())
