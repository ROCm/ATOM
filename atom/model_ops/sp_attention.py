# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ulysses SP attention: each rank runs the impl on its head slice over the
whole sequence, between the exchanges of `atom.distributed.ulysses_sp`.

An FP8 o-projection takes the output as per-token FP8. Its quantizer has one
scale over *all* heads of each token while a rank owns only a head shard, so
the shards' maxima are gathered before quantizing.
"""

import torch
import triton
import triton.language as tl
from aiter import dtypes
from aiter.dist.parallel_state import get_pcp_group
from aiter.ops.quant import per_token_quant_hip

from atom.config import get_current_atom_config
from atom.distributed.token_transport import (
    gather_heads,
    head_exchange_view,
    takes_head_exchange,
)
from atom.distributed.ulysses_sp import exchange_columns, sp_is_enabled
from atom.utils import mark_spliting_op


def ulysses_attention(
    attn, query, key, value, positions, q_scale, qkv, output_transform=None
):
    """Run this rank's head slice over the full sequence, then restore tokens.

    Called inside the attention custom op, keeping collectives and padding
    slices opaque to Dynamo.
    """
    group = get_pcp_group()
    # The impl's declared fields cover the fused qkv, indexer columns included.
    source = torch.cat([query, key, value], -1) if qkv is None else qkv
    packed = exchange_columns(source, group, attn.sp_columns)

    # Global positions and attention metadata exclude the divisibility pad.
    real_tokens = positions.shape[-1]
    padded_tokens = packed.shape[0]
    if real_tokens < padded_tokens:
        packed = packed[:real_tokens]

    # Attention accepts column views, so avoid another Q/K/V copy.
    q_width = attn.num_heads * attn.head_dim
    kv_width = attn.num_kv_heads * attn.head_dim
    out = attn.impl.forward(
        query=packed[:, :q_width],
        key=packed[:, q_width : q_width + kv_width],
        value=packed[:, q_width + kv_width : q_width + 2 * kv_width],
        position=positions,
        q_scale=q_scale,
        # Preserve the caller's choice of fused or separate Q/K/V kernels.
        qkv=packed if qkv is not None else None,
    )

    # The head exchange restores the pad in its own send buffer.
    if output_transform is not None:
        return output_transform(out, padded_tokens)
    return gather_heads(out, group, padded_tokens, attn.sp_head_exchange)


def sp_fp8_attention_output_supported() -> bool:
    """Whether SP attention can hand an FP8 o-projection per-token FP8 output.

    Check at construction: querying it from forward would introduce a graph
    break.
    """
    if not sp_is_enabled() or dtypes.fp8 != torch.float8_e4m3fn:
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return (
        get_current_atom_config().torch_dtype == torch.bfloat16
        and get_gfx_runtime() == "gfx950"
    )


# Rows past ROWS are the divisibility pad: they read as zero rows.
@triton.jit(do_not_specialize=["ROWS"])
def _head_amax_kernel(X, AMAX, ROWS, STRIDE, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    live = (col < WIDTH) & (row < ROWS)
    x = tl.load(X + row * STRIDE + col, live, other=0).to(tl.float32)
    # The maximum of BF16 values is exact in float32: no scale or epsilon
    # travels over the wire, only this maximum.
    amax = tl.max(tl.abs(x), 0)
    tl.store(AMAX + row, amax)


# One build for every rank.
@triton.jit(do_not_specialize=["ROWS", "TOKENS", "RANK"])
def _quantize_global_amax_kernel(
    X,
    AMAX,
    Q,
    SCALE,
    ROWS,
    TOKENS,
    STRIDE,
    RANK,
    WIDTH: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    amax = tl.full((), 0, tl.float32)
    for peer in tl.static_range(WORLD):
        amax = tl.maximum(amax, tl.load(AMAX + peer * TOKENS + row))
    # Match AITER data_to_per_row_scale and scaled_quant_vgpr_impl exactly:
    # float32 reciprocal constant, multiply, then AMD's approximate reciprocal.
    # In particular the per-token HIP quantizer has NO nonzero epsilon floor.
    scale = amax * (1.0 / 448.0)
    inv_scale = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [scale], dtype=tl.float32, is_pure=True, pack=1
    )
    live = (col < WIDTH) & (row < ROWS)
    x = tl.load(X + row * STRIDE + col, live, other=0).to(tl.float32)
    scaled = x * inv_scale
    # Use the same instruction and operand order as AITER, including zero-row
    # and nonfinite behavior, instead of silently adding a new clipping policy.
    clipped = tl.inline_asm_elementwise(
        "v_med3_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        [scaled, tl.full((), -448.0, tl.float32), tl.full((), 448.0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Q + row * WIDTH + col, clipped.to(Q.dtype.element_ty), col < WIDTH)
    local_tokens = TOKENS // WORLD
    if (row >= RANK * local_tokens) & (row < (RANK + 1) * local_tokens):
        tl.store(SCALE + row - RANK * local_tokens, scale)


def head_amax(x: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """Exact row maxima of this rank's attention heads, zero past ``x``'s rows
    up to ``rows``."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    rows = x.shape[0] if rows is None else rows
    amax = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    _head_amax_kernel[(rows,)](
        x,
        amax,
        x.shape[0],
        x.stride(0),
        x.shape[1],
        triton.next_power_of_2(x.shape[1]),
    )
    return amax


def quantize_with_gathered_amax(
    x: torch.Tensor,
    amax: torch.Tensor,
    world: int,
    rank: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a head shard with global scales, returning this rank's scales.

    ``amax`` holds every rank's maxima of the padded rows; ``x`` may hold only
    the leading ones, the rest quantize as zero rows.
    """
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert amax.is_contiguous() and amax.dtype == torch.float32
    assert amax.numel() % world == 0
    tokens = amax.numel() // world
    assert tokens % world == 0 and x.shape[0] <= tokens
    assert dtypes.fp8 == torch.float8_e4m3fn, "FP8 transport requires E4M3FN"
    shape = (tokens, x.shape[1])
    q = torch.empty(shape, device=x.device, dtype=dtypes.fp8) if out is None else out
    assert q.shape == shape and q.dtype == dtypes.fp8 and q.is_contiguous()
    scale = torch.empty((tokens // world, 1), device=x.device, dtype=torch.float32)
    _quantize_global_amax_kernel[(tokens,)](
        x,
        amax,
        q,
        scale,
        x.shape[0],
        tokens,
        x.stride(0),
        rank,
        x.shape[1],
        world,
        triton.next_power_of_2(x.shape[1]),
    )
    return q, scale


def gather_heads_fp8(
    x: torch.Tensor, group, rows: int | None = None, ca=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace BF16 head all-to-all followed by o_proj's FP8 quantizer.

    One small amax all-gather precedes the FP8 payload exchange. Viewing pairs
    of FP8 bytes as BF16 is a raw transport view, with no arithmetic; this
    supports both AITER custom all-gather and PyNccl all-to-all.
    """
    x = x.reshape(x.shape[0], -1)
    world = group.world_size
    rows = x.shape[0] if rows is None else rows
    assert world > 1 and x.shape[1] % 2 == 0
    # Piecewise compilation reuses the initially traced graph at other token
    # counts. Keep this runtime choice inside the opaque attention op so decode
    # cannot inherit the prefill branch or change the op's tuple schema.
    if not takes_head_exchange(rows, world):
        payload = gather_heads(x, group, rows)
        return per_token_quant_hip(payload, quant_dtype=dtypes.fp8)
    amax = group.device_communicator.all_gatherv(head_amax(x, rows))
    # Quantize straight into the exchange's registered pool when it takes it.
    scratch = head_exchange_view(ca, x, rows, dtypes.fp8)
    q, scale = quantize_with_gathered_amax(
        x,
        amax,
        world,
        group.rank_in_group,
        out=scratch,
    )
    payload = q.view(torch.bfloat16)
    output = (
        ca.head_exchange(payload, registered=True)
        if scratch is not None
        else gather_heads(payload, group)
    )
    return output.view(dtypes.fp8), scale


def _attention_fp8_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(q.shape, device=q.device, dtype=dtypes.fp8),
        torch.empty((q.shape[0], 1), device=q.device, dtype=torch.float32),
    )


# Not an eager_region: TR_HYBRID records it inline in the decode token graphs,
# which is valid only while a token bucket is one (bs, q=1) request bucket.
# Config refuses spec decoding under SP, and prefill token graphs are recorded
# only for a builder with `supports_token_graph` (DeepSeek-V4.1's, an attention
# SP refuses); ModelRunner also refuses any such builder under SP.
@mark_spliting_op(is_custom=True, gen_fake=_attention_fp8_fake, mutates_args=[])
def sp_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    group = get_pcp_group()
    return ulysses_attention(
        attn,
        q,
        k,
        v,
        positions,
        None,
        qkv,
        output_transform=lambda out, rows: gather_heads_fp8(
            out, group, rows, attn.sp_head_exchange
        ),
    )
