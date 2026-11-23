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
from aiter.ops.quant import dynamic_per_token_scaled_quant

from atom.config import get_current_atom_config
from atom.distributed.pcp_utils import pcp_pad_dense
from atom.distributed.ulysses_sp import (
    ALLGATHER_MAX_PAYLOAD_BYTES,
    all_to_all_blocks,
    exchange_columns,
    sp_is_enabled,
)
from atom.utils import envs, mark_spliting_op

# Output rows per rank from which an attention output takes the prefill head
# exchange (FP8 with a gathered amax, the four-rank exchange kernel); decode
# stays below it on the all-gather.
HEAD_EXCHANGE_MIN_LOCAL_ROWS = 2048


def ulysses_attention(attn, query, key, value, positions, q_scale, qkv):
    """Run this rank's head slice over the full sequence. Returns the impl's
    output and the padded sequence length, for `gather_heads`.

    Called inside the attention custom op, keeping collectives and padding
    slices opaque to Dynamo.
    """
    # The impl's declared fields cover the fused qkv, indexer columns included.
    source = torch.cat([query, key, value], -1) if qkv is None else qkv
    packed = exchange_columns(source, get_pcp_group(), attn.sp_columns)

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
    return out, padded_tokens


def head_exchange_comm(group):
    """The communicator of the four-rank head exchange, unless the env opts
    out. Static: a layer decides it once, when built."""
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return None
    return group.device_communicator.ca_comm


def head_exchange_view(ca, x: torch.Tensor, rows: int, dtype=None):
    """The ``[rows, x.shape[1]]`` buffer of ``dtype`` that ``ca`` exchanges
    from: ``x`` itself when it fits as is, else a registered staging view.
    None when the exchange does not take it."""
    dtype = x.dtype if dtype is None else dtype
    if ca is None:
        return None
    if (rows, dtype) == (x.shape[0], x.dtype):
        view = x
    else:
        view = ca.registered_input_view((rows, x.shape[1]), dtype)
    if view is not None and ca.should_head_exchange(view):
        return view
    return None


def gather_heads(x, rows, ca=None, scale=None):
    """``[S, H/W*D] -> [S/W, H*D]``: give up the sequence, regain all heads.

    ``x`` may hold only the leading rows of S = ``rows``; the rest are sent as
    zeros. ``ca`` is the layer's `head_exchange_comm`. With ``scale``, the
    result is per-token FP8 and its scales go to ``scale``; a long output then
    travels as FP8, quantized with the gathered row maxima. Every choice
    depends only on shapes, so all ranks pick the same collective.
    """
    group = get_pcp_group()
    w, rank = group.world_size, group.rank_in_group
    x = x.reshape(x.shape[0], -1)
    s_local = rows // w
    small = rows * x.shape[1] * x.element_size() <= ALLGATHER_MAX_PAYLOAD_BYTES
    long_rows = not small and rows >= w * HEAD_EXCHANGE_MIN_LOCAL_ROWS
    fp8_wire = long_rows and scale is not None
    view = None
    if long_rows:
        view = head_exchange_view(ca, x, rows, dtypes.fp8 if fp8_wire else None)
    registered = view is not None and view is not x
    if fp8_wire:
        amax = group.device_communicator.all_gatherv(head_amax(x, rows))
        # FP8 bytes travel in pairs as BF16, a raw view with no arithmetic.
        x = quantize_with_gathered_amax(x, amax, w, rank, scale, out=view)
        x = x.view(torch.bfloat16)
    elif registered:
        # Pad in the registered pool: this copy replaces the exchange's own.
        view[: x.shape[0]].copy_(x)
        view[x.shape[0] :].zero_()
        x = view
    else:
        x = pcp_pad_dense(x, rows - x.shape[0])

    if view is not None:
        out = ca.head_exchange(x, registered=registered)
    elif small:
        out = group.device_communicator.all_gather(x, 1)
        out = out[rank * s_local : (rank + 1) * s_local]
    else:
        send = x.reshape(w, s_local, x.shape[1]).contiguous()
        # Block i is head group i of our token chunk: concatenate on heads.
        out = all_to_all_blocks(send, group).transpose(0, 1).reshape(s_local, -1)

    if fp8_wire:
        return out.view(dtypes.fp8)
    if scale is None:
        return out
    q = torch.empty(out.shape, device=out.device, dtype=dtypes.fp8)
    dynamic_per_token_scaled_quant(q, out, scale)
    return q


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
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize a head shard with global scales, writing this rank's to ``scale``.

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
    assert scale.shape == (tokens // world, 1) and scale.dtype == torch.float32
    assert scale.is_contiguous()
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
    return q


def _attention_fp8_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor | None,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(q.shape, device=q.device, dtype=dtypes.fp8)


# Not an eager_region: TR_HYBRID records it inline in the decode token graphs,
# which is valid only while a token bucket is one (bs, q=1) request bucket.
# Config refuses spec decoding under SP, and prefill token graphs are recorded
# only for a builder with `supports_token_graph` (DeepSeek-V4.1's, an attention
# SP refuses); ModelRunner also refuses any such builder under SP.
@mark_spliting_op(is_custom=True, gen_fake=_attention_fp8_fake, mutates_args=["scale"])
def sp_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    qkv: torch.Tensor | None,
    scale: torch.Tensor,
) -> torch.Tensor:
    attn = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    out, rows = ulysses_attention(attn, q, k, v, positions, None, qkv)
    return gather_heads(out, rows, attn.sp_head_exchange, scale)
