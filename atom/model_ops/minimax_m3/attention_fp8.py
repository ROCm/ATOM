# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Experimental, numerically equivalent FP8 transport for M3 attention output.

The o-projection quantizer has one scale over *all* heads of each token. The
attention producer owns only a head shard, so its local amax must be exchanged
before quantization. Quantizing shards independently would change that contract.
"""

import torch
import triton
import triton.language as tl
from aiter import dtypes

from atom.config import get_current_atom_config
from atom.distributed.ulysses_sp import (
    _all_gather_tokens,
    get_sp_group,
    get_sp_world_size,
    ulysses_attention,
    ulysses_gather_heads,
)
from atom.utils import envs, mark_spliting_op


def supports_m3_attention_fp8(query_width: int) -> bool:
    """Startup-only gate for the exact hardware/layout measured by this probe.

    Cache this boolean on the attention module: querying the architecture from
    forward would introduce a graph break. Online weight quantization happens
    after construction, so the o_proj quantization contract is checked later.
    """
    if (
        not envs.ATOM_SP_ATTN_FP8
        or get_sp_world_size() != 4
        or query_width != 8192
        or dtypes.fp8 != torch.float8_e4m3fn
        or get_current_atom_config().torch_dtype != torch.bfloat16
    ):
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    from atom.plugin.prepare import is_plugin_mode

    return not is_plugin_mode() and get_gfx_runtime() == "gfx950"


@triton.jit
def _head_amax_kernel(
    X, AMAX, WIDTH: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + col, col < WIDTH, other=0).to(tl.float32)
    # max(abs(BF16)) is itself exactly representable as BF16. No scale or
    # epsilon is rounded to BF16; only this exact maximum travels over the wire.
    amax = tl.max(tl.abs(x), 0)
    tl.store(AMAX + row, amax)


@triton.jit
def _quantize_global_amax_kernel(
    X,
    AMAX,
    Q,
    SCALE,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    amax = tl.full((), 0, tl.float32)
    for peer in tl.static_range(WORLD):
        amax = tl.maximum(amax, tl.load(AMAX + peer * TOKENS + row).to(tl.float32))
    # Match AITER data_to_per_row_scale and scaled_quant_vgpr_impl exactly:
    # float32 reciprocal constant, multiply, then AMD's approximate reciprocal.
    # In particular the per-token HIP quantizer has NO nonzero epsilon floor.
    scale = amax * (1.0 / 448.0)
    inv_scale = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [scale], dtype=tl.float32, is_pure=True, pack=1
    )
    x = tl.load(X + row * STRIDE + col, col < WIDTH, other=0).to(tl.float32)
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


def head_amax(x: torch.Tensor) -> torch.Tensor:
    """Exact BF16 row maxima of this rank's attention heads."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    amax = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.bfloat16)
    _head_amax_kernel[(x.shape[0],)](
        x, amax, x.shape[1], x.stride(0), triton.next_power_of_2(x.shape[1])
    )
    return amax


def quantize_with_gathered_amax(
    x: torch.Tensor, amax: torch.Tensor, world: int, rank: int,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a head shard with global scales, returning this rank's scales."""
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.stride(1) == 1
    assert x.shape[0] % world == 0
    assert amax.numel() == world * x.shape[0] and amax.is_contiguous()
    assert dtypes.fp8 == torch.float8_e4m3fn, "M3 FP8 transport requires E4M3FN"
    q = torch.empty(x.shape, device=x.device, dtype=dtypes.fp8) if out is None else out
    assert q.shape == x.shape and q.dtype == dtypes.fp8 and q.is_contiguous()
    scale = torch.empty((x.shape[0] // world, 1), device=x.device, dtype=torch.float32)
    _quantize_global_amax_kernel[(x.shape[0],)](
        x,
        amax,
        q,
        scale,
        x.shape[0],
        x.shape[1],
        x.stride(0),
        world,
        rank,
        triton.next_power_of_2(x.shape[1]),
    )
    return q, scale


def gather_heads_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace BF16 head all-to-all followed by o_proj's FP8 quantizer.

    One small BF16 amax all-gather precedes the FP8 payload exchange. Viewing
    pairs of FP8 bytes as BF16 is a raw transport view, with no arithmetic;
    this supports both AITER custom all-gather and PyNccl all-to-all.
    """
    x = x.reshape(x.shape[0], -1)
    world = get_sp_world_size()
    assert world > 1 and x.shape[1] % 2 == 0
    # Piecewise compilation reuses the initially traced graph at other token
    # counts. Keep this runtime choice inside the opaque attention op so decode
    # cannot inherit the prefill branch or change the op's tuple schema.
    if x.shape[0] // world < 2048:
        from aiter.ops.quant import per_token_quant_hip

        return per_token_quant_hip(ulysses_gather_heads(x), quant_dtype=dtypes.fp8)
    amax = _all_gather_tokens(head_amax(x))
    if envs.ATOM_SP_HEAD_EXCHANGE:
        from atom.distributed.sp_head_exchange import (
            exchange_heads,
            head_exchange_communicator,
        )
        from atom.distributed.sp_registered_buffer import registered_input_view

        ca = head_exchange_communicator(x)
        scratch = registered_input_view(get_sp_group(), x.shape, dtypes.fp8) if ca is not None else None
        if scratch is not None:
            q, scale = quantize_with_gathered_amax(
                x, amax, world, get_sp_group().rank_in_group, out=scratch[0]
            )
            output = exchange_heads(q.view(torch.bfloat16), ca, registered=True)
            return output.view(dtypes.fp8), scale
    q, scale = quantize_with_gathered_amax(x, amax, world, get_sp_group().rank_in_group)
    output = ulysses_gather_heads(q.view(torch.bfloat16)).view(dtypes.fp8)
    return output, scale


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


@mark_spliting_op(is_custom=True, gen_fake=_attention_fp8_fake, mutates_args=[])
def minimax_m3_attention_fp8(
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
    return ulysses_attention(
        attn, q, k, v, positions, None, qkv, output_transform=gather_heads_fp8
    )
