# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Column-sharded latent tail of the Kimi-K3 MoE block for decode.

``KimiSparseMoeBlock._apply_routed_norm_and_up_proj`` normalizes the routed
latent and projects it to the full hidden size on every rank. ``latent_tail``
projects only the rank's own up-projection rows and adds its columns of the
shared expert output: ``rmsnorm(latent) @ up_proj_rows.T + shared``,
accumulated in fp32 and rounded to bf16 once.

The normalization is the block's own RMSNorm call. The projection is split
over K; all splits of an output tile run on one XCD (the hardware deals
programs to the eight XCDs round-robin by id), so the partials and the arrival
counter stay coherent in that XCD's L2 without a device-scope fence. The last
split to arrive sums the partials in split order and resets the counter. This
is the split-K reduction of aiter's Triton GEMMs (``_sum_splits_on_xcd``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.layernorm import rmsnorm2d_fwd_
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace

MAX_TOKENS = 64
_SPLIT, _BLOCK_N, _BLOCK_K = 7, 32, 256
_COUNTERS = 1024
_arrival_counters: dict[torch.device, torch.Tensor] = {}


@triton.jit
def _latent_tail_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    o_ptr,
    part_ptr,
    cnt_ptr,
    T,
    stride_x,
    stride_w,
    stride_s,
    stride_o,
    L: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = (pid // 8 // SPLIT) * 8 + pid % 8
    split = pid // 8 % SPLIT
    if tile < N // BLOCK_N:
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs_m[None, :] < T
        shared = tl.load(
            s_ptr + offs_m[None, :] * stride_s + offs_n[:, None], mask=mask, other=0.0
        )
        K_SPLIT: tl.constexpr = L // SPLIT
        acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        for k0 in tl.range(split * K_SPLIT, split * K_SPLIT + K_SPLIT, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            w = tl.load(
                w_ptr + offs_n[:, None] * stride_w + offs_k[None, :],
                cache_modifier=".cg",
            )
            x = tl.load(
                x_ptr + offs_m[None, :] * stride_x + offs_k[:, None],
                mask=mask,
                other=0.0,
            )
            acc += tl.dot(w, x)

        tile_offs = tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        parts = part_ptr + tile * SPLIT * BLOCK_N * BLOCK_M
        tl.store(parts + split * BLOCK_N * BLOCK_M + tile_offs, acc)
        # The partial must reach L2 before this split is counted.
        tl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
        )
        tl.debug_barrier()
        if tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="cta") == SPLIT - 1:
            acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            for j in tl.static_range(SPLIT):
                acc += tl.load(
                    parts + j * BLOCK_N * BLOCK_M + tile_offs, cache_modifier=".cv"
                )
            y = acc + shared.to(tl.float32)
            tl.store(
                o_ptr + offs_m[None, :] * stride_o + offs_n[:, None],
                y.to(tl.bfloat16),
                mask=mask,
            )
            tl.store(cnt_ptr + tile, 0)


def fits(latent_size: int, columns: int) -> bool:
    """Whether the kernel tiles a ``latent_size`` latent onto ``columns``
    up-projection rows."""
    return (
        latent_size % (_SPLIT * _BLOCK_K) == 0
        and columns % _BLOCK_N == 0
        and columns // _BLOCK_N <= _COUNTERS
    )


def _counters(device: torch.device) -> torch.Tensor:
    cnt = _arrival_counters.get(device)
    if cnt is None:
        # Zeroed once; every launch leaves them at zero. A graph capture only
        # records a fill, so the first call has to be eager.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("latent_tail must run once before graph capture")
        cnt = torch.zeros(_COUNTERS, dtype=torch.int32, device=device)
        _arrival_counters[device] = cnt
    return cnt


def _latent_tail_op(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    T, L = latent.shape
    N = up_proj_rows.shape[0]
    if not 1 <= T <= MAX_TOKENS:
        raise ValueError(f"latent_tail takes 1..{MAX_TOKENS} tokens, got {T}")
    assert up_proj_rows.shape[1] == L and shared.shape == (T, N) and fits(L, N)
    assert up_proj_rows.stride(1) == 1 and shared.stride(1) == 1

    x = rmsnorm2d_fwd_(latent, norm_weight, eps, L)
    out = torch.empty((T, N), dtype=torch.bfloat16, device=latent.device)
    block_m = max(16, triton.next_power_of_2(T))
    tiles = N // _BLOCK_N
    parts = torch.empty(
        tiles * _SPLIT * _BLOCK_N * block_m, dtype=torch.float32, device=x.device
    )
    grid = (triton.cdiv(tiles, 8) * 8 * _SPLIT,)
    _latent_tail_kernel[grid](
        x,
        up_proj_rows,
        shared,
        out,
        parts,
        _counters(x.device),
        T,
        x.stride(0),
        up_proj_rows.stride(0),
        shared.stride(0),
        out.stride(0),
        L=L,
        N=N,
        SPLIT=_SPLIT,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=4,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return out


def _latent_tail_op_fake(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    T, N = latent.shape[0], up_proj_rows.shape[0]
    return latent.new_empty((T, N), dtype=torch.bfloat16)


direct_register_custom_op(
    op_name="kimi_k3_latent_tail",
    op_func=_latent_tail_op,
    mutates_args=[],
    fake_impl=_latent_tail_op_fake,
)


@mark_trace
def latent_tail(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    up_proj_rows: torch.Tensor,
    shared: torch.Tensor,
) -> torch.Tensor:
    """One rank's columns of the routed RMSNorm and up-projection plus shared.

    ``latent`` is ``[T, L]`` (T up to ``MAX_TOKENS``), ``up_proj_rows`` the
    rank's ``[N, L]`` rows of the up-projection weight, ``shared`` the rank's
    ``[T, N]`` columns of the shared expert output at any row stride. Returns
    ``[T, N]`` bf16. Run it once outside graph capture before capturing it.
    """
    return torch.ops.aiter.kimi_k3_latent_tail(
        latent, norm_weight, eps, up_proj_rows, shared
    )
