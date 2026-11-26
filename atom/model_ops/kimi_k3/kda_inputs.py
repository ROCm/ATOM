# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The KDA prefill's per-sequence inputs, gathered in one kernel."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _gather_kda_inputs_kernel(
        tok_ptr,  # [n] source token of each output row
        q_ptr,  # q, k, v, gate: [1, T, H, D], each token's H * D dense
        k_ptr,
        v_ptr,
        g_ptr,
        b_ptr,  # beta [1, T, H], each token's H dense
        qo_ptr,  # [1, n, H, D] each
        ko_ptr,
        vo_ptr,
        go_ptr,
        bo_ptr,  # [1, n, H] float32
        stride_q,
        stride_k,
        stride_v,
        stride_g,
        stride_b,
        HD: tl.constexpr,
        H: tl.constexpr,
        BH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        src = tl.load(tok_ptr + row).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        dst = row * HD + offs
        tl.store(qo_ptr + dst, tl.load(q_ptr + src * stride_q + offs))
        tl.store(ko_ptr + dst, tl.load(k_ptr + src * stride_k + offs))
        tl.store(vo_ptr + dst, tl.load(v_ptr + src * stride_v + offs))
        tl.store(go_ptr + dst, tl.load(g_ptr + src * stride_g + offs))
        if tl.program_id(1) == 0:
            h = tl.arange(0, BH)
            beta = tl.load(b_ptr + src * stride_b + h, mask=h < H)
            tl.store(bo_ptr + row * H + h, beta.to(tl.float32), mask=h < H)


# Elements of a token's H * D row per program.
_BLOCK = 512


def _dense_tokens(x: torch.Tensor, inner: tuple[int, ...]) -> bool:
    """``[1, T, *inner]`` with each token's elements dense."""
    if x.dim() != 2 + len(inner) or x.shape[0] != 1 or tuple(x.shape[2:]) != inner:
        return False
    step = 1
    for size, stride in zip(reversed(inner), reversed(x.stride()[2:])):
        if size != 1 and stride != step:
            return False
        step *= size
    return True


def gather_kda_inputs(tokens, q, k, v, gate, beta):
    """``x.index_select(1, tokens)`` for each of ``q, k, v, gate, beta``, and
    ``beta`` widened to fp32 as `KimiKDAAttention._run_kda` takes it, in one
    kernel. Copies only: bit-identical to the torch ops."""
    inner = tuple(q.shape[2:])
    if (
        not _HAS_TRITON
        or not q.is_cuda
        or tokens.shape[0] == 0
        or len(inner) != 2
        or (inner[0] * inner[1]) % _BLOCK
        or not all(_dense_tokens(x, inner) for x in (q, k, v, gate))
        or not _dense_tokens(beta, inner[:1])
    ):
        q, k, v, gate, beta = (x.index_select(1, tokens) for x in (q, k, v, gate, beta))
        return q, k, v, gate, beta.float()
    return _gather_inputs(tokens, q, k, v, gate, beta)


def _gather_inputs(tokens, q, k, v, gate, beta):
    n = int(tokens.shape[0])
    H, D = q.shape[2], q.shape[3]
    outs = [x.new_empty((1, n, H, D)) for x in (q, k, v, gate)]
    beta_out = torch.empty((1, n, H), dtype=torch.float32, device=beta.device)
    _gather_kda_inputs_kernel[(n, H * D // _BLOCK)](
        tokens,
        q,
        k,
        v,
        gate,
        beta,
        *outs,
        beta_out,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        gate.stride(1),
        beta.stride(1),
        HD=H * D,
        H=H,
        BH=triton.next_power_of_2(H),
        BLOCK=_BLOCK,
    )
    return (*outs, beta_out)
