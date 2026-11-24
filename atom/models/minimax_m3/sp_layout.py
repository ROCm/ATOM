# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout kernels of the MiniMax-M3 sequence-parallel prefill hand-offs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _a2a_unpack_kernel(
    q_ptr,
    s_ptr,
    oq_ptr,
    os_ptr,
    rows,
    W: tl.constexpr,
    QW: tl.constexpr,
    SW: tl.constexpr,
):
    # int32 words; (row, r): block r of the all-to-all receive buffer -> columns
    # [r * QW, (r + 1) * QW) of the row
    row = tl.program_id(0)
    r = tl.program_id(1)
    qo = tl.arange(0, QW)
    q = tl.load(q_ptr + (r * rows + row).to(tl.int64) * QW + qo)
    tl.store(oq_ptr + (row * W + r).to(tl.int64) * QW + qo, q)
    so = tl.arange(0, SW)
    s = tl.load(s_ptr + (r * rows + row).to(tl.int64) * SW + so)
    tl.store(os_ptr + (row * W + r).to(tl.int64) * SW + so, s)


def a2a_unpack(
    recv_q: torch.Tensor, recv_s: torch.Tensor, world: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The all-to-all receive buffers ``[world * rows, d]`` codes / ``[world * rows,
    d / 32]`` scales (block r = rank r's heads of this rank's rows) -> ``[rows,
    world * d]`` / ``[rows, world * d / 32]``, one launch for both. Same bytes as
    ``x.view(w, rows, d).transpose(0, 1).reshape(rows, w * d)`` (a strided copy
    per tensor)."""
    d = recv_q.shape[1]
    ds = recv_s.shape[1]
    assert recv_q.is_contiguous() and recv_s.is_contiguous()
    assert recv_q.shape[0] == world * rows
    qw, sw = d // 4, ds // 4
    if d % 4 or ds % 4 or qw & (qw - 1) or sw & (sw - 1):
        # a width the kernel's power-of-two blocks do not cover
        return (
            recv_q.view(world, rows, d).transpose(0, 1).reshape(rows, world * d),
            recv_s.view(world, rows, ds).transpose(0, 1).reshape(rows, world * ds),
        )
    out_q = torch.empty((rows, world * d), dtype=recv_q.dtype, device=recv_q.device)
    out_s = torch.empty((rows, world * ds), dtype=recv_s.dtype, device=recv_s.device)
    if rows:
        _a2a_unpack_kernel[(rows, world)](
            recv_q.view(torch.int32),
            recv_s.view(torch.int32),
            out_q.view(torch.int32),
            out_s.view(torch.int32),
            rows,
            W=world,
            QW=qw,
            SW=sw,
        )
    return out_q, out_s
