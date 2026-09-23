# SPDX-License-Identifier: MIT
"""Compact FP4/FP8 row codecs shared by cache writers and sparse readers.

Rows contain value bytes followed by scale bytes. Main KV uses E2M1 with
E4M3 group16 scales; index rows use E2M1 with E8M0 group32 scales; SWA uses
E4M3 with E8M0 group32 scales. Only requested rows are decoded.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_TL_DTYPE


@triton.jit
def _e8m0(code):
    bits = code.to(tl.uint32) << 23
    return tl.where(
        code == 0,
        2.0**-127,
        tl.where(code == 255, float("nan"), bits.to(tl.float32, bitcast=True)),
    )


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(
        mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2.0, (mag - 4.0) * 2.0)
    )
    return tl.where(code >= 8, -value, value)


@triton.jit
def load_mixed_rows(pool, tagged_rows, dims, valid, D: tl.constexpr):
    """A nonnegative int64 address: byte offset * 2, low bit selects FP8."""
    fp8 = (tagged_rows & 1)[:, None] != 0
    base = (tagged_rows >> 1)[:, None]
    d = dims[None, :]
    mask = valid[:, None] & (d < D)
    raw = tl.load(pool + base + tl.where(fp8, d, d // 2), mask, other=0)
    code = (raw >> ((d % 2) * 4)) & 15
    # The value byte is an activation this process quantized, so its encoding
    # follows the device. The FP4 scale below stays E4M3 on every device: it is
    # the checkpoint's format, not a choice this build gets to make.
    value = tl.where(
        fp8, raw.to(FP8_TL_DTYPE, bitcast=True).to(tl.float32), _e2m1(code)
    )
    scale = tl.load(
        pool + base + tl.where(fp8, D + d // 32, D // 2 + d // 16), mask, other=127
    )
    multiplier = tl.where(
        fp8, _e8m0(scale), scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    )
    return tl.where(mask, value * multiplier, 0).to(tl.bfloat16)


def pack_rows(values, scales):
    """Interleave existing native value/scale bytes without quantizing again."""
    return torch.cat((values.view(torch.uint8), scales.view(torch.uint8)), dim=-1)


@triton.jit
def _write_window(
    values,
    scales,
    pool,
    positions,
    batches,
    cu,
    slots,
    ring_start,
    SLOT_BYTES: tl.constexpr,
    WINDOW: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    batch = tl.load(batches + t)
    # A padding row owns no request, so there is nowhere for it to write and
    # nothing at `slots[-1]` to read. The grid is the forward's width because
    # that is the width a captured replay runs.
    if batch < 0:
        return
    end = tl.load(cu + batch + 1)
    if t < end - WINDOW:
        return
    slot = tl.load(slots + batch).to(tl.int64)
    pos = tl.load(positions + t)
    row = ring_start + slot * SLOT_BYTES + (pos % WINDOW) * (D + D // 32)
    d = tl.arange(0, BLOCK)
    v = tl.load(values + t * D + d, d < D, other=0)
    s = tl.load(scales + t * (D // 32) + d, d < D // 32, other=127)
    tl.store(pool + row + d, v, d < D)
    tl.store(pool + row + D + d, s, d < D // 32)


def write_packed_window(values, scales, pool, step, window, dim):
    if step.width:
        _write_window[(step.width,)](
            values.view(torch.uint8),
            scales.view(torch.uint8),
            pool,
            step.positions,
            step.batch_ids,
            step.cu_seqlens_q,
            step.slots,
            window.ring_start,
            window.slot_rows,
            window.ring_slots,
            dim,
            triton.next_power_of_2(dim),
        )


@triton.jit
def _gather_prefix(
    pool, addresses, ptr, out, first, end, CAPACITY: tl.constexpr, D: tl.constexpr
):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.load(ptr + first)
    count = tl.load(ptr + end) - start
    valid = (i < count) & (i < CAPACITY)
    row = tl.load(addresses + start + i, valid, other=0)
    value = load_mixed_rows(pool, row, tl.arange(0, D), valid, D)
    tl.store(
        out + i[:, None] * D + tl.arange(0, D)[None, :], value, i[:, None] < CAPACITY
    )


def gather_prefix_rows(pool, addresses, ptr, output, first, end):
    """Fill bounded scratch with only one query tile's selected history rows."""
    if output.shape[0]:
        _gather_prefix[(triton.cdiv(output.shape[0], 16),)](
            pool,
            addresses,
            ptr,
            output,
            first,
            end,
            output.shape[0],
            output.shape[1],
        )
