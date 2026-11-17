# SPDX-License-Identifier: MIT
"""Compact FP4/FP8 row codecs shared by cache writers and sparse readers.

Source buffers store contiguous data and scale planes. Main KV uses E2M1 with
E4M3 group16 scales; index rows use E2M1 with E8M0 group32 scales; SWA uses
E4M3 with E8M0 group32 scales. Only requested rows are decoded.
"""

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
def _read_plane(values, scales, rows, dims, valid, D: tl.constexpr, FP4: tl.constexpr):
    rows = rows.to(tl.int64)
    if FP4:
        raw = tl.load(
            values + rows[:, None] * (D // 2) + dims[None, :] // 2,
            valid[:, None],
            other=0,
        )
        value = _e2m1((raw >> ((dims[None, :] % 2) * 4)) & 15)
        code = tl.load(
            scales + rows[:, None] * (D // 16) + dims[None, :] // 16,
            valid[:, None],
            other=0,
        )
        scale = code.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        raw = tl.load(
            values + rows[:, None] * D + dims[None, :], valid[:, None], other=0
        )
        value = raw.to(FP8_TL_DTYPE, bitcast=True).to(tl.float32)
        code = tl.load(
            scales + rows[:, None] * (D // 32) + dims[None, :] // 32,
            valid[:, None],
            other=127,
        )
        scale = _e8m0(code)
    return (value * scale).to(tl.bfloat16)


@triton.jit(do_not_specialize=["first", "end", "CAPACITY"])
def _gather_prefix(
    pool,
    addresses,
    ptr,
    out,
    first,
    end,
    CAPACITY,
    main_scale,
    window,
    window_scale,
    D: tl.constexpr,
):
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    start = tl.load(ptr + first)
    count = tl.load(ptr + end) - start
    valid = (i < count) & (i < CAPACITY)
    row = tl.load(addresses + start + i, valid, other=0)
    fp8 = (row & 1) != 0
    dims = tl.arange(0, D)
    main = _read_plane(pool, main_scale, row >> 1, dims, valid & ~fp8, D, True)
    swa = _read_plane(window, window_scale, row >> 1, dims, valid & fp8, D, False)
    value = tl.where(fp8[:, None], swa, main)
    tl.store(
        out + i[:, None] * D + tl.arange(0, D)[None, :], value, i[:, None] < CAPACITY
    )


def gather_prefix_rows(pool, addresses, ptr, output, first, end):
    """Fill bounded scratch with one query tile's selected history.

    A ((main, scale), (window, scale)) source uses tagged row IDs;
    the low bit selects FP8 SWA, and the remaining bits are source-local rows.
    """
    (main, main_scale), (window, window_scale) = pool
    if output.shape[0]:
        _gather_prefix[(triton.cdiv(output.shape[0], 16),)](
            main,
            addresses,
            ptr,
            output,
            first,
            end,
            output.shape[0],
            main_scale,
            window,
            window_scale,
            output.shape[1],
        )
