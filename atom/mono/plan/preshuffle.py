# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Addresses in an E4M3 plane in AITER's (16, 16) preshuffle,
``[N/16][K/32][2][16 rows][16 B]``: ATOM's per-token FP8 linears as the
production GEMM holds them, which ``gemv``'s ``w8`` loads read in place.

As ``arith``: a kernel calls these on its traced operands, a CPU test on ints,
so the test checks what the kernel runs. Only non-negative operands reach ``//``
and ``%``, where Python's floor and the device's truncation agree. No FlyDSL
import.
"""

ROWS = 16
# dwords from a lane's 8 fp8 of one q step to the next's: 32 K columns, a K
# block (``w8_offset``)
W8_QSTEP = 128


def w8_offset(row0, r, col, ldk):
    """The byte of element (row0 + r, col) of an ``[N, ldk]`` E4M3 plane in
    AITER's (16, 16) preshuffle, ``[N/16][K/32][2][16 rows][16 B]`` (``row0`` a
    multiple of 16, r < 16). The 8 elements from a ``col`` multiple of 8 are
    contiguous: a lane's operand of a q step."""
    return (
        ((row0 // ROWS * (ldk // 32) + col // 32) * 2 + col % 32 // 16) * 256
        + r * 16
        + col % 16
    )


def w8_lane_dwords(row0, lane, kcol, ldk):
    """The dwords of a lane's four q-step operands of chunk ``kcol`` in a
    preshuffled E4M3 plane (``w8_offset``): row row0 + lane % 16, 8 elements
    from kcol + 32 q + 8 (lane / 16), as ``gemv.chunk_ops`` takes them."""
    base = w8_offset(row0, lane % ROWS, kcol + 8 * (lane // ROWS), ldk) // 4
    return [base + W8_QSTEP * q for q in range(4)]
