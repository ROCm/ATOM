# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MFMA issue orders shared by the gfx950 MXFP8 FlyDSL GEMMs (pure Python)."""


def row_stationary_order(n_a: int, n_b: int) -> list[tuple[int, int]]:
    """(A tile, B tile) issue order of one quadrant, row by row: A tile i stays
    for ``n_b`` consecutive MFMAs while the B tiles serpentine, so adjacent MFMAs
    switch one operand (fewer than the 2x2 zig-zag). Every accumulator still
    takes its K-steps in order, so the output is bitwise the zig-zag order's."""
    cols = list(range(n_b))
    return [(i, j) for i in range(n_a) for j in (cols[::-1] if i % 2 else cols)]
