# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Addresses and work assignment of the routed-MoE tail + all-reduce launch (``kernels/moe_tail_ar_gfx1250.py``).

Plain arithmetic on Python ints or FlyDSL values, shared by the kernel and by the host-side address audit
(``p6/tests/test_layout_audit.py``), so the audit checks the expressions the kernel runs.

Work: ``MAX_M_TILES * n_tiles`` tile workgroups (256 threads), m-major: workgroup ``m * n_tiles + c`` runs the
production GEMM2 tile of m-tile ``m`` (= one expert: an expert has at most 8 rows) and column tile ``c``, and for
``m`` < tokens then gathers token ``m``'s packs of column tile ``c`` and pushes them (to the three peers' k5 inboxes
and to this rank's own-row scratch). Then ``tokens`` receivers.
"""

RANKS = 4
HIDDEN = 6144
ROW_PACKS = HIDDEN // 8  # 768 16-byte packs (8 bf16) a row
RWAVES = 8  # k5: waves of a row
SLAB = 96  # k5: packs a wave, one round counter
BUFS = 3  # k5: inbox buffers (rounds mod 3)
WAVE = 32
TILE_M = 16  # GEMM2 m-tile = the route's expert alignment
MAX_M_TILES = 32  # one a lane
NO_TILE = 1 << 20


def n_tiles(tile_n):
    return HIDDEN // tile_n


def tile_wgs(tile_n):
    return MAX_M_TILES * n_tiles(tile_n)


def grid(tile_n, tokens):
    """Workgroups of a launch: the tile workgroups, then one receiver a row."""
    return tile_wgs(tile_n) + tokens


def tile_packs(tile_n):
    return tile_n // 8


def pack_col(cblk, pc, tile_n):
    """Pack index within a row of pack ``pc`` of column tile ``cblk``."""
    return cblk * tile_packs(tile_n) + pc


def pack_index(t, pcol):
    """k5's pack index of row ``t``, pack column ``pcol``."""
    return t * ROW_PACKS + pcol


def slab(t, pcol):
    """k5's round-counter index (send counters; + ctr_recv for the receive counters)."""
    return t * RWAVES + pcol // SLAB


def ctr_recv(max_rows):
    return max_rows * RWAVES


def ctr_err(max_rows):
    return 2 * max_rows * RWAVES


def ctr_words(max_rows):
    return 2 * max_rows * RWAVES + 1


def inbox_bytes(slot):
    return BUFS * RANKS * slot * 16


def inbox_offset(rnd, sender, slot, p):
    """Byte offset in a rank's k5 inbox of pack ``p`` sent by ``sender`` in round ``rnd``."""
    return ((rnd * RANKS + sender) * slot + p) * 16


def g_row_base(row):
    """Byte offset in the grouped GEMM2 output of route row ``row``."""
    return row * (HIDDEN * 2)


def g_pack_off(pcol):
    """Byte offset within a grouped output row (a per-row descriptor) of pack column ``pcol``."""
    return pcol * 16


def xown_off(p):
    """Byte offset in this rank's own-row scratch (bf16 [tokens, 6144], behind the grouped output) of pack ``p``."""
    return p * 16


def out_byte(t, w, lane, j):
    """Byte offset in the E4M3 output of the 8 values receiver wave ``w``, lane ``lane`` holds for plane ``j``."""
    return t * HIDDEN + w * 768 + 8 * lane + 256 * j


def scale_byte(t, w, lane, j):
    return t * (HIDDEN // 32) + (w * 768 + 8 * lane + 256 * j) // 32


# ---- host model of a tile workgroup's work (the kernel computes the same lane-wise) ----
def tile_of(rows, ids, mt, n_experts, i32_m):
    """``rows`` / ``ids``: [tokens][4] ints. (expert, valid rows) of m-tile ``mt``, or None when it does not run."""
    expert, nrow = 0, 0
    for t in range(len(rows)):
        for k in range(4):
            r = rows[t][k]
            if 0 <= r < i32_m and r // TILE_M == mt:
                expert |= ids[t][k]
                nrow = max(nrow, r % TILE_M + 1)
    return (expert, nrow) if nrow > 0 and 0 <= expert < n_experts else None
