# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reference launches for ``test_mono_hi_bridge``: each of the fused bridge's bodies (``hi_bridge``) as a kernel
of its own, the separate launches the fused begin-step / end-step replace. Loaded by the test with the layout's jit helpers in its namespace (``_ring_in``, ``_req_words``,
...); not a module of its own."""

# ruff: noqa: F821  (the layout's jit helpers are injected into this module's namespace by the test)

import triton
import triton.language as tl


@triton.jit
def _import_rings_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos, rows,
    WIN: tl.constexpr, DIM: tl.constexpr, RB: tl.constexpr, NB: tl.constexpr = 1, BM: tl.constexpr = 1,
):  # fmt: skip
    """Program (m, s, b): ``_ring_in`` (grid (NMF, WIN, NB); ``rows`` = NB requests' rows)."""
    _ring_in(tl.program_id(0), tl.program_id(2), tl.program_id(1), rows // NB, pool, positions, batch_ids, slots,
             rings, layer_ids, pos_out, bounds_out, ring_start0, slot_rows, ring_slots, num_slots, max_pos, WIN, DIM,
             RB, BM)  # fmt: skip


@triton.jit
def _export_rings_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos,
    WIN: tl.constexpr, DIM: tl.constexpr, RQ: tl.constexpr = 0, BM: tl.constexpr = 1,
):  # fmt: skip
    """Program (m, j): ``_ring_out`` (``RQ``: rows a request; 0: one request, the grid's rows)."""
    rq = tl.num_programs(1) if RQ == 0 else RQ
    _ring_out(tl.program_id(0), tl.program_id(1), rq, pool, positions, batch_ids, slots, rings, layer_ids, ring_start0,
              slot_rows, ring_slots, num_slots, max_pos, WIN, DIM, BM)  # fmt: skip


@triton.jit
def _req_words_kernel(
    positions, batch_ids, slots, num_slots, max_pos, req, pos_out, rows,
    NB: tl.constexpr, BM: tl.constexpr, RQP: tl.constexpr,
):  # fmt: skip
    """Program b: ``_req_words`` of request b (grid (NB,); ``rows`` = NB requests' rows); program 0 also
    ``_dead_words``."""
    b = tl.program_id(0)
    _req_words(b, rows // NB, positions, batch_ids, slots, num_slots, max_pos, req, RQP)
    if b == 0:
        _dead_words(req, pos_out, rows, NB, BM)


@triton.jit
def _engram_fold_kernel(rq, cursor, cursor_stride, slots, positions, ids_src, ids_dst, tmap, dead, cand, cand_stride,
                        cand_req_stride, live, NGRAM1: tl.constexpr, RPOW: tl.constexpr):  # fmt: skip
    """Program b: ``_eng_fold`` of request b (the CPU check's)."""
    _eng_fold(tl.program_id(0), rq, cursor, cursor_stride, slots, positions, ids_src, ids_dst, tmap, dead, cand,
              cand_stride, cand_req_stride, live, NGRAM1, RPOW)  # fmt: skip


@triton.jit
def _engram_hist_kernel(
    cursor, cursor_stride, slots, positions, hist, NGRAM1: tl.constexpr
):
    """One program: ``_eng_hist``."""
    _eng_hist(cursor, cursor_stride, slots, positions, hist, NGRAM1)


@triton.jit
def _engram_rows_kernel(src, src_stride, dst, KIN: tl.constexpr, BLOCK: tl.constexpr):
    """Program j: ``_eng_row`` (the separate-kernel reference of the fused begin-step)."""
    _eng_row(tl.program_id(0), src, src_stride, dst, KIN, BLOCK)


@triton.jit
def _expand_rows_kernel(hidden, hidden_stride, h_in, emb_ids, emb_rows, emb_per, HIDDEN: tl.constexpr,
                        BLOCK: tl.constexpr, NHC: tl.constexpr, EMB: tl.constexpr):  # fmt: skip
    """Program j: ``_expand_row`` (the CPU check's)."""
    _expand_row(tl.program_id(0).to(tl.int64), hidden, hidden_stride, h_in, emb_ids, emb_rows, emb_per, HIDDEN, BLOCK,
                NHC, EMB)  # fmt: skip


@triton.jit
def _collapse_rows_kernel(gate, res, out, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr,
                          NMIX: tl.constexpr):  # fmt: skip
    """Program (j, c): ``_collapse_chunk`` (the CPU check's; launched with ``enable_fp_fusion=False`` as the end-step
    is)."""
    _collapse_chunk(
        tl.program_id(0), tl.program_id(1), gate, res, out, HIDDEN, BLOCK, NHC, NMIX
    )


@triton.jit
def _gates_planar_kernel(gate, pre, post, comb, NMIX: tl.constexpr, NHC: tl.constexpr):
    """Program j: ``_planar``."""
    _planar(tl.program_id(0), gate, pre, post, comb, NMIX, NHC)


@triton.jit
def _atom_to_mf_kernel(
    res, pend, pre, post, comb, h_in, pend_out, gin,
    HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NMIX: tl.constexpr, NHC: tl.constexpr,
):  # fmt: skip
    """Program (j, c): ATOM's state row j -> the hi runtime's plain layer input; c < 4
    copies stream c of the residual, c == 4 the pending output and the gates
    (pre | post | comb, comb[i_in][o_out] row-major as ATOM keeps it)."""
    j = tl.program_id(0)
    c = tl.program_id(1)
    for o in range(0, HIDDEN, BLOCK):
        d = o + tl.arange(0, BLOCK)
        if c < NHC:
            at = (j * NHC + c) * HIDDEN + d
            tl.store(h_in + at, tl.load(res + at))
        else:
            tl.store(pend_out + j * HIDDEN + d, tl.load(pend + j * HIDDEN + d))
    if c == NHC:
        g = tl.arange(0, 32)
        v = tl.load(pre + j * NHC + g, mask=g < NHC, other=0.0)
        v += tl.load(
            post + j * NHC + (g - NHC), mask=(g >= NHC) & (g < 2 * NHC), other=0.0
        )
        v += tl.load(
            comb + j * NHC * NHC + (g - 2 * NHC),
            mask=(g >= 2 * NHC) & (g < NMIX),
            other=0.0,
        )
        tl.store(gin + j * NMIX + g, v, mask=g < NMIX)


@triton.jit
def _aux_tap_kernel(
    r_a, aux, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, NHC: tl.constexpr
):
    """Program j: ``_tap``."""
    _tap(tl.program_id(0), r_a, aux, HIDDEN, BLOCK, NHC)
