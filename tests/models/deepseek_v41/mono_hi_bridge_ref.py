# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reference launches for ``test_mono_hi_bridge``: each of the fused bridge's bodies (``hi_bridge``) as a kernel
of its own, the separate launches the fused begin-step / end-step replace, and the per-request verdict the in-graph
post-verify is checked against. Loaded by the test with the layout's jit helpers in its namespace (``_ring_in``,
``_req_words``, ...); not a module of its own."""

# ruff: noqa: F821  (the layout's jit helpers are injected into this module's namespace by the test)

import triton
import triton.language as tl


@triton.jit
def _import_rings_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids, pos_out, bounds_out,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos, rows,
    WIN: tl.constexpr, DIM: tl.constexpr, RB: tl.constexpr, NB: tl.constexpr = 1, BM: tl.constexpr = 1,
    KV8: tl.constexpr = 0,
):  # fmt: skip
    """Program (m, s, b): ``_ring_in`` (grid (NMF, WIN, NB); ``rows`` = NB requests' rows)."""
    _ring_in(tl.program_id(0), tl.program_id(2), tl.program_id(1), rows // NB, pool, positions, batch_ids, slots,
             rings, layer_ids, pos_out, bounds_out, ring_start0, slot_rows, ring_slots, num_slots, max_pos, WIN, DIM,
             RB, BM, KV8)  # fmt: skip


@triton.jit
def _export_rings_kernel(
    pool, positions, batch_ids, slots, rings, layer_ids,
    ring_start0, slot_rows, ring_slots, num_slots, max_pos,
    WIN: tl.constexpr, DIM: tl.constexpr, RQ: tl.constexpr = 0, BM: tl.constexpr = 1, KV8: tl.constexpr = 0,
):  # fmt: skip
    """Program (m, j): ``_ring_out`` (``RQ``: rows a request; 0: one request, the grid's rows)."""
    rq = tl.num_programs(1) if RQ == 0 else RQ
    _ring_out(tl.program_id(0), tl.program_id(1), rq, pool, positions, batch_ids, slots, rings, layer_ids, ring_start0,
              slot_rows, ring_slots, num_slots, max_pos, WIN, DIM, BM, KV8)  # fmt: skip


@triton.jit
def _kv8_store_kernel(src, pool, DIM: tl.constexpr):
    """Program r: bf16 row r of ``src`` [rows, DIM] -> ATOM's MXFP8 row r of ``pool`` (``_kv8_store``)."""
    r = tl.program_id(0)
    v = tl.load(src + r.to(tl.int64) * DIM + tl.arange(0, DIM))
    _kv8_store(pool, r.to(tl.int64), v, r >= 0, DIM)


@triton.jit
def _kv8_load_kernel(pool, dst, DIM: tl.constexpr):
    """Program r: ATOM's MXFP8 row r of ``pool`` -> bf16 row r of ``dst`` [rows, DIM] (``_kv8_load``)."""
    r = tl.program_id(0)
    tl.store(dst + r.to(tl.int64) * DIM + tl.arange(0, DIM), _kv8_load(pool, r.to(tl.int64), r >= 0, DIM))


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
def _expand_rows_kernel(
    hidden,
    hidden_stride,
    h_in,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    NHC: tl.constexpr,
):
    """Program j: ``_expand_row`` (the CPU check's)."""
    _expand_row(
        tl.program_id(0).to(tl.int64), hidden, hidden_stride, h_in, HIDDEN, BLOCK, NHC
    )


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


@triton.jit(do_not_specialize=["num_spec_steps", "seed"])
def _head_verdict_kernel(
    out_ptr,  # int32 [requests, num_spec_steps + 1]
    verdict_ptr,  # int32 [3, requests]: accepted drafts, rejected, anchor row
    cu_num_draft_ptr,  # [requests]
    input_ids_ptr,  # the step's input ids; draft r is input_ids[target_idx[r] + 1]
    target_idx_ptr,  # [verify rows]: each verify row's row of the step
    bonus_idx_ptr,  # [requests]: each request's bonus row
    head_ids_ptr,  # [rows of the step]: the head's id a row
    uniform_ptr,  # [verify rows] f32 (SYNTHETIC and not DRAW)
    cond_rates_ptr,  # [num_spec_steps] f32 (SYNTHETIC)
    seed,
    num_spec_steps,
    SYNTHETIC: tl.constexpr,
    DRAW: tl.constexpr,
):
    """The verdict reference the in-graph post-verify (``round.post_verify._post_verify_kernel``) is checked against,
    a program a request: ATOM's ``rejection_greedy_sample_kernel`` / ``rejection_synthetic_sample_kernel`` +
    ``_finish_request`` on the head's ids read through the row indices (the two ``index_select``), the uniforms given
    (ATOM's) or drawn here (``DRAW``, as the post-verify draws them).
    """
    req = tl.program_id(0)
    if req == 0:
        start = 0
    else:
        start = tl.load(cu_num_draft_ptr + req - 1)
    n = tl.load(cu_num_draft_ptr + req) - start
    INVALID: tl.constexpr = -1
    row_out = out_ptr + req * (num_spec_steps + 1)
    rejected = False
    last = -1
    for pos in range(n):
        if rejected:
            oid = INVALID
        else:
            r = start + pos
            vrow = tl.load(target_idx_ptr + r)
            target = tl.load(head_ids_ptr + vrow).to(tl.int32)
            draft = tl.load(input_ids_ptr + vrow + 1).to(tl.int32)
            if SYNTHETIC:
                if DRAW:
                    u = tl.rand(seed, r)
                else:
                    u = tl.load(uniform_ptr + r)
                if u < tl.load(cond_rates_ptr + pos):
                    oid = draft
                else:
                    oid = target
                    rejected = True
            else:
                oid = target
                if draft != target:
                    rejected = True
            last += 1
        tl.store(row_out + pos, oid)
    accepted = last
    brow = tl.load(bonus_idx_ptr + req)
    if rejected:
        bonus = INVALID
    else:
        bonus = tl.load(head_ids_ptr + brow).to(tl.int32)
        accepted += 1
    tl.store(row_out + n, bonus)
    for pos in range(n + 1, num_spec_steps + 1):
        tl.store(row_out + pos, INVALID)
    nreq = tl.num_programs(0)
    tl.store(verdict_ptr + req, accepted)
    tl.store(verdict_ptr + nreq + req, num_spec_steps - accepted)
    tl.store(verdict_ptr + 2 * nreq + req, brow - n + accepted)
