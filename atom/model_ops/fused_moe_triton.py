# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py
# Copyright 2023 The vLLM team.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import prod

import torch
import triton
import triton.language as tl
from aiter import ActivationType
from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
from aiter.ops.triton.utils._triton.arch_info import get_arch

from atom.utils import envs

if (
    envs.ATOM_USE_TRITON_GEMM
    or envs.ATOM_USE_TRITON_MOE
    or envs.ATOM_USE_TRITON_MOE_DECODE
    # The EP path reaches these same helpers from inside the modular kernel, so
    # it must pull the imports in too -- otherwise they are NameErrors at call
    # time when only the EP flag is set.
    or envs.ATOM_USE_TRITON_MOE_EP
):
    # Module level, not function local: `_ep_scatter_expt_data_kernel` calls these
    # from inside a @triton.jit body, so they must be resolvable in the kernel's
    # global namespace when it is compiled on first launch.
    from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
        _expt_data_compute_stage1,
        _expt_data_compute_stage2,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        moe_gemm_a4w4,
        mxfp4_quant,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import (
        moe_gemm_a8w4,
    )
    from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
        moe_gemm_a16w4,
    )
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, downcast_to_static_fp8
    from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

from atom.model_ops.moe import MoEActivationQuant


def _swizzle_mxfp4(
    w1,
    w1_scale,
    w2,
    w2_scale,
    w_dtype,
    N_1,
    K_1,
    N_2,
    K_2,
    TP=1,
):
    """Weight swizzle for mxfp4 moe, used for aiter triton mxfp4 moe kernels.

    The arch -> SWIZZLE_MX_SCALE label decision lives in aiter
    (``shuffle_scale_moe(..., return_layout=True)``), so this stays arch-agnostic.
    """
    assert envs.ATOM_USE_TRITON_GEMM or envs.ATOM_USE_TRITON_MOE

    # Transposing for expected layout of aiter triton kernels
    w1_triton_layout = w1.transpose(-2, -1)
    w1_scale_triton_layout = w1_scale.transpose(-2, -1)
    w2_triton_layout = w2.transpose(-2, -1)
    w2_scale_triton_layout = w2_scale.transpose(-2, -1)

    if N_1 % 32 == 0 and K_1 % (32 * 8) == 0:
        w1_scale_triton_layout, w1_swizzle_layout = shuffle_scale_moe(
            w1_scale_triton_layout, return_layout=True
        )
    else:
        w1_swizzle_layout = None

    if N_2 % 32 == 0 and K_2 % (32 * 8) == 0:
        w2_scale_triton_layout, w2_swizzle_layout = shuffle_scale_moe(
            w2_scale_triton_layout, return_layout=True
        )
    else:
        w2_swizzle_layout = None

    return (
        w1_triton_layout,
        w1_scale_triton_layout,
        w1_swizzle_layout,
        w2_triton_layout,
        w2_scale_triton_layout,
        w2_swizzle_layout,
    )


@triton.jit
def _ep_gate_prep_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids, row-major
    ExpertMap,  # (E_map,) int32, global id -> local id, -1 if not owned here
    NumLocalTokens,  # (1,) int32, valid row count R (device-side)
    GateValid,  # (G,) int32 out
    ExptIndx,  # (G,) int32 out, local id or SENTINEL
    PartialHist,  # (n_ctas, N_BINS) int32 out, this CTA's private per-bin counts
    n_gates,
    e_map_numel,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,  # next_pow2(SENTINEL + 1)
    BLOCK: tl.constexpr,
):
    """Kernel A: gate gating + per-CTA private histogram. No atomics.

    Replaces ~9 elementwise launches that all share the G = M*topk axis:

        ids   = dispatch_ids.long().clamp_(0, E_map-1)
        local = expert_map[ids]
        local = where(row < R, local, -1)
        gate_valid = (local >= 0).reshape(-1).int()
        expt_indx  = where(local < 0, SENTINEL, local).reshape(-1).int()

    The histogram is built as a one-hot reduction inside each CTA rather than
    with `atomic_add` into shared bins: at prefill G is ~786k against 49 bins,
    which serialises badly. Each CTA writes its own row of PartialHist, and the
    cross-CTA combine happens in the scan kernel. This is the same two-level
    scheme aiter's sort_tokens gets from `bitmatrix.sum(partials_block_size=...)`
    -- we just build the partials directly instead of via a bitmatrix, which we
    do not have post-dispatch.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_gates

    # Garbage rows can hold out-of-range ids; clamp so the ExpertMap gather stays
    # in bounds. The last slot of ExpertMap is the sentinel entry, which is -1.
    ids = tl.load(DispatchIds + offs, mask=mask, other=0)
    ids = tl.minimum(tl.maximum(ids, 0), e_map_numel - 1)
    local = tl.load(ExpertMap + ids, mask=mask, other=-1)

    # Rows past R hold garbage from the over-allocated receive buffer.
    row = offs // TOPK
    r = tl.load(NumLocalTokens)
    valid = (local >= 0) & (row < r) & mask

    tl.store(GateValid + offs, valid.to(tl.int32), mask=mask)
    expt = tl.where(valid, local, SENTINEL).to(tl.int32)
    tl.store(ExptIndx + offs, expt, mask=mask)

    # Private per-bin counts: (BLOCK, N_BINS) one-hot reduced along the gate axis.
    # Out-of-range lanes are excluded via `mask`, or they would all land in bin 0.
    bins = tl.arange(0, N_BINS)
    onehot = ((expt[:, None] == bins[None, :]) & mask[:, None]).to(tl.int32)
    tl.store(PartialHist + pid * N_BINS + bins, tl.sum(onehot, 0))


@triton.jit
def _ep_scan_partials_kernel(
    PartialHist,  # (n_ctas, N_BINS) int32 in
    CtaBase,  # (n_ctas, N_BINS) int32 out, exclusive scan down the CTA axis
    Hist,  # (N_BINS,) int32 out, column totals
    n_ctas,
    N_BINS: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    """Kernel A': exclusive scan of PartialHist down the CTA axis, per bin.

    Grid is (N_BINS,) -- one program per bin, so the bins run in parallel and
    each does a strided sequential scan over n_ctas. Gives every (CTA, bin) pair
    its own write cursor, which is what lets the scatter avoid atomics entirely.

    Separate launch by necessity: A's axis is G and this one's is the bin axis, so
    no CTA arrangement serves both, and the scan needs every CTA's partial counts.
    """
    e = tl.program_id(0)
    acc = tl.zeros([1], dtype=tl.int32)
    for c0 in range(0, n_ctas, C_BLOCK):
        offs = c0 + tl.arange(0, C_BLOCK)
        m = offs < n_ctas
        v = tl.load(PartialHist + offs * N_BINS + e, mask=m, other=0)
        tl.store(CtaBase + offs * N_BINS + e, tl.cumsum(v, 0) - v + acc, mask=m)
        acc += tl.sum(v, 0)
    tl.store(Hist + e, tl.sum(acc, 0))


@triton.jit
def _ep_scatter_body(
    pid,  # gate-axis CTA index (NOT program_id -- v2 offsets it)
    ExptIndx,  # (G,) int32 in
    DispatchWeights,  # (M, topk) f32 in, read flat as (G,)
    CtaBase,  # (n_ctas, N_BINS) int32 in, from A'
    Hist,  # (N_BINS,) int32 in, from A'
    GatherIndx,  # (G,) int32 out == topk_indx
    ScatterIndx,  # (G,) int32 out == gate_indx (inverse permutation)
    GateScal,  # (G,) f32 out, router weights in sorted order
    n_gates,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Kernel B's body: the atomic-free scatter into expert-sorted order.

    Takes `pid` as an argument rather than reading `tl.program_id(0)`, because it
    occupies the upper part of v2's wider grid and must be shifted down by the
    expt_data CTA count.

    Deterministic, atomic-free scatter into expert-sorted order.

    Destination of gate `i` in CTA `c` with bin `e`:

        pos = bin_base[e] + cta_base[c][e] + rank_of_i_within_(c, e)

    `bin_base` is the exclusive cumsum of Hist over bins -- only N_BINS (64)
    elements, so every CTA recomputes it rather than reading a separate buffer.
    `rank` comes from an exclusive cumsum of the same one-hot matrix Kernel A
    built; recomputing it is cheaper than storing (BLOCK, N_BINS) per CTA, and is
    the same trade aiter's P23 makes when it re-reads topk_ids.

    Because the sentinel is the largest bin, live gates tile [0, sum(hist[:E]))
    and dead gates land in the tail -- so the matmul, which only reads
    GatherIndx[start_m + j] for j < hist[e], never sees them.

    Both permutations and the permuted weights fall out of the same `pos`, so
    GateScal cannot drift out of alignment, and outputs are int32 directly (no
    trailing .int() copies).
    """
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_gates

    expt = tl.load(ExptIndx + idx, mask=mask, other=0)
    bins = tl.arange(0, N_BINS)
    h = tl.load(Hist + bins)
    bin_base = tl.cumsum(h, 0) - h
    cta = tl.load(CtaBase + pid * N_BINS + bins)

    onehot = ((expt[:, None] == bins[None, :]) & mask[:, None]).to(tl.int32)
    rank = tl.sum((tl.cumsum(onehot, 0) - onehot) * onehot, 1)
    base = tl.sum((bin_base + cta)[None, :] * onehot, 1)
    pos = base + rank

    tl.store(GatherIndx + pos, idx.to(tl.int32), mask=mask)
    tl.store(ScatterIndx + idx, pos.to(tl.int32), mask=mask)
    w = tl.load(DispatchWeights + idx, mask=mask, other=0.0)
    tl.store(GateScal + pos, w.to(tl.float32), mask=mask)


@triton.jit
def _ep_scatter_expt_data_kernel(
    # --- scatter half, pid >= N_EXPTS ---
    ExptIndx,  # (G,) int32 in
    DispatchWeights,  # (M, topk) f32 in
    CtaBase,  # (n_ctas, N_BINS) int32 in
    Hist,  # (N_BINS,) int32 in -- shared by BOTH halves
    GatherIndx,  # (G,) int32 out
    ScatterIndx,  # (G,) int32 out
    GateScal,  # (G,) f32 out
    n_gates,  # also read by stage1 for TokenStart[N_EXPTS]
    # --- expt_data half, pid < N_EXPTS ---
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw
    TileStart,  # (N_EXPTS+1,) int32 out == token_offs_pad
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map
    max_num_tiles,
    N_EXPTS: tl.constexpr,  # == blocks1a, the expt_data CTA count
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,  # next_pow2(N_EXPTS), stage1's tile width
    EQUAL_A: tl.constexpr,  # N_EXPTS == BLOCK_A
    N_BINS: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
):
    """v2 Kernel B+C: scatter and ExptData in ONE launch, split by CTA index.

    Grid is ``(N_EXPTS + n_ctas,)``, borrowing the layout of aiter's
    ``_combined_routing_fused`` (``blocks1a + blocks1b`` with an ``if pid <
    blocks1a`` split). Legal because, given ``Hist``, the two halves are
    independent: both only *read* the histogram, and their writes are disjoint --
    scatter owns GatherIndx/ScatterIndx/GateScal, expt_data owns
    TokenStart/TileStart/MDTileInfo.

    Two deliberate differences from the aiter reference:

    1. **No grid-wide dependency, so no barrier before the split.** The reference
       must build its histogram inside the same launch, which it can only do
       because ``_sum_bitmatrix_rows_fused`` has no ``pid`` in it: *every* CTA
       recomputes the whole histogram from the compact bitmatrix and plain-stores
       identical values. Its ``tl.debug_barrier()`` is intra-CTA and would not
       have made a cooperative reduction safe. We have no bitmatrix, so our
       histogram comes from ``_ep_gate_prep`` + ``_ep_scan_partials`` -- a real
       cross-CTA reduction, hence a hard launch boundary that stays in place. It
       also means Hist is already visible here and needs no barrier.
    2. **stage1 is guarded by the branch.** The reference calls it for all pids,
       including the gate-axis half; at our prefill that is ~3000 CTAs each
       redundantly prefix-summing 48 bins for nothing. ``pid == 0``, which owns
       stage1's terminal writes and the 0xFFFFFFFF tail memset, is still inside
       the guard.

    The one barrier we do keep is *within* the expt_data half: stage1 writes
    TileStart lane-wise and stage2 reads TileStart[pid] as a scalar, so the
    producing lane need not be the consuming one. `pid` is CTA-uniform, so the
    branch is uniform and the barrier is well-formed.

    Cost of branch fusion: one register/LDS budget and one `num_warps` for both
    halves, sized by whichever is heavier (the scatter's (GATE_BLOCK, N_BINS)
    one-hot). Measure before trusting it -- see _test/bench_ep_sort.py.
    """
    pid = tl.program_id(0)
    if pid >= N_EXPTS:
        _ep_scatter_body(
            pid - N_EXPTS,
            ExptIndx,
            DispatchWeights,
            CtaBase,
            Hist,
            GatherIndx,
            ScatterIndx,
            GateScal,
            n_gates,
            N_BINS=N_BINS,
            BLOCK=GATE_BLOCK,
        )
    else:
        # Hist has N_BINS entries but stage1 is told N_EXPTS, so it masks the
        # sentinel bin out of both prefix sums -- the matmul must schedule no
        # tile for gates this rank does not own.
        _expt_data_compute_stage1(
            pid,
            Hist,
            N_EXPTS,
            TokenStart,
            TileStart,
            MDTileInfo,
            max_num_tiles,
            n_gates,
            tile_dim_log2,
            BLOCK_A,
            EQUAL_A,
        )
        tl.debug_barrier()
        # Last statement in the kernel on purpose: stage2 early-returns for
        # empty experts, so nothing may follow it in this branch.
        _expt_data_compute_stage2(pid, Hist, TileStart, MDTileInfo, tile_dim_log2)


@triton.jit
def _ep_mesh_build_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32
    Mesh,  # (E, M) uint8 out: slot+1 if this token picked expert e, else 0
    Hist,  # (E,) int32 out
    GateValid,  # (G,) int32 out, written by program 0 only
    M,
    e_map_numel,
    TOPK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    """v1 phase 1 -- mirrors flydsl's MoeSortingMultiPhaseKernel_P0_v2.

    Grid is (E,): one program per LOCAL expert, which scans every gate and writes
    its own contiguous mesh row. That layout is the whole point -- phase 2 then
    walks a single row and emits a contiguous run, so it needs no rank arithmetic
    and its stores coalesce. flydsl calls this `p_expert_mesh` [expert, tokens].

    The trade versus v2: E-fold redundant reads of DispatchIds/ExpertMap, in
    exchange for exact per-expert counts with no atomics and no cross-CTA
    histogram barrier (each program owns a row, so `Hist[e]` is just its own
    count). The row is written in full including zeros, so no memset is needed.

    ASSUMPTION -- a token's topk selections must be DISTINCT experts. The mesh
    holds one slot per (expert, token) cell, so if a token named the same expert
    twice the second write overwrites the first and that gate is silently dropped
    (measured: hist 27 vs 28 on synthetic duplicate ids). Real topk selects
    distinct experts, and v2 has no such constraint because each gate is
    independent -- so if a router ever emits duplicates, v2 is the correct path.
    flydsl's decode mesh is `int` rather than `unsigned char`, which would allow a
    per-slot bitmask instead; worth revisiting if duplicates are ever possible.

    GateValid is gate-indexed, not expert-indexed, so program 0 alone emits it
    while it already has `local` in registers. Avoids both a memset and a race.
    """
    e = tl.program_id(0)
    cnt = tl.zeros([1], dtype=tl.int32)
    r = tl.load(NumLocalTokens)
    for t0 in range(0, M, T_BLOCK):
        rows = t0 + tl.arange(0, T_BLOCK)
        rmask = rows < M
        sel = tl.zeros([T_BLOCK], dtype=tl.int32)
        for sl in tl.static_range(TOPK):
            g = rows * TOPK + sl
            ids = tl.load(DispatchIds + g, mask=rmask, other=0)
            ids = tl.minimum(tl.maximum(ids, 0), e_map_numel - 1)
            local = tl.load(ExpertMap + ids, mask=rmask, other=-1)
            live = (local >= 0) & (rows < r) & rmask
            sel = tl.where(live & (local == e), sl + 1, sel)
            if e == 0:
                tl.store(GateValid + g, live.to(tl.int32), mask=rmask)
        tl.store(Mesh + e * M + rows, sel.to(tl.uint8), mask=rmask)
        cnt += tl.sum((sel > 0).to(tl.int32), 0)
    tl.store(Hist + e, tl.sum(cnt, 0))


@triton.jit
def _ep_mesh_scatter_kernel(
    Mesh,  # (E, M) uint8 in
    Hist,  # (E,) int32 in
    DispatchWeights,  # (M, topk) f32 in, read flat
    GatherIndx,  # (G,) int32 out
    ScatterIndx,  # (G,) int32 out, PRE-ZEROED (dead gates keep 0)
    GateScal,  # (G,) f32 out
    M,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    TOPK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    """v1 phase 2 -- mirrors flydsl's MoeSortingMultiPhaseKernel_P23.

    Grid is (E,). Each program redundantly cumsums the E-element histogram to get
    its own base (48 elements, cheaper than a separate buffer + launch -- the same
    reason v2's scatter recomputes `bin_base`), then walks its mesh row and emits
    a CONTIGUOUS run of GatherIndx/GateScal. That contiguity is what v2 lacks:
    there a CTA's 256 gates scatter to up to 49 distant regions.

    Dead gates never appear in the mesh, so their ScatterIndx entry is left at the
    pre-zeroed 0 -- harmless because reduce_grouped clamps masked slots via
    indx_valid before dereferencing.
    """
    e = tl.program_id(0)
    bins = tl.arange(0, E_PAD)
    h = tl.load(Hist + bins, mask=bins < E, other=0)
    base = tl.sum(tl.where(bins < e, h, 0), 0)  # exclusive prefix over experts

    run = tl.zeros([1], dtype=tl.int32)
    for t0 in range(0, M, T_BLOCK):
        rows = t0 + tl.arange(0, T_BLOCK)
        rmask = rows < M
        sel = tl.load(Mesh + e * M + rows, mask=rmask, other=0).to(tl.int32)
        hit = sel > 0
        # Position within this expert's contiguous region.
        rank = tl.cumsum(hit.to(tl.int32), 0) - hit.to(tl.int32)
        pos = base + run + rank
        gate = rows * TOPK + (sel - 1)
        tl.store(GatherIndx + pos, gate.to(tl.int32), mask=hit)
        tl.store(ScatterIndx + gate, pos.to(tl.int32), mask=hit)
        w = tl.load(DispatchWeights + gate, mask=hit, other=0.0)
        tl.store(GateScal + pos, w.to(tl.float32), mask=hit)
        run += tl.sum(hit.to(tl.int32), 0)


def ep_sort_routing_v1(
    dispatch_weights,
    dispatch_ids,
    expert_map,
    num_local_experts,
    num_local_tokens,
    M,
    topk,
    n_gates,
):
    """Mesh prep+sort: 2 kernels, mirroring flydsl's opus_moe_sorting.

    P0-equivalent: grid (E,), each program writes its own [expert, token] mesh row
                   and its exact count -- no atomics, no histogram barrier.
    P23-equivalent: grid (E,), redundant E-element cumsum for the base, then walk
                   the row emitting a CONTIGUOUS run.

    Costs an (E, M) uint8 mesh and E-fold redundant reads of dispatch_ids, and
    buys coalesced stores plus no rank arithmetic -- the two things that make
    flydsl's sort 9.4 ms where the no-mesh scatter's is 30 ms.
    """
    device = dispatch_ids.device
    E = num_local_experts
    T_BLOCK = 256
    mesh = torch.empty((E, M), dtype=torch.uint8, device=device)
    hist = torch.empty(E, dtype=torch.int32, device=device)
    gate_valid = torch.empty(n_gates, dtype=torch.int32, device=device)
    _ep_mesh_build_kernel[(E,)](
        dispatch_ids,
        expert_map,
        num_local_tokens,
        mesh,
        hist,
        gate_valid,
        M,
        expert_map.numel(),
        TOPK=topk,
        T_BLOCK=T_BLOCK,
    )

    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    # Zeroed: dead gates are absent from the mesh, so phase 2 never writes them.
    # reduce_grouped clamps masked slots via indx_valid, so 0 is safe.
    gate_indx = torch.zeros(n_gates, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates, dtype=torch.float32, device=device)
    _ep_mesh_scatter_kernel[(E,)](
        mesh,
        hist,
        dispatch_weights,
        topk_indx,
        gate_indx,
        gate_scal,
        M,
        E=E,
        E_PAD=triton.next_power_of_2(E),
        TOPK=topk,
        T_BLOCK=T_BLOCK,
    )
    return hist, topk_indx, gate_indx, gate_scal, gate_valid


def ep_sort_routing_v2(
    dispatch_weights,
    dispatch_ids,
    expert_map,
    num_local_experts,
    num_local_tokens,
    M,
    topk,
    n_gates,
    expt_data_bufs,
):
    """No-mesh prep+sort with the scatter and ExptData stages fused: 3 kernels.

    A  : gating + per-CTA private histogram          (grid: n_ctas)
    A' : per-bin exclusive scan down the CTA axis    (grid: N_BINS)
    B+C: scatter | ExptData, split by CTA index      (grid: N_EXPTS + n_ctas)

    A and A' cannot join anything -- A' consumes every CTA's partials, which is a
    grid-wide reduction. But B and C are both pure consumers of `hist` with
    disjoint outputs, so they collapse into one launch. See
    `_ep_scatter_expt_data_kernel` for why this mirrors aiter's
    `_combined_routing_fused` layout without needing its barrier.

    `expt_data_bufs` is the tuple `_compute_expt_data_internal` returns; the
    caller allocates it (it is allocation-only and version-independent) so v1 can
    keep launching the standalone `_expt_data_only_kernel` unchanged.

    Returns the same 5-tuple as v1; the ExptData buffers are filled in place.
    """
    device = dispatch_ids.device
    sentinel = num_local_experts
    GATE_BLOCK = 256
    n_bins = triton.next_power_of_2(sentinel + 1)
    n_ctas = triton.cdiv(n_gates, GATE_BLOCK)
    token_offs_raw, token_offs_pad, block_pid_map, blocks1, BLOCK_A, block_m_log2 = (
        expt_data_bufs
    )
    # blocks1a. _compute_expt_data_internal returns blocks1 == n_expts_tot; the
    # fused grid assumes that, since it recovers the gate CTA as pid - N_EXPTS.
    assert blocks1 == num_local_experts, f"{blocks1} != {num_local_experts}"

    gate_valid = torch.empty(n_gates, dtype=torch.int32, device=device)
    expt_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    partial_hist = torch.empty((n_ctas, n_bins), dtype=torch.int32, device=device)
    _ep_gate_prep_kernel[(n_ctas,)](
        dispatch_ids,
        expert_map,
        num_local_tokens,
        gate_valid,
        expt_indx,
        partial_hist,
        n_gates,
        expert_map.numel(),
        TOPK=topk,
        SENTINEL=sentinel,
        N_BINS=n_bins,
        BLOCK=GATE_BLOCK,
    )

    cta_base = torch.empty_like(partial_hist)
    hist_full = torch.empty(n_bins, dtype=torch.int32, device=device)
    _ep_scan_partials_kernel[(n_bins,)](
        partial_hist, cta_base, hist_full, n_ctas, N_BINS=n_bins, C_BLOCK=128
    )

    topk_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_indx = torch.empty(n_gates, dtype=torch.int32, device=device)
    gate_scal = torch.empty(n_gates, dtype=torch.float32, device=device)
    # hist_full, not hist_full[:E]: the scatter half needs the sentinel bin to
    # place dead gates in the tail, and stage1 masks at N_EXPTS anyway.
    _ep_scatter_expt_data_kernel[(num_local_experts + n_ctas,)](
        expt_indx,
        dispatch_weights,
        cta_base,
        hist_full,
        topk_indx,
        gate_indx,
        gate_scal,
        n_gates,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
        block_pid_map.shape[0],
        N_EXPTS=num_local_experts,
        tile_dim_log2=block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(num_local_experts == BLOCK_A),
        N_BINS=n_bins,
        GATE_BLOCK=GATE_BLOCK,
    )
    return hist_full, topk_indx, gate_indx, gate_scal, gate_valid


def routing_from_dispatched(
    dispatch_weights: torch.Tensor,
    dispatch_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
    num_local_tokens: torch.Tensor,
):
    """Build triton RoutingData / gather / scatter from mori-dispatched rows.

    The EP path cannot use aiter's ``routing()``: that starts from router logits,
    but after the all-to-all the top-k choice is already made and the rows have
    been permuted across ranks. This is ``routing_torch``'s second half --
    everything from ``(expt_scal, expt_indx)`` onward -- adapted for three facts
    about the post-dispatch buffer:

    1. Rows are per-token: mori sends one copy per (token, destination rank), so
       a row carries the full top-k tuple with only *some* entries owned here.
       Non-local entries go to a sentinel bin that is sliced off the histogram,
       so the matmul schedules no block for them.
    2. The flat gate index must stay ``row * topk + slot``, because the matmul
       recovers the activation row as ``gather_idx // N_EXPTS_ACT``. Non-local
       entries are therefore **masked, never compacted** -- compacting would
       break that arithmetic and silently read the wrong rows.
    3. ``num_local_tokens`` is a device tensor and rows past it hold garbage from
       the over-allocated receive buffer. Masking them the same way keeps this
       function sync-free and its shapes static (so it stays cudagraph-safe).
       It is REQUIRED, not optional: the mori buffer always has M > R, so
       skipping the row mask would fold garbage rows into the histogram as live
       gates -- silently wrong rather than an error.

    Returns ``(routing_data, gather_indx, scatter_indx, gate_valid)`` -- the first
    three match ``routing()``; ``gate_valid`` is the extra piece EP needs, since
    ``routing()`` never produces dead gates.
    """
    from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
        _expt_data_only_kernel,
    )
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        RoutingData,
        _compute_expt_data_internal,
    )

    M, topk = dispatch_ids.shape
    device = dispatch_ids.device
    n_gates = M * topk

    # gate_valid is in flat gate order (row * topk + slot) -- the same layout
    # scatter_indx uses, so reduce_grouped's .view(-1, n_expts_act) lines up
    # slot-for-slot. A dead slot's sorted position is never written by the GEMM
    # (the sentinel keeps the matmul off it), so the reduce must be told to skip
    # it rather than sum uninitialized memory.

    # Same derivation as routing_torch. Note n_gates counts every gate slot while
    # only ~1/topk are live under EP, so this overstates real per-expert
    # occupancy and picks larger tiles than the work needs. That is a
    # perf/tiling concern, not correctness: the matmul wraps its gather with
    # `offs_x_m % hist[e]` and masks stores with `offs_m < hist[e]`, so a
    # mostly-empty tile recomputes a live row rather than reading garbage.
    #
    # Hoisted above the version dispatch because it is allocation-only (no
    # kernel) and version-independent -- v2 needs the buffers up front so it can
    # fill them in the same launch as the scatter.
    tokens_per_expt = max(1, n_gates // max(num_local_experts, 1))
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data_bufs = _compute_expt_data_internal(
        num_local_experts, n_gates, block_m, device
    )
    token_offs_raw, token_offs_pad, block_pid_map, blocks1, BLOCK, block_m_log2 = (
        expt_data_bufs
    )

    version = envs.ATOM_EP_SORT_VERSION
    assert version in (1, 2), f"ATOM_EP_SORT_VERSION must be 1 or 2, got {version}"
    if version == 2:
        hist_full, topk_indx, gate_indx, gate_scal, gate_valid = ep_sort_routing_v2(
            dispatch_weights,
            dispatch_ids,
            expert_map,
            num_local_experts,
            num_local_tokens,
            M,
            topk,
            n_gates,
            expt_data_bufs,
        )
        hist = hist_full[:num_local_experts]
    else:
        hist_full, topk_indx, gate_indx, gate_scal, gate_valid = ep_sort_routing_v1(
            dispatch_weights,
            dispatch_ids,
            expert_map,
            num_local_experts,
            num_local_tokens,
            M,
            topk,
            n_gates,
        )
        hist = hist_full[:num_local_experts]

        # No fill_(-1): stage1's pid==0 memsets [tile_off_last, max_num_tiles)
        # and stage2 writes exactly [0, tile_off_last), so the union covers the
        # whole buffer. Verified with a poison fill -- zero survivors, including
        # empty experts where stage2 returns early.
        _expt_data_only_kernel[(blocks1,)](
            hist,
            num_local_experts,
            token_offs_raw,
            token_offs_pad,
            block_pid_map,
            block_pid_map.shape[0],
            n_gates,
            block_m_log2,
            BLOCK,
            EQUAL_BLOCK=(num_local_experts == BLOCK),
        )
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)

    routing_data = RoutingData(
        block_m, gate_scal, hist, num_local_experts, topk, expt_data
    )
    return routing_data, topk_indx, gate_indx, gate_valid


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def triton_kernel_moe_forward(
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    activation: ActivationType = ActivationType.Silu,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
) -> torch.Tensor:
    routing_data, gather_idx, scatter_idx = routing(
        gating_output, topk, sm_first=not renormalize
    )

    output = torch.empty_like(hidden_states)

    return triton_kernel_fused_experts(
        output,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_idx,
        scatter_idx,
        topk=topk,
        activation=activation,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        a13_scale=a13_scale,
        a2_scale=a2_scale,
        w13_swizzle_layout=w13_swizzle_layout,
        w2_swizzle_layout=w2_swizzle_layout,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        swiglu_limit=swiglu_limit,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        act_quant=act_quant,
    )


# This is a triton implementation of the fused_experts function
def triton_kernel_fused_experts(
    output_tensor: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,  # Tensor or triton_kernels.Tensor
    w2,  # Tensor or triton_kernels.Tensor
    routing_data,  # RoutingData
    gather_indx,  # GatherIndx -> tensor
    scatter_indx,  # ScatterIndx -> tensor
    topk: int,
    activation: ActivationType = ActivationType.Silu,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w13_swizzle_layout: torch.Tensor | None = None,
    w2_swizzle_layout: torch.Tensor | None = None,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    intermediate_cache: torch.Tensor | None = None,
    act_quant: MoEActivationQuant = MoEActivationQuant.BF16,
) -> torch.Tensor:
    # type check, uint8 means mxfp4
    assert hidden_states.dtype == torch.bfloat16
    assert w1_bias is None or w1_bias.dtype == torch.float32
    assert w2_bias is None or w2_bias.dtype == torch.float32

    # Shape check
    # Changes to weight handling before this function, therefore shape check change
    assert hidden_states.ndim == 2

    # aiter kernels expect 2d inputs/outputs
    M, K = hidden_states.shape[-2:]
    E, _, N = w1.shape

    if global_num_experts == -1:
        global_num_experts = E

    half_N = N // 2

    if intermediate_cache is None:
        intermediate_cache = torch.empty(
            (M * topk, half_N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    # Add batch_dim to output buffer because matmul_ogs expects 3D output
    intermediate_cache = _resize_cache(intermediate_cache, (M * topk, half_N))

    output_tensor = _resize_cache(output_tensor, (M, K))

    gammas = routing_data.gate_scal if routing_data else None

    if activation == ActivationType.Swiglu:
        # SwiGLU (GPT OSS): fused activation with interleaved [gate, up] layout
        if act_quant == MoEActivationQuant.FP8:
            assert a13_scale is not None
            assert a2_scale is not None

            quant_dtype = torch.float8_e4m3fn
            if get_arch() == "gfx942":
                quant_dtype = torch.float8_e4m3fnuz

            hidden_states = downcast_to_static_fp8(hidden_states, a13_scale)
            interm_cache = moe_gemm_a8w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                a13_scale,
                a2_scale,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                out_dtype=quant_dtype,
                apply_swiglu=True,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                swiglu_add_residual=True,
            )
            output_tensor = moe_gemm_a8w4(
                interm_cache,
                w2,
                None,
                w2_scale,
                a2_scale,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
        else:
            interm_cache = moe_gemm_a16w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=True,
                alpha=swiglu_alpha,
                limit=swiglu_limit,
                swiglu_add_residual=True,  # gpt-oss `(up + 1)`
            )
            output_tensor = moe_gemm_a16w4(
                interm_cache,
                w2,
                None,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
    else:
        # SiLU (DeepSeek): concatenated [gate | up] layout, manual activation.
        # The activation precision selects the routed GEMM: MXFP4 activations
        # (a4w4) when act_quant is FP4, otherwise bf16 activations (a16w4).
        if act_quant == MoEActivationQuant.FP8:
            raise NotImplementedError(
                "SiLU activation with FP8 act_quant is not implemented in the "
                "triton MoE kernel. Only the SwiGLU branch supports FP8 "
                "activations (moe_gemm_a8w4)."
            )
        if act_quant == MoEActivationQuant.FP4:
            hidden_states_fp4, hidden_states_mx_scale = mxfp4_quant(hidden_states)
            raw_intermediate = moe_gemm_a4w4(
                hidden_states_fp4,
                w1,
                hidden_states_mx_scale,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=False,
            )
        else:
            raw_intermediate = moe_gemm_a16w4(
                hidden_states,
                w1,
                None,
                w13_scale,
                None,
                None,
                w1_bias,
                routing_data,
                gather_indx=gather_indx,
                gammas=gammas if apply_router_weight_on_input else None,
                swizzle_mx_scale=w13_swizzle_layout,
                apply_swiglu=False,
            )

        raw_2d = raw_intermediate.view(M * topk, N)
        intermediate_cache = intermediate_cache.view(M * topk, half_N)
        fused_clamp_act_mul(
            raw_2d,
            out=intermediate_cache,
            swiglu_limit=swiglu_limit,
            activation="silu",
            dtype_quant=None,
        )

        if act_quant == MoEActivationQuant.FP4:
            intermediate_fp4, intermediate_mx_scale = mxfp4_quant(intermediate_cache)
            output_tensor = moe_gemm_a4w4(
                intermediate_fp4,
                w2,
                intermediate_mx_scale,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )
        else:
            output_tensor = moe_gemm_a16w4(
                intermediate_cache,
                w2,
                None,
                w2_scale,
                None,
                None,
                w2_bias,
                routing_data,
                scatter_indx=scatter_indx,
                gammas=None if apply_router_weight_on_input else gammas,
                swizzle_mx_scale=w2_swizzle_layout,
            )

        return output_tensor

    output_tensor = output_tensor.view(M, K)
    return output_tensor


def triton_kernel_fused_experts_a8w4_silu_gguu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_swizzle_layout,
    w2_swizzle_layout,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """Decode-only A8W4 MoE for SiLU models, GGUU (separated ``[gate|up]``).

    GGUU keeps gate and up as contiguous halves, so the per-block SiLU cannot be
    fused into GEMM1's write-back (a tile spans only gate *or* only up). The
    activation and quant therefore run as a separate step:

        MXFP8 quant -> GEMM1(a8w4, no swiglu, bf16 [gate|up]) ->
        fused_clamp_act_mul(SiLU(gate)*up on the halves) ->
        MXFP8 quant -> GEMM2(a8w4).

    The intermediate is re-quantized with ``downcast_to_mxfp`` (same op as the x
    path) so GEMM2 sees the identical activation-scale format. Weights are in the
    preshuffled a8w4 layout with w13 gate/up separated.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16

    gammas = routing_data.gate_scal if routing_data else None

    x_fp8, x_scale = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)

    # GEMM1: raw bf16 [gate|up] output; no fused activation for the separated layout.
    interm = moe_gemm_a8w4(
        x_fp8,
        w1,
        x_scale,
        w13_scale,
        a13_scale,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=w13_swizzle_layout,
        apply_swiglu=False,
        out_dtype=torch.bfloat16,
        preshuffled=True,
    )

    # Standalone SiLU(gate)*up over the contiguous halves, then MXFP8 quant.
    interm_act = fused_clamp_act_mul(
        interm, swiglu_limit=swiglu_limit, activation="silu"
    )
    interm_fp8, interm_scale = downcast_to_mxfp(
        interm_act, torch.float8_e4m3fn, axis=-1
    )

    output_tensor = moe_gemm_a8w4(
        interm_fp8,
        w2,
        interm_scale,
        w2_scale,
        a2_scale,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=w2_swizzle_layout,
        preshuffled=True,
    )

    return output_tensor


def triton_kernel_fused_experts_a8w4_silu_gugu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_swizzle_layout,
    w2_swizzle_layout,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
    apply_router_weight_on_input: bool = False,
    gate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """A8W4 MoE for SiLU models, GUGU (interleaved ``[gate, up]``).

    Interleaved is the a8w4 kernel's native layout: ``_swiglu`` splits
    ``reshape(M, N // 2, 2)`` on the trailing axis, i.e. adjacent gate/up pairs,
    so a BLOCK_N tile carries both halves and the activation fuses into GEMM1's
    write-back. ``out_mx_quant=True`` folds the MXFP8 requant in with it, so the
    whole layer is two launches:

        MXFP8 quant -> GEMM1(a8w4, fused SiLU + MX requant) -> GEMM2(a8w4)

    versus four on the GGUU path (GEMM1 -> fused_clamp_act_mul ->
    downcast_to_mxfp -> GEMM2), which needs the separate steps precisely because
    a tile there spans only gate *or* only up.

    ``alpha=1.0`` with ``swiglu_add_residual=False`` is plain SiLU (``s * linear``).
    GPT-OSS uses ``swiglu_add_residual=True`` for its ``s * (linear + 1)`` variant,
    which would be wrong for DeepSeek-V4.
    """
    assert hidden_states.ndim == 2
    assert hidden_states.dtype == torch.bfloat16

    gammas = routing_data.gate_scal if routing_data else None

    # Only gfx1250's gluon kernel consumes the WMMA-preshuffled weight; the
    # CDNA triton kernel takes a plain (E, K, N) weight.
    _preshuffled = get_arch() == "gfx1250"

    x_fp8, x_scale = downcast_to_mxfp(hidden_states, torch.float8_e4m3fn, axis=-1)

    # GEMM1: SiLU(gate)*up fused into write-back, emitting (fp8 e4m3, ue8m0)
    # directly. out_mx_quant requires split_k == 1 and no scatter_indx, both of
    # which hold for a GEMM1-style call.
    interm_fp8, interm_scale = moe_gemm_a8w4(
        x_fp8,
        w1,
        x_scale,
        w13_scale,
        a13_scale,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=w13_swizzle_layout,
        apply_swiglu=True,
        alpha=1.0,
        limit=swiglu_limit,
        swiglu_add_residual=False,
        out_mx_quant=True,
        out_dtype=torch.float8_e4m3fn,
        preshuffled=_preshuffled,
    )

    return moe_gemm_a8w4(
        interm_fp8,
        w2,
        interm_scale,
        w2_scale,
        a2_scale,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=w2_swizzle_layout,
        preshuffled=_preshuffled,
        # Only GEMM2 feeds reduce_grouped, so the mask belongs here. GEMM1's
        # dead slots are already skipped by the sentinel histogram.
        gate_valid=gate_valid,
    )


def triton_kernel_fused_experts_a4w4_silu_gugu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_data,
    gather_indx,
    scatter_indx,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_swizzle_layout,
    w2_swizzle_layout,
    a13_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    swiglu_limit: float = 10.0,
    apply_router_weight_on_input: bool = False,
    gate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """A4W4 MoE for SiLU models, GUGU -- same signature as the a8w4 twin.

    Identical to ``triton_kernel_fused_experts_a8w4_silu_gugu`` except that the
    activations are MXFP4 rather than MXFP8. The WEIGHTS are unchanged: both are
    w4, so the same ``process_weights_after_loading`` output feeds either path
    and no extra weight prep or memory is needed.

    Costs one launch more than a8w4. ``moe_gemm_a4w4`` has no ``out_mx_quant``,
    so the intermediate must be re-quantised by a separate ``mxfp4_quant``
    instead of being folded into GEMM1's write-back:

        mxfp4_quant -> GEMM1(a4w4, fused SiLU) -> mxfp4_quant -> GEMM2(a4w4)

    versus a8w4's two launches. Measured on gfx950 that costs 1.22-1.28x overall
    (see _test/ep_moe_bench_report.md): the a4w4 GEMMs are genuinely faster
    (2771 vs 3154 us at conc256 prefill, the halved weight traffic paying off)
    but the doubled quant is far more expensive than the GEMM saving. Kept behind
    ATOM_USE_TRITON_MOE_EP_A4W4 so it is measurable on gfx1250, where the
    trade-off may differ.

    ``moe_gemm_a4w4`` has no ``preshuffled`` parameter, so unlike a8w4 there is no
    gfx1250 pre-shuffled weight variant to select here.
    """
    gammas = routing_data.gate_scal if routing_data else None

    x_fp4, x_scale = mxfp4_quant(hidden_states)
    interm = moe_gemm_a4w4(
        x_fp4,
        w1,
        x_scale,
        w13_scale,
        a13_scale,
        None,
        w1_bias,
        routing_data,
        gather_indx=gather_indx,
        gammas=gammas if apply_router_weight_on_input else None,
        swizzle_mx_scale=w13_swizzle_layout,
        apply_swiglu=True,
        alpha=1.0,
        limit=swiglu_limit,
        swiglu_add_residual=False,
    )
    # The launch a8w4 avoids via out_mx_quant=True.
    interm_fp4, interm_scale = mxfp4_quant(interm)
    return moe_gemm_a4w4(
        interm_fp4,
        w2,
        interm_scale,
        w2_scale,
        a2_scale,
        None,
        w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        gammas=None if apply_router_weight_on_input else gammas,
        swizzle_mx_scale=w2_swizzle_layout,
        # Only GEMM2 feeds reduce_grouped, so the mask belongs here. GEMM1's
        # dead slots are already skipped by the sentinel histogram.
        gate_valid=gate_valid,
    )
