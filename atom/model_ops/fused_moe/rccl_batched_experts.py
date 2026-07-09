# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Batched grouped-expert MoE compute for the native RCCL all2all backend.

The default RCCL path feeds the dispatched, token-sorted rows straight into
aiter ``fused_moe`` (a single token-major GEMM with internal expert masking).
This module implements an alternative: re-sort the dispatched rows into a dense
per-expert grid ``[local_experts, C, hidden]`` (zero-padding short experts) and
run a **strided-batched** w4a8 (MXFP8 activation x MXFP4 weight) GEMM per expert,
matching either the FlyDSL ``compile_mxfp8_gemm`` batch kernel or AITER's
Triton ``batched_gemm_a8wfp4`` kernel.

Selected via ``ATOM_RCCL_MOE_IMPL=flydsl_batched_gemm`` or
``ATOM_RCCL_MOE_IMPL=triton_batched_gemm`` (default ``default``).

Layout summary (E = local experts, C = per-expert capacity, H = hidden,
I = intermediate / inter_dim):

    dispatched rows (topk==1)                 [R, H]
      -> resort_to_batched                    [E, C, H]  (+ row_map, counts)
      -> gate_up batched w4a8 GEMM            [E, C, 2I]
      -> SwiGLU                                [E, C, I]
      -> down batched w4a8 GEMM              [E, C, H]
      -> unsort_from_batched                  [R, H]
      -> (caller) finalize / all2all back

The re-sort / un-sort math is routing-value independent in shape (C is fixed
from the capacity), so it is CUDA-graph safe and unit-testable without a GPU.
"""

import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger("atom")


def _use_fused(device) -> bool:
    """Triton fused kernels on CUDA unless ATOM_RCCL_FUSED_PACK disables them.

    Shares the toggle with the HT/LL pack fusion so a single env var flips the
    whole RCCL batched path back to the pure-torch reference (used by the
    mocked-dist CPU unit tests and for A/B).
    """
    return device.type == "cuda" and os.getenv("ATOM_RCCL_FUSED_PACK", "1") not in (
        "0",
        "false",
        "False",
    )


# --------------------------------------------------------------------------- #
# Fused resort / unsort Triton kernels.
#
# resort_to_batched and unsort_from_batched were each ~a dozen small torch ops
# (an [R, E] one-hot alloc + scatter_ + cumsum + gather for the slot assignment,
# then index_copy_ scatters; and for unsort an arange/where/scatter_/index_select
# gather-map). These kernels collapse each into two / one launch(es):
#   _resort_count_kernel : per-expert valid-row histogram (only needed to size C
#                          on the HT/eager path; == returned counts).
#   _resort_slot_kernel  : cheap atomic slot claim over [R] ints -> dst slot per
#                          row (-1 = dropped). Atomic serialization is decoupled
#                          from the big activation copy.
#   _resort_copy_kernel  : atomic-free 2D-tiled copy of each kept row's H values
#                          into the exact [E*C] grid (+ row_map). Runs at plain
#                          index_copy_ bandwidth; no dump tail.
#   _unsort_scatter_kernel: one program per grid slot; if occupied, scatter its
#                           H-row straight to out[row_map[slot]] (replaces the
#                           inv/covered build + index_select + masked multiply).
# All are CUDA-graph safe: static shapes, no host sync, no data-dependent
# indexing. Atomic slot order within an expert is nondeterministic but only the
# (row <-> slot) mapping matters (unsort inverts by the same row_map), so exact.
# --------------------------------------------------------------------------- #


@triton.jit
def _resort_count_kernel(
    ids_ptr,  # (R,) int global expert id per dispatched row (-1 = padding)
    counts_ptr,  # (E,) int32 per-expert valid-row histogram (pre-zeroed)
    R,
    E,
    ep_rank,
):
    """One program per dispatched row: atomically bump counts[local_e] for valid
    rows. Only used to size C on the HT/eager path (C = round32(max count));
    counts (clamped to C) are also the returned per-expert token counts."""
    r = tl.program_id(0)
    if r >= R:
        return
    gid = tl.load(ids_ptr + r).to(tl.int64)
    local_e = gid - ep_rank * E
    valid = (gid != -1) & (local_e >= 0) & (local_e < E)
    safe_e = tl.where(valid, local_e, 0)
    tl.atomic_add(counts_ptr + safe_e, tl.where(valid, 1, 0).to(tl.int32))


@triton.jit
def _resort_slot_kernel(
    ids_ptr,  # (R,) int global expert id per dispatched row (-1 = padding)
    cursor_ptr,  # (E,) int32 per-expert running slot counter (pre-zeroed)
    slot2row_ptr,  # (E*C,) int64 out: source row per grid slot (pre-filled -1)
    R,
    E,
    C,
    ep_rank,
):
    """CHEAP slot assignment (no H-row copy). One program per dispatched row:
    atomically claim the next slot in its local expert block; for kept rows write
    the INVERSE map slot2row[e*C+slot] = r. Dropped rows (padding / non-local /
    overflow) write nothing, so their slots keep the pre-filled -1. This map is
    both the copy kernel's source index AND the finalize row_map. Touches only
    int data, so the atomic serialization is decoupled from the big copy."""
    r = tl.program_id(0)
    if r >= R:
        return
    gid = tl.load(ids_ptr + r).to(tl.int64)
    local_e = gid - ep_rank * E
    valid = (gid != -1) & (local_e >= 0) & (local_e < E)
    safe_e = tl.where(valid, local_e, 0)
    slot = tl.atomic_add(cursor_ptr + safe_e, tl.where(valid, 1, 0).to(tl.int32))
    slot = slot.to(tl.int64)
    keep = valid & (slot < C)
    if keep:
        tl.store(slot2row_ptr + safe_e * C + slot, r)


@triton.jit
def _resort_copy_kernel(
    slot2row_ptr,  # (E*C,) int64 source row per grid slot (-1 = padding)
    a1_ptr,  # (R, H) dispatched activations
    grid_ptr,  # (E*C, H) out grid — this kernel OWNS the full write (no pre-zero)
    EC,
    H,
    BLOCK_H: tl.constexpr,
):
    """Grid-SLOT-major, atomic-free copy that owns the ENTIRE grid write: program
    (slot, h_block) reads its source row from slot2row and either copies that
    BLOCK_H chunk (occupied slot) or writes zeros (padding slot, slot2row == -1).
    Writing every slot exactly once here removes the separate ~E*C*H torch.zeros
    pre-zero (the old code zeroed the whole grid then overwrote ~half of it)."""
    slot = tl.program_id(0)
    if slot >= EC:
        return
    row = tl.load(slot2row_ptr + slot)
    h0 = tl.program_id(1) * BLOCK_H
    offs = h0 + tl.arange(0, BLOCK_H)
    mask = offs < H
    if row < 0:  # padding slot: zero-fill (grid is not pre-zeroed)
        tl.store(
            grid_ptr + slot * H + offs,
            tl.zeros((BLOCK_H,), grid_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        vals = tl.load(a1_ptr + row * H + offs, mask=mask)
        tl.store(grid_ptr + slot * H + offs, vals, mask=mask)


@triton.jit
def _unsort_scatter_kernel(
    flat_ptr,  # (E*C, H) expert outputs (row-major grid)
    rowmap_ptr,  # (E*C,) int64 source output-row per slot
    counts_ptr,  # (E,) int64 real token count per expert
    out_ptr,  # (num_rows, H) out (pre-zeroed)
    ec,  # E*C
    C,
    H,
    BLOCK_H: tl.constexpr,
):
    """One program per grid slot (e, s). Occupied iff s < counts[e]; if so, copy
    its H-row to out[row_map[slot]]. Each occupied slot maps to a unique output
    row, so there are no colliding writes; padding slots do nothing (out stays
    zero)."""
    slot = tl.program_id(0)
    if slot >= ec:
        return
    e = slot // C
    s = slot - e * C
    cnt = tl.load(counts_ptr + e)
    if s >= cnt:
        return
    dst = tl.load(rowmap_ptr + slot)  # output row
    for h0 in tl.range(0, H, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(flat_ptr + slot * H + offs, mask=mask)
        tl.store(out_ptr + dst * H + offs, vals, mask=mask)


# --------------------------------------------------------------------------- #
# Pure re-sort / un-sort (no GPU, no collectives) — unit-testable.
# --------------------------------------------------------------------------- #


def resort_to_batched(
    dispatch_a1: torch.Tensor,
    dispatch_ids: torch.Tensor,
    local_num_experts: int,
    capacity: Optional[int],
    ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter dispatched rows into a dense ``[E, C, H]`` per-expert grid.

    Args:
        dispatch_a1: ``[R, H]`` dispatched activations (one row = one token
            routed to exactly one expert; the topk==1 layout the RCCL prepare
            step produces).
        dispatch_ids: ``[R, 1]`` or ``[R]`` GLOBAL expert id per dispatched row.
            Rows with id ``-1`` are empty padding and are ignored.
        local_num_experts: number of experts owned by this rank (E).
        capacity: fixed per-expert row budget (C). Pass an int for a static
            shape (LL / CUDA-graph path). Pass ``None`` for the HT (eager) path:
            C is then the data-dependent ``max_num_token_per_local_expert`` this
            step, so no expert overflows and no compute is wasted on padding.
        ep_rank: this rank's EP index; local expert = global_id - ep_rank * E.

    Returns:
        batched: ``[E, C, H]`` zero-padded grid. ``batched[e, s]`` is the s-th
            token routed to local expert e (s < count[e]); the rest is zero.
        row_map: ``[E, C]`` int64. ``row_map[e, s]`` is the source row index in
            ``dispatch_a1`` that landed at ``[e, s]`` (clamped for empty slots;
            pair validity is tracked separately via ``counts``).
        counts: ``[E]`` int64, number of real tokens per local expert.

    With an explicit ``capacity`` the shapes depend only on (E, C, H) — never on
    routing values — so it is CUDA-graph safe; overflow rows are dropped. With
    ``capacity=None`` the shape depends on the data (a ``.item()`` host sync), so
    it is eager-only, but exact (no drops, no wasted pad).
    """
    if dispatch_ids.dim() == 2:
        ids = dispatch_ids.reshape(-1)
    else:
        ids = dispatch_ids
    R, H = dispatch_a1.shape
    device = dispatch_a1.device
    E = local_num_experts

    if _use_fused(device):
        return _resort_to_batched_fused(dispatch_a1, ids, E, capacity, ep_rank)

    # ------------------------------------------------------------------ #
    # Pure-torch reference (CPU / no-Triton / ATOM_RCCL_FUSED_PACK=0).
    # ------------------------------------------------------------------ #
    # Global -> local expert index. Non-local / padding rows (id == -1, or not
    # owned by this rank) map outside [0, E) and are routed to a dump lane.
    local_e = ids.to(torch.int64) - ep_rank * E  # [R]
    valid = (ids != -1) & (local_e >= 0) & (local_e < E)  # [R] bool

    # Per-expert slot via one-hot + exclusive cumsum over the R rows (same trick
    # as the LL dispatch). onehot is [R, E]; slot = # earlier rows to same expert.
    safe_e = torch.where(valid, local_e, torch.zeros_like(local_e))  # in [0, E)
    onehot = torch.zeros(R, E, dtype=torch.int64, device=device)
    onehot.scatter_(1, safe_e.unsqueeze(1), valid.to(torch.int64).unsqueeze(1))
    slot = (onehot.cumsum(0) - onehot).gather(1, safe_e.unsqueeze(1)).squeeze(1)  # [R]

    # HT (eager) path: size C to the actual max tokens per local expert so every
    # row fits and nothing is dropped or padded beyond need. per_expert = column
    # sums of the one-hot; C = max (host sync, allowed in eager).
    if capacity is None:
        # HT/eager: C = max tokens per local expert, rounded UP to a multiple of
        # 32. The FlyDSL w4a8 kernel indexes the A-side scale with a per-32
        # stride ((M+31)//32) and shuffle_scale_w4 requires the token dim to be a
        # multiple of 32; rounding also guarantees C >= 32 so an expert that
        # received 0 tokens still gets a valid all-padding grid, not zero-width.
        per_expert = onehot.sum(0)  # [E]
        C = int(per_expert.max().item()) if E > 0 else 0
        C = ((max(C, 1) + 31) // 32) * 32
    else:
        # LL/fixed path: caller sized C deliberately (graph_bs*topk); use as-is.
        C = capacity

    # A row is kept iff it is valid AND fits the capacity.
    keep = valid & (slot < C)  # [R] bool

    # Flat destination index into [E*C]; dropped rows -> unique dump lane so the
    # scatter never collides and shapes stay static.
    in_slot = safe_e * C + slot  # [R]
    dump = E * C + torch.arange(R, device=device)  # [R] unique
    dst = torch.where(keep, in_slot, dump)  # [R]

    # Scatter activations into [E*C + R, H]; keep only the [E*C] head.
    grid = torch.zeros((E * C + R, H), dtype=dispatch_a1.dtype, device=device)
    grid.index_copy_(0, dst, dispatch_a1)
    batched = grid[: E * C].reshape(E, C, H)

    # row_map[e, s] = source row that landed at (e, s); default 0 for empties.
    src_rows = torch.arange(R, device=device)
    rowmap_flat = torch.zeros(E * C + R, dtype=torch.int64, device=device)
    rowmap_flat.index_copy_(0, dst, src_rows)
    row_map = rowmap_flat[: E * C].reshape(E, C)

    counts = torch.zeros(E, dtype=torch.int64, device=device)
    counts.scatter_add_(0, safe_e, keep.to(torch.int64))

    return batched, row_map, counts


def _resort_to_batched_fused(
    dispatch_a1: torch.Tensor,
    ids: torch.Tensor,  # (R,) global expert id
    E: int,
    capacity: Optional[int],
    ep_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton fused resort: atomic slot-assign + single scatter. See the kernels
    above. Matches the pure-torch reference (per-expert token multiset + counts;
    intra-expert order differs but unsort inverts by the same row_map)."""
    R, H = dispatch_a1.shape
    device = dispatch_a1.device
    ids_i = ids.to(torch.int64).contiguous()
    a1_c = dispatch_a1.contiguous()

    from atom.utils import envs as _envs

    if capacity is None and _envs.ATOM_FORCE_BALANCE_FOR_BENCH:
        # BENCH force-balance: routing is a perfect round-robin, so every local
        # expert receives exactly ceil(R / E) rows -- the max per-expert count is
        # deterministic. Compute C directly (round up to 32) and SKIP the
        # histogram kernel + the .item() D2H entirely.
        C = ((((R + E - 1) // max(E, 1)) + 31) // 32) * 32 if E > 0 else 32
    elif capacity is None:
        # HT/eager only: the grid shape [E, C, H] is data-dependent, so C = max
        # tokens per local expert must be known host-side. Run a cheap histogram
        # kernel + a single [E] max/.item() (the tiny D2H is unavoidable for a
        # data-dependent shape; the old design ran a full C=R slot-assign here).
        hist = torch.zeros(E, dtype=torch.int32, device=device)
        if R > 0:
            _resort_count_kernel[(R,)](ids_i, hist, R, E, ep_rank)
        C = int(hist.max().item()) if E > 0 else 0
        C = ((max(C, 1) + 31) // 32) * 32
    else:
        # LL / fixed-capacity (CUDA-graph): C is a compile-time int, so NO count
        # kernel and NO .item() — the assign-scatter cursor gives counts on device.
        C = capacity

    # Grid is exactly [E*C] (no dump tail). Two decoupled kernels:
    #  1. slot-assign: cheap atomic cursor over [R] ints -> inverse map
    #     slot2row[grid_slot] = source row (kept rows only; padding stays -1).
    #     The atomic serialization touches only tiny int data.
    #  2. copy (grid-slot-major): owns the FULL [E*C, H] grid write -- copies each
    #     occupied slot's source row, zero-fills padding slots. This absorbs the
    #     grid zero-init (no separate ~E*C*H torch.zeros) and the row_map (slot2row
    #     IS the row_map). slot2row's -1 pre-fill is only [E*C] ints (~0.5MB) vs
    #     the 947MB grid zero it replaces.
    EC = E * C
    grid = torch.empty((EC, H), dtype=dispatch_a1.dtype, device=device)
    slot2row = torch.full((EC,), -1, dtype=torch.int64, device=device)
    cursor = torch.zeros(E, dtype=torch.int32, device=device)
    if R > 0:
        _resort_slot_kernel[(R,)](ids_i, cursor, slot2row, R, E, C, ep_rank)
        n_hblk = (H + 1024 - 1) // 1024
        _resort_copy_kernel[(EC, n_hblk)](slot2row, a1_c, grid, EC, H, BLOCK_H=1024)
    else:
        grid.zero_()  # no rows: whole grid is padding
    counts = torch.clamp(cursor.to(torch.int64), max=C)
    batched = grid.reshape(E, C, H)
    row_map = slot2row.reshape(E, C)
    return batched, row_map, counts


def unsort_from_batched(
    batched_out: torch.Tensor,
    row_map: torch.Tensor,
    counts: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    """Inverse of :func:`resort_to_batched`: scatter ``[E, C, H]`` back to ``[R, H]``.

    Args:
        batched_out: ``[E, C, H]`` per-expert expert outputs.
        row_map: ``[E, C]`` source-row index for each grid slot (from resort).
        counts: ``[E]`` real token count per expert (slots ``>= counts[e]`` are
            padding and must not be written back).
        num_rows: R, the dispatched row count (output leading dim).

    Returns ``[R, H]`` in dispatched-row order; untouched rows are zero.
    """
    E, C, H = batched_out.shape
    device = batched_out.device
    flat = batched_out.reshape(E * C, H)
    ec = E * C

    if _use_fused(device):
        # One program per grid slot: if occupied (s < counts[e]), copy its H-row
        # straight to out[row_map[slot]]. Replaces the slot arange + valid mask +
        # inv/covered scatter build + index_select + masked multiply.
        out = torch.zeros((num_rows, H), dtype=flat.dtype, device=device)
        _unsort_scatter_kernel[(ec,)](
            flat.contiguous(),
            row_map.reshape(-1).contiguous(),
            counts.contiguous(),
            out,
            ec,
            C,
            H,
            BLOCK_H=1024,
        )
        return out

    # ------------------------------------------------------------------ #
    # Pure-torch reference (CPU / no-Triton / ATOM_RCCL_FUSED_PACK=0).
    # ------------------------------------------------------------------ #
    # Valid-slot mask: slot s of expert e is real iff s < counts[e]. Built from
    # a fixed [E, C] arange compared to counts -> STATIC shape (no host sync).
    slot_ids = torch.arange(C, device=device).unsqueeze(0).expand(E, C)  # [E, C]
    valid = (slot_ids < counts.unsqueeze(1)).reshape(-1)  # [E*C] bool

    raw = row_map.reshape(-1)  # [E*C] output row each slot belongs to

    # Build inv[r] = flat slot index that holds output row r, WITHOUT boolean-mask
    # indexing (which yields a data-dependent shape and breaks CUDA-graph
    # capture). Instead scatter over ALL ec slots with a fixed shape:
    #   - valid slot  -> writes its own index at inv[row_map[slot]]
    #   - invalid slot -> routed to a dump row (num_rows) so it never clobbers a
    #     real row. inv/covered are sized num_rows+1; the dump tail is discarded.
    # Each real output row is the unique target of exactly one valid slot, so the
    # scatter has no colliding writes among valid slots.
    slot_idx = torch.arange(ec, device=device)  # [E*C]
    dst = torch.where(
        valid, raw, torch.full_like(raw, num_rows)
    )  # [E*C] -> [0,num_rows]

    inv = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
    covered = torch.zeros(num_rows + 1, dtype=torch.bool, device=device)
    inv.scatter_(0, dst, slot_idx)
    covered.scatter_(0, dst, torch.ones_like(dst, dtype=torch.bool))
    inv = inv[:num_rows]
    covered = covered[:num_rows]

    out = flat.index_select(0, inv)  # [num_rows, H]
    out = out * covered.unsqueeze(1).to(flat.dtype)  # zero any uncovered row
    return out


# --------------------------------------------------------------------------- #
# Batched expert MLP: quantize -> gate_up w4a8 GEMM -> SwiGLU -> down w4a8 GEMM.
# --------------------------------------------------------------------------- #


def _f32_to_e8m0_capture_safe(x: torch.Tensor) -> torch.Tensor:
    """CUDA-graph-safe f32 -> e8m0 (matches fp4_utils.f32_to_e8m0 bit-for-bit).

    The vendored helper does ``exponent[round_case] += 1`` /
    ``exponent[nan_case] = 0xFF`` — boolean-MASK in-place assignment, which on
    ROCm raises "operation not permitted when stream is capturing" during CUDA
    graph capture. Reexpress with ``torch.where`` (pure elementwise, static
    shape) so it is capturable.
    """
    u32 = x.view(torch.int32)
    exponent = ((u32 >> 23) & 0xFF).view(torch.uint32).to(torch.int32)
    nan_case = exponent == 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent = torch.where(round_case, exponent + 1, exponent)
    exponent = torch.where(nan_case, torch.full_like(exponent, 0xFF), exponent)
    from .flydsl_kernels.fp4_utils import fp8_e8m0

    return exponent.to(torch.uint8).view(fp8_e8m0)


def _e8m0_to_f32_capture_safe(scale_e8m0_biased: torch.Tensor) -> torch.Tensor:
    """CUDA-graph-safe e8m0 -> f32 (matches fp4_utils.e8m0_to_f32 bit-for-bit).

    Same reason as above: the vendored helper uses masked in-place writes for the
    zero / nan special cases; rewrite with ``torch.where``.
    """
    b = scale_e8m0_biased.view(torch.uint8)
    zero_case = b == 0
    nan_case = b == 0xFF
    f = b.to(torch.int32) << 23
    f = torch.where(zero_case, torch.full_like(f, 0x00400000), f)
    f = torch.where(nan_case, torch.full_like(f, 0x7F800001), f)
    return f.view(torch.float32)


@triton.jit
def _mxfp8_quant_shuffle_kernel(
    x_ptr,  # (E*M, K) bf16 activations
    codes_ptr,  # (E*M, K) uint8 out (fp8 e4m3 codes, row-major)
    scale_ptr,  # (E*M*Kd,) uint8 out (e8m0 scale, shuffle_scale_w4 gate_up=False)
    M,
    K,
    Kd,  # K // 32 (blocks per row)
    K1,  # Kd // 8
    per_expert,  # M * Kd  (scale elems per expert)
    BLOCK: tl.constexpr,  # 32 (quant block along K)
    BLOCK_KD: tl.constexpr,  # next pow2 >= Kd (arange must be pow2)
):
    """One program per ROW (e, m): quantize all Kd 1x32 blocks of the row at once
    and write the fp8 codes (row-major) + the Kd e8m0 scale bytes directly into
    the shuffle_scale_w4(gate_up=False) layout. Whole quant + scale-shuffle in a
    SINGLE kernel, no torch ops before the GEMM. Per-row (not per-block) grid so
    each program does K work — good occupancy vs a 14M-block launch.

    Replicates fp4_utils.per_1x32_f8_quant + f32_to_e8m0/e8m0_to_f32 inline
    (dtype_max = 2**8 = 256 for e4m3; no host sync, no masked writes) and the
    scale index permutation (verified bit-identical to shuffle_scale_w4)."""
    r = tl.program_id(0)  # flat row in [0, E*M)
    e = r // M
    m = r - e * M

    # Load the full row as [BLOCK_KD, 32]; mask the padded tail (Kd not pow2).
    kb = tl.arange(0, BLOCK_KD)  # [BLOCK_KD]
    kb_mask = kb < Kd
    j = tl.arange(0, BLOCK)  # [32]
    offs = kb[:, None] * BLOCK + j[None, :]  # [BLOCK_KD, 32]
    x = tl.load(x_ptr + r * K + offs, mask=kb_mask[:, None], other=0.0).to(
        tl.float32
    )  # [BLOCK_KD, 32]

    # --- e8m0 scale per block from max_abs / 256 (inline f32_to_e8m0) ---
    max_abs = tl.max(tl.abs(x), axis=1)  # [BLOCK_KD]
    v = max_abs * (1.0 / 256.0)
    u32 = v.to(tl.int32, bitcast=True)
    exponent = (u32 >> 23) & 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent = tl.where(round_case, exponent + 1, exponent)
    nan_case = exponent == 0xFF
    exponent = tl.where(nan_case, 0xFF, exponent)  # [Kd] e8m0 byte

    # --- divisor = e8m0_to_f32(exponent) (inline; power of two) ---
    zero_case = exponent == 0
    f_bits = exponent << 23
    f_bits = tl.where(zero_case, 0x00400000, f_bits)
    f_bits = tl.where(nan_case, 0x7F800001, f_bits)
    scale_f32 = f_bits.to(tl.float32, bitcast=True)  # [Kd]

    # --- codes = round-to-fp8(x / scale), stored row-major ---
    y = x / scale_f32[:, None]  # [BLOCK_KD, 32]
    codes = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(codes_ptr + r * K + offs, codes, mask=kb_mask[:, None])

    # --- scale bytes -> shuffle_scale_w4(gate_up=False) destinations (per kb) ---
    n_lane = m % 16
    t = m // 16
    n_pack = t % 2
    n1 = t // 2
    k_lane = kb % 4
    uu = kb // 4
    k_pack = uu % 2
    k1 = uu // 2
    off = ((((n1 * K1 + k1) * 4 + k_lane) * 16 + n_lane) * 2 + k_pack) * 2 + n_pack
    tl.store(scale_ptr + e * per_expert + off, exponent.to(tl.uint8), mask=kb_mask)


@triton.jit
def _mxfp8_quant_plain_kernel(
    x_ptr,  # (E*M, K) bf16 activations
    codes_ptr,  # (E*M, K) uint8 out (fp8 e4m3 codes, row-major)
    scale_ptr,  # (E*M, Kd) uint8 out (plain e8m0 scales)
    K,
    Kd,  # K // 32 (blocks per row)
    BLOCK: tl.constexpr,  # 32 (quant block along K)
    BLOCK_KD: tl.constexpr,  # next pow2 >= Kd
):
    """Plain per-1x32 MXFP8 quant for AITER Triton batched_gemm_a8wfp4.

    Unlike `_mxfp8_quant_shuffle_kernel`, this writes scales directly as
    [E, M, K//32]. AITER's Triton A8W4 kernel consumes non-shuffled scales.
    """
    r = tl.program_id(0)  # flat row in [0, E*M)

    kb = tl.arange(0, BLOCK_KD)
    kb_mask = kb < Kd
    j = tl.arange(0, BLOCK)
    offs = kb[:, None] * BLOCK + j[None, :]
    x = tl.load(x_ptr + r * K + offs, mask=kb_mask[:, None], other=0.0).to(
        tl.float32
    )

    max_abs = tl.max(tl.abs(x), axis=1)
    v = max_abs * (1.0 / 256.0)
    u32 = v.to(tl.int32, bitcast=True)
    exponent = (u32 >> 23) & 0xFF
    round_case = ((u32 & 0x400000) > 0) & (
        ((u32 & 0x200000) > 0) | ((u32 & 0x1FFFFF) > 0) | (exponent > 0)
    )
    exponent = tl.where(round_case, exponent + 1, exponent)
    nan_case = exponent == 0xFF
    exponent = tl.where(nan_case, 0xFF, exponent)

    zero_case = exponent == 0
    f_bits = exponent << 23
    f_bits = tl.where(zero_case, 0x00400000, f_bits)
    f_bits = tl.where(nan_case, 0x7F800001, f_bits)
    scale_f32 = f_bits.to(tl.float32, bitcast=True)

    y = x / scale_f32[:, None]
    codes = y.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(codes_ptr + r * K + offs, codes, mask=kb_mask[:, None])
    tl.store(scale_ptr + r * Kd + kb, exponent.to(tl.uint8), mask=kb_mask)


def _mxfp8_quant_batched(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[E, M, K]`` bf16 tensor to MXFP8 e4m3 (row-major codes) +
    ``shuffle_scale_w4``-shuffled e8m0 block scales, per expert.

    The FlyDSL batched kernel expects the A-side activation scale to be run
    through ``shuffle_scale_w4`` (gate_up=False; verified against a dequant
    reference: the unshuffled scale gives rel err ~0.5, the shuffled ~0.003).
    Codes stay row-major. Returns ``(codes[E, M, K] fp8, scale_shuffled[E, ...])``.

    The GPU path is a SINGLE Triton kernel (``_mxfp8_quant_shuffle_kernel``): one
    program per 1x32 block computes the fp8 codes AND writes the e8m0 scale
    directly in the shuffled layout — no torch ops before the GEMM. It replicates
    per_1x32_f8_quant + f32_to_e8m0/e8m0_to_f32 + shuffle_scale_w4 inline and is
    CUDA-graph safe (no host sync, no masked writes). Falls back to a pure-torch
    reference on CPU / no-Triton / ATOM_RCCL_FUSED_PACK=0.
    """
    E, M, K = x.shape
    block = 32
    kd = K // block

    if not _use_fused(x.device):
        # Pure-torch reference (matches per_1x32_f8_quant + shuffle_scale_w4).
        from .flydsl_kernels import fp4_utils

        x2d = x.reshape(E * M, K)
        xb = x2d.reshape(-1, block)
        max_abs = torch.amax(torch.abs(xb.float()), 1)
        scale_e8m0 = _f32_to_e8m0_capture_safe(max_abs / 256.0)
        scale_f32 = _e8m0_to_f32_capture_safe(scale_e8m0)
        y = xb.float() / scale_f32.view(-1, 1)
        codes = (
            y.to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .view(E, M, K)
            .view(torch.float8_e4m3fn)
        )
        scale2d = scale_e8m0.view(E * M, kd).view(torch.uint8)
        scale_shuf = fp4_utils.shuffle_scale_w4(scale2d, E, False).view(E, M, kd)
        return codes, scale_shuf

    codes = torch.empty((E * M, K), dtype=torch.uint8, device=x.device)
    scale_shuf = torch.empty((E * M * kd,), dtype=torch.uint8, device=x.device)
    block_kd = triton.next_power_of_2(kd)
    # One program per row (E*M): each quantizes all kd blocks of its K-row.
    _mxfp8_quant_shuffle_kernel[(E * M,)](
        x.reshape(E * M, K).contiguous(),
        codes,
        scale_shuf,
        M,
        K,
        kd,
        kd // 8,
        M * kd,
        BLOCK=block,
        BLOCK_KD=block_kd,
    )
    return (
        codes.view(E, M, K).view(torch.float8_e4m3fn),
        scale_shuf.view(E, M, kd),
    )


def _mxfp8_quant_batched_plain(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [E, M, K] bf16 to plain MXFP8 codes/scales.

    Returns codes [E, M, K] as fp8 and e8m0 scales [E, M, K//32] without the
    FlyDSL/aiter scale shuffle. This is the activation format expected by
    AITER's Triton `batched_gemm_a8wfp4`.
    """
    E, M, K = x.shape
    block = 32
    kd = K // block

    if not _use_fused(x.device):
        x2d = x.reshape(E * M, K)
        xb = x2d.reshape(-1, block)
        max_abs = torch.amax(torch.abs(xb.float()), 1)
        scale_e8m0 = _f32_to_e8m0_capture_safe(max_abs / 256.0)
        scale_f32 = _e8m0_to_f32_capture_safe(scale_e8m0)
        y = xb.float() / scale_f32.view(-1, 1)
        codes = (
            y.to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .view(E, M, K)
            .view(torch.float8_e4m3fn)
        )
        return codes, scale_e8m0.view(E, M, kd).view(torch.uint8)

    codes = torch.empty((E * M, K), dtype=torch.uint8, device=x.device)
    scales = torch.empty((E * M, kd), dtype=torch.uint8, device=x.device)
    block_kd = triton.next_power_of_2(kd)
    _mxfp8_quant_plain_kernel[(E * M,)](
        x.reshape(E * M, K).contiguous(),
        codes,
        scales,
        K,
        kd,
        BLOCK=block,
        BLOCK_KD=block_kd,
    )
    return codes.view(E, M, K).view(torch.float8_e4m3fn), scales.view(E, M, kd)


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


def _swiglu(gate_up: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """SwiGLU on the last dim, respecting the gate/up weight layout.

    The gate_up projection output is either SEPARATED ([all gates | all ups]) or
    INTERLEAVED ([gate0, up0, gate1, up1, ...]). Which one depends on how the
    expert weight ROWS are ordered, and that differs by weight format — so the
    caller passes it explicitly rather than reading a global env (which would
    conflate the fp4 and block-fp8 paths, whose weights are laid out
    differently). Using the wrong split silently produces coherent-but-wrong
    output. Returns ``silu(gate) * up`` with shape ``[..., I]``.
    """
    import torch.nn.functional as F

    if interleaved:
        gate = gate_up[..., 0::2]
        up = gate_up[..., 1::2]
    else:
        gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _block_fp8_gemm_batched(
    a: torch.Tensor,  # [E, M, K] bf16 activations
    w: torch.Tensor,  # [E, N, K] fp8 weights (block-scaled)
    w_scale: torch.Tensor,  # [E, N//128, K//128] fp32 block scales
) -> torch.Tensor:
    """Per-expert block-scaled w8a8 (fp8) GEMM: C[e] = A[e] @ W[e]^T, bf16 out.

    Uses aiter's Triton ``gemm_a8w8_blockscale`` (128x128 weight blocks, per-
    token-group activation scale) looped over the E experts. Activations are
    dynamically quantized to fp8 with per-1x128 block scales to match the
    weight blocking. Eager-only (a Python loop over experts).
    """
    from aiter import get_hip_quant, QuantType
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale,
    )

    E, M, _K = a.shape
    N = w.shape[1]
    quant = get_hip_quant(QuantType.per_1x128)
    out = torch.empty((E, M, N), dtype=torch.bfloat16, device=a.device)
    for e in range(E):
        # x: (M, K) fp8, x_scale: (M, K//block_k) fp32
        xq, xs = quant(a[e].contiguous(), quant_dtype=torch.float8_e4m3fn)
        gemm_a8w8_blockscale(xq, w[e], xs, w_scale[e], dtype=torch.bfloat16, y=out[e])
    return out


def batched_block_fp8_mlp(
    batched_a: torch.Tensor,  # [E, C, H] bf16 activations
    w13: torch.Tensor,  # [E, 2I, H] fp8 (gate+up)
    w13_scale: torch.Tensor,  # [E, 2I//128, H//128] fp32
    w2: torch.Tensor,  # [E, H, I] fp8 (down)
    w2_scale: torch.Tensor,  # [E, H//128, I//128] fp32
    activation,
) -> torch.Tensor:
    """Two-GEMM expert MLP for BLOCK-FP8 (w8a8) expert weights (e.g. DeepSeek-V4
    with ``quant_method: fp8``). gate_up -> SwiGLU -> down, all in [E, C, *].
    """
    gate_up = _block_fp8_gemm_batched(batched_a, w13, w13_scale)  # [E, C, 2I]
    # Block-fp8 (1x128) expert weights keep the checkpoint's SEPARATED
    # [gate | up] row layout (the gate/up interleave only applies to the 1x32
    # MXFP8 path), so split by halves, not by even/odd.
    act = _swiglu(gate_up, interleaved=False)  # [E, C, I]
    out = _block_fp8_gemm_batched(act.to(batched_a.dtype), w2, w2_scale)  # [E, C, H]
    return out


def batched_w4a8_mlp(
    batched_a: torch.Tensor,  # [E, C, H] bf16 activations (zero-padded)
    w13: torch.Tensor,  # [E, 2I, H/2] CK-preshuffled MXFP4 (gate+up)
    w13_scale: torch.Tensor,  # [E, 2I, H/32] e8m0
    w2: torch.Tensor,  # [E, H, I/2] CK-preshuffled MXFP4 (down)
    w2_scale: torch.Tensor,  # [E, H, I/32] e8m0
    activation,  # ActivationType (Silu/Swiglu) — SiluAndMul used
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
) -> torch.Tensor:
    """Run the two-GEMM expert MLP over the ``[E, C, H]`` grid.

    Steps (E experts, C capacity, H hidden, I inter_dim):
      1. quantize batched_a -> MXFP8
      2. gate_up: [E,C,H] x w13[E,2I,H/2] -> [E,C,2I]
      3. SwiGLU: [E,C,2I] -> [E,C,I]
      4. quantize -> MXFP8
      5. down: [E,C,I] x w2[E,H,I/2] -> [E,C,H]
    Returns ``[E, C, H]`` bf16.
    """
    E = w13.shape[0]

    # The model's weight scales arrive FLATTENED to 2D ([E*N, K/32]) because
    # aiter's moe_shuffle_scale reshapes to (-1, last) before shuffling. The
    # FlyDSL batched GEMM indexes the scale per expert, so it needs a 3D
    # [E, N, K/32] view. The shuffle is within-expert and expert-major, so a
    # plain reshape by E is its exact inverse. (No-op if already 3D.)
    def _scale_3d(s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 3:
            return s
        return s.reshape(E, s.shape[0] // E, s.shape[1])

    w13_scale = _scale_3d(w13_scale)
    w2_scale = _scale_3d(w2_scale)

    from atom.utils import envs

    if envs.ATOM_RCCL_MOE_IMPL == "triton_batched_gemm":
        from aiter.ops.triton.gemm.batched.batched_gemm_a8wfp4 import (
            batched_gemm_a8wfp4,
        )

        out_dtype = batched_a.dtype
        aq, as_ = _mxfp8_quant_batched_plain(batched_a)
        gate_up = batched_gemm_a8wfp4(
            aq,
            w13.view(torch.uint8),
            as_,
            w13_scale.view(torch.uint8),
            dtype=out_dtype,
            a_dtype="fp8",
        )
        # Plain checkpoint layout stores gate and up as [gate | up].
        act = _swiglu(gate_up, interleaved=False)
        actq, act_s = _mxfp8_quant_batched_plain(act.to(batched_a.dtype))
        return batched_gemm_a8wfp4(
            actq,
            w2.view(torch.uint8),
            act_s,
            w2_scale.view(torch.uint8),
            dtype=out_dtype,
            a_dtype="fp8",
        )

    from atom.model_ops.fused_moe.flydsl_batched_gemm import (
        batched_mxfp8_mxfp4_gemm,
    )

    # 1-2. gate_up GEMM.
    aq, as_ = _mxfp8_quant_batched(batched_a)
    gate_up = batched_mxfp8_mxfp4_gemm(
        aq, w13, as_, w13_scale, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )  # [E, C, 2I]

    # 3. SwiGLU. The MXFP4 expert weights are stored gate/up INTERLEAVED
    # (aiter shuffle with is_guinterleave=True), so split by even/odd columns.
    act = _swiglu(gate_up, interleaved=True)  # [E, C, I]

    # 4-5. down GEMM.
    actq, act_s = _mxfp8_quant_batched(act.to(batched_a.dtype))
    out = batched_mxfp8_mxfp4_gemm(
        actq, w2, act_s, w2_scale, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
    )  # [E, C, H]
    return out


def batched_expert_compute(
    dispatch_a1: torch.Tensor,
    dispatch_ids: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    activation,
    local_num_experts: int,
    capacity: Optional[int],
    ep_rank: int,
) -> torch.Tensor:
    """End-to-end batched grouped-expert compute in dispatched-row space.

    resort dispatched rows -> [E, C, H] -> two-GEMM MLP -> unsort back to
    ``[R, H]`` (dispatched-row order). The caller (modular kernel) then runs the
    unchanged finalize / all2all-back + weighted combine.

    ``capacity=None`` (HT / eager) sizes C to the actual max tokens per local
    expert; an int gives a fixed, CUDA-graph-safe C (LL path).
    """
    R = dispatch_a1.shape[0]
    batched, row_map, counts = resort_to_batched(
        dispatch_a1, dispatch_ids, local_num_experts, capacity, ep_rank
    )
    # Dispatch by expert weight format: fp8 block-quantized -> w8a8 blockscale
    # GEMM; fp4 (packed MXFP4) -> the FlyDSL batched w4a8 GEMM.
    if w13.dtype in _FP8_DTYPES:
        out_grid = batched_block_fp8_mlp(
            batched, w13, w13_scale, w2, w2_scale, activation
        )
    else:
        out_grid = batched_w4a8_mlp(batched, w13, w13_scale, w2, w2_scale, activation)
    return unsort_from_batched(out_grid, row_map, counts, num_rows=R)
