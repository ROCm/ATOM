# SPDX-License-Identifier: MIT
# p1/k2-draft (ATOM_P1_DRAFT_MARKOV_PERSIST, default off): the sharded greedy DSpark Markov block -- all T positions --
# in ONE persistent launch per rank, replacing `MiniMaxM3DSpark._markov_atomic`'s T `_markov_step_atomic` launches,
# T aiter all-gathers of the [B] packed words and the closing `_markov_last_atomic`.
#
# Every workgroup owns one 256-wide vocab tile of this rank's W2 shard and loads it ONCE (registers); then, per
# position, the same GEMV / tile argmax / (max, lowest id) packing / atomic_max fold as `_markov_step_atomic`, folded
# per counter group (S groups of WGs, tile % S, to cut the per-address atomic contention: S=4 is 6-8 us/block faster
# than one fold). The last workgroup of a group to arrive (a gpu-scope counter; its increment depends on every row's
# returned atomic_max, so no cache fence is needed) reads the group's [B] packed words and pushes them into every
# rank's slot buffer with 64-bit system-scope atomic exchanges (IPC-mapped peers). Consumers spin on the slots
# themselves (data-as-flag: EMPTY until pushed); the winner is the max of the 4 x S words -- the max `_ranks_best_id`
# takes after the all-gather of the per-rank folds (max is associative). A row
# no tile hit is pushed as NOHIT (low word all ones -> id 0, the id an EMPTY word gives; NOHIT sorts below every real
# packed value, whose high word is never 0x80000000 = a NaN max, which never hits). Same values, same tie rule ->
# the same ids, bit for bit (p1/k2-draft J1/J3: TP4 eager + graph replay, random and tie-heavy inputs, B 1-16).
#
# Buffers are double-buffered by the launch epoch's parity: each launch resets the other parity's slots, packed
# words and counters (the previous launch's, finished on every rank: TP collectives separate two drafts), and the
# last arriver of position 0 advances the epoch. Every rank must launch it the same number of times (TP lockstep).
# Co-residency: the grid (cdiv(V_local, 256) = 196 at V_local 50016) must fit the GPU's CUs at one WG per CU.
import triton
import triton.language as tl

from atom.model_ops.tile_argmax import argmax_lowest_id
from atom.models.minimax_m3.packed_argmax import pack_max_id

EMPTY = -(2**63)
NOHIT = -(2**63) + 0xFFFFFFFF
CNT_LINE = 32  # int32 words per counter (one 128 B line each)
MAXB = 64  # rows (requests) per block: one row tile
REPLICAS = 4  # copies of every pushed word (consumers spread over them)
BLOCK_V = 256
BLOCK_K = 64
_E = tl.constexpr(EMPTY)
_NH = tl.constexpr(NOHIT)
_CL = tl.constexpr(CNT_LINE)


@triton.jit
def _p1mb_wait_prev(slot_ptr, par, kprev, rep, offs_row, row_mask,
                    T: tl.constexpr, NS: tl.constexpr, R: tl.constexpr, MAXB: tl.constexpr, BLOCK_ROW: tl.constexpr):
    """Spin until every source's (rank x counter group) packed word of position kprev has landed in this WG's
    replica; the winner's id."""
    offs_s = tl.arange(0, NS)
    sp = slot_ptr + ((par * T + kprev) * R + rep) * (NS * MAXB)
    ptrs = sp + offs_s[:, None] * MAXB + offs_row[None, :]
    m = row_mask[None, :] & (offs_s[:, None] < NS)
    v = tl.atomic_add(ptrs, tl.zeros([NS, BLOCK_ROW], tl.int64), mask=m, sem="relaxed", scope="sys")
    v = tl.where(m, v, _NH)
    pending = tl.sum((v == _E).to(tl.int32))
    while pending > 0:
        v = tl.atomic_add(ptrs, tl.zeros([NS, BLOCK_ROW], tl.int64), mask=m, sem="relaxed", scope="sys")
        v = tl.where(m, v, _NH)
        pending = tl.sum((v == _E).to(tl.int32))
    best = tl.max(v, axis=0)
    return (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)


@triton.jit
def _p1mb_push(dst_ptr, par, k, src, got, row_mask, T: tl.constexpr, NS: tl.constexpr, R: tl.constexpr,
               MAXB: tl.constexpr, BLOCK_ROW: tl.constexpr):
    offs_row = tl.arange(0, BLOCK_ROW)
    offs_r = tl.arange(0, R)
    base = dst_ptr + (par * T + k) * (R * NS * MAXB) + src * MAXB
    ptrs = base + offs_r[:, None] * (NS * MAXB) + offs_row[None, :]
    vals = tl.broadcast_to(got[None, :], [R, BLOCK_ROW])
    tl.atomic_xchg(ptrs, vals, mask=row_mask[None, :] & (offs_r[:, None] < R), sem="relaxed", scope="sys")


@triton.jit
def _p1_markov_block_kernel(
    base_ptr, anchor_ptr, out_ptr, w1_ptr, w2_ptr,
    pub_ptr,  # local int64 [2, T, S, MAXB]: this rank's packed best per position and counter group (atomic_max)
    cnt_ptr,  # local int32 [2 * T * S + 1, CNT_LINE]: arrival counters; the last line counts position-0 starts
    ep_ptr,  # local int64 [16]: word 0 = launch epoch
    slot_ptr,  # local int64 [2, T, R, W * S, MAXB]: every source's packed words, pushed (EMPTY = not yet)
    s0_ptr, s1_ptr, s2_ptr, s3_ptr,  # every rank's slot buffer (s{me} == slot_ptr)
    num_rows, vocab_size, rank,
    stride_base_pos, stride_base_row, stride_anchor, stride_out_row, stride_out_pos,
    stride_w1, stride_w2, vocab_offset, me,
    T: tl.constexpr, W: tl.constexpr, R: tl.constexpr, MAXB: tl.constexpr,
    BLOCK_ROW: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_K: tl.constexpr, S: tl.constexpr = 1,
):
    tl.static_assert(BLOCK_K * 4 == 256)
    tile = tl.program_id(0)
    nt = tl.num_programs(0)
    offs_row = tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size

    ep = tl.atomic_add(ep_ptr, 0, sem="relaxed", scope="gpu")
    par = (ep & 1).to(tl.int32)
    opar = 1 - par

    # this WG's W2^T tile, 4 chunks [BLOCK_K, BLOCK_V], loaded once (`markov_logits_tile`'s masks; rank 256)
    offs_k = tl.arange(0, BLOCK_K)
    wb = w2_ptr + (vocab_offset + offs_v[None, :]).to(tl.int64) * stride_w2 + offs_k[:, None]
    wm = v_mask[None, :] & (offs_k[:, None] < rank)
    w2_0 = tl.load(wb, mask=wm, other=0.0)
    w2_1 = tl.load(wb + BLOCK_K, mask=wm, other=0.0)
    w2_2 = tl.load(wb + 2 * BLOCK_K, mask=wm, other=0.0)
    w2_3 = tl.load(wb + 3 * BLOCK_K, mask=wm, other=0.0)

    # reset the other parity (the previous launch's; finished on every rank)
    offs_m = tl.arange(0, MAXB)
    for j in range(tile, T * R * W * S, nt):
        tl.atomic_xchg(slot_ptr + opar * (T * R * W * S * MAXB) + j * MAXB + offs_m,
                       tl.full([MAXB], _E, tl.int64), sem="relaxed", scope="sys")
    if tile < T * S:
        tl.atomic_xchg(pub_ptr + (opar * T * S + tile) * MAXB + offs_m, tl.full([MAXB], _E, tl.int64),
                       sem="relaxed", scope="gpu")
    if tile == 0:
        offs_t = tl.arange(0, 64)
        tl.atomic_xchg(cnt_ptr + (opar * T * S + offs_t) * _CL, tl.zeros([64], tl.int32), mask=offs_t < T * S,
                       sem="relaxed", scope="gpu")

    rep = tile % R
    grp = tile % S  # counter group: WG ids go round-robin over the XCDs, so a group spans them
    ngrp = (nt - grp + S - 1) // S
    for k in tl.static_range(T):
        if k == 0:
            prev = tl.load(anchor_ptr + offs_row * stride_anchor, mask=row_mask, other=0).to(tl.int32)
        else:
            prev = _p1mb_wait_prev(slot_ptr, par, k - 1, rep, offs_row, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
            if tile == 0:
                tl.store(out_ptr + offs_row * stride_out_row + (k - 1) * stride_out_pos, prev, mask=row_mask)
        eb = w1_ptr + prev[:, None].to(tl.int64) * stride_w1 + offs_k[None, :]
        em = row_mask[:, None] & (offs_k[None, :] < rank)
        acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
        acc = tl.dot(tl.load(eb, mask=em, other=0.0), w2_0, acc)
        acc = tl.dot(tl.load(eb + BLOCK_K, mask=em, other=0.0), w2_1, acc)
        acc = tl.dot(tl.load(eb + 2 * BLOCK_K, mask=em, other=0.0), w2_2, acc)
        acc = tl.dot(tl.load(eb + 3 * BLOCK_K, mask=em, other=0.0), w2_3, acc)
        base = tl.load(
            base_ptr + k * stride_base_pos + offs_row[:, None].to(tl.int64) * stride_base_row + offs_v[None, :],
            mask=row_mask[:, None] & v_mask[None, :], other=0.0)
        vals = tl.where(v_mask[None, :], base.to(tl.float32) + acc, float("-inf"))
        tile_max, tile_idx = argmax_lowest_id(vals, offs_v[None, :], v_mask[None, :], 2147483647)
        hit = tile_idx != 2147483647
        packed = pack_max_id(tile_max, tile_idx + vocab_offset)
        pubk = pub_ptr + ((par * T + k) * S + grp) * MAXB
        # ordering by data dependency, no cache fences: the counter's increment depends (through a WG reduction) on
        # every row's returned atomic_max, so it is issued only after all of them completed at the atomics'
        # coherence point; the increment is 1 unless an impossible returned value shows up.
        old = tl.atomic_max(pubk + offs_row, packed, mask=row_mask & hit, sem="relaxed", scope="gpu")
        dep = tl.sum(tl.where((row_mask & hit) & (old == 7), 1, 0))
        c = tl.atomic_add(cnt_ptr + ((par * T + k) * S + grp) * _CL, 1 + dep * 0 + (dep > 1 << 30).to(tl.int32),
                          sem="relaxed", scope="gpu")
        if k == 0:
            if S == 1:
                started = c
            else:
                # the epoch may advance only once EVERY WG has read it (all started), not just this group's
                started = tl.atomic_add(cnt_ptr + (2 * T * S) * _CL, 1, sem="relaxed", scope="gpu")
            if started == nt - 1:
                if S > 1:
                    tl.atomic_xchg(cnt_ptr + (2 * T * S) * _CL, 0, sem="relaxed", scope="gpu")
                tl.atomic_add(ep_ptr, 1, sem="relaxed", scope="gpu")
        if c == ngrp - 1:
            got = tl.atomic_add(pubk + offs_row, tl.zeros([BLOCK_ROW], tl.int64), mask=row_mask, sem="relaxed",
                                scope="gpu")
            got = tl.where(got == _E, _NH, got)
            src = me * S + grp
            _p1mb_push(s0_ptr, par, k, src, got, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
            if W > 1:
                _p1mb_push(s1_ptr, par, k, src, got, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
            if W > 2:
                _p1mb_push(s2_ptr, par, k, src, got, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
                _p1mb_push(s3_ptr, par, k, src, got, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
    if tile == 0:
        last = _p1mb_wait_prev(slot_ptr, par, T - 1, rep, offs_row, row_mask, T, W * S, R, MAXB, BLOCK_ROW)
        tl.store(out_ptr + offs_row * stride_out_row + (T - 1) * stride_out_pos, last, mask=row_mask)


class _Ptr:
    """A raw device address as a Triton pointer argument."""

    def __init__(self, addr, dtype):
        self._a, self.dtype = int(addr), dtype

    def data_ptr(self):
        return self._a


class MarkovBlockState:
    """One rank's buffers for `_p1_markov_block_kernel` at block length T. ``peer_slots``: the W ranks' slot
    buffer addresses in rank order (own included), filled by `connect` (IPC) or left local for W = 1."""

    def __init__(self, T: int, W: int, device, S: int = 4):
        import torch

        assert 1 <= T <= 8 and W in (1, 2, 4) and S in (1, 2, 4, 8)
        self.T, self.W, self.S = T, W, S
        n_slot = 2 * T * REPLICAS * W * S * MAXB
        n_pub = 2 * T * S * MAXB
        # one allocation (the slots are IPC-exported): slot | pub | epoch
        self.mem = torch.full((n_slot + n_pub + 16,), EMPTY, dtype=torch.int64, device=device)
        self.slot = self.mem[:n_slot]
        self.pub = self.mem[n_slot:n_slot + n_pub]
        self.ep = self.mem[n_slot + n_pub:]
        self.ep.zero_()
        self.cnt = torch.zeros((2 * T * S + 1) * CNT_LINE, dtype=torch.int32, device=device)
        self.peer_slots = [self.slot.data_ptr()] * 4

    def connect(self, rank: int, cpu_group) -> None:
        """Exchange the slot buffers' IPC handles with the other ranks (a collective on ``cpu_group``)."""
        from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144.ipc import _alloc_handle, _open_handle, exchange

        base, raw = _alloc_handle(self.slot.data_ptr())
        every = exchange((raw, self.slot.data_ptr() - base), rank, cpu_group)
        assert len(every) == self.W
        self.peer_slots = [self.slot.data_ptr() if s == rank else _open_handle(h) + off
                           for s, (h, off) in enumerate(every)] + [self.slot.data_ptr()] * (4 - self.W)


def markov_block(state: MarkovBlockState, base_logits, anchor_ids, w1, w2, out_ids, vocab_offset: int, me: int):
    """Greedy Markov block: base_logits [B, T, V_local] (any strides), anchor_ids [B], out_ids [B, T] (written)."""
    import torch

    bs, T, v_local = base_logits.shape
    assert T == state.T and bs <= MAXB and w2.shape[1] == 4 * BLOCK_K
    block_row = min(MAXB, max(16, triton.next_power_of_2(bs)))
    s = [_Ptr(a, torch.int64) for a in state.peer_slots]
    _p1_markov_block_kernel[(triton.cdiv(v_local, BLOCK_V),)](
        base_logits, anchor_ids, out_ids, w1, w2, state.pub, state.cnt, state.ep, state.slot, s[0], s[1], s[2], s[3],
        bs, v_local, w2.shape[1], base_logits.stride(1), base_logits.stride(0), anchor_ids.stride(0),
        out_ids.stride(0), out_ids.stride(1), w1.stride(0), w2.stride(0), vocab_offset, me,
        T=T, W=state.W, R=REPLICAS, MAXB=MAXB, BLOCK_ROW=block_row, BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
        S=state.S, num_warps=8, num_stages=1,
    )
