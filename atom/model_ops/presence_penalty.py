# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI `presence_penalty` (vLLM semantics) on decode and speculative-verify rows.

A request's logits lose `presence_penalty` on every token it has generated so far
(its prompt does not count), before temperature and top-k / top-p. Each request
holds a slot: one byte a vocabulary entry, set once the token is emitted. A verify
row also counts the drafts before it -- row `i` of a request (query tokens
`[anchor, d_0 .. d_{n-1}]`) predicts the token after `d_{i-1}`, so `d_0 .. d_{i-1}`
are already output there -- which is what makes the rows' accept/reject draws the
penalized target distribution.

Slots belong to the runner (the scheduler does not tell it when a request ends):
least recently scheduled first out, twice `max_num_seqs` of them, so a running
request, scheduled every step, is never the one recycled.
"""

from collections import OrderedDict

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _apply_presence_penalty_kernel(
    logits_ptr,
    logits_stride,
    input_ids_ptr,
    row_seq_ptr,
    row_pos_ptr,
    seq_start_ptr,
    slot_ptr,
    penalty_ptr,
    mask_ptr,
    mask_stride,
    vocab_start,
    vocab_local,
    MAX_EXTRA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inb = cols < vocab_local
    s = tl.load(row_seq_ptr + row)
    p = tl.load(penalty_ptr + s)
    slot = tl.load(slot_ptr + s).to(tl.int64)
    pos = tl.load(row_pos_ptr + row)
    start = tl.load(seq_start_ptr + s)
    present = (
        tl.load(mask_ptr + slot * mask_stride + vocab_start + cols, mask=inb, other=0)
        != 0
    )
    # the drafts this row's context already holds: input ids start + 1 .. start + pos
    for k in tl.static_range(1, MAX_EXTRA + 1):
        tok = tl.load(input_ids_ptr + start + k, mask=k <= pos, other=-1)
        present = present | ((cols + vocab_start) == tok)
    ptrs = logits_ptr + row.to(tl.int64) * logits_stride + cols
    x = tl.load(ptrs, mask=inb).to(tl.float32)
    x = x - tl.where(present, p, 0.0)
    tl.store(ptrs, x.to(logits_ptr.dtype.element_ty), mask=inb)


@triton.jit
def _mark_tokens_kernel(
    tokens_ptr,
    tokens_stride,
    num_tokens,
    slot_ptr,
    mask_ptr,
    mask_stride,
    vocab,
    BLOCK: tl.constexpr,
):
    s = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tok = tl.load(
        tokens_ptr + s * tokens_stride + offs, mask=offs < num_tokens, other=-1
    )
    slot = tl.load(slot_ptr + s).to(tl.int64)
    tl.store(
        mask_ptr + slot * mask_stride + tok,
        tl.full([BLOCK], 1, tl.uint8),
        mask=(tok >= 0) & (tok < vocab),
    )


@triton.jit
def _record_draft_kernel(
    ids_ptr, ctx_ptr, pos_ptr, rows, CTX: tl.constexpr, BLOCK: tl.constexpr
):
    # Each row's new draft id into its context (slot 1 + pos), then pos + 1.
    r = tl.arange(0, BLOCK)
    m = r < rows
    pos = tl.load(pos_ptr + r, mask=m, other=0)
    tok = tl.load(ids_ptr + r, mask=m, other=-1)
    tl.store(ctx_ptr + r * CTX + 1 + pos, tok, mask=m & (pos < CTX - 1))
    tl.store(pos_ptr + r, pos + 1, mask=m)


class DraftPresencePenalty:
    """ATOM_MTP_DRAFT_PRESENCE_PENALTY: the verifier's presence penalty on the
    draft passes' argmax.

    Row `r` of a draft pass is the step's `r`-th request. Before the argmax it
    loses the request's penalty on every token in its slot's mask (all it has
    emitted, this step's tokens included: they are marked before the drafter
    runs) and on the drafts already made this step, the set its verify row will
    penalize. The buffers are fixed, since the draft passes are recorded: `begin`
    stages a step's slots and penalties before the passes, each pass then
    penalizes and records its ids (`apply`, `record`), so a replayed mid-step
    reads how many drafts precede it from `pos`."""

    def __init__(self, state, max_rows: int, max_drafts: int, device):
        assert (
            max_drafts <= state.max_extra
        ), "the penalty kernel counts max_extra drafts"
        self.state = state
        self.max_rows = int(max_rows)
        self.ctx_width = int(max_drafts) + 1
        self.slots = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        self.penalties = torch.zeros(self.max_rows, dtype=torch.float32, device=device)
        self.ctx = torch.full(
            (self.max_rows * self.ctx_width,), -1, dtype=torch.int32, device=device
        )
        self.pos = torch.zeros(self.max_rows, dtype=torch.int32, device=device)
        self.row_seq = torch.arange(self.max_rows, dtype=torch.int32, device=device)
        self.seq_start = self.row_seq * self.ctx_width

    def begin(self, step: "PresencePenaltyStep | None", scheduled_bs: int):
        """A step's requests (`step`'s slots and penalties, rows in batch order)
        before its draft passes; no penalty without a step (prefill, none set)."""
        self.penalties.zero_()
        self.pos.zero_()
        self.ctx.fill_(-1)
        if step is None:
            return
        n = min(int(scheduled_bs), step.slots.numel(), self.max_rows)
        self.slots[:n].copy_(step.slots[:n])
        self.penalties[:n].copy_(step.penalties[:n])

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Subtract the penalties in place from a draft pass's `[rows, V]` logits."""
        rows = logits.shape[0]
        assert rows <= self.max_rows and logits.stride(1) == 1
        block = 4096
        _apply_presence_penalty_kernel[(rows, triton.cdiv(logits.shape[1], block))](
            logits,
            logits.stride(0),
            self.ctx,
            self.row_seq,
            self.pos,
            self.seq_start,
            self.slots,
            self.penalties,
            self.state.mask,
            self.state.mask.stride(0),
            0,
            logits.shape[1],
            MAX_EXTRA=self.state.max_extra,
            BLOCK=block,
        )
        return logits

    def record(self, ids: torch.Tensor):
        """A draft pass's `[rows]` ids: they count for the passes after it."""
        rows = ids.numel()
        _record_draft_kernel[(1,)](
            ids.view(-1),
            self.ctx,
            self.pos,
            rows,
            CTX=self.ctx_width,
            BLOCK=triton.next_power_of_2(max(rows, 2)),
        )


class PresencePenaltyStep:
    """One decode forward's rows: which request each logits row belongs to, its
    position in the request's query, and the requests' slots and penalties."""

    def __init__(self, state, slots, penalties, seq_start, row_seq, row_pos, input_ids):
        self.state = state
        self.slots = slots
        self.penalties = penalties
        self.seq_start = seq_start
        self.row_seq = row_seq
        self.row_pos = row_pos
        self.num_rows = row_seq.numel()
        self.input_ids = input_ids
        self._anchors_marked = False

    def _mark_anchors(self):
        # A decode query starts at the request's last emitted token: the first
        # token after prefill reaches the mask here, the rest are already in it.
        if self._anchors_marked:
            return
        self._anchors_marked = True
        anchors = torch.index_select(self.input_ids, 0, self.seq_start.long())
        self._mark(anchors.view(-1, 1))

    def apply(self, logits: torch.Tensor, vocab_start: int = 0) -> torch.Tensor:
        """Subtract the penalties in place from `logits` `[rows, V_local]`, the
        vocab slice starting at `vocab_start` (a TP shard's or the whole vocab)."""
        self._mark_anchors()
        rows = min(self.num_rows, logits.shape[0])
        if rows == 0:
            return logits
        assert logits.stride(1) == 1
        block = 4096
        _apply_presence_penalty_kernel[(rows, triton.cdiv(logits.shape[1], block))](
            logits,
            logits.stride(0),
            self.input_ids,
            self.row_seq,
            self.row_pos,
            self.seq_start,
            self.slots,
            self.penalties,
            self.state.mask,
            self.state.mask.stride(0),
            vocab_start,
            logits.shape[1],
            MAX_EXTRA=self.state.max_extra,
            BLOCK=block,
        )
        return logits

    def mark(self, tokens: torch.Tensor):
        """Record emitted tokens: `tokens` `[seqs, T]`, -1 where nothing was emitted."""
        self._mark_anchors()
        self._mark(tokens)

    def _mark(self, tokens: torch.Tensor):
        tokens = tokens.reshape(tokens.shape[0], -1)
        if tokens.shape[0] == 0:
            return
        assert tokens.stride(1) == 1
        _mark_tokens_kernel[(tokens.shape[0],)](
            tokens,
            tokens.stride(0),
            tokens.shape[1],
            self.slots,
            self.state.mask,
            self.state.mask.stride(0),
            self.state.mask.shape[1],
            BLOCK=triton.next_power_of_2(max(tokens.shape[1], 2)),
        )


class PresencePenaltyState:
    def __init__(self, vocab_size: int, max_num_seqs: int, max_extra: int, device):
        self.num_slots = 2 * max_num_seqs
        self.mask = torch.zeros(
            self.num_slots, vocab_size, dtype=torch.uint8, device=device
        )
        self.max_extra = max(int(max_extra), 1)
        self.device = device
        self._slot_of: OrderedDict = OrderedDict()
        self._free = list(range(self.num_slots - 1, -1, -1))

    def _slots_for(self, req_ids) -> np.ndarray:
        """Each request's slot, allocating (and clearing) those not yet seen;
        recycles the least recently scheduled request outside this batch."""
        in_batch = set(req_ids)
        slots = np.empty(len(req_ids), dtype=np.int32)
        fresh = []
        for i, rid in enumerate(req_ids):
            slot = self._slot_of.get(rid)
            if slot is None:
                if not self._free:
                    for old in self._slot_of:
                        if old not in in_batch:
                            self._free.append(self._slot_of.pop(old))
                            break
                slot = self._free.pop()
                self._slot_of[rid] = slot
                fresh.append(slot)
            else:
                self._slot_of.move_to_end(rid)
            slots[i] = slot
        if fresh:
            idx = torch.tensor(fresh, dtype=torch.long).to(
                self.device, non_blocking=True
            )
            self.mask.index_fill_(0, idx, 0)
        return slots

    def begin_step(self, batch, input_ids: torch.Tensor):
        """The step's penalty rows, or None when no request in it has a penalty.
        Requests are given slots on every step that schedules them, prefill too,
        so their first decode finds a cleared mask; only decode steps penalize."""
        pens = getattr(batch, "presence_penalties", None)
        if pens is None or batch.is_dummy_run or not np.any(pens != 0):
            return None
        slots = self._slots_for(batch.req_ids)
        if batch.total_tokens_num_prefill > 0:
            return None
        q = np.asarray(batch.num_scheduled_tokens, dtype=np.int32)
        if q.max() - 1 > self.max_extra:
            raise ValueError(
                f"presence_penalty: {q.max()} query tokens a request, the kernel "
                f"counts at most {self.max_extra} drafts"
            )
        seq_start = np.zeros(len(q), dtype=np.int32)
        np.cumsum(q[:-1], out=seq_start[1:])
        row_seq = np.repeat(np.arange(len(q), dtype=np.int32), q)
        row_pos = np.arange(int(q.sum()), dtype=np.int32) - np.repeat(seq_start, q)
        host = torch.from_numpy(
            np.concatenate(
                [
                    slots,
                    np.asarray(pens, dtype=np.float32).view(np.int32),
                    seq_start,
                    row_seq,
                    row_pos,
                ]
            )
        ).pin_memory()
        dev = host.to(self.device, non_blocking=True)
        bs, n = len(q), len(row_seq)
        return PresencePenaltyStep(
            self,
            dev[:bs],
            dev[bs : 2 * bs].view(torch.float32),
            dev[2 * bs : 3 * bs],
            dev[3 * bs : 3 * bs + n],
            dev[3 * bs + n :],
            input_ids,
        )
