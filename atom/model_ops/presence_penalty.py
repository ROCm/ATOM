# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""OpenAI `presence_penalty`, with vLLM's semantics.

Before sampling, a request's logits lose its penalty on every token it has
generated so far; its prompt does not count. A verification row also counts
the drafts before it: the row checks the draft that follows them, so they are
output there. The bonus row counts every draft.

A request's generated tokens are bits in its row of `PresencePenalties.seen`.
The step that emits a token sets its bit after the token broadcast, so every
rank holds the same bits. A request claims a row the first time it is
scheduled and gives it back when the engine releases it (finished, aborted or
refused). A preempted request keeps its row. The table starts with
`max_num_seqs` rows; only preempted requests can need more, and then it doubles.

`ModelRunner.prepare_sample` stages each logits row's table row, penalty and
draft count with the other sampling parameters (`stage`). `TokenConstraints`
applies them before sampling and commits the emitted ids
(`triton_presence_penalty.py`).
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PresenceRows:
    """One step's staged rows: views of its `sampling` publication."""

    slots: torch.Tensor  # [rows] int32: the request's table row, -1 for none
    penalties: torch.Tensor  # [rows] float32
    drafts_before: torch.Tensor  # [rows] int32: the request's drafts before it
    emit_slots: torch.Tensor  # [requests] int32: -1 where no token is emitted
    fresh: torch.Tensor  # [claimed] int32: rows to clear before use
    num_slots: int  # the table rows the step addresses


class PresencePenalties:
    """The per-request bit table of generated tokens, and its row bookkeeping
    on the host. Every rank runs the same bookkeeping on the same batches."""

    def __init__(self, vocab_size: int, num_slots: int, device):
        self.vocab_size = vocab_size
        self.seen = torch.zeros(
            num_slots, -(-vocab_size // 32), dtype=torch.int32, device=device
        )
        self._num_slots = num_slots
        self._slot_of: dict[int, int] = {}
        self._free = list(range(num_slots - 1, -1, -1))

    @staticmethod
    def staging_size(max_bs: int, rows_per_request: int) -> int:
        """int32 entries `stage` writes at most: three per logits row, one per
        request and one per claimed row."""
        return max_bs * (3 * rows_per_request + 2)

    def stage(self, batch, lengths, buffer) -> tuple[PresenceRows, int]:
        """Write the step's rows into `buffer` (int32, `sampling` publication)
        and return their views and the count to publish. `lengths` is each
        request's query length on a verification step, None when every request
        has one logits row."""
        bs = batch.total_seqs_num
        penalties = batch.presence_penalties[:bs]
        slots = np.full(bs, -1, dtype=np.int32)
        fresh = []
        for i in np.flatnonzero(penalties):
            slots[i], claimed = self._claim(batch.req_ids[i])
            if claimed:
                fresh.append(slots[i])
        q = np.ones(bs, np.int64) if lengths is None else np.asarray(lengths, np.int64)
        rows = int(q.sum())
        emit = slots.copy()
        if batch.is_final_chunk is not None:
            # A middle prefill chunk's sampled token is not output.
            emit[~np.asarray(batch.is_final_chunk[:bs], dtype=bool)] = -1
        out = buffer.np
        out[:rows] = np.repeat(slots, q)
        out[rows : 2 * rows] = np.repeat(penalties, q).view(np.int32)
        out[2 * rows : 3 * rows] = np.arange(rows) - np.repeat(np.cumsum(q) - q, q)
        e = 3 * rows
        out[e : e + bs] = emit
        out[e + bs : e + bs + len(fresh)] = fresh
        count = e + bs + len(fresh)
        gpu = buffer.gpu
        return (
            PresenceRows(
                slots=gpu[:rows],
                penalties=gpu[rows : 2 * rows].view(torch.float32),
                drafts_before=gpu[2 * rows : e],
                emit_slots=gpu[e : e + bs],
                fresh=gpu[e + bs : count],
                num_slots=self._num_slots,
            ),
            count,
        )

    def release(self, request_ids) -> None:
        for request_id in request_ids:
            slot = self._slot_of.pop(request_id, None)
            if slot is not None:
                self._free.append(slot)

    def _claim(self, request_id) -> tuple[int, bool]:
        slot = self._slot_of.get(request_id)
        if slot is not None:
            return slot, False
        if not self._free:
            # Running requests fit; preempted ones keep their rows.
            n = self._num_slots
            self._free = list(range(2 * n - 1, n - 1, -1))
            self._num_slots = 2 * n
        slot = self._free.pop()
        self._slot_of[request_id] = slot
        return slot, True

    def table(self, num_slots: int) -> torch.Tensor:
        """The device table with at least `num_slots` rows; grown on the stream
        that reads it, so earlier steps' kernels finish on the old one first."""
        if self.seen.shape[0] < num_slots:
            grown = self.seen.new_zeros(num_slots, self.seen.shape[1])
            grown[: self.seen.shape[0]] = self.seen
            self.seen = grown
        return self.seen
