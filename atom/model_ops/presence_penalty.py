# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""OpenAI `presence_penalty`, with vLLM's semantics.

Before sampling, a request's logits lose its penalty on every token it has
generated so far; its prompt does not count. A verification row also counts
the drafts before it: the row checks the draft that follows them, so they are
output there. The bonus row counts every draft.

A request's generated tokens are bits in its row of `PresencePenalties.seen`.
The step that emits a token sets its bit after the token broadcast, so every
rank holds the same bits. A request claims a row on its first step that
samples it, the step that clears the row before it penalizes. The request gives
the row back when the engine releases it (finished, aborted or refused) or
preempts it; a resumed request marks its earlier output again on the step that
claims its new row.

The table has a fixed size, allocated before the memory profile: two rows per
`max_num_seqs`. A finished or aborted request releases its row right after the
step's postprocess, before the next forward. A preempted request releases it in
the next forwarded batch, before that batch claims rows. So only running
requests hold rows when a step claims; the second half is headroom.

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
    resumed_slots: torch.Tensor  # [ids] int32: the row of each id below
    resumed_ids: torch.Tensor  # [ids] int32: resumed requests' earlier output


class PresencePenalties:
    """The per-request bit table of generated tokens, and its row bookkeeping
    on the host. Every rank runs the same bookkeeping on the same batches."""

    def __init__(self, vocab_size: int, max_num_seqs: int, device):
        self.vocab_size = vocab_size
        num_slots = 2 * max_num_seqs
        self.seen = torch.zeros(
            num_slots, -(-vocab_size // 32), dtype=torch.int32, device=device
        )
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
        has one logits row. Only a row that is sampled gets a table row: a
        middle prefill chunk's token is not output."""
        bs = batch.total_seqs_num
        penalties = batch.presence_penalties[:bs]
        sampled = penalties != 0
        if batch.is_final_chunk is not None:
            sampled &= np.asarray(batch.is_final_chunk[:bs], dtype=bool)
        slots = np.full(bs, -1, dtype=np.int32)
        fresh, resumed_slots, resumed_ids = [], [], []
        for i in np.flatnonzero(sampled):
            request_id = batch.req_ids[i]
            slots[i], claimed = self._claim(request_id)
            earlier = batch.presence_resumed.get(request_id)
            if claimed or earlier is not None:
                fresh.append(slots[i])
            if earlier is not None:
                resumed_slots.append(np.full(len(earlier), slots[i], np.int32))
                resumed_ids.append(earlier)
        q = np.ones(bs, np.int64) if lengths is None else np.asarray(lengths, np.int64)
        rows = int(q.sum())
        out = buffer.np
        out[:rows] = np.repeat(slots, q)
        out[rows : 2 * rows] = np.repeat(penalties, q).view(np.int32)
        out[2 * rows : 3 * rows] = np.arange(rows) - np.repeat(np.cumsum(q) - q, q)
        e = 3 * rows
        out[e : e + bs] = slots
        out[e + bs : e + bs + len(fresh)] = fresh
        count = e + bs + len(fresh)
        gpu = buffer.gpu
        resumed = [gpu[:0], gpu[:0]]
        if resumed_ids:
            # Rare and of any length: a blocking copy of its own, not the
            # fixed-size publication.
            resumed = [
                torch.from_numpy(np.concatenate(part)).to(gpu.device)
                for part in (resumed_slots, resumed_ids)
            ]
        return (
            PresenceRows(
                slots=gpu[:rows],
                penalties=gpu[rows : 2 * rows].view(torch.float32),
                drafts_before=gpu[2 * rows : e],
                emit_slots=gpu[e : e + bs],
                fresh=gpu[e + bs : count],
                resumed_slots=resumed[0],
                resumed_ids=resumed[1],
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
            raise RuntimeError(
                "presence_penalty table full: more requests hold rows than "
                "two per max_num_seqs"
            )
        slot = self._free.pop()
        self._slot_of[request_id] = slot
        return slot, True
