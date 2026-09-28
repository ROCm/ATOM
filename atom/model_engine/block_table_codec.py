# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Ship a forward's block tables to the workers as appends alone.

Every forward RPC broadcasts one `ScheduledBatch` to every worker, and its
`block_tables` are the bulk of it: one row per running request, the whole row
every step, growing one block per decode. At 50 seqs x 100k context that is
~313k ids -- 1.2 MiB pickled and unpickled per rank per step -- to say
something whose new content is 50 ids.

The rows are `BlockTable`s, which carry a `version` that only a non-append
mutation redraws (see `atom/model_engine/sequence.py`). So the encoder does
not have to *infer* that a row merely grew, and never compares ids to find
out: it remembers `(version, length)` per request from the last step it sent,
and an unchanged version proves that everything up to that length is what the
workers already hold. Anything else -- a table cleared and refilled, a prefix
privatised in place by `BlockManager.disown_claimed_prefix`, a request seen
for the first time -- redraws the version and is sent in full.

The workers rebuild their rows from their own cache. Both sides are always on
and decoding is driven by the payload's type, so there is no setting a worker
could hold differently from the scheduler: whichever form a batch arrives in
is the form it is read in, and a step the encoder cannot account for arrives
as whole tables and resets both caches at once.
"""

import copy
import logging
from dataclasses import dataclass

import numpy as np

from atom.model_engine.sequence import BlockTable

logger = logging.getLogger("atom")

# The one RPC whose payload is large enough and repetitive enough to be worth
# encoding. Named here rather than in the transport so that `async_proc` need
# not know anything about what a forward carries.
FORWARD_RPC = "forward"

_I32 = np.dtype(np.int32)


@dataclass(frozen=True)
class BlockTableDelta:
    """One step's block tables, as a shared prefix length plus new ids.

    Row `i` is ``workers_row_i[: base_lengths[i]]`` followed by
    ``tail_values[tail_offsets[i] : tail_offsets[i + 1]]``. A `base_lengths` of
    0 makes the row self-contained, which is how a request's first step, and
    every step after a table was disturbed, travel.
    """

    base_lengths: np.ndarray  # int32[num_rows]
    tail_offsets: np.ndarray  # int32[num_rows + 1]
    tail_values: np.ndarray  # int32[tail_offsets[-1]]


class BlockTableDeltaEncoder:
    """Scheduler side: replace a batch's block tables with a `BlockTableDelta`.

    One encoder per broadcast channel. It tracks what that channel's workers
    were last told, so it must see every forward on that channel and no other.
    """

    def __init__(self):
        # req_id -> (version, length) as last published to the workers.
        self._published: dict[int, tuple[int, int]] = {}
        self._full_send_reason: str | None = None

    def encode_rpc(self, func_name: str, args: tuple) -> tuple:
        """Encode `args` if this is a forward, otherwise pass them through."""
        if func_name != FORWARD_RPC or not args:
            return args
        return (self.encode(args[0]), *args[1:])

    def encode(self, batch):
        rows = getattr(batch, "block_tables", None)
        req_ids = getattr(batch, "req_ids", None)
        if not rows or req_ids is None:
            # A warmup or dummy batch carries no tables at all.
            return self._send_whole(batch, "batch has no block tables")
        if len(rows) != len(req_ids):
            # `ScheduledBatch` drops rows for seqs with an empty table, which
            # would leave row i describing some other request. Downstream
            # consumers index the rows by batch position too, so this is not
            # expected to happen -- say so rather than silently sending full
            # tables forever.
            return self._send_whole(
                batch, f"{len(rows)} rows for {len(req_ids)} requests"
            )
        if not all(isinstance(row, BlockTable) for row in rows):
            # Rollout/disaggregation callers may build batches by hand.
            return self._send_whole(batch, "rows are not BlockTables")

        num_rows = len(rows)
        base_lengths = np.empty(num_rows, dtype=_I32)
        tail_offsets = np.empty(num_rows + 1, dtype=_I32)
        tail_offsets[0] = 0
        tails: list[np.ndarray] = []
        published: dict[int, tuple[int, int]] = {}

        for i, (req_id, row) in enumerate(zip(req_ids, rows, strict=True)):
            req_id = int(req_id)
            row_len = len(row)
            base = 0
            last = self._published.get(req_id)
            if last is not None:
                last_version, last_len = last
                # `last_len <= row_len` cannot fail while the version holds --
                # an unchanged version means only appends -- and is the one
                # cheap guard against a mutator that forgot to redraw it.
                if last_version == row.version and last_len <= row_len:
                    base = last_len
            base_lengths[i] = base
            # Zero-copy view of the ids this step added; `np.concatenate`
            # below is the only pass over them.
            tails.append(
                np.frombuffer(
                    row, dtype=_I32, count=row_len - base, offset=_I32.itemsize * base
                )
            )
            tail_offsets[i + 1] = tail_offsets[i] + (row_len - base)
            published[req_id] = (row.version, row_len)

        wire_batch = copy.copy(batch)
        wire_batch.block_tables = BlockTableDelta(
            base_lengths=base_lengths,
            tail_offsets=tail_offsets,
            tail_values=(
                np.concatenate(tails, dtype=_I32)
                if tail_offsets[-1]
                else np.empty(0, dtype=_I32)
            ),
        )
        # Published only once the encoded batch exists: if building it raises,
        # the workers were told nothing and the cache must not claim otherwise.
        self._published = published
        self._full_send_reason = None
        return wire_batch

    def _send_whole(self, batch, reason: str):
        """Fall back to the plain batch, and drop what the workers were told.

        The workers key off the payload's type, so an un-encoded batch also
        resets their side; the two caches cannot drift apart.
        """
        self._published.clear()
        if reason != self._full_send_reason:
            # Once per reason, not once per step: this is the difference
            # between "the optimization is off" and a log flood.
            logger.debug("block-table delta off, sending whole tables: %s", reason)
            self._full_send_reason = reason
        return batch


class BlockTableDeltaDecoder:
    """Worker side: rebuild versioned rows from a `BlockTableDelta`."""

    def __init__(self):
        self._rows: dict[int, BlockTable] = {}

    def decode_rpc(self, func_name: str, args: list) -> list:
        """Decode `args[0]` in place if this is an encoded forward."""
        if func_name == FORWARD_RPC and args:
            args[0] = self.decode(args[0])
        return args

    def decode(self, batch):
        delta = getattr(batch, "block_tables", None)
        if not isinstance(delta, BlockTableDelta):
            # Whole tables arrived, so whatever was cached is superseded --
            # the encoder cleared its side for the same step.
            self._rows.clear()
            return batch

        req_ids = batch.req_ids
        if len(req_ids) != len(delta.base_lengths):
            raise RuntimeError(
                f"block-table delta has {len(delta.base_lengths)} rows for "
                f"{len(req_ids)} requests"
            )

        rows: list[BlockTable] = []
        cached: dict[int, BlockTable] = {}
        for i, req_id in enumerate(req_ids):
            req_id = int(req_id)
            base = int(delta.base_lengths[i])
            start = int(delta.tail_offsets[i])
            end = int(delta.tail_offsets[i + 1])
            if base:
                previous = self._rows.get(req_id)
                if previous is None or len(previous) != base:
                    raise RuntimeError(
                        f"missing block-table prefix for request {req_id}: "
                        f"need {base}, have "
                        f"{None if previous is None else len(previous)}"
                    )
                # Appending into the row the previous batch handed out would
                # rewrite history the token processor may still be reading, so
                # a row that grows is copied first. A row that did not grow is
                # shared, which is the decode-step-with-no-new-block case.
                row = previous
                if end != start:
                    row = BlockTable(previous)
                    # This is the next immutable snapshot in the same append
                    # lineage. Equal version + length still means equal ids.
                    row.version = previous.version
            else:
                row = BlockTable()
            if end > start:
                row.frombytes(memoryview(delta.tail_values[start:end]).cast("B"))
            rows.append(row)
            cached[req_id] = row

        batch.block_tables = rows
        self._rows = cached
        return batch
