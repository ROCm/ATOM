# SPDX-License-Identifier: MIT
"""Accepted-prefix selection for CSA2 tails and Engram history.

The window has physical slack, so rejecting rows only changes visibility.
Small state that cannot be reconstructed from KV is staged per input prefix;
the sampler selects one prefix per request before checkpointing or drafting.
"""

import torch


class TentativeState:
    def __init__(self, cache, step):
        self.cache, self.step = cache, step
        self.request_indices = {span.slot: i for i, span in enumerate(step.requests)}
        batch, width = len(step.requests), step.max_length
        options = {"device": cache.pool.device}
        shape = (len(cache.tail_indices), batch, width, cache.geometry.head_dim)
        self.values = torch.zeros(shape, dtype=torch.float32, **options)
        self.scores = torch.zeros_like(self.values)
        self.cursors = torch.empty(
            batch, width, cache.geometry.history_size + 1, dtype=torch.int64, **options
        )
        self.tails_written, self.histories_written = set(), set()
        self.finished = False

    def stage_tail(self, owner, span, rows):
        if rows is None or rows.scores is None:
            raise ValueError("Tentative compression requires per-token projections")
        expected = (1, span.length, self.cache.geometry.head_dim)
        if rows.values.shape != expected or rows.scores.shape != expected:
            raise ValueError(
                "Tentative compressor projections do not match the request"
            )
        i, owner_index = self.request_indices[span.slot], self.cache.tail_indices[owner]
        # An odd committed cursor leaves the preceding even-position row as
        # the incomplete group. Even cursors have no compressor tail.
        keep = (
            torch.arange(span.position, span.end, device=rows.values.device) % 2 == 0
        )[:, None]
        self.values[owner_index, i, : span.length] = torch.where(
            keep, rows.values[0], 0
        )
        self.scores[owner_index, i, : span.length] = torch.where(
            keep, rows.scores[0], 0
        )
        self.tails_written.add((owner, span.slot))

    def stage_history(self, span, compressed_ids):
        ids = torch.as_tensor(
            compressed_ids, dtype=torch.int64, device=self.cursors.device
        )
        if ids.shape != (span.length,):
            raise ValueError(
                "Tentative Engram history requires one compressed ID per token"
            )
        history = self.cache.cursor[span.slot, 1:]
        prefixes = torch.cat((history, ids)).unfold(0, history.numel(), 1)[1:]
        i = self.request_indices[span.slot]
        self.cursors[i, : span.length, 0] = torch.arange(
            span.position + 1, span.end + 1, device=ids.device
        )
        self.cursors[i, : span.length, 1:] = prefixes
        self.histories_written.add(span.slot)

    def finish(self):
        expected_tails = {
            (owner, span.slot)
            for owner in self.cache.tail_indices
            for span in self.step.requests
        }
        if self.tails_written != expected_tails or self.histories_written != set(
            self.request_indices
        ):
            raise RuntimeError(
                "Tentative state is missing compressor or Engram prefixes"
            )
        self.finished = True

    def commit(self, accepted_lengths):
        """Lengths include the guaranteed target input: zero drafts means one."""
        if not self.finished:
            raise RuntimeError("Cannot commit before the target forward finishes")
        lengths = torch.as_tensor(
            accepted_lengths, dtype=torch.int64, device=self.cursors.device
        )
        if lengths.shape != (len(self.step.requests),):
            raise ValueError("Accepted lengths must match the scheduled requests")
        limits = torch.tensor(
            [span.length for span in self.step.requests], device=lengths.device
        )
        torch._assert_async(
            ((lengths >= 1) & (lengths <= limits)).all(),
            "Accepted prefix is outside the verification span",
        )
        batch = torch.arange(lengths.numel(), device=lengths.device)
        slots = self.step.slots.long()
        self.cache.cursor[slots] = self.cursors[batch, lengths - 1]
        for name, source in (
            ("tail_values", self.values),
            ("tail_scores", self.scores),
        ):
            self.cache.state.view(name)[:, slots, 0] = source[:, batch, lengths - 1]
