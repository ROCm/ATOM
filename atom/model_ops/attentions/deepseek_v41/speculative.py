# SPDX-License-Identifier: MIT
"""Accepted-prefix selection for Engram history.

The window has physical slack, so rejecting rows only changes visibility, and
the compressor's ring is widened by the same slack -- a rejected round's writes
land past the window the next round reads, so neither needs a rollback. Engram
history is the one thing left that cannot be reconstructed from KV: it is
staged per input prefix and the sampler picks one per request before
checkpointing or drafting.
"""

import torch


class TentativeState:
    def __init__(self, cache, step):
        self.cache, self.step = cache, step
        self.request_indices = {span.slot: i for i, span in enumerate(step.requests)}
        batch, width = step.scheduled_bs, step.max_q_len
        self.cursors = torch.empty(
            batch,
            width,
            cache.geometry.history_size + 1,
            dtype=torch.int64,
            device=cache.pool.device,
        )
        self.histories_written = set()

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

    def commit(self, accepted_lengths):
        """Lengths include the guaranteed target input: zero drafts means one."""
        if self.histories_written != set(self.request_indices):
            raise RuntimeError("Tentative state is missing an Engram prefix")
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
        # The scheduled prefix, not the forward's width: a padding request owns
        # no slot, and the 0 standing in for one is a live request's.
        slots = self.step.slots[: self.step.scheduled_bs].long()
        self.cache.cursor[slots] = self.cursors[batch, lengths - 1]
