# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-side publication and lifetime of a Mooncake chunked prefill.

The block table is allocated once. Publications describe completed forwards,
not the scheduler's (possibly already advanced) live sequence. Each consumer
has its own cursor into this table. No network or GPU dependency lives here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefillHandoff:
    """Generation state passed from P to D after prefill; KV travels via RDMA."""

    req_id: str | int  # D-local request ID for receive completion.
    first_token_id: int  # First output token sampled by P.
    draft_token_ids: tuple[int, ...] = ()  # Drafts for D to verify.
    prefix_cache_hit_tokens: int = 0  # P-side prefix hits for metrics.

    @classmethod
    def from_wire(cls, req_id, data):
        """Validate the wire handoff and attach D's local request ID."""
        if not isinstance(data, dict):
            raise TypeError("prefill handoff must be a dictionary")
        first = data.get("first_token_id")
        drafts = data.get("draft_token_ids", [])
        cached = data.get("prefix_cache_hit_tokens", 0)
        if type(first) is not int or first < 0:
            raise ValueError("invalid prefill first token")
        if not isinstance(drafts, (list, tuple)) or any(
            type(x) is not int or x < 0 for x in drafts
        ):
            raise ValueError("invalid prefill draft tokens")
        if type(cached) is not int or cached < 0:
            raise ValueError("invalid prefill cache hit count")
        return cls(req_id, first, tuple(drafts), cached)


class ChunkedPrefill:
    """Track chunk publication and source lifetime for one prefill on a P worker.

    The full block table stays fixed; each consumer advances its own cursor.
    A condition variable coordinates chunk/handoff waits and reader exit.
    Callers handle GPU synchronization and RDMA.
    """

    def __init__(
        self,
        req_id,
        block_ids,
        num_tokens,
        block_size,
        slot_index=-1,
        swa_block_ids=(),
        timeout=60,
        prompt_digest=None,
    ):
        if num_tokens <= 0 or block_size <= 0:
            raise ValueError("chunked prefill requires positive token/block counts")
        if len(block_ids) != (num_tokens + block_size - 1) // block_size:
            raise ValueError("chunked prefill block table does not cover the prompt")
        self.req_id = req_id
        self.prompt_digest = prompt_digest
        self.block_ids = tuple(block_ids)
        self.num_tokens = num_tokens
        self.block_size = block_size
        self.slot_index = slot_index
        self.swa_block_ids = list(swa_block_ids)
        self.timeout = timeout
        self.cv = threading.Condition()
        self.num_ready_blocks = 0  # Published prefix blocks; reads must wait on event.
        self.event: Any = None
        self.handoff: dict | None = None
        self.cancelled = False
        self.readers = 0  # Claimed send tasks still active or queued.
        self.expected_readers = 0  # Total consumers served by this P rank.
        self.completed_readers = 0  # Exited send tasks, including failures.
        self.claims: set[tuple] = set()  # Retain identities to deduplicate sends.
        self.updated = time.monotonic()

    def publish(self, end_tokens, event):
        """Publish a forward boundary and event; only the final block may be partial."""
        with self.cv:
            if self.cancelled:
                return
            end_tokens = min(end_tokens, self.num_tokens)
            num_blocks = (
                (end_tokens + self.block_size - 1) // self.block_size
                if end_tokens == self.num_tokens
                else end_tokens // self.block_size
            )
            if num_blocks < self.num_ready_blocks:
                raise ValueError("prefill publication moved backwards")
            self.num_ready_blocks = num_blocks
            self.event = event
            self.updated = time.monotonic()
            self.cv.notify_all()

    def finish(self, handoff, *, slot_index=None, swa_block_ids=None):
        """Publish final sampling and slot/SWA state; RDMA may still be in flight."""
        with self.cv:
            if not self.cancelled:
                if slot_index is not None:
                    self.slot_index = slot_index
                # SWA is mutable per-request state. Transfer its final slot
                # once the final handoff is available, using the current slot.
                if swa_block_ids is not None:
                    self.swa_block_ids = list(swa_block_ids)
                self.handoff = dict(handoff)
                self.updated = time.monotonic()
            self.cv.notify_all()

    def cancel(self):
        """Cancel waits and new claims; in-flight RDMA must still drain."""
        with self.cv:
            self.cancelled = True
            self.cv.notify_all()

    def acquire(self, consumer_identity, expected_readers):
        """Claim one consumer's reads of this producer rank's source blocks.

        The identity deduplicates write requests; expected_readers counts all
        consumers that must finish before the source allocation can be freed.
        """
        with self.cv:
            # The original reader owns the terminal notification even after
            # cancellation. Failing a duplicate now could let D free its pages
            # while that reader is still writing them.
            if consumer_identity in self.claims:
                return False
            if self.cancelled:
                raise RuntimeError("prefill was cancelled")
            if self.expected_readers and self.expected_readers != expected_readers:
                raise ValueError("inconsistent consumer fan-out")
            if len(self.claims) >= expected_readers:
                raise ValueError("too many consumers for one prefill")
            self.expected_readers = expected_readers
            self.claims.add(consumer_identity)
            self.readers += 1
            return True

    def release(self):
        """Release a reader when its send task exits, on success or failure."""
        with self.cv:
            self.readers -= 1
            self.completed_readers += 1
            self.cv.notify_all()

    def source_safe(self):
        """Apply the terminal timeout and check source safety without freeing blocks."""
        with self.cv:
            # A missing consumer must not pin HBM indefinitely. Active RDMA
            # readers always drain before reclamation, even after a timeout.
            if (
                self.handoff is not None
                and time.monotonic() - self.updated >= self.timeout
            ):
                self.cancelled = True
                self.cv.notify_all()
            return self.readers == 0 and (
                self.cancelled
                or (
                    self.handoff is not None
                    and self.expected_readers > 0
                    and self.completed_readers == self.expected_readers
                )
            )

    def wait_chunk(self, src_block_offset, src_blocks_per_dst_block=1):
        """Return source block IDs, their GPU event, and the exclusive end offset.

        Offsets index the full producer block table, including any cached prefix.
        ``src_blocks_per_dst_block`` is the number of producer blocks represented
        by one consumer block.

        DCP consumers need whole groups of producer blocks except at the final
        tail. Holding back an incomplete group prevents rewriting a destination
        page while a later chunk fills the rest of it.
        """
        with self.cv:

            def ready_src_block_end():
                num_blocks = self.num_ready_blocks
                if num_blocks == len(self.block_ids):
                    return num_blocks
                return num_blocks - num_blocks % src_blocks_per_dst_block

            ready = self.cv.wait_for(
                lambda: self.cancelled or ready_src_block_end() > src_block_offset,
                timeout=self.timeout,
            )
            if not ready or self.cancelled:
                raise RuntimeError("prefill chunk cancelled or timed out")
            src_block_end = ready_src_block_end()
            return (
                list(self.block_ids[src_block_offset:src_block_end]),
                self.event,
                src_block_end,
            )

    def wait_handoff(self):
        """Wait for the final handoff, raising on cancellation or timeout."""
        with self.cv:
            ready = self.cv.wait_for(
                lambda: self.cancelled or self.handoff is not None,
                timeout=self.timeout,
            )
            if not ready or self.cancelled:
                raise RuntimeError("prefill handoff cancelled or timed out")
            return dict(self.handoff)
