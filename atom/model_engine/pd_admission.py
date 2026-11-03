# SPDX-License-Identifier: MIT
"""Decode-side admission credits, held through the first scheduled forward."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("atom")


@dataclass
class PendingReceive:
    sequence: Any
    blocks: frozenset[int]
    ready_at: float | None = None
    cancelled: bool = False


class PdAdmissionController:
    """Credits never free KV or decide whether a remote writer is safe."""

    @classmethod
    def from_config(cls, config, connector, metrics):
        options = (getattr(config, "kv_transfer_config", None) or {}).get(
            "pd_admission", {}
        )
        if not isinstance(options, dict):
            raise TypeError("pd_admission must be an object")
        if type(options.get("enabled", False)) is not bool:
            raise ValueError("pd_admission.enabled must be a boolean")
        if not options.get("enabled", False):
            return None
        unknown = options.keys() - {
            "enabled",
            "max_pending_requests",
            "max_pending_kv_bytes",
            "max_pending_kv_fraction",
        }
        if unknown:
            raise ValueError(f"Unknown pd_admission options: {sorted(unknown)}")
        if not getattr(connector, "supports_pd_admission", False):
            raise ValueError("pd_admission requires a Mooncake consumer")
        if any(
            getattr(config, name, 1) != 1
            for name in (
                "pipeline_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise ValueError("pd_admission requires D PP=1, DCP=1 and PCP=1")
        block_bytes = getattr(config, "pd_admission_block_bytes", 0)
        if block_bytes <= 0:
            raise ValueError("pd_admission requires a single paged KV pool layout")
        count = options.get("max_pending_requests", 4)
        byte_limit = options.get("max_pending_kv_bytes")
        fraction = options.get(
            "max_pending_kv_fraction", None if byte_limit is not None else 0.15
        )
        for name, value in (
            ("max_pending_requests", count),
            ("max_pending_kv_bytes", byte_limit),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(
                    f"pd_admission.{name} must be a positive integer or null"
                )
        if fraction is not None:
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not math.isfinite(fraction)
                or not 0 < fraction <= 1
            ):
                raise ValueError(
                    "pd_admission.max_pending_kv_fraction must be in (0, 1]"
                )
            if byte_limit is not None:
                raise ValueError("Set only one PD byte limit: bytes or fraction")
            byte_limit = max(1, int(config.num_kvcache_blocks * block_bytes * fraction))
        if count is None and byte_limit is None:
            raise ValueError("pd_admission requires at least one limit")
        return cls(count, byte_limit, block_bytes, metrics)

    def __init__(self, max_requests, max_bytes, block_bytes, metrics=None):
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.block_bytes = block_bytes
        self.metrics = metrics
        self.pending: dict[str, PendingReceive] = {}
        self._block_refs: dict[int, int] = {}
        self.rejected = {"requests": 0, "bytes": 0}
        self.oversized_admissions = 0
        self._publish()
        logger.info(
            "PD admission enabled: pending_requests=%s, pending_kv_bytes=%s "
            "per D rank, block_bytes=%d; credits held until first forward",
            max_requests,
            max_bytes,
            block_bytes,
        )

    @property
    def pending_bytes(self):
        return len(self._block_refs) * self.block_bytes

    def can_admit(self, receive_blocks: int) -> bool:
        reason = None
        if self.max_requests is not None and len(self.pending) >= self.max_requests:
            reason = "requests"
        elif (
            self.max_bytes is not None
            and self.pending_bytes + receive_blocks * self.block_bytes > self.max_bytes
            and self.pending
        ):
            # With no pending receive, allow a single large request. The normal
            # allocator still enforces the hard capacity and lookahead checks.
            reason = "bytes"
        if reason is None:
            return True
        self.rejected[reason] += 1
        if self.metrics is not None:
            self.metrics.pd_admission_rejected.labels(
                **self.metrics._labels, reason=reason
            ).inc()
        return False

    def admit(self, seq, block_ids):
        key = str(seq.id)
        if key in self.pending:
            raise RuntimeError(f"Duplicate PD admission for sequence {key}")
        blocks = frozenset(block_ids)
        if (
            self.max_bytes is not None
            and len(blocks) * self.block_bytes > self.max_bytes
        ):
            self.oversized_admissions += 1
            if self.metrics is not None:
                self.metrics.pd_admission_oversized.inc()
        self.pending[key] = PendingReceive(seq, blocks)
        for block in blocks:
            self._block_refs[block] = self._block_refs.get(block, 0) + 1
        self._publish()

    def received(self, req_id):
        entry = self.pending.get(str(req_id))
        if entry is not None and entry.ready_at is None:
            entry.ready_at = time.perf_counter()
            self._publish()

    def cancel(self, seq):
        entry = self.pending.get(str(seq.id))
        if entry is not None and entry.sequence is seq:
            entry.cancelled = True
            self._publish()

    def scheduled(self, sequences):
        for seq in sequences:
            entry = self.pending.get(str(seq.id))
            if entry is None:
                continue
            if entry.ready_at is None or entry.cancelled:
                raise RuntimeError(f"Scheduled unfinished PD receive {seq.id}")
            if self.metrics is not None:
                self.metrics.pd_admission_ready_wait.observe(
                    time.perf_counter() - entry.ready_at
                )
            self.release(seq)

    def release(self, seq):
        key = str(seq.id)
        entry = self.pending.get(key)
        if entry is None or entry.sequence is not seq:
            return
        del self.pending[key]
        for block in entry.blocks:
            refs = self._block_refs[block] - 1
            if refs:
                self._block_refs[block] = refs
            else:
                del self._block_refs[block]
        self._publish()

    def _publish(self):
        if self.metrics is None:
            return
        counts = {"receiving": 0, "ready": 0, "cancel_draining": 0}
        for entry in self.pending.values():
            state = (
                "cancel_draining"
                if entry.cancelled
                else ("ready" if entry.ready_at is not None else "receiving")
            )
            counts[state] += 1
        for state, count in counts.items():
            self.metrics.pd_admission_pending.labels(
                **self.metrics._labels, state=state
            ).set(count)
        self.metrics.pd_admission_bytes.set(self.pending_bytes)
