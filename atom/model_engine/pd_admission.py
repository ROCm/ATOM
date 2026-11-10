# SPDX-License-Identifier: MIT
"""Decode-side admission credits, held through the first scheduled forward."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from atom.model_engine.capabilities import pcp_group_size_of

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
    def from_config(cls, config, connector):
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
            "max_pending_kv_bytes",
            "max_pending_kv_fraction",
        }
        if unknown:
            raise ValueError(f"Unknown pd_admission options: {sorted(unknown)}")
        if not getattr(connector, "supports_pd_admission", False):
            raise ValueError("pd_admission requires a Mooncake consumer")
        if (
            any(
                getattr(config, name, 1) != 1
                for name in ("pipeline_parallel_size", "decode_context_parallel_size")
            )
            # Ulysses SP runs over the PCP group, so this also covers SP.
            or pcp_group_size_of(config) != 1
        ):
            raise ValueError("pd_admission requires D PP=1, DCP=1, PCP=1 and SP=1")
        block_bytes = getattr(config, "pd_admission_block_bytes", 0)
        if block_bytes <= 0:
            raise ValueError("pd_admission requires a single paged KV pool layout")
        byte_limit = options.get("max_pending_kv_bytes")
        fraction = options.get(
            "max_pending_kv_fraction", None if byte_limit is not None else 0.15
        )
        if byte_limit is not None and (type(byte_limit) is not int or byte_limit <= 0):
            raise ValueError(
                "pd_admission.max_pending_kv_bytes must be a positive integer or null"
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
        if byte_limit is None:
            raise ValueError("pd_admission requires a receive-byte limit")
        return cls(byte_limit, block_bytes)

    def __init__(self, max_bytes, block_bytes):
        self.max_bytes = max_bytes
        self.block_bytes = block_bytes
        self.pending: dict[str, PendingReceive] = {}
        self._block_refs: dict[int, int] = {}
        # Requests deferred at least once, per limit.
        self.deferred = {"bytes": 0}
        self.oversized_admissions = 0
        logger.info(
            "PD admission enabled: pending_kv_bytes=%d per D rank, "
            "block_bytes=%d; credits held until first forward",
            max_bytes,
            block_bytes,
        )

    @property
    def pending_bytes(self):
        return len(self._block_refs) * self.block_bytes

    def can_admit(self, seq, receive_blocks: int) -> bool:
        # With no pending receive, allow a single request past the soft
        # limit. The normal allocator still enforces the hard capacity and
        # lookahead checks.
        if (
            not self.pending
            or self.pending_bytes + receive_blocks * self.block_bytes <= self.max_bytes
        ):
            return True
        reason = "bytes"
        # One deferral per request and limit, however many passes it waits.
        if getattr(seq, "_pd_deferred_by", None) is None:
            seq._pd_deferred_by = set()
        if reason not in seq._pd_deferred_by:
            seq._pd_deferred_by.add(reason)
            self.deferred[reason] += 1
        return False

    def admit(self, seq, block_ids):
        key = str(seq.id)
        if key in self.pending:
            raise RuntimeError(f"Duplicate PD admission for sequence {key}")
        seq._pd_deferred_by = None
        blocks = frozenset(block_ids)
        if len(blocks) * self.block_bytes > self.max_bytes:
            self.oversized_admissions += 1
        self.pending[key] = PendingReceive(seq, blocks)
        for block in blocks:
            self._block_refs[block] = self._block_refs.get(block, 0) + 1

    def received(self, req_id):
        entry = self.pending.get(str(req_id))
        if entry is not None and entry.ready_at is None:
            entry.ready_at = time.perf_counter()

    def cancel(self, seq):
        entry = self.pending.get(str(seq.id))
        if entry is not None and entry.sequence is seq:
            entry.cancelled = True

    def scheduled(self, sequences):
        for seq in sequences:
            entry = self.pending.get(str(seq.id))
            if entry is None:
                continue
            if entry.ready_at is None or entry.cancelled:
                raise RuntimeError(f"Scheduled unfinished PD receive {seq.id}")
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
