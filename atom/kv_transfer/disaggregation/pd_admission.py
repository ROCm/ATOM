# SPDX-License-Identifier: MIT
"""Decode-side admission credits, held through the first scheduled forward."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from atom.model_engine.capabilities import pcp_group_size_of

logger = logging.getLogger("atom")


@dataclass
class PendingReceive:
    sequence: Any
    blocks: frozenset[int]
    ready: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class ReceivePlan:
    """The destination range a PD receive writes: `block_table[offset:num_blocks]`."""

    offset: int
    num_blocks: int


class PdAdmissionController:
    """Credits never free KV or decide whether a remote writer is safe."""

    @classmethod
    def from_config(cls, config, connector, block_manager):
        options = (config.kv_transfer_config or {}).get("pd_admission", {})
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
            "min_free_kv_fraction",
        }
        if unknown:
            raise ValueError(f"Unknown pd_admission options: {sorted(unknown)}")
        if connector is None or not connector.supports_pd_admission:
            raise ValueError("pd_admission requires a Mooncake consumer")
        if (
            config.pipeline_parallel_size != 1
            or config.decode_context_parallel_size != 1
            # Ulysses SP runs over the PCP group, so this also covers SP.
            or pcp_group_size_of(config) != 1
        ):
            raise ValueError("pd_admission requires D PP=1, DCP=1, PCP=1 and SP=1")
        # Credits are counted in pages, so the paged KV must be the only pool.
        entry_bytes = list(config.pool_entry_bytes.values())
        if len(entry_bytes) != 1 or entry_bytes[0] <= 0:
            raise ValueError("pd_admission requires a single paged KV pool layout")
        block_bytes = entry_bytes[0]
        byte_limit = options.get("max_pending_kv_bytes")
        fraction = options.get(
            "max_pending_kv_fraction", None if byte_limit is not None else 0.15
        )
        free_fraction = options.get("min_free_kv_fraction")
        if byte_limit is not None and (type(byte_limit) is not int or byte_limit <= 0):
            raise ValueError(
                "pd_admission.max_pending_kv_bytes must be a positive integer or null"
            )
        for name, value, closed in (
            ("max_pending_kv_fraction", fraction, True),
            ("min_free_kv_fraction", free_fraction, False),
        ):
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not (0 < value <= 1 if closed else 0 < value < 1)
            ):
                interval = "(0, 1]" if closed else "(0, 1)"
                raise ValueError(f"pd_admission.{name} must be in {interval}")
        if fraction is not None:
            if byte_limit is not None:
                raise ValueError("Set only one PD byte limit: bytes or fraction")
            byte_limit = max(1, int(config.num_kvcache_blocks * block_bytes * fraction))
        reserve_blocks = (
            None
            if free_fraction is None
            else math.ceil(config.num_kvcache_blocks * free_fraction)
        )
        if byte_limit is None and reserve_blocks is None:
            raise ValueError("pd_admission requires at least one limit")
        return cls(
            byte_limit,
            block_bytes,
            connector,
            block_manager,
            reserve_blocks=reserve_blocks,
        )

    def __init__(
        self, max_bytes, block_bytes, connector, block_manager, *, reserve_blocks=None
    ):
        self.max_bytes = max_bytes
        self.block_bytes = block_bytes
        # The receive range is the connector's prefix-skip rule over the
        # allocator's destination table; neither is the scheduler's to know.
        self._connector = connector
        self._block_manager = block_manager
        # Free blocks kept for the running batch's decode growth.
        self.reserve_blocks = reserve_blocks
        self.pending: dict[str, PendingReceive] = {}
        self._block_refs: dict[int, int] = {}
        # Requests deferred, once per admission wait and limit, for the metrics
        # snapshot. `_deferred_by` holds the limits each waiting request hit.
        self.deferred = {"bytes": 0, "free_kv": 0}
        self._deferred_by: dict[str, set[str]] = {}
        logger.info(
            "PD admission enabled: pending_kv_bytes=%s per D rank, "
            "reserved_free_blocks=%s, block_bytes=%d; credits held until first "
            "forward",
            max_bytes,
            reserve_blocks,
            block_bytes,
        )

    @property
    def pending_bytes(self):
        return len(self._block_refs) * self.block_bytes

    @property
    def pending_states(self) -> dict[str, int]:
        """Pending requests by state; a cancelled one drains whether or not ready."""
        states = dict.fromkeys(("receiving", "ready", "cancel_draining"), 0)
        for entry in self.pending.values():
            if entry.cancelled:
                states["cancel_draining"] += 1
            elif entry.ready:
                states["ready"] += 1
            else:
                states["receiving"] += 1
        return states

    def check(self, seq, probe) -> tuple[bool, ReceivePlan | None]:
        """Whether `seq` may allocate now, and the range a later `commit` charges.

        Takes no credit and publishes nothing, so a caller that does not reach
        `commit` (allocation or detach declined) just drops the result. The
        plan is `None` when this is not a PD receive.
        """
        num_blocks = self._block_manager.num_pool_blocks(len(seq))
        offset = self._connector.pd_receive_block_offset(
            seq, num_blocks, probe.num_cached_blocks
        )
        if offset is None:
            return True, None
        plan = ReceivePlan(offset, num_blocks)
        reason = None
        # With no pending receive, allow a single request past either soft
        # limit. The normal allocator still enforces the hard capacity and
        # lookahead checks.
        if (
            self.max_bytes is not None
            and self.pending
            and self.pending_bytes
            + (num_blocks - offset - self._charged_hits(probe, offset))
            * self.block_bytes
            > self.max_bytes
        ):
            reason = "bytes"
        elif (
            self.reserve_blocks is not None
            # Every block the request allocates, not only the receive targets:
            # a local prefix hit on a cached-but-free block leaves the free
            # list like a receive target does.
            and self._block_manager.kv.num_free - num_blocks < self.reserve_blocks
            and self.pending
        ):
            reason = "free_kv"
        if reason is None:
            return True, plan
        # One deferral per admission wait and limit, however many passes it
        # takes; a repeated pass allocates nothing.
        key = str(seq.id)
        reasons = self._deferred_by.get(key)
        if reasons is None:
            reasons = self._deferred_by[key] = set()
        if reason not in reasons:
            reasons.add(reason)
            self.deferred[reason] += 1
        return False, plan

    def _charged_hits(self, probe, offset: int) -> int:
        """Prefix hits inside the receive range that pending credits already hold.

        Only a full transfer over a local hit (no usable page geometry) writes
        into hit blocks; fresh and cached-but-free blocks hold no credit.
        """
        lookup = self._block_manager.kv.lookup
        return sum(
            lookup(h) in self._block_refs
            for h in probe.block_hashes[offset : probe.num_cached_blocks]
        )

    def commit(self, seq, plan: ReceivePlan) -> None:
        """Charge the range `check` approved, before the receive is published.

        That range is exact only while `allocate` keeps the probe's hit and
        block count. It does for the layouts `from_config` accepts; supporting
        per-request state (a joint boundary, a disowned prefix) has to
        re-measure the receive from `seq.block_table` here.
        """
        key = str(seq.id)
        if key in self.pending:
            raise RuntimeError(f"Duplicate PD admission for sequence {key}")
        self._deferred_by.pop(key, None)
        blocks = frozenset(seq.block_table[plan.offset : plan.num_blocks])
        self.pending[key] = PendingReceive(seq, blocks)
        for block in blocks:
            self._block_refs[block] = self._block_refs.get(block, 0) + 1

    def received(self, req_id):
        entry = self.pending.get(str(req_id))
        if entry is not None:
            entry.ready = True

    def cancel(self, seq):
        key = str(seq.id)
        # A request rejected while deferred never reaches `commit`.
        self._deferred_by.pop(key, None)
        entry = self.pending.get(key)
        if entry is not None and entry.sequence is seq:
            entry.cancelled = True

    def scheduled(self, sequences):
        # Runs on every batch, and most batches decode with nothing pending.
        if not self.pending:
            return
        for seq in sequences:
            entry = self.pending.get(str(seq.id))
            if entry is None:
                continue
            if not entry.ready or entry.cancelled:
                raise RuntimeError(f"Scheduled unfinished PD receive {seq.id}")
            self.release(seq)
            if not self.pending:
                return

    def release(self, seq):
        key = str(seq.id)
        self._deferred_by.pop(key, None)
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
