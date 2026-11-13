# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's offload schedulers as this frontend drives them.

What only the vLLM plugin asks of a scheduler lives here, on top of ATOM's own
classes, rather than in them: which loads a build declined, whether a recent
save still holds blocks it will give back, and refusing a save whose source
blocks vLLM no longer holds.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler
from atom.kv_transfer.offload.mp.scheduler import LMCacheMPConnectorScheduler

logger = logging.getLogger("atom")


def _save_source_is_backed(block_ids, saved: int, aligned: int, block: int) -> bool:
    """Whether the block table still covers every token the save claims."""
    end_block = -(-int(aligned) // block)  # ceil div
    if len(block_ids) < end_block:
        return False
    return all(
        block_ids[index] is not None for index in range(int(saved) // block, end_block)
    )


class PluginSchedulerMixin:
    """The plugin's view of an ATOM offload scheduler; listed before it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init_plugin_state()

    def _init_plugin_state(self) -> None:
        #: Loads queued for this step's build that it did not dispatch. A
        #: request parked on its load is released this step rather than
        #: after the promise grace.
        self.declined_loads: set[str] = set()
        #: Lease key -> when its blocks were last leased or a save of its
        #: request was dispatched; what `has_active_save_leases` ages.
        self._lease_fresh_at: dict[int, float] = {}
        self.total_unbacked_saves = 0

    def build_connector_meta(self):
        queued = set(self._reqs_need_recv)
        meta = super().build_connector_meta()
        dispatched = set()
        now = time.monotonic()
        for request in getattr(meta, "requests", ()):
            sid = str(request.req_id)
            if request.load_spec is not None:
                dispatched.add(sid)
            if request.save_spec is not None:
                entry = self._save_tracker.get(sid)
                if entry is not None and id(entry[0]) in self._save_lease_blocks:
                    self._lease_fresh_at[id(entry[0])] = now
        self.declined_loads = queued - dispatched
        return meta

    def enforce_transfer_deadlines(self) -> None:
        """Bound transfers that never report, every step: vLLM calls no
        `process_completions`, where ATOM's engine has the MP scheduler do it.
        Only MP bounds its transfers this way."""

    def activate_block_leases(self, seq, block_ids: frozenset[int]) -> None:
        super().activate_block_leases(seq, block_ids)
        if id(seq) in self._save_lease_blocks:
            self._lease_fresh_at[id(seq)] = time.monotonic()

    def has_active_save_leases(self, max_age_s: float) -> bool:
        """Whether blocks are leased for a save that may land soon.

        Such blocks are held by a refcount rather than by the request, so
        nothing about the request says they exist; they are what lets vLLM
        wait a step instead of preempting a running request. Only leases
        taken, or saves dispatched, in the last ``max_age_s``: an older one
        is a save that is stuck rather than landing.
        """
        fresh = self._lease_fresh_at
        for key in [key for key in fresh if key not in self._save_lease_blocks]:
            del fresh[key]
        oldest = time.monotonic() - max_age_s
        return any(at > oldest for at in fresh.values())

    def _save_source_backed(self, seq, saved: int, aligned: int, block_ids) -> bool:
        """Refuse a save whose source blocks vLLM has taken back.

        A preemption clears the request's block table while its save is still
        queued. Emitting anyway ships a descriptor the worker cannot honour
        ("needs N blocks for [start, end), got 0") and kills the transfer; the
        tokens are simply not resident to read. A finished request gets no
        blocks back, so its save is retired rather than retried every step.
        """
        block = int(getattr(self, "virtual_block_size", self.block_size))
        if _save_source_is_backed(block_ids, saved, aligned, block):
            return True
        self.total_unbacked_saves += 1
        if getattr(seq, "_offload_finished", False):
            self._retire_finished_save(str(seq.id), seq)
        logger.warning(
            "Offload save of seq %s covers [%d, %d) but its block table holds "
            "%d blocks; skipping (total unbacked saves=%d)",
            seq.id,
            saved,
            aligned,
            len(block_ids),
            self.total_unbacked_saves,
        )
        return False

    def _build_save_request(
        self, seq, saved, aligned, operation, block_ids, is_last_prefill
    ):
        if not self._save_source_backed(seq, saved, aligned, block_ids):
            return None
        return super()._build_save_request(
            seq, saved, aligned, operation, block_ids, is_last_prefill
        )


class PluginMPScheduler(PluginSchedulerMixin, LMCacheMPConnectorScheduler):
    """LMCache MP scheduler as the plugin drives it."""

    def enforce_transfer_deadlines(self) -> None:
        self._enforce_transfer_deadlines()


class PluginDenseScheduler(PluginSchedulerMixin, DenseOffloadScheduler):
    """In-process offload scheduler as the plugin drives it."""


__all__ = ["PluginDenseScheduler", "PluginMPScheduler", "PluginSchedulerMixin"]
