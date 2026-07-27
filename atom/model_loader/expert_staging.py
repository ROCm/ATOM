# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Batched staging for MoE expert weights.

On a large MoE checkpoint each expert's weights arrive as their own tensor, so
the per-expert `weight_loader` issues one small host-to-device copy per
(expert, shard) — tens of thousands of them, latency-bound rather than
bandwidth-bound. The pool coalesces every arrival for one fused parameter into
a CPU staging buffer and writes the result back with a single large copy.

Deliberately free of AITER and of `atom.config`: the pool talks to a MoE module
only through the small protocol below, so it can be unit-tested on a plain CPU
runner.

    stage_expert_weight(param, staging, loaded_weight, local_expert_id,
                        shard_id, weight_name) -> bool
    expected_batched_arrivals(param) -> int | None
    _map_global_expert_id_to_local_expert_id(global_expert_id) -> int
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("atom")


@dataclass
class StagingEntry:
    """One in-flight fused parameter's staging buffer and arrival count."""

    param: torch.nn.Parameter
    staging: torch.Tensor
    moe: Any
    expected: int
    arrived: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class ExpertStagingPool:
    """Coalesces per-expert arrivals for a fused MoE parameter.

    `resolve_moe` maps a full parameter name to the module that owns it (or
    None); the loader supplies it because only the loader holds the model.
    """

    def __init__(self, resolve_moe: Callable[[str], Any]):
        self._resolve_moe = resolve_moe
        self._entries: dict[int, StagingEntry] = {}
        self._declined: set[int] = set()
        self._lock = threading.Lock()
        self._batchable: dict[int, bool] = {}

    def is_batchable(self, param: torch.nn.Parameter, full_param_name: str) -> bool:
        """Whether arrivals for this parameter should go through the pool."""
        pid = id(param)
        if pid not in self._batchable:
            moe = self._resolve_moe(full_param_name)
            expected = (
                moe.expected_batched_arrivals(param)
                if moe is not None and hasattr(moe, "stage_expert_weight")
                else None
            )
            self._batchable[pid] = bool(expected)
        return self._batchable[pid]

    def stage(
        self,
        param: torch.nn.Parameter,
        full_param_name: str,
        shard_id: str,
        global_expert_id: int,
        loaded_weight: torch.Tensor,
    ) -> None:
        """Stage one arrival, falling back to the per-expert loader if needed.

        Runs on a worker thread, concurrently with other arrivals for the same
        and for other parameters.
        """
        pid = id(param)
        with self._lock:
            declined = pid in self._declined
            entry = None if declined else self._entries.get(pid)
        if declined:
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        # Map to this rank's local expert id BEFORE touching the entry table.
        # Under expert parallelism every rank iterates all global experts, but a
        # non-local expert contributes nothing to this rank's staging. If such a
        # straggler ran after the param already reached `expected` and flushed
        # (which drops its entry), creating an entry here would leave a fresh,
        # never-filled entry that is miscounted as "under-filled" at the end of
        # loading. Return early so non-local shards never create entries.
        moe = self._resolve_moe(full_param_name)
        local_eid = moe._map_global_expert_id_to_local_expert_id(global_expert_id)
        if local_eid == -1:
            return

        if entry is None:
            entry = self._get_or_create_entry(param, moe)
            if entry is None:  # declined while we were allocating
                self._direct_load(
                    param, full_param_name, shard_id, global_expert_id, loaded_weight
                )
                return

        staged = moe.stage_expert_weight(
            param=param,
            staging=entry.staging,
            loaded_weight=loaded_weight,
            local_expert_id=local_eid,
            shard_id=shard_id,
            weight_name=full_param_name,
        )
        if not staged:
            with self._lock:
                self._declined.add(pid)
                self._entries.pop(pid, None)
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        with entry.lock:
            entry.arrived += 1
            complete = entry.arrived >= entry.expected
        if complete:
            self._flush(entry)
            with self._lock:
                if self._entries.get(pid) is entry:
                    del self._entries[pid]

    def take_pending(self) -> list[StagingEntry]:
        """Drain and return entries that never reached their arrival count."""
        with self._lock:
            pending = list(self._entries.values())
            self._entries.clear()
        return pending

    # ── internals ─────────────────────────────────────────────────────────

    def _get_or_create_entry(self, param, moe) -> StagingEntry | None:
        pid = id(param)
        # Allocate outside the lock: the buffer is parameter-sized and pinning
        # it is slow enough that holding the lock would serialize every layer.
        candidate = StagingEntry(
            param=param,
            staging=self._allocate_staging(param),
            moe=moe,
            expected=moe.expected_batched_arrivals(param),
        )
        with self._lock:
            if pid in self._declined:
                return None
            return self._entries.setdefault(pid, candidate)

    @staticmethod
    def _allocate_staging(param: torch.nn.Parameter) -> torch.Tensor:
        """A zeroed host buffer shaped like `param`.

        Zero-initialised on purpose: a slot that is only partially written (a
        padded MXFP4 shard, say) must read back as zero, matching what the
        parameter itself was initialised to.
        """

        def _alloc(pinned: bool) -> torch.Tensor:
            try:
                t = torch.empty(
                    param.data.shape,
                    dtype=param.data.dtype,
                    device="cpu",
                    pin_memory=pinned,
                )
            except NotImplementedError:
                # Packed dtypes (fp4x2) have no CPU implementation; stage the
                # raw bytes instead and let the flush re-view the parameter.
                t = torch.empty(
                    param.data.shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pinned,
                )
            return t.zero_()

        try:
            return _alloc(torch.cuda.is_available())
        except RuntimeError as e:
            logger.warning("Pinned staging alloc failed (%s); using unpinned.", e)
            return _alloc(False)

    @staticmethod
    def _direct_load(
        param, full_param_name, shard_id, global_expert_id, loaded_weight
    ) -> None:
        param.weight_loader(
            param, loaded_weight, full_param_name, shard_id, global_expert_id
        )

    @staticmethod
    def _flush(entry: StagingEntry) -> None:
        param, staging = entry.param, entry.staging
        if staging.dtype != param.data.dtype:
            param.data.view(torch.uint8).copy_(staging)
        else:
            param.data.copy_(staging)
