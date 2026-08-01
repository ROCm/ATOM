# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Batched staging for MoE expert weights.

On a large MoE checkpoint each expert's weights arrive as their own tensor, so
the per-expert `weight_loader` issues one small host-to-device copy per
(expert, shard) — tens of thousands of them, latency-bound rather than
bandwidth-bound. The pool coalesces every arrival for one fused parameter into
a CPU staging buffer and writes the result back with a single large copy.

Ownership rule: the pool writes back only the (expert slot, shard) regions it
actually staged. Any other loader path may write the same parameter as long as
it touches different regions, which is what lets a checkpoint that stores
routed experts one way and shared experts another -- Qwen3.5 BF16 stacks the
routed experts into a single tensor -- load correctly. Before writing such a
parameter, that other path calls `decline` so the two never race for a region.

Deliberately free of AITER and of `atom.config`: the pool talks to a MoE module
only through the protocol below, so it can be unit-tested on a plain CPU
runner.

    stage_expert_weight(param, staging, loaded_weight, local_expert_id,
                        shard_id, weight_name) -> bool
    expected_batched_arrivals(param) -> int | None
    _map_global_expert_id_to_local_expert_id(global_expert_id) -> int
    is_batched_expert_slot(local_expert_id) -> bool
    flush_staged(param, staging, filled) -> None
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("atom")

# `zero_` on a packed dtype only reaches `fill_cpu` -- which has no kernel for
# it -- once the tensor is big enough for torch's parallel path (GRAIN_SIZE,
# 32768 elements). Below that it takes a memset fast path and succeeds, so a
# small probe would answer "zeroable" for every dtype and defeat the point.
_ZERO_PROBE_NUMEL = 1 << 15


@functools.cache
def _cpu_zeroable(dtype: torch.dtype) -> bool:
    """Whether a staging-sized CPU tensor of `dtype` can be `zero_`d.

    Cached per dtype: a checkpoint has a handful of dtypes and tens of
    thousands of tensors, and the probe allocates (briefly) to answer.
    """
    try:
        torch.empty(_ZERO_PROBE_NUMEL, dtype=dtype, device="cpu").zero_()
    except NotImplementedError:
        return False
    except RuntimeError:
        # Some dtypes cannot even be allocated flat on CPU. Treat that the same
        # way -- raw bytes stage correctly for anything.
        return False
    return True


@dataclass
class StagingEntry:
    """One in-flight fused parameter's staging buffer and its filled regions."""

    param: torch.nn.Parameter
    staging: torch.Tensor
    moe: Any
    expected: int
    name: str
    filled: set[tuple[int, str]] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def complete(self) -> bool:
        return len(self.filled) >= self.expected

    def missing_description(self) -> str:
        got = len(self.filled)
        slots = sorted({slot for slot, _ in self.filled})
        span = f"{slots[0]}..{slots[-1]}" if slots else "none"
        return (
            f"{self.name}: {got}/{self.expected} (expert slot, shard) regions "
            f"staged, slots {span}"
        )


@dataclass
class StagingReport:
    """What the pool still held when loading finished."""

    incomplete: list[str] = field(default_factory=list)


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
        # Staging-allocation accounting, reported once at flush time. Pinning
        # host memory runs at ~4 GB/s, so a buffer built and then dropped is
        # not free; `_discarded_*` is how much of that the allocation race
        # costs, which is deliberately tolerated (see `_get_or_create_entry`).
        self._alloc_count = 0
        self._alloc_bytes = 0
        self._alloc_seconds = 0.0
        self._discarded_count = 0
        self._discarded_bytes = 0

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

        # This slot is somebody else's (a fused shared expert), but the
        # parameter as a whole is still batchable -- do not poison it the way
        # an unstageable shard below does.
        if not moe.is_batched_expert_slot(local_eid):
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        if entry is None:
            entry = self._get_or_create_entry(param, moe, full_param_name)
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
            # The shard shape or dtype is not one the batched path handles, and
            # that verdict holds for every arrival of this parameter, so hand
            # the whole thing over -- writing back what already landed first.
            self.decline(param)
            self._direct_load(
                param, full_param_name, shard_id, global_expert_id, loaded_weight
            )
            return

        with entry.lock:
            entry.filled.add((local_eid, shard_id))
            complete = entry.complete
        if complete:
            with self._lock:
                claimed = self._entries.pop(pid, None) is entry
            if claimed:
                with entry.lock:
                    moe.flush_staged(param, entry.staging, entry.filled)

    def decline(self, param: torch.nn.Parameter) -> None:
        """Hand this parameter over to another loader path.

        Called before any write that does not go through the pool. Whatever has
        already been staged is written back first: the other path is about to
        write different regions of the same parameter, and dropping the buffer
        would silently lose the arrivals that did land.
        """
        pid = id(param)
        with self._lock:
            self._declined.add(pid)
            entry = self._entries.pop(pid, None)
        if entry is not None:
            with entry.lock:
                entry.moe.flush_staged(param, entry.staging, entry.filled)

    def flush_pending(self) -> StagingReport:
        """Write back and drop every entry still held, after the drain.

        An entry that is still here never reached its arrival count. Its staged
        regions are written back regardless -- partial data beats none, and the
        report tells the caller exactly what is missing.
        """
        with self._lock:
            pending = list(self._entries.values())
            self._entries.clear()
        report = StagingReport()
        for entry in pending:
            with entry.lock:
                entry.moe.flush_staged(entry.param, entry.staging, entry.filled)
                report.incomplete.append(entry.missing_description())
        logger.info(
            "Staging buffers: %d allocated (%.1f GiB, %.2fs pinning), "
            "%d dropped to allocation races (%.1f GiB)",
            self._alloc_count,
            self._alloc_bytes / 1024**3,
            self._alloc_seconds,
            self._discarded_count,
            self._discarded_bytes / 1024**3,
        )
        return report

    # ── internals ─────────────────────────────────────────────────────────

    def _get_or_create_entry(self, param, moe, name: str) -> StagingEntry | None:
        pid = id(param)
        # Allocate outside the lock, and speculatively: concurrent first
        # arrivals for one parameter each build a buffer and all but one are
        # dropped, which looks wasteful and is -- 341 of 573 buffers on
        # DeepSeek-R1 MXFP4. Publishing the entry first and letting one thread
        # allocate while the others wait removes that waste and measured
        # *worse*: it idles fifteen workers at every new parameter, and on a
        # cold cache those workers are what keeps the device busy (cold load
        # 154s -> 183s). Wasted host bandwidth is the cheaper of the two.
        t0 = time.perf_counter()
        staging = self._allocate_staging(param)
        alloc_seconds = time.perf_counter() - t0
        candidate = StagingEntry(
            param=param,
            staging=staging,
            moe=moe,
            expected=moe.expected_batched_arrivals(param),
            name=name,
        )
        nbytes = staging.numel() * staging.element_size()
        with self._lock:
            self._alloc_count += 1
            self._alloc_bytes += nbytes
            self._alloc_seconds += alloc_seconds
            if pid in self._declined:
                return None
            winner = self._entries.setdefault(pid, candidate)
            if winner is not candidate:
                self._discarded_count += 1
                self._discarded_bytes += nbytes
            return winner

    @staticmethod
    def _allocate_staging(param: torch.nn.Parameter) -> torch.Tensor:
        """A zeroed host buffer shaped like `param`.

        Zero-initialised on purpose: a slot that is only partially written (a
        padded MXFP4 shard, say) must read back as zero, matching what the
        parameter itself was initialised to.
        """
        # Pick the dtype before allocating. Discovering it the other way --
        # allocate, let `zero_` raise, allocate again -- costs a full pinned
        # allocation per parameter, and host pinning runs at ~4.2 GB/s cold.
        # On DeepSeek-R1 MXFP4 every routed-expert parameter is packed, so the
        # discarded first buffer was ~40% of the loader worker pool's time.
        dtype = param.data.dtype if _cpu_zeroable(param.data.dtype) else torch.uint8

        def _alloc(pinned: bool) -> torch.Tensor:
            try:
                t = torch.empty(
                    param.data.shape,
                    dtype=dtype,
                    device="cpu",
                    pin_memory=pinned,
                )
                # Kept inside the try: `_cpu_zeroable` probes one size, and a
                # dtype it clears there could still fail here. Falling back
                # costs what this function used to cost every time, which is
                # the right price for being wrong rarely.
                t.zero_()
            except NotImplementedError:
                # Stage the raw bytes instead and let the flush re-view the
                # parameter as uint8.
                t = torch.empty(
                    param.data.shape,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pinned,
                )
                t.zero_()
            return t

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
