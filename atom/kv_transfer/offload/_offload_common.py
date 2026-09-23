# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Shared machinery for ATOM's dense and hybrid offload families.

The public ``lmcache_offload`` shell selects either ordinary dense raw-block KV
or DSV4 PAGE+SLOT storage. Both families share the worker-side executor, role,
completion, and LMCache-engine plumbing here; family modules retain only their
payload mapping and PAGE/SLOT policy.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadCompletionId,
    SaveCompletionId,
)
from atom.kv_transfer.offload import config as offcfg

logger = logging.getLogger("atom")
_VALID_KV_ROLES = {"offload", "kv_both", "kv_producer", "kv_consumer"}


def tokens_to_tensor(tokens: list[int]) -> torch.Tensor:
    """Materialize a request's token ids as an int64 CPU tensor.

    ``torch.tensor(list_of_int)`` unboxes every element through the CPython
    API while holding the GIL, and offload runs on save/load worker threads
    that contend for it with the forward loop. numpy builds the buffer in C
    and ``from_numpy`` adopts it without a copy; the resulting values and
    dtype are identical.

    Two measurements, because they disagree and the smaller one is the one to
    plan against. A microbenchmark at M3's longest requests (32768 ids, three
    threads spinning on the GIL) gives 16.1 ms per call for ``torch.tensor``
    against 1.8 ms here -- 8.8x. In the server, across a 180 s window at 440
    save calls per rank, the same substitution moved the conversion from
    5.07 s to 1.71 s -- 3.0x. The gap is request length: the benchmark uses
    the longest requests, the server sees a distribution, and the numpy path's
    fixed cost is a larger share of a short one. The in-server figure is what
    the connector's own budget moved by, and it was 24% of the whole save cost
    before the change.

    The tensor aliases the fresh numpy buffer, which nothing else holds, so
    LMCache owns it outright.
    """
    return torch.from_numpy(np.asarray(tokens, dtype=np.int64))


def validated_kv_role(kvc: dict) -> str:
    role = kvc.get("kv_role", "offload")
    if role not in _VALID_KV_ROLES:
        raise ValueError(
            f"invalid kv_role {role!r}; expected one of {sorted(_VALID_KV_ROLES)}"
        )
    return role


def pp_aware_rank_and_world(config, tp) -> tuple[int, int]:
    """Return the LMCache rank identity for this worker under PP.

    PP stages hold disjoint layer slices, so the bytes they offload are not
    interchangeable. The TP rank alone repeats on every stage, which would make
    all stages share one engine namespace and one IPC socket. Fold the stage
    index in so each stage gets its own.
    """
    pp_rank = int(
        getattr(getattr(config, "parallel_config", None), "pipeline_parallel_rank", 0)
        or 0
    )
    pp_size = int(getattr(config, "pipeline_parallel_size", 1) or 1)
    return pp_rank * tp.world_size + tp.rank_in_group, pp_size * tp.world_size


def build_offload_engine(
    config,
    *,
    engine_id: str,
    block_size: int,
    bytes_per_block: int,
    gpu_connector_factory,
    world: int,
    rank: int,
    cfg=None,
):
    """Build + post_init a per-rank LMCache engine for opaque uint8 offload.

    ``gpu_connector_factory(cfg, meta)`` builds the LMCache
    ``GPUConnectorInterface`` once the validated chunk size and uint8 metadata
    exist. Returns ``(engine, cfg, meta)``. The metadata forces uint8 shapes;
    ``fmt`` is a tensor-accepting ``MemoryFormat`` purely to satisfy the
    LocalCPU allocator.
    """
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.memory_management import MemoryFormat

    from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata

    if cfg is None:
        cfg = offcfg.build_lmcache_config(getattr(config, "kv_transfer_config", None))
    # Only the worker engine allocates the CPU pool, so the per-stage split is
    # applied here and not on the scheduler-side lookup clients.
    offcfg.scale_cpu_size_for_pp(cfg, config)
    base_meta = offcfg.build_lmcache_metadata(config, cfg, world, rank)
    meta = ATOMRawBytesLMCacheMetadata(
        base_meta, atom_block_size=int(block_size), bytes_per_block=int(bytes_per_block)
    )
    gpu_connector = gpu_connector_factory(cfg, meta)
    engine = LMCacheEngineBuilder.get_or_create(
        engine_id, cfg, meta, gpu_connector, lambda t, s: None, lambda o, s: o
    )
    engine.fmt = MemoryFormat.KV_2LTD
    engine.post_init()
    return engine, cfg, meta


class OffloadWorkerMixin:
    """Executor plumbing + completion reporting shared by offload workers.

    Subclasses call :meth:`_init_worker_common` from ``__init__`` and use the
    ``_save_executor`` / ``_load_executor`` + the ``_done_save`` / ``_done_load``
    / ``_failed_load`` tallies. Override :meth:`_on_load_fail` for connectors that
    hold a lookup pin to release on failure.
    """

    is_producer = False

    def _init_worker_common(
        self,
        config,
        *,
        save_workers: int | None = None,
        load_workers: int | None = None,
        thread_name_prefix: str = "offload",
    ) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        # Separate executors so a load (on the TTFT critical path) never queues
        # behind fire-and-forget saves. OFFLOAD_COPY_WORKERS tunes the save pool,
        # OFFLOAD_LOAD_WORKERS the load pool.
        n_save = (
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
            if save_workers is None
            else int(save_workers)
        )
        if n_save <= 0:
            raise ValueError("offload save worker count must be positive")
        # A single load thread saturates once the HBM pool is small enough for
        # the CPU tier to serve real traffic: measured 143s of `retrieve` inside
        # a 163s window (88% duty cycle) on the radix workload at 7900 blocks,
        # which turned a +57.8pp hit-rate win into a 4% throughput loss. The
        # byte-copy path keeps its staging buffers and CUDA streams in
        # thread-local state (`_BlockGpuConnector._thread_state`), so extra load
        # threads are independent; each costs one more
        # `gpu_staging_buffer_bytes` allocation per rank.
        n_load = (
            int(os.environ.get("OFFLOAD_LOAD_WORKERS", "1"))
            if load_workers is None
            else int(load_workers)
        )
        if n_load <= 0:
            raise ValueError("offload load worker count must be positive")
        # Kept alongside the pools so callers that want to report the widths --
        # the startup banner does -- need not reach into ThreadPoolExecutor's
        # private `_max_workers`.
        self.save_workers = n_save
        self.load_workers = n_load
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save, thread_name_prefix=f"{thread_name_prefix}-save"
        )
        self._load_executor = ThreadPoolExecutor(
            max_workers=n_load, thread_name_prefix=f"{thread_name_prefix}-load"
        )
        self._lock = threading.Lock()
        self._done_save: set[SaveCompletionId] = set()
        self._done_load: set[LoadCompletionId] = set()
        self._failed_load: set[LoadCompletionId] = set()
        self._connector_completions: set[ConnectorCompletion] = set()
        # GPU blocks a failed load left unfilled, and the copy jobs still
        # running per request. Both exist for the vLLM plugin path: vLLM needs
        # the block ids to truncate `num_computed_tokens` at the first block a
        # load could not supply, and needs a fence it can hold a preemption
        # against. See `take_load_error_blocks` / `wait_for_requests`.
        self._failed_load_blocks: set[int] = set()
        self._inflight_jobs: dict[str, set] = {}

    def close(self) -> None:
        """Join the save/load executors at worker teardown.

        `ThreadPoolExecutor` threads are non-daemon, so a process that exits
        without joining them either hangs on interpreter shutdown or logs a
        `threads can no longer be started` error as atexit tears the pools down
        out from under an in-flight copy. `ModelRunner.exit()` calls this before
        it destroys the distributed env. Idempotent -- a second call finds the
        executors already shut down. Subclasses that own further resources
        (K3's state tier) override and call `super().close()` last.
        """
        for name in ("_save_executor", "_load_executor"):
            executor = getattr(self, name, None)
            if executor is not None:
                executor.shutdown(wait=True)

    # -- in-flight job tracking (preemption fence) -----------------------
    def _track_job(self, req_id, future) -> None:
        """Remember one submitted copy job so a preemption can wait on it.

        vLLM frees a preempted request's blocks inside `schedule()` and may hand
        them to another request in the same step. A save still reading them then
        persists the new occupant's bytes under the old request's key -- a
        poisoned cache entry, not a lost one. `wait_for_requests` is the fence,
        and it needs the futures the submit sites otherwise discard.
        """

        sid = str(req_id)
        with self._lock:
            self._inflight_jobs.setdefault(sid, set()).add(future)
        # Runs inline when the job already finished; `_lock` is released above,
        # so the callback can retake it.
        future.add_done_callback(lambda done, sid=sid: self._untrack_job(sid, done))

    def _untrack_job(self, sid: str, future) -> None:
        with self._lock:
            pending = self._inflight_jobs.get(sid)
            if pending is None:
                return
            pending.discard(future)
            if not pending:
                del self._inflight_jobs[sid]

    def wait_for_requests(self, req_ids) -> None:
        """Block until every copy job for `req_ids` has stopped touching HBM.

        Called from the preemption fence, which runs before the forward that
        would overwrite the freed blocks. `_guard` already swallows job
        exceptions, so `result()` is only ever a join.
        """

        jobs: set = set()
        with self._lock:
            for req_id in req_ids or ():
                jobs |= self._inflight_jobs.get(str(req_id), set())
        for job in jobs:
            try:
                job.result()
            except Exception:  # pragma: no cover - `_guard` swallows job errors
                logger.exception("offload: in-flight job raised while fencing")

    def take_load_error_blocks(self) -> set[int]:
        """Drain the GPU blocks that failed loads left holding no valid KV."""

        with self._lock:
            blocks = set(self._failed_load_blocks)
            self._failed_load_blocks.clear()
        return blocks

    def _record_load_error_blocks(self, req) -> None:
        """Record the block range a failed load was supposed to fill.

        The caller must hold `self._lock`. The range is the one the load owned:
        `[hbm_cached_tokens, lmcache_cached_tokens)`. Everything below the HBM
        frontier is already valid, so reporting it would make vLLM discard a
        prefix that is fine -- and those lower blocks may be shared with another
        request, whose computed count would then be truncated too.

        The token-to-block grid is the VIRTUAL block size, the same one
        `BlockGPUConnector` maps chunks with: under DCP one scheduler block id
        covers one virtual global block while the codec moves a rank-local
        physical page, so the physical size would index the wrong entries.
        """

        load_spec = getattr(req, "load_spec", None)
        block_ids = list(getattr(req, "block_ids", ()) or ())
        block_size = int(
            getattr(self, "virtual_block_size", None)
            or getattr(self, "block_size", 0)
            or 0
        )
        if load_spec is None or not block_ids or block_size <= 0:
            return
        start = max(0, int(load_spec.hbm_cached_tokens)) // block_size
        end = -(-max(0, int(load_spec.lmcache_cached_tokens)) // block_size)
        self._failed_load_blocks.update(block_ids[start:end])

    @staticmethod
    def _load_completion_id(req) -> LoadCompletionId:
        return getattr(req, "load_operation", None) or req.req_id

    @staticmethod
    def _save_completion_id(req) -> SaveCompletionId:
        return getattr(req, "save_operation", None) or req.req_id

    def _on_load_fail(self, req_id) -> None:
        """Release the LMCache lookup pin held by a failed load."""

        self._lookup_unpin(req_id)

    def _lookup_unpin(self, req_id) -> None:
        """Best-effort release of one worker-side LMCache lookup pin."""

        engine = getattr(self, "_engine", None)
        if engine is None:
            return
        try:
            engine.lookup_unpin(str(req_id))
        except Exception:  # optional third-party cleanup boundary
            logger.debug(
                "LMCache offload: lookup unpin failed for req=%s",
                req_id,
                exc_info=True,
            )

    @staticmethod
    def _profile_enabled() -> bool:
        # Same env-flag semantics as `atom_lmcache_staging._env_flag` (kept
        # inline rather than imported: that module pulls in torch and this one
        # is torch-free). Strip, and read the empty string as OFF -- `VAR=` is
        # how a shell clears a flag inline, and a bare membership test would
        # read "" as ON (not in the false set), the opposite of what the
        # operator wrote; `VAR="off "` had the same trap.
        raw = os.environ.get("OFFLOAD_PROFILE", "0").strip().lower()
        return bool(raw) and raw not in {"0", "false", "no", "off"}

    def _last_gpu_connector_transfer_stats(self) -> dict[str, int | float]:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "last_transfer_stats"):
            return {}
        try:
            return dict(gpu_connector.last_transfer_stats())
        except Exception:  # optional instrumentation hook
            logger.debug(
                "LMCache offload: transfer stats collection failed",
                exc_info=True,
            )
            return {}

    def _reset_gpu_connector_transfer_stats(self) -> None:
        gpu_connector = getattr(getattr(self, "_engine", None), "gpu_connector", None)
        if gpu_connector is None or not hasattr(gpu_connector, "reset_transfer_stats"):
            return
        try:
            gpu_connector.reset_transfer_stats()
        except Exception:  # optional instrumentation hook
            logger.debug(
                "LMCache offload: transfer stats reset failed",
                exc_info=True,
            )

    def _guard(self, kind: str, fn, req) -> None:
        """Run a copy job off the RPC thread, tallying success/failure."""
        try:
            fn(req)
        except Exception:
            logger.exception(
                "offload %s failed for %s",
                getattr(fn, "__name__", kind),
                getattr(req, "req_id", req),
            )
            rid = getattr(req, "req_id", req)
            if kind == "load":
                self._on_load_fail(rid)
                with self._lock:
                    self._failed_load.add(self._load_completion_id(req))
                    self._record_load_error_blocks(req)
            else:
                # Layouts with a richer success/failure protocol override this
                # hook.  Legacy layouts still report a terminal save so a
                # whole-request deferred free cannot leak.
                self._record_save_failure(req)

    def _record_save_failure(self, req) -> None:
        with self._lock:
            self._done_save.add(self._save_completion_id(req))

    def get_finished(self) -> KVConnectorOutput:
        with self._lock:
            dl, fl, ds = self._drain_common_completions_locked()
            completions = set(self._connector_completions)
            self._connector_completions.clear()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=dl,
            failed_loading=fl,
            finished_saving=ds,
            connector_completions=completions,
        )

    def _drain_common_completions_locked(
        self,
    ) -> tuple[set[LoadCompletionId], set[LoadCompletionId], set[SaveCompletionId]]:
        """Drain base completion sets while the caller holds ``self._lock``."""

        done_load = set(self._done_load)
        failed_load = set(self._failed_load)
        done_save = set(self._done_save)
        self._done_load.clear()
        self._failed_load.clear()
        self._done_save.clear()
        return done_load, failed_load, done_save

    def get_finished_recv_blocks(self) -> list[int]:
        return []


class StateOffloadFace(ABC):
    """The KDA state-tier surface, implemented only by the layout that hosts it.

    `MultiConnector` and the delegating shell must tell a tier-hosting impl
    (kimi_k3) from one with no state tier (dense / dsv4-page) so they route
    state stores and loads to the right sub. They used to probe by method
    *presence* -- but the shell defines the whole face unconditionally,
    forwarding through `getattr` to `_impl`, so presence could not distinguish
    them and every state call fell through to a dense impl's no-tier defaults
    (stores recorded failed, loads dropped, reports empty). Make the face an
    explicit type: only `KimiK3OffloadScheduler` inherits it, so
    `isinstance(impl, StateOffloadFace)` is the honest predicate.

    Narrow on purpose -- only the four tier methods. Other impl attributes the
    scheduler reads (e.g. `max_pending_saves`) stay plain shell forwards rather
    than joining this face, which is exclusively the KDA state-tier contract.
    """

    @abstractmethod
    def enqueue_state_loads(self, loads) -> bool: ...
    @abstractmethod
    def enqueue_state_stores(self, stores) -> bool: ...
    @abstractmethod
    def take_state_reports(self) -> tuple[set[int], set[int]]: ...
    @abstractmethod
    def take_state_source_releases(self) -> set: ...


class OffloadSchedulerMixin(ABC):
    """Layout-independent scheduler policy shared by dense and DSV4 offload.

    Subclasses own lookup construction, metadata serialization, and any
    state-checkpoint policy. This mixin contains only token-frontier and load
    handoff mechanics whose invariants are identical for both layouts.
    """

    # The tier-hit memo answers three methods that any scheduler may reach
    # before `_init_offload_statistics` has run (the hand-built schedulers in
    # the tests, and any future partial construction). It is an optimisation,
    # so its default has to be "remember nothing" rather than an AttributeError
    # halfway through a lookup. `_remember_tier_hit` replaces the None with a
    # per-instance dict on first use.
    _tier_hit_memo: dict | None = None
    _tier_memo_steps = 32
    _tier_retry_steps = 32
    total_lookups_skipped_by_memo = 0

    # Save/load lifecycle contract. Declared abstract so a missing forwarder is
    # a construction-time TypeError, not a silent no-op behind the delegating
    # shell -- the failure mode that let DSV4 ship without abandon_save, and
    # later without the `source_blocks_released` terminal that is its only exit
    # for a request whose save completed normally. The
    # bodies differ by layout (dense keeps one save per request; DSV4 keeps a
    # set plus a SLOT sidecar), so each impl supplies its own; the contract
    # detail lives on those concrete overrides.
    @abstractmethod
    def save_finished(self, req_id) -> None: ...
    @abstractmethod
    def abandon_save(self, req_id) -> None: ...
    @abstractmethod
    def release_stalled_save(self, seq) -> None: ...
    @abstractmethod
    def source_blocks_released(self, seq) -> None: ...
    @abstractmethod
    def load_failed(self, req_id) -> bool: ...
    @abstractmethod
    def load_finished(self, req_id) -> bool: ...
    @abstractmethod
    def cancel_pending_load(self, seq) -> None: ...

    def send_finished(self, req_id) -> None:
        """Offload backends own saves and loads, but no P/D send claims."""

    def _init_offload_statistics(self) -> None:
        """Initialize layout-independent scheduler counters."""

        self.total_load_requests = 0
        self.total_loaded_tokens = 0
        self.total_load_failures = 0
        self.total_save_requests = 0
        self.total_saved_tokens = 0
        self._load_inflight_tokens: dict[object, int] = {}
        self._save_inflight_tokens: dict[object, int] = {}
        # req_id -> the sequence whose external-tier load failed. One
        # external-tier attempt per request; see `_repeat_load_suppressed`.
        self._load_failed_seqs: dict[str, object] = {}
        self.total_suppressed_load_retries = 0
        # Repeats of `get_num_new_matched_tokens` served from the remembered
        # tier hit instead of a fresh external-tier lookup.
        self.total_lookups_skipped_by_memo = 0
        self._init_tier_hit_memo()
        # Early block-release observability. Populated by layouts that support
        # exact source-block leases; unsupported layouts leave these at 0.
        self.total_early_released_blocks = 0  # freed at request-finish, not save-gated
        self.total_leased_source_blocks = 0  # ever protected by a save lease
        self.total_source_safe_released_blocks = 0  # freed once their save reported
        self.total_abnormal_lease_reclaims = 0  # freed by stall timeout, no report

    def process_completions(self, output: KVConnectorOutput) -> KVConnectorOutput:
        """Apply offload-specific completions and expose plain request IDs."""

        loaded = {
            value.req_id if hasattr(value, "req_id") else value
            for value in output.finished_loading
            if self.load_finished(value) is not False
        }
        failed = {
            value.req_id if hasattr(value, "req_id") else value
            for value in output.failed_loading
            if self.load_failed(value) is not False
        }
        terminal_saves = set()
        callback = getattr(self, "connector_completion", None)
        for completion in output.connector_completions:
            handled = callback(completion) if callback is not None else False
            if handled is False:
                logger.warning(
                    "Ignoring unhandled offload completion channel %s",
                    completion.channel,
                )
                continue
            # ``True`` means this connector-owned event is also a terminal save
            # for scheduler deferred-free purposes. ``None`` is a handled
            # non-terminal milestone such as PAGE source-safety.
            if handled is True:
                value = completion.operation_id
                terminal_saves.add(value.req_id if hasattr(value, "req_id") else value)
        for value in output.finished_saving:
            self.save_finished(value)
            terminal_saves.add(value.req_id if hasattr(value, "req_id") else value)

        output.finished_loading = loaded
        output.failed_loading = failed
        output.finished_saving = terminal_saves
        output.connector_completions.clear()
        return output

    def _track_load_statistics(self, operation, tokens: int) -> None:
        self._load_inflight_tokens[operation] = max(0, int(tokens))

    def _track_save_statistics(self, operation, tokens: int) -> None:
        self._save_inflight_tokens[operation] = max(0, int(tokens))

    def _refresh_save_reclaim_clock(self, seq) -> None:
        """Give every newly dispatched save a full source-retention window.

        A finished, deferred request dispatches its chunks serially, but the
        clock is stamped once at park time -- so generation k+1 would inherit
        the remains of generation 1's window and be abandoned by
        `_reconcile_stalled_deferred_saves` mid-copy. Restart it per dispatch;
        holding older work longer is the safe direction.
        """
        now = time.monotonic()
        if getattr(seq, "_deferred_save_at", None) is not None:
            seq._deferred_save_at = now
        lease_times = getattr(self, "_save_lease_at", {})
        if id(seq) in lease_times:
            lease_times[id(seq)] = now

    def _finish_load_statistics(self, operation, *, succeeded: bool) -> None:
        if operation not in self._load_inflight_tokens:
            return
        tokens = self._load_inflight_tokens.pop(operation)
        if succeeded:
            self.total_load_requests += 1
            self.total_loaded_tokens += tokens
        else:
            self.total_load_failures += 1

    def _cancel_load_statistics(self, operation) -> None:
        """Forget an operation retired by request cleanup without a terminal."""

        self._load_inflight_tokens.pop(operation, None)

    def _finish_save_statistics(self, operation) -> None:
        if operation not in self._save_inflight_tokens:
            return
        tokens = self._save_inflight_tokens.pop(operation)
        self.total_save_requests += 1
        self.total_saved_tokens += tokens

    def record_early_release(self, n: int) -> None:
        """Count blocks `deallocate_partial` freed immediately at request-finish.

        Called by the scheduler right after it frees the non-protected part of
        a finished request's `block_table` under early block release, so
        `get_statistics()`'s `early_released_blocks` reflects blocks a save
        never needed to hold, not just the ones a lease later returns.
        """
        self.total_early_released_blocks += max(0, int(n))

    def _cancel_save_statistics(self, operation) -> None:
        """Forget a save retired without a terminal (scheduler abandon).

        Mirror of `_cancel_load_statistics`: the bytes were never persisted, so
        this must not bump `total_save_requests`/`total_saved_tokens` the way
        `_finish_save_statistics` does -- it only drops the inflight-tokens entry
        so the pending gauge does not leak.
        """

        self._save_inflight_tokens.pop(operation, None)

    def get_statistics(self) -> dict[str, int]:
        """Return cumulative counters and exact-operation queue depths."""

        statistics = {
            "load_requests": self.total_load_requests,
            "loaded_tokens": self.total_loaded_tokens,
            "load_failures": self.total_load_failures,
            "save_requests": self.total_save_requests,
            "saved_tokens": self.total_saved_tokens,
            "loads_pending": len(self._load_inflight_tokens),
            "saves_pending": len(self._save_inflight_tokens),
            "suppressed_load_retries": self.total_suppressed_load_retries,
            "lookups_skipped_by_memo": self.total_lookups_skipped_by_memo,
        }
        if hasattr(self, "total_early_released_blocks"):
            statistics.update(
                early_released_blocks=self.total_early_released_blocks,
                leased_source_blocks=self.total_leased_source_blocks,
                source_safe_released_blocks=self.total_source_safe_released_blocks,
                blocks_waiting_for_store=self.blocks_waiting_for_store(),
                abnormal_lease_reclaims=self.total_abnormal_lease_reclaims,
            )
        return statistics

    def blocks_waiting_for_store(self) -> int:
        """PAGE blocks source-safe but not yet store-terminal, if supported."""

        return 0

    def save_abandon_timeout_s(self) -> float:
        """Seconds a deferred save may sit before the engine reclaims it.

        The value is LMCache knowledge -- it is derived from LMCache's own pin
        timeout -- so it lives on the offload connector, and the scheduler asks
        the connector for it rather than re-deriving the same env math on its own.
        See `offload_save_abandon_timeout_s` for the safety argument. Concrete on
        the mixin because every offload variant answers it identically.
        """

        return offload_save_abandon_timeout_s()

    @property
    def max_pending_saves(self) -> int | None:
        """Running-plus-queued save bound this connector enforces, else None.

        The public read of the per-connector `_max_pending_saves` that
        `max_pending_saves(kvc, save_workers)` computes from
        `kv_connector_extra_config` and `OFFLOAD_COPY_WORKERS`. The state leg
        (`Scheduler._state_store_pending_cap`) shares this exact number with the
        KV leg's `_may_emit_save` so both legs pin the same slice of the pool,
        and honours a per-connector `"max_pending_saves"` override the env reader
        never sees. None when the connector does not bound its save queue
        (`_may_emit_save` always True, as on dense) -- the scheduler then falls
        back to the env reader. Exposed so the scheduler never reaches through
        the delegating shell's `_impl` for it.
        """
        return getattr(self, "_max_pending_saves", None)

    def _chunk_floor(self, tokens: int) -> int:
        chunk = int(self.chunk_size or 256)
        return (max(0, int(tokens)) // chunk) * chunk

    def _loadable_hit(self, hit: int, num_prompt: int) -> int:
        """Turn a lookup hit into a length the external tier can actually serve.

        Two steps, in this order:

        * A hit covering the whole prompt leaves nothing to compute, so step
          back one token.
        * Floor to a chunk. `retrieve` resolves the tier at chunk granularity,
          so a `hit` that is not a chunk multiple names tokens the tier does
          not hold and the load's `ret_mask[hbm:lmc].all()` check can never
          pass. The decrement above is exactly how that happens in practice:
          a prompt whose length is a multiple of the chunk size lands on a
          boundary and stepping back one token walks off it -- which is why the
          floor must come second. Flooring costs at most one chunk of
          re-prefill and makes the spec satisfiable; without it such a request
          can never load, only fail.
        """

        hit = int(hit)
        if hit == int(num_prompt):
            hit -= 1
        return self._chunk_floor(hit)

    # -- the one expensive input: how much of this prompt the tier holds ---
    def _init_tier_hit_memo(self) -> None:
        """Remember each waiting request's tier hit, so it is asked once.

        Both schedulers ask `get_num_new_matched_tokens` before allocating, and
        every allocation failure returns the request to the head of the waiting
        queue unchanged -- so a full KV cache turns one question into one
        question per step. The question is not cheap: it copies the prompt,
        hashes it a chunk at a time on the scheduler thread (~5k hashes for a
        645k-token prompt) and blocks on the tier's reply. The step rate
        collapses, the running requests cannot finish, the cache never drains,
        and the engine livelocks with the GPUs idle.

        Only the hit is remembered -- the single costly quantity. Everything
        downstream of it (the load spec, the save floors, the decline rules) is
        arithmetic, and is re-derived on every call against the frontier of the
        moment, so no verdict and no side effect is ever replayed.

        The hit is a function of the prompt, which is fixed, and of the tier's
        contents, which can only gain a prefix while this request waits. So the
        entry goes stale in one direction, and the budgets below bound how long
        that can last: `OFFLOAD_LOOKUP_MEMO_STEPS` caps the replays of an
        answer, and `OFFLOAD_LOOKUP_RETRY_STEPS` caps how long a non-answer (a
        tier timeout, which costs `lmcache.mp.lookup_timeout` seconds of
        scheduler thread each time it is retried) suppresses the next attempt.
        """

        # sid -> (weak ref to the sequence asked about, hit or None, budget).
        self._tier_hit_memo: dict[str, tuple[object, int | None, int]] = {}
        self._tier_memo_steps = self._positive_env("OFFLOAD_LOOKUP_MEMO_STEPS", 32)
        self._tier_retry_steps = self._positive_env("OFFLOAD_LOOKUP_RETRY_STEPS", 32)

    @staticmethod
    def _positive_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            value = -1
        if value < 0:
            logger.warning(
                "LMCache offload scheduler: invalid %s=%r; using %d",
                name,
                raw,
                default,
            )
            return default
        return value

    @staticmethod
    def _weak_seq(seq):
        """A reference that does not keep an aborted request's prompt alive.

        A request can leave the waiting queue without ever reaching
        `request_finished` -- `Scheduler._reject_aborted_waiting` is one such
        path -- and an entry holding the sequence strongly would pin its whole
        `token_ids` list for the life of the process.
        """

        try:
            return weakref.ref(seq)
        except TypeError:
            # Sequence views may use __slots__ without __weakref__. Falling
            # back to a strong reference keeps the memo correct; the entry is
            # then released by the ordinary lifecycle drops below.
            return lambda seq=seq: seq

    def _remembered_tier_hit(self, seq, sid: str) -> tuple[bool, int | None]:
        """`(answered, hit)` -- spending one step of this entry's budget.

        `answered` False means the caller must run the lookup itself.
        """

        memo = self._tier_hit_memo
        entry = memo.get(sid) if memo else None
        if entry is None:
            return False, None
        ref, hit, budget = entry
        if ref() is not seq or budget <= 0:
            memo.pop(sid, None)
            return False, None
        memo[sid] = (ref, hit, budget - 1)
        self.total_lookups_skipped_by_memo += 1
        return True, hit

    def _remember_tier_hit(self, seq, sid: str, hit: int | None) -> None:
        budget = self._tier_retry_steps if hit is None else self._tier_memo_steps
        if self._tier_hit_memo is None:
            self._tier_hit_memo = {}
        self._tier_hit_memo[sid] = (
            self._weak_seq(seq),
            None if hit is None else int(hit),
            int(budget),
        )

    def _last_tier_hit(self, seq, sid: str) -> int | None:
        """The raw hit behind this step's answer -- before floor and caps.

        A subclass that has the last word on an answer (`_MPOffloadScheduler`
        refuses a whole-prompt hit that its block size cannot host) needs the
        length the tier reported, which the load spec no longer carries once
        `_loadable_hit` has floored it. Reading it back off the lookup client
        would only work on the steps that ran a lookup. Does not spend the
        memo's budget: this is a read of the answer already given.
        """

        entry = (self._tier_hit_memo or {}).get(sid)
        if entry is None or entry[0]() is not seq:
            return None
        return entry[1]

    def _forget_tier_hit(self, sid: str) -> None:
        if self._tier_hit_memo:
            self._tier_hit_memo.pop(sid, None)

    def _ensure_lookup_pin(self, seq, sid: str, spec) -> bool:
        """Re-take the worker-side lookup pin before a load is committed.

        An answer served from the remembered hit carries no pin: the metadata
        dispatch unpins every lookup that did not become a load
        (`lookup_requests_in_step` is the worker's *unpin* list). An answer does
        not need one -- the request is only waiting, and nothing reads the tier.
        A transfer does: it must not read an entry that nothing is holding. So
        the step that turns a load spec into a dispatched retrieve asks the tier
        for real, and the load stands only if the tier still holds at least what
        the spec promised. If it holds less -- evicted in the steps since -- the
        load is dropped and the request prefills, which is what would have
        happened with no tier at all.

        Confirming rather than re-deriving is deliberate: by this point the spec
        may have been shaped by something this mixin does not own (the unaligned
        handoff moves `hbm_cached_tokens` to a chunk boundary the prefill is
        walking towards), and rebuilding it here would quietly erase that.

        That is one lookup per committed load, which is what the tier cost was
        before this connector remembered anything; the repeats the livelock was
        made of are the steps that never get here.

        `_fresh_tier_lookup` belongs to the two concrete schedulers
        (`ChunkedOffloadSchedulerBase`, `DSV4OffloadScheduler`); this mixin only
        sequences it.
        """

        # `_lookup_results` is the dense scheduler's live-pin carrier. The DSV4
        # scheduler releases every pin at the end of the step it was taken in,
        # so it has none and always answers this with a fresh lookup.
        pending = getattr(self, "_lookup_results", {}).get(sid)
        if pending is not None and pending[0] is seq:
            return True
        if self._lookup_client is None:
            # Nothing to confirm: with no client no lookup ever ran, so this
            # spec did not come from one and holds no pin to re-take.
            return True
        hit = self._fresh_tier_lookup(seq, sid)
        if hit is None or int(hit) < int(spec.lmcache_cached_tokens):
            logger.debug(
                "[OFFLOAD-LOOKUP] seq=%s load dropped: tier now holds %s, "
                "spec promised %d",
                seq.id,
                hit,
                int(spec.lmcache_cached_tokens),
            )
            self._clear_pending_load(sid)
            return False
        return True

    def _repeat_load_suppressed(self, seq, sid: str) -> bool:
        """True once this request has spent its one external-tier attempt.

        Asking again repeats the same lookup against the same tier state, which
        is how a single failure becomes a permanent one: `load_failed` clears
        the pending load and the lookup memo, so the next scheduler pass hits,
        parks the request in WAITING_FOR_REMOTE_KVS again, and fails again --
        forever, holding the request's KV blocks and its concurrency slot the
        whole time. Prefilling normally instead is exactly what would have
        happened with no external tier at all.
        """

        if self._load_failed_seqs.get(sid) is seq:
            self.total_suppressed_load_retries += 1
            return True
        return False

    def _record_failed_load_attempt(self, sid: str) -> None:
        """Spend the attempt against the sequence that actually suffered it."""

        failed_seq = self._load_lifecycles.get(sid)
        if failed_seq is not None:
            self._load_failed_seqs[sid] = failed_seq

    def _release_failed_load_attempt(self, sid: str, seq) -> None:
        """Drop the mark when its sequence is done with the request ID."""

        if self._load_failed_seqs.get(sid) is seq:
            self._load_failed_seqs.pop(sid, None)

    def _lmcache_hit_save_floor(self, load_spec) -> int:
        if load_spec is None:
            return 0
        return self._chunk_floor(load_spec.lmcache_cached_tokens)

    def _set_save_frontier(self, sid: str, seq, saved: int) -> None:
        saved = self._chunk_floor(saved)
        if sid not in self._save_tracker:
            self._save_tracker[sid] = [seq, saved]
        else:
            self._save_tracker[sid][0] = seq
            self._save_tracker[sid][1] = saved

    def _maybe_start_unaligned_handoff(
        self,
        seq,
        load_spec,
        hbm: int,
        lmc: int,
        chunk: int,
    ) -> bool:
        boundary = ((hbm + chunk - 1) // chunk) * chunk
        remaining_after_boundary = lmc - boundary
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if boundary <= hbm or remaining_after_boundary < min_load:
            return False

        sid = str(seq.id)
        load_spec.hbm_cached_tokens = boundary
        load_spec.can_load = True
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.add(sid)
        seq.offload_loaded_tokens = hbm
        seq.offload_handoff_boundary_tokens = boundary
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF] seq=%s hbm_cached=%d boundary=%d "
            "lmc_cached=%d need_after_boundary=%d min_load=%d chunk=%d",
            seq.id,
            hbm,
            boundary,
            lmc,
            remaining_after_boundary,
            min_load,
            chunk,
        )
        return True

    def _claim_after_load(self, seq, hbm: int, lmc: int) -> int:
        """How far the request may call itself cached once the load lands.

        The transfer's end, for every layout whose claim and transfer share a
        boundary. A layout that aims them at different places overrides this --
        `kimi_k3` transfers the chunk *covering* its state boundary and may only
        claim the boundary, and claiming the rounded-up figure there would have
        the forward skip tokens the recurrent state does not cover.

        A seam rather than four inlined `max`es because getting it wrong is
        silent: the request simply starts further along than its state
        supports, and the output is wrong with no exception anywhere.
        """
        return max(int(hbm), int(lmc))

    def should_park_partial_prefill_for_load(self, seq) -> bool:
        if not self._do_load:
            return False
        sid = str(seq.id)
        if sid not in self._handoff_loads:
            return False
        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            self._handoff_loads.discard(sid)
            return False
        boundary = int(getattr(seq, "offload_handoff_boundary_tokens", 0) or 0)
        hbm = int(getattr(seq, "num_cached_tokens", 0))
        if boundary > 0 and hbm < boundary:
            return False

        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, load_spec
        )
        if not should_load:
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False

        load_spec.can_load = True
        self._reqs_need_recv[sid] = seq
        self._handoff_loads.discard(sid)
        seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
        logger.debug(
            "[OFFLOAD-LOAD-HANDOFF-READY] seq=%s hbm_cached=%d "
            "lmc_cached=%d offload_loaded=%d need=%d",
            seq.id,
            hbm,
            lmc,
            seq.offload_loaded_tokens,
            need,
        )
        return True

    def _mark_load_skip(
        self,
        seq,
        reason: str,
        hbm: int,
        lmc: int,
        need: int,
        chunk: int,
    ) -> None:
        seq.offload_loaded_tokens = hbm
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        logger.debug(
            "[OFFLOAD-LOAD-SKIP] seq=%s hbm_cached=%d lmc_cached=%d "
            "need=%d min_load=%d chunk=%d reason=%s",
            seq.id,
            hbm,
            lmc,
            need,
            min_load,
            chunk,
            reason,
        )

    def should_park_for_load_after_alloc(self, seq) -> bool:
        if not self._do_load:
            return False
        sid = str(seq.id)
        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            return False
        should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
            seq, load_spec
        )
        if not should_load:
            if (
                reason == "unaligned_hbm_prefill"
                and self._maybe_start_unaligned_handoff(seq, load_spec, hbm, lmc, chunk)
            ):
                return False
            self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
            self._clear_pending_load(sid)
            return False
        seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
        return True

    def _save_frontier(self, seq) -> int:
        computed = min(
            int(
                getattr(
                    seq,
                    "_offload_finished_cached_tokens",
                    getattr(seq, "num_cached_tokens", 0),
                )
            ),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        return self._chunk_floor(computed)

    def _has_pending_save(self, seq) -> bool:
        sid = str(seq.id)
        entry = self._save_tracker.get(sid)
        if entry is None:
            return False
        return self._save_frontier(seq) > int(entry[1])

    def _has_active_load(self, seq) -> bool:
        """Return whether this concrete request lifecycle still owns a load."""

        active = self._active_load_operations.get(str(seq.id))
        return active is not None and active[0] is seq


def max_pending_saves(kvc, save_workers: int) -> int:
    """Return the maximum running-plus-queued worker save operations."""

    extra = (kvc or {}).get("kv_connector_extra_config", kvc or {}) or {}
    configured = extra.get("max_pending_saves")
    if configured is None:
        configured = os.environ.get(
            "OFFLOAD_MAX_PENDING_SAVES",
            str(max(2, 2 * save_workers)),
        )
        try:
            capacity = int(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError("max pending saves must be a positive integer") from exc
    else:
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise ValueError("max pending saves must be a positive integer")
        capacity = configured
    if capacity <= 0:
        raise ValueError("max pending saves must be a positive integer")
    return capacity


# Seconds added on top of LMCache's own pin timeout before the engine reclaims a
# save that never reported. See `offload_save_abandon_timeout_s` for why the sum
# and not the timeout itself is the safe window.
_SAVE_ABANDON_MARGIN_S = 30.0

# What LMCache's pin monitor uses when `LMCACHE_EC_PIN_TIMEOUT_SEC` is unset.
_LMCACHE_PIN_TIMEOUT_DEFAULT_S = 300.0

# Memoised result of `offload_save_abandon_timeout_s`; the reconciler asks for it
# on a 1ms poll and the answer is a process constant.
_save_abandon_timeout_s: float | None = None


def offload_save_abandon_timeout_s() -> float:
    """Seconds a deferred offload save may sit before the engine reclaims it.

    Blocks are freed on `finished_saving`, so a lost report leaves them deferred
    forever: `has_pending_kv_work()` never clears and the engine busy-loops with
    every GPU idle.

    Reclaiming cannot race a live copy. `OffloadWorkerMixin._guard` reports on
    both the success and the exception path, so a report is lost only when
    `store()` neither returns nor raises -- it is parked inside LMCache. Then
    either the parked save is not copying (LMCache force-unpinned its source
    after `pin_timeout_sec`) or a save queued behind it never reached `store()`.
    Both cases are safe once that window has passed.

    Derived from LMCache's own `LMCACHE_EC_PIN_TIMEOUT_SEC` rather than a knob of
    its own, because that ordering IS the safety argument -- two independent env
    vars could be set the wrong way round with nothing to say so. This lives on
    the offload connector, not the scheduler: it is LMCache knowledge, and the
    scheduler now asks the connector for it (`save_abandon_timeout_s`).
    Non-positive disables reclamation.
    """
    global _save_abandon_timeout_s
    if _save_abandon_timeout_s is not None:
        return _save_abandon_timeout_s
    pin = _LMCACHE_PIN_TIMEOUT_DEFAULT_S
    raw = os.environ.get("LMCACHE_EC_PIN_TIMEOUT_SEC")
    if raw is not None:
        try:
            pin = float(raw)
        except ValueError:
            logger.warning(
                "invalid LMCACHE_EC_PIN_TIMEOUT_SEC=%r; assuming LMCache's %.0fs "
                "default for the offload save abandon window",
                raw,
                _LMCACHE_PIN_TIMEOUT_DEFAULT_S,
            )
    _save_abandon_timeout_s = pin + _SAVE_ABANDON_MARGIN_S if pin > 0 else 0.0
    return _save_abandon_timeout_s
