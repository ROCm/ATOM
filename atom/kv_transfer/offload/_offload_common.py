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
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadCompletionId,
    SaveCompletionId,
)
from atom.kv_transfer.offload import config as offcfg

logger = logging.getLogger("atom")
_VALID_KV_ROLES = {"offload", "kv_both", "kv_producer", "kv_consumer"}


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
        thread_name_prefix: str = "offload",
    ) -> None:
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        # Separate executors so a load (on the TTFT critical path) never queues
        # behind fire-and-forget saves. OFFLOAD_COPY_WORKERS tunes the save pool.
        n_save = (
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
            if save_workers is None
            else int(save_workers)
        )
        if n_save <= 0:
            raise ValueError("offload save worker count must be positive")
        self._save_executor = ThreadPoolExecutor(
            max_workers=n_save, thread_name_prefix=f"{thread_name_prefix}-save"
        )
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{thread_name_prefix}-load"
        )
        self._lock = threading.Lock()
        self._done_save: set[SaveCompletionId] = set()
        self._done_load: set[LoadCompletionId] = set()
        self._failed_load: set[LoadCompletionId] = set()

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
            else:
                # A failed save just loses this offload opportunity; still report
                # finished_saving so the scheduler releases any deferred free.
                with self._lock:
                    self._done_save.add(self._save_completion_id(req))

    def get_finished(self) -> KVConnectorOutput:
        with self._lock:
            dl, fl, ds = self._drain_common_completions_locked()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=dl,
            failed_loading=fl,
            finished_saving=ds,
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

    # Save/load lifecycle contract. Declared abstract so a missing forwarder is
    # a construction-time TypeError, not a silent no-op behind the delegating
    # shell -- the failure mode that let DSV4 ship without abandon_save. The
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
    def load_failed(self, req_id) -> bool: ...
    @abstractmethod
    def load_finished(self, req_id) -> bool: ...
    @abstractmethod
    def cancel_pending_load(self, seq) -> None: ...

    def _init_offload_statistics(self) -> None:
        """Initialize layout-independent scheduler counters."""

        self.total_load_requests = 0
        self.total_loaded_tokens = 0
        self.total_load_failures = 0
        self.total_save_requests = 0
        self.total_saved_tokens = 0
        self._load_inflight_tokens: dict[object, int] = {}
        self._save_inflight_tokens: dict[object, int] = {}

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
            if callback is None or callback(completion) is False:
                logger.warning(
                    "Ignoring unhandled offload completion channel %s",
                    completion.channel,
                )
                continue
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

        return {
            "load_requests": self.total_load_requests,
            "loaded_tokens": self.total_loaded_tokens,
            "load_failures": self.total_load_failures,
            "save_requests": self.total_save_requests,
            "saved_tokens": self.total_saved_tokens,
            "loads_pending": len(self._load_inflight_tokens),
            "saves_pending": len(self._save_inflight_tokens),
        }

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
            int(getattr(seq, "num_cached_tokens", 0)),
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
