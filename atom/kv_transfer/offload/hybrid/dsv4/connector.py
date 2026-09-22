# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 standalone LMCache CPU/NVMe KV-offload connector.

Design:

* **Use LMCache engine orchestration** — worker-side save/load calls
  ``CacheEngine.store()`` / ``CacheEngine.retrieve()`` so LMCache owns chunking,
  key generation, lookup pins, and storage-manager put/get.
* **ATOM-owned PAGE/SLOT layout** — LMCache owns token chunks and storage while
  :class:`DSV4PageSlotCodec` gathers/scatters DSV4 ``block_regions`` and full
  request SLOT regions through one layout codec.
* **Fail-closed DSV4 PAGE+SLOT** — stateful DSV4 stores token-chunked PAGE bytes
  through ``LMCacheEngine`` and one complete request SLOT as an AOS1 sidecar.
  The sidecar is published only after PAGE coverage reaches the same boundary.
* **Snapshot-before-forward, publish in background** — ``start_load_kv`` gathers
  each live Active SLOT into connector-owned staging on the caller's CUDA stream
  before ``forward`` is enqueued. Background workers wait for that snapshot and
  then publish it without reading the live SLOT again. Saves serialize per request
  while different requests can use the configured worker pool in parallel;
  completions are polled in ``get_finished``.
* **Cross-process hit lookup** — scheduler (EngineCore process) queries worker hits
  via LMCache's ZMQ ``LookupClient``/``LookupServer`` (no homegrown mirror).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import ceil, isfinite
from numbers import Integral

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadOperationId,
    ReqId,
    SaveCompletionId,
    SaveOperationId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import (
    OffloadSchedulerMixin,
    OffloadWorkerMixin,
    build_offload_engine,
    max_pending_saves,
    pp_aware_rank_and_world,
    tokens_to_tensor,
    validated_kv_role,
)
from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    HEADER_BYTES,
    DSV4CheckpointCodec,
    DSV4CheckpointCorruptionError,
    DSV4CheckpointError,
    DSV4CheckpointKey,
    DSV4CheckpointStore,
    DSV4PageSlotCodec,
)
from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    DSV4StagingAdmission,
    _BoundedLRUSet,
    _chained_prefix_hashes,
    _committed_sidecar_capacity,
    _compute_slot_fingerprint,
    build_dsv4_profile,
    select_pending_sidecar_boundary,
    sidecar_boundary_tokens,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
    SlotLoadSpec,
    SlotSaveSpec,
)
from atom.kv_transfer.offload.save_admission import (
    PrefixDemandKey,
    PrefixDemandTracker,
    SaveBlockReservation,
    load_save_admission_config,
)

logger = logging.getLogger("atom")

DSV4_CHECKPOINT_SAVE_CHANNEL = "atom.dsv4.checkpoint.save"
# PAGE saves need a terminal channel of their own because ``finished_saving``
# cannot express failure: it is one undifferentiated set, so a save rejected on
# admission or killed inside the executor reaches the scheduler looking exactly
# like one that landed. The scheduler advances its saved watermark when a save
# is *emitted*, so with no failure signal it can never learn to take that
# advance back, and the skipped range becomes a permanent hole in the prefix.
DSV4_PAGE_SAVE_CHANNEL = "atom.dsv4.page.save"


def _wait_for_publication(
    probe,
    *,
    timeout_s: float,
    poll_interval_s: float,
    clock=time.monotonic,
    sleep=time.sleep,
) -> bool:
    """Poll ``probe`` until publication is visible or the deadline expires."""

    timeout_s = float(timeout_s)
    poll_interval_s = float(poll_interval_s)
    if not isfinite(timeout_s):
        raise ValueError("publication timeout must be finite")
    if not isfinite(poll_interval_s):
        raise ValueError("publication poll interval must be finite")
    if timeout_s < 0:
        raise ValueError("publication timeout must be nonnegative")
    if poll_interval_s <= 0:
        raise ValueError("publication poll interval must be positive")

    deadline = clock() + timeout_s
    while True:
        if bool(probe()):
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(poll_interval_s, remaining))


def _env_nonnegative_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _env_positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class _SaveCandidate:
    sid: str
    seq: object
    generation: int
    saved: int
    computed: int
    aligned: int
    sidecar_candidate: tuple[int, int] | None
    observed_count: int
    reusable_tokens: int
    enqueued_at: float
    finished: bool
    held_blocks: int

    @property
    def dirty_tokens(self) -> int:
        return max(0, self.aligned - self.saved)

    @property
    def page_save_due(self) -> bool:
        return self.aligned > self.saved


@dataclass(frozen=True)
class _SlotSaveSnapshot:
    """RPC-thread result carried to one background sidecar save."""

    staging_id: int | None
    ready_event: object | None
    snapshot_ok: bool
    source_completion_uncertain: bool = False


class _SlotStagingSyncError(RuntimeError):
    """GPU completion was not confirmed, so the staging row is unsafe."""


class _SlotLoadBatchReservation:
    """One staging row shared by serial SLOT loads in a worker batch."""

    def __init__(self, staging_id: int, users: int) -> None:
        self.staging_id = staging_id
        self._remaining = users
        self._reusable = True
        self._lock = threading.Lock()

    @property
    def reusable(self) -> bool:
        with self._lock:
            return self._reusable

    def finish(self, reusable: bool) -> str | None:
        """Retire one user and return the last-user cleanup action."""
        with self._lock:
            if self._remaining <= 0:
                raise RuntimeError("SLOT load batch reservation already retired")
            self._reusable = self._reusable and reusable
            self._remaining -= 1
            if self._remaining:
                return None
            return "release" if self._reusable else "quarantine"


# =====================================================================
# Worker side
# =====================================================================
class DSV4OffloadConnector(OffloadWorkerMixin, KVConnectorBase):
    # Offload is a *consumer* from the scheduler's POV (it loads KV back). Saves
    # are fire-and-forget on the worker and must NOT be reported as
    # finished_sending (the scheduler frees blocks on finished_sending — a P/D
    # producer semantic that would wrongly deallocate live offload blocks).
    is_producer = False

    def __init__(self, config) -> None:
        self._config = config
        raw_block_size = config.kv_cache_block_size
        if isinstance(raw_block_size, bool) or not isinstance(raw_block_size, Integral):
            # Preserve the public configuration error contract.
            raise ValueError("DSV4 block size must be an integer")  # noqa: TRY004
        self.block_size = int(raw_block_size)
        self.virtual_block_size: int | None = None
        self.profile = None
        self.chunk_size: int | None = None
        self._publication_timeout_s = _env_nonnegative_float(
            "OFFLOAD_PUBLICATION_TIMEOUT_S",
            5.0,
        )
        self._publication_poll_interval_s = _env_positive_float(
            "OFFLOAD_PUBLICATION_POLL_INTERVAL_S",
            0.01,
        )
        self._publication_clock = time.monotonic
        self._publication_sleep = time.sleep

        # Copy daemons: keep GPU<->host copies off the RPC thread. SEPARATE
        # executors for LOAD vs SAVE so a load (on the TTFT critical path — a
        # parked seq is waiting for it) never queues behind a backlog of fire-
        # and-forget saves (Phase 4 root cause: with one shared serial daemon, a
        # reload sat behind ~N filler saves -> request hung well past timeout).
        # The ATOM LMCache GPU connector owns per-thread staging streams.
        # OFFLOAD_COPY_WORKERS tunes the SAVE pool only.
        n_save_workers = int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
        # The SLOT load path is *not* thread-safe against itself: a worker batch
        # shares one staging row across its loads (`_SlotLoadBatchReservation`)
        # precisely because the load executor runs them in submission order. A
        # second load thread would run two loads through the same staging row
        # concurrently and corrupt both. OFFLOAD_LOAD_WORKERS therefore does not
        # apply to DSV4 -- honour it loudly rather than silently.
        n_load_env = int(os.environ.get("OFFLOAD_LOAD_WORKERS", "1"))
        if n_load_env != 1:
            logger.warning(
                "ATOM DSV4 offload: ignoring OFFLOAD_LOAD_WORKERS=%d; the SLOT "
                "load path shares one staging row per worker batch and requires "
                "a serial load executor",
                n_load_env,
            )
        self._max_pending_saves = max_pending_saves(kvc, n_save_workers)
        self._save_admission = threading.BoundedSemaphore(self._max_pending_saves)
        # Terminal PAGE-save outcomes, drained by ``get_finished`` onto
        # ``DSV4_PAGE_SAVE_CHANNEL``. Every request carrying a ``save_spec``
        # lands in exactly one of these, on every path out of the save, so the
        # TP aggregator always reaches a verdict instead of stranding the key.
        self._done_page_save: set = set()
        self._failed_page_save: set = set()
        self._init_worker_common(
            config,
            save_workers=n_save_workers,
            load_workers=1,
            thread_name_prefix="lmc-offload",
        )
        # Kept separate from _done_save: PAGE completion releases deferred
        # blocks, while SLOT completion controls scheduler-side boundary commit.
        self._done_sidecar_save: set[SaveCompletionId] = set()
        self._failed_sidecar_save: set[SaveCompletionId] = set()
        self._pending_save_ops: dict[ReqId, int] = {}
        self._pending_legacy_save_ops: dict[ReqId, int] = {}
        self._save_req_locks: dict[ReqId, threading.Lock] = {}

        self._engine = None
        self._codec: DSV4PageSlotCodec | None = None
        self._lookup_server = None
        self._checkpoint_codec: DSV4CheckpointCodec | None = None
        self._slot_store: DSV4CheckpointStore | None = None
        self._slot_staging: torch.Tensor | None = None
        self._slot_admission: DSV4StagingAdmission | None = None

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        self._rank = rank
        cfg = offcfg.build_lmcache_config(
            getattr(self._config, "kv_transfer_config", None)
        )
        self.profile = build_dsv4_profile(
            self._config,
            chunk_size=int(cfg.chunk_size),
        )
        self.block_size = self.profile.block_size
        self.virtual_block_size = self.profile.hash_block_size

        # num_blocks is the scheduler-visible block count, threaded from the
        # model runner. MLA stores its KV token-major, so the codec cannot infer
        # this count from tensor.shape[0] (the page-size-1 physical row count).
        block_regions = getattr(transfer_tensors, "block_regions", None)
        stateful_page = False
        if not kv_caches and block_regions:
            page_num_blocks = (
                num_blocks
                if num_blocks is not None
                else getattr(transfer_tensors, "num_blocks", None)
            )
            if page_num_blocks is None:
                raise ValueError(
                    "LMCache offload PAGE regions require a num_blocks value"
                )
            # Reject a half-described V4 PAGE/SLOT setup before creating or
            # starting an LMCache engine.
            stateful_page = self._validate_stateful_page_slot_geometry(transfer_tensors)
            slot_regions = (
                getattr(transfer_tensors, "swa_block_regions", None) or ()
                if stateful_page
                else ()
            )
            slot_count = int(getattr(transfer_tensors, "num_slots", 0) or 0)
            self._codec = DSV4PageSlotCodec(
                page_regions=block_regions,
                slot_regions=slot_regions,
                num_blocks=page_num_blocks,
                num_slots=slot_count,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            if not self._codec.has_fused_chunk_major_staging:
                raise RuntimeError(
                    "DSV4 PAGE/SLOT offload requires Triton fused staging at "
                    "worker startup"
                )
        else:
            raise ValueError(
                "hybrid PAGE+SLOT offload requires empty kv_caches and "
                "transfer_tensors.block_regions; use offload_layout='dense' "
                "for ordinary MHA/MLA KV caches"
            )
        self._engine, cfg, meta = build_offload_engine(
            self._config,
            engine_id=f"{offcfg.lmcache_engine_id(self._config)}-{rank}",
            block_size=self.virtual_block_size,
            bytes_per_block=self._codec.bytes_per_block,
            gpu_connector_factory=lambda cfg, _meta: BlockGPUConnector(
                self._codec,
                self.block_size,
                chunk_size=int(cfg.chunk_size),
                virtual_block_size=self.virtual_block_size,
            ),
            world=world,
            rank=rank,
            cfg=cfg,
        )
        self.chunk_size = int(cfg.chunk_size)
        base_meta = meta._atom_base_metadata
        gpu_connector = self._engine.gpu_connector
        self._validate_and_log_storage_backends(cfg)
        if stateful_page:
            self._initialize_slot_sidecar(
                transfer_tensors,
                model_name=str(
                    getattr(
                        base_meta,
                        "model_name",
                        getattr(self._config, "model", "atom-model"),
                    )
                ),
                tp_size=world,
                tp_rank=rank,
                geometry_validated=True,
            )
            self._require_slot_components()

        # ZMQ lookup server so the scheduler process can query our hit counts.
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_server = LookupClientFactory.create_lookup_server(
                self._engine, meta
            )
        except Exception as e:  # noqa: BLE001  # optional third-party service
            logger.warning("LMCache offload: lookup server not started: %s", e)

        logger.info(
            "LMCache offload worker rank=%d: bytes_per_block=%d chunk=%d "
            "gpu_staging_chunk_bytes=%d gpu_staging_buffer_chunks=%d "
            "gpu_staging_buffer_bytes=%d release_gpu_staging=%s "
            "save=%s load=%s",
            rank,
            self._codec.bytes_per_block,
            self.chunk_size,
            gpu_connector.gpu_staging_chunk_bytes,
            gpu_connector.gpu_staging_buffer_chunks,
            gpu_connector.gpu_staging_buffer_bytes,
            gpu_connector.release_gpu_staging_after_transfer,
            self._do_save,
            self._do_load,
        )

    def _validate_stateful_page_slot_geometry(self, transfer_tensors) -> bool:
        """Preflight a stateful PAGE's full SLOT metadata before engine start."""
        regions = getattr(transfer_tensors, "swa_block_regions", None) or []
        slot_markers = getattr(transfer_tensors, "slot_regions", None) or []
        num_slots = getattr(transfer_tensors, "num_slots", 0)
        expected_count = getattr(
            transfer_tensors,
            "expected_full_slot_region_count",
            None,
        )
        stateful_page = bool(
            regions or slot_markers or num_slots or expected_count is not None
        )
        if not stateful_page:
            return False
        if not regions:
            raise ValueError(
                "Stateful PAGE offload requires full per-request SLOT regions "
                "in swa_block_regions (legacy field name)"
            )
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, Integral)
            or expected_count <= 0
        ):
            raise ValueError(
                "Stateful PAGE offload expected_full_slot_region_count must be "
                "a positive integer"
            )
        expected_count = int(expected_count)
        if len(regions) != expected_count:
            raise ValueError(
                f"Stateful PAGE offload expected {expected_count} full per-request "
                f"SLOT regions, got {len(regions)}"
            )
        if isinstance(num_slots, bool) or not isinstance(num_slots, Integral):
            # Preserve the public configuration error contract.
            raise ValueError(  # noqa: TRY004
                "Stateful PAGE offload num_slots must be an integer"
            )
        num_slots = int(num_slots)
        if num_slots <= 0:
            raise ValueError(
                f"Stateful PAGE offload num_slots must be > 0, got {num_slots}"
            )

        # Per-region address, byte geometry, and reverse-index validation is
        # owned by the immutable DSV4PageSlotCodec snapshot constructed next.
        return True

    def _slot_staging_slots(self) -> int:
        kvc = getattr(self._config, "kv_transfer_config", {}) or {}
        extra = kvc.get("kv_connector_extra_config", kvc) or {}
        configured = extra.get("slot_sidecar_staging_slots")
        if configured is None:
            configured = os.environ.get("OFFLOAD_SLOT_STAGING_SLOTS", "1")
            try:
                count = int(configured)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SLOT sidecar staging count must be an integer"
                ) from exc
        else:
            if isinstance(configured, bool) or not isinstance(configured, Integral):
                raise ValueError("SLOT sidecar staging count must be an integer")
            count = int(configured)
        if count <= 0:
            raise ValueError(f"SLOT sidecar staging count must be > 0, got {count}")
        return count

    def _initialize_slot_sidecar(
        self,
        transfer_tensors,
        *,
        model_name: str,
        tp_size: int,
        tp_rank: int,
        geometry_validated: bool = False,
    ) -> None:
        """Build the stateful V4 SLOT runtime after the PAGE engine exists."""
        if self._engine is None:
            raise RuntimeError("SLOT sidecar initialization requires an LMCache engine")
        if not geometry_validated and not self._validate_stateful_page_slot_geometry(
            transfer_tensors
        ):
            raise ValueError("SLOT sidecar initialization requires a stateful PAGE")
        num_slots = getattr(transfer_tensors, "num_slots", None)
        codec = self._codec
        if not isinstance(codec, DSV4PageSlotCodec) or codec.slot_bytes <= 0:
            raise RuntimeError(
                "SLOT sidecar initialization requires the unified DSV4 PAGE/SLOT codec"
            )

        staging_slots = self._slot_staging_slots()
        staging = torch.empty(
            (staging_slots, codec.slot_bytes),
            dtype=torch.uint8,
            device=codec.device,
        )
        hf_config = getattr(self._config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None) or []
        model_tag = getattr(self._config, "model_tag", None) or model_name
        fingerprint = _compute_slot_fingerprint(
            model_tag=str(model_tag),
            page_namespace=model_name,
            kv_dtype=str(getattr(self._config, "kv_cache_dtype", "auto")),
            compress_ratios=compress_ratios,
            block_size=self.block_size,
            kv_head_dim=self.profile.kv_head_dim,
            index_head_dim=self.profile.index_head_dim,
            num_slots=num_slots,
            slot_regions=codec.slot_regions,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        checkpoint_codec = DSV4CheckpointCodec(
            fingerprint=fingerprint,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        admission = DSV4StagingAdmission(staging_slots)
        store = DSV4CheckpointStore(
            self._engine,
            checkpoint_codec=checkpoint_codec,
            model_name=model_name,
        )

        # Publish only a complete runtime. A partially initialized stateful
        # PAGE connector must fail startup rather than silently drop SLOT data.
        self._checkpoint_codec = checkpoint_codec
        self._slot_staging = staging
        self._slot_admission = admission
        self._slot_store = store
        logger.info(
            "LMCache offload PAGE+SLOT registered rank=%d "
            "page_bytes_per_block=%d slot_bytes=%d slot_staging_slots=%d "
            "fingerprint=%s",
            tp_rank,
            int(self._codec.bytes_per_block),
            int(codec.slot_bytes),
            staging_slots,
            fingerprint.hex()[:12],
        )

    def _require_slot_components(self):
        codec = getattr(self, "_codec", None)
        store = getattr(self, "_slot_store", None)
        admission = getattr(self, "_slot_admission", None)
        missing = [
            name
            for name, component in (
                ("codec", codec),
                ("store", store),
                ("admission", admission),
            )
            if component is None
        ]
        if missing:
            raise RuntimeError(
                "Stateful PAGE offload has incomplete SLOT components: "
                + ", ".join(missing)
            )
        return codec, store, admission

    @staticmethod
    def _slot_payload_bytes(slot_codec) -> int:
        """Return the unified codec's SLOT width."""

        value = getattr(slot_codec, "slot_bytes", None)
        if value is None or int(value) <= 0:
            raise RuntimeError("DSV4 codec does not expose a positive SLOT width")
        return int(value)

    def _slot_staging_view(self, staging_id: int) -> torch.Tensor:
        """Return one connector-owned SLOT temp row."""

        staging = getattr(self, "_slot_staging", None)
        if staging is None:
            raise RuntimeError("connector-owned SLOT staging is unavailable")
        if isinstance(staging_id, bool) or not isinstance(staging_id, Integral):
            raise ValueError("SLOT staging id must be an integer")  # noqa: TRY004
        staging_id = int(staging_id)
        if not 0 <= staging_id < int(staging.shape[0]):
            raise ValueError(
                f"SLOT staging id {staging_id} outside pool "
                f"[0, {int(staging.shape[0])})"
            )
        return staging[staging_id]

    def _require_checkpoint_codec(self) -> DSV4CheckpointCodec:
        checkpoint_codec = getattr(self, "_checkpoint_codec", None)
        if not isinstance(checkpoint_codec, DSV4CheckpointCodec):
            # This is a connector initialization invariant, not input validation.
            raise RuntimeError("DSV4 checkpoint codec is unavailable")  # noqa: TRY004
        return checkpoint_codec

    def _validate_and_log_storage_backends(self, cfg) -> None:
        """Report the realized LMCache tier topology and validate NVMe startup."""
        storage_manager = getattr(self._engine, "storage_manager", None)
        backend_names: list[str] = []
        if storage_manager is not None:
            list_backends = getattr(storage_manager, "list_backends", None)
            if callable(list_backends):
                backend_names = sorted(str(name) for name in list_backends())
            else:
                storage_backends = getattr(storage_manager, "storage_backends", {})
                backend_names = sorted(str(name) for name in storage_backends)

        local_disk = getattr(cfg, "local_disk", None)
        disk_size_gib = float(getattr(cfg, "max_local_disk_size", 0.0) or 0.0)
        disk_configured = bool(local_disk) and disk_size_gib > 0
        if disk_configured and "LocalDiskBackend" not in backend_names:
            raise RuntimeError(
                "LMCache local-disk offload was configured but LocalDiskBackend "
                f"was not created on rank {self._rank}; backends={backend_names}"
            )

        logger.info(
            "LMCache offload worker rank=%d storage: backends=%s "
            "local_cpu=%s max_local_cpu_gib=%s local_disk=%s "
            "max_local_disk_gib=%s store_location=%s retrieve_locations=%s",
            self._rank,
            backend_names,
            getattr(cfg, "local_cpu", None),
            getattr(cfg, "max_local_cpu_size", None),
            local_disk,
            getattr(cfg, "max_local_disk_size", None),
            getattr(cfg, "store_location", None),
            getattr(cfg, "retrieve_locations", None),
        )

    # -- per-step (RPC thread): enqueue PAGE work, snapshot SLOT ----------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        load_requests = [
            req
            for req in metadata.requests
            if (
                getattr(req, "load_spec", None) is not None
                or getattr(req, "slot_load_spec", None) is not None
            )
            and self._do_load
        ]
        loading_lookup_ids = {str(req.req_id) for req in load_requests}
        for lookup_id in metadata.lookup_requests_in_step:
            if str(lookup_id) not in loading_lookup_ids:
                self._lookup_unpin(lookup_id)

        # The load executor is serial, so one row can serve every SLOT load in
        # submission order. Keep it reserved across the batch so saves cannot
        # steal it, and release it only after the last terminal load.
        slot_loads = [
            req
            for req in load_requests
            if getattr(req, "slot_load_spec", None) is not None
        ]
        load_reservation = None
        if slot_loads:
            staging_id = None
            try:
                _, _, admission = self._require_slot_components()
                staging_id = admission.try_acquire()
            except Exception:
                logger.debug(
                    "LMCache offload: SLOT load staging reservation failed",
                    exc_info=True,
                )
            if staging_id is None:
                for req in slot_loads:
                    self._finish_rejected_load(req, None)
            else:
                load_reservation = _SlotLoadBatchReservation(
                    staging_id,
                    len(slot_loads),
                )

        reserved_loads: list[
            tuple[LMCacheReqMeta, _SlotLoadBatchReservation | None]
        ] = []
        for req in load_requests:
            if (
                getattr(req, "slot_load_spec", None) is not None
                and load_reservation is None
            ):
                continue
            reservation = (
                load_reservation
                if getattr(req, "slot_load_spec", None) is not None
                else None
            )
            reserved_loads.append((req, reservation))

        for req, reservation in reserved_loads:
            try:
                self._load_executor.submit(
                    self._guard,
                    "load",
                    self._do_load_req,
                    req,
                    reservation,
                )
            except Exception:
                logger.exception(
                    "LMCache offload: failed to submit load req=%s",
                    req.req_id,
                )
                self._finish_rejected_load(req, reservation)

        save_ready_event = None
        for req in metadata.requests:
            if (
                getattr(req, "save_spec", None) is not None
                or getattr(req, "slot_save_spec", None) is not None
            ) and self._do_save:
                if not self._save_admission.acquire(blocking=False):
                    self._finish_unadmitted_save(req)
                    continue
                req_lock = self._begin_save_operation(
                    req.req_id,
                    getattr(req, "save_operation", None),
                )
                try:
                    # Metadata is dispatched before this batch's forward. Copy
                    # the live Active SLOT into connector-owned staging on the
                    # current stream, then let the forward mutate the shared
                    # PAGE/SLOT backing only after the snapshot has been issued.
                    slot_snapshot = self._prepare_slot_save(req)
                except Exception:
                    logger.exception(
                        "LMCache offload: failed to prepare save req=%s",
                        req.req_id,
                    )
                    self._finish_rejected_save(req, None, None)
                    continue
                producer_event = (
                    slot_snapshot.ready_event
                    if slot_snapshot is not None
                    and slot_snapshot.ready_event is not None
                    else None
                )
                if (
                    producer_event is None
                    and getattr(req, "save_spec", None) is not None
                ):
                    if save_ready_event is None:
                        # PAGE-only and SLOT-admission-failure saves still need
                        # the forward producer fence before asynchronous store.
                        try:
                            candidate_event = torch.cuda.Event()
                            candidate_event.record(torch.cuda.current_stream())
                            save_ready_event = candidate_event
                        except Exception:
                            logger.exception(
                                "LMCache offload: PAGE save fence creation failed "
                                "req=%s",
                                req.req_id,
                            )
                            self._finish_rejected_save(
                                req,
                                None,
                                slot_snapshot,
                            )
                            continue
                    producer_event = save_ready_event
                try:
                    self._save_executor.submit(
                        self._run_save_req,
                        req,
                        producer_event,
                        slot_snapshot,
                        req_lock,
                        True,
                    )
                except Exception:
                    logger.exception(
                        "LMCache offload: failed to submit save req=%s",
                        req.req_id,
                    )
                    self._finish_rejected_save(
                        req,
                        producer_event,
                        slot_snapshot,
                    )

    def _report_page_save_locked(self, req: LMCacheReqMeta, *, succeeded: bool) -> None:
        """Record one request's terminal PAGE outcome. Caller holds ``_lock``.

        Only a request that carries a ``save_spec`` moved the scheduler's
        watermark, so only that one needs a verdict -- a SLOT-only save has
        nothing to roll back.
        """
        if getattr(req, "save_spec", None) is None:
            return
        completion_id = self._save_completion_id(req)
        if succeeded:
            self._done_page_save.add(completion_id)
        else:
            self._failed_page_save.add(completion_id)

    def _report_page_save(self, req: LMCacheReqMeta, *, succeeded: bool) -> None:
        """Lock-taking form of :meth:`_report_page_save_locked`.

        ``_lock`` is a plain ``Lock``, so this must never be called while it is
        already held; the in-executor path reports through the ``_locked`` form
        inside the section it already owns.
        """
        with self._lock:
            self._report_page_save_locked(req, succeeded=succeeded)

    def _finish_unadmitted_save(
        self,
        req: LMCacheReqMeta,
        *,
        reason: str = "max_pending_saves",
    ) -> None:
        """Terminally reject a save before retaining any operation state."""

        self._report_page_save(req, succeeded=False)
        completion_id = self._save_completion_id(req)
        with self._lock:
            save_operation = getattr(req, "save_operation", None)
            if (
                save_operation is not None
                or self._pending_legacy_save_ops.get(req.req_id, 0) == 0
            ):
                self._done_save.add(completion_id)
            if getattr(req, "slot_save_spec", None) is not None:
                self._failed_sidecar_save.add(completion_id)
        logger.warning(
            "LMCache offload: save rejected req=%s reason=%s capacity=%d",
            req.req_id,
            reason,
            self._max_pending_saves,
        )

    def _release_slot_staging(
        self,
        req_id,
        staging_id: int | None,
        *,
        operation: str,
    ) -> bool:
        if staging_id is None:
            return True
        try:
            self._slot_admission.release(staging_id)
            return True
        except Exception:
            logger.exception(
                "LMCache offload: SLOT %s release failed req=%s",
                operation,
                req_id,
            )
            return False

    def _quarantine_slot_staging(
        self,
        req_id,
        staging_id: int,
        *,
        operation: str,
    ) -> bool:
        try:
            self._slot_admission.quarantine(staging_id)
            logger.error(
                "LMCache offload: quarantined SLOT %s staging id=%s req=%s",
                operation,
                staging_id,
                req_id,
            )
            return True
        except Exception:
            logger.exception(
                "LMCache offload: SLOT %s quarantine failed req=%s",
                operation,
                req_id,
            )
            return False

    def _finish_slot_load_reservation(
        self,
        req_id,
        reservation: _SlotLoadBatchReservation | None,
        *,
        reusable: bool,
    ) -> bool:
        if reservation is None:
            return reusable
        try:
            action = reservation.finish(reusable)
        except Exception:
            logger.exception(
                "LMCache offload: SLOT load reservation accounting failed req=%s",
                req_id,
            )
            return False
        if action == "release":
            return self._release_slot_staging(
                req_id,
                reservation.staging_id,
                operation="load batch",
            )
        if action == "quarantine":
            self._quarantine_slot_staging(
                req_id,
                reservation.staging_id,
                operation="load batch",
            )
            return False
        return reusable

    def _finish_rejected_load(
        self,
        req: LMCacheReqMeta,
        reservation: _SlotLoadBatchReservation | None,
    ) -> None:
        self._finish_slot_load_reservation(
            req.req_id,
            reservation,
            reusable=True,
        )
        self._lookup_unpin(req.req_id)
        self._complete_load(req, succeeded=False)
        slot_spec = getattr(req, "slot_load_spec", None)
        if slot_spec is not None:
            logger.warning(
                "LMCache offload: SLOT sidecar load failed rank=%s "
                "req=%s boundary=%d reason=staging_or_submission "
                "error_type=none",
                getattr(self, "_rank", "?"),
                req.req_id,
                slot_spec.boundary_tokens,
            )

    def _finish_rejected_save(
        self,
        req: LMCacheReqMeta,
        producer_event,
        slot_snapshot: _SlotSaveSnapshot | None,
    ) -> None:
        """Safely retire an RPC snapshot when executor submission fails."""
        self._report_page_save(req, succeeded=False)
        try:
            gpu_complete = True
            try:
                if producer_event is not None:
                    producer_event.synchronize()
            except Exception:
                gpu_complete = False
                logger.exception(
                    "LMCache offload: rejected save fence failed req=%s",
                    req.req_id,
                )
            if slot_snapshot is not None and slot_snapshot.staging_id is not None:
                if gpu_complete:
                    self._release_slot_staging(
                        req.req_id,
                        slot_snapshot.staging_id,
                        operation="rejected save",
                    )
                else:
                    self._quarantine_slot_staging(
                        req.req_id,
                        slot_snapshot.staging_id,
                        operation="rejected save",
                    )
            with self._lock:
                if getattr(req, "slot_save_spec", None) is not None:
                    self._failed_sidecar_save.add(self._save_completion_id(req))
            slot_spec = getattr(req, "slot_save_spec", None)
            if slot_spec is not None:
                logger.warning(
                    "LMCache offload: SLOT sidecar save failed rank=%s "
                    "req=%s boundary=%d reason=executor_rejected "
                    "error_type=none",
                    getattr(self, "_rank", "?"),
                    req.req_id,
                    slot_spec.boundary_tokens,
                )
        finally:
            self._finish_save_operation(
                req.req_id,
                getattr(req, "save_operation", None),
            )
            self._save_admission.release()

    def _prepare_slot_save(
        self,
        req: LMCacheReqMeta,
    ) -> _SlotSaveSnapshot | None:
        """Issue the full-slot D2D snapshot on the RPC current stream."""
        spec = getattr(req, "slot_save_spec", None)
        if spec is None:
            return None
        staging_id = self._reserve_slot_staging()
        if staging_id is None:
            return _SlotSaveSnapshot(None, None, False)
        return self._snapshot_reserved_slot_save(
            req,
            source_group=int(spec.source_group),
            staging_id=staging_id,
        )

    def _reserve_slot_staging(self) -> int | None:
        """Reserve one connector-owned row without issuing a GPU copy yet."""

        try:
            _, _, admission = self._require_slot_components()
            return admission.try_acquire()
        except Exception:  # noqa: BLE001
            return None

    def _snapshot_reserved_slot_save(
        self,
        req: LMCacheReqMeta,
        *,
        source_group: int,
        staging_id: int,
    ) -> _SlotSaveSnapshot:
        """Gather ``source_group`` into an already reserved staging row."""

        try:
            slot_codec, _, admission = self._require_slot_components()
        except Exception:  # noqa: BLE001
            return _SlotSaveSnapshot(staging_id, None, False)

        stream = None
        ready_event = None
        try:
            stream = torch.cuda.current_stream()
            ready_event = torch.cuda.Event()
            row = self._slot_staging_view(staging_id)
            slot_codec.gather_slot(source_group, row, stream=stream)
            ready_event.record(stream)
            return _SlotSaveSnapshot(staging_id, ready_event, True)
        except Exception:  # noqa: BLE001
            # A copy kernel may have been issued before the exception. If an
            # event can still be recorded, let the background finalizer wait it
            # before returning the row to admission.
            if stream is not None and ready_event is not None:
                try:
                    ready_event.record(stream)
                    return _SlotSaveSnapshot(staging_id, ready_event, False)
                except Exception:
                    logger.debug(
                        "LMCache offload: failed to record SLOT cleanup event",
                        exc_info=True,
                    )
            cleanup_complete = True
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception:
                    cleanup_complete = False
                    logger.exception(
                        "LMCache offload: SLOT snapshot cleanup sync failed req=%s",
                        req.req_id,
                    )
            if cleanup_complete:
                try:
                    admission.release(staging_id)
                except Exception:
                    logger.exception(
                        "LMCache offload: SLOT snapshot cleanup release failed req=%s",
                        req.req_id,
                    )
            else:
                self._quarantine_slot_staging(
                    req.req_id,
                    staging_id,
                    operation="snapshot cleanup",
                )
            return _SlotSaveSnapshot(
                None,
                None,
                False,
                source_completion_uncertain=not cleanup_complete,
            )

    def _guard(self, kind: str, fn, req, *args) -> None:
        try:
            fn(req, *args)
        except Exception:
            logger.exception(
                "LMCache offload: %s failed for %s", fn.__name__, req.req_id
            )
            with self._lock:
                if kind == "load":
                    self._complete_load_locked(req, succeeded=False)
                elif getattr(req, "slot_save_spec", None) is not None:
                    self._failed_sidecar_save.add(self._save_completion_id(req))
                if kind == "save":
                    # Backstop. ``_do_save_req`` normally reports its own PAGE
                    # verdict and swallows its errors, so this only fires if it
                    # died outside that handling -- but a save that reports
                    # nothing would sit in the TP aggregator forever, so the
                    # failure has to be recorded here too. A duplicate report is
                    # harmless: the aggregator is failure-dominant per worker.
                    self._report_page_save_locked(req, succeeded=False)

    def _complete_load_locked(self, req: LMCacheReqMeta, *, succeeded: bool) -> None:
        completion_id = self._load_completion_id(req)
        if succeeded and completion_id not in self._failed_load:
            self._done_load.add(completion_id)
            return
        self._done_load.discard(completion_id)
        self._failed_load.add(completion_id)

    def _complete_load(self, req: LMCacheReqMeta, *, succeeded: bool) -> None:
        with self._lock:
            self._complete_load_locked(req, succeeded=succeeded)

    def _begin_save_operation(
        self,
        req_id: ReqId,
        save_operation: SaveOperationId | None = None,
    ) -> threading.Lock:
        completion_id = save_operation or req_id
        with self._lock:
            self._done_save.discard(completion_id)
            self._pending_save_ops[req_id] = self._pending_save_ops.get(req_id, 0) + 1
            if save_operation is None:
                self._pending_legacy_save_ops[req_id] = (
                    self._pending_legacy_save_ops.get(req_id, 0) + 1
                )
            return self._save_req_locks.setdefault(req_id, threading.Lock())

    def _finish_save_operation(
        self,
        req_id: ReqId,
        save_operation: SaveOperationId | None = None,
    ) -> None:
        with self._lock:
            pending = self._pending_save_ops.get(req_id, 0)
            if pending <= 0:
                logger.error(
                    "LMCache offload: duplicate save completion req=%s",
                    req_id,
                )
                return
            if pending > 1:
                self._pending_save_ops[req_id] = pending - 1
            else:
                self._pending_save_ops.pop(req_id, None)
                self._save_req_locks.pop(req_id, None)

            if save_operation is not None:
                self._done_save.add(save_operation)
                return

            legacy_pending = self._pending_legacy_save_ops.get(req_id, 0)
            if legacy_pending > 1:
                self._pending_legacy_save_ops[req_id] = legacy_pending - 1
            else:
                self._pending_legacy_save_ops.pop(req_id, None)
                self._done_save.add(req_id)

    def _run_save_req(
        self,
        req: LMCacheReqMeta,
        producer_event,
        slot_snapshot: _SlotSaveSnapshot | None,
        req_lock: threading.Lock,
        admission_owned: bool = False,
    ) -> None:
        try:
            with req_lock:
                self._guard(
                    "save",
                    self._do_save_req,
                    req,
                    producer_event,
                    slot_snapshot,
                )
        finally:
            try:
                self._finish_save_operation(
                    req.req_id,
                    getattr(req, "save_operation", None),
                )
            finally:
                if admission_owned:
                    self._save_admission.release()

    # -- copy daemon thread ----------------------------------------------
    def _load_page(self, req: LMCacheReqMeta) -> bool:
        ls = getattr(req, "load_spec", None)
        if ls is None:
            return True
        hbm = int(ls.hbm_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        toks = req.token_ids[:lmc]
        t_total0 = time.perf_counter()
        if lmc <= hbm:
            return True
        chunk_size = int(self.chunk_size or 256)
        if hbm % chunk_size != 0:
            logger.warning(
                "LMCache offload: HBM prefix is not chunk-aligned req=%s "
                "hbm=%d chunk=%d; re-prefill",
                req.req_id,
                hbm,
                chunk_size,
            )
            return False

        mask = torch.ones(len(toks), dtype=torch.bool)
        mask[:hbm] = False

        t_retrieve0 = time.perf_counter()
        self._reset_gpu_connector_transfer_stats()
        ret_mask = self._engine.retrieve(
            tokens_to_tensor(toks),
            mask=mask,
            block_ids=req.block_ids,
            req_id=str(req.req_id),
        )
        retrieve_ms = (time.perf_counter() - t_retrieve0) * 1000
        transfer_stats = self._last_gpu_connector_transfer_stats()
        loaded = bool(ret_mask[hbm:lmc].all().item())
        total_ms = (time.perf_counter() - t_total0) * 1000
        if self._profile_enabled():
            logger.info(
                "[OFFLOAD-LOAD-PROF] rank=%s req=%s hbm=%d lmc=%d "
                "retrieved=%d status=%s chunks=%d groups=%d "
                "max_chunk_bytes=%d max_group_bytes=%d "
                "gpu_staging_chunk_bytes=%d gpu_staging_buffer_chunks=%d "
                "gpu_staging_buffer_bytes=%d total_bytes=%d "
                "pack_ms=%.2f copy_ms=%.2f sync_ms=%.2f "
                "transfer_ms=%.2f effective_gbps=%.2f "
                "retrieve_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                hbm,
                lmc,
                int(ret_mask.sum().item()),
                "ok" if loaded else "miss",
                int(transfer_stats.get("chunks", 0)),
                int(transfer_stats.get("groups", 0)),
                int(transfer_stats.get("max_chunk_bytes", 0)),
                int(transfer_stats.get("max_group_bytes", 0)),
                int(transfer_stats.get("gpu_staging_chunk_bytes", 0)),
                int(transfer_stats.get("gpu_staging_buffer_chunks", 0)),
                int(transfer_stats.get("gpu_staging_buffer_bytes", 0)),
                int(transfer_stats.get("total_bytes", 0)),
                float(transfer_stats.get("pack_ms", 0.0)),
                float(transfer_stats.get("copy_ms", 0.0)),
                float(transfer_stats.get("sync_ms", 0.0)),
                float(transfer_stats.get("transfer_ms", 0.0)),
                float(transfer_stats.get("effective_gbps", 0.0)),
                retrieve_ms,
                total_ms,
            )
        return loaded

    def _slot_key(self, boundary_block_hash: int) -> DSV4CheckpointKey:
        return self._require_checkpoint_codec().make_key(
            boundary_block_hash=boundary_block_hash,
        )

    @staticmethod
    def _cpu_byte_view(blob) -> memoryview:
        if isinstance(blob, torch.Tensor):
            if blob.dtype is not torch.uint8:
                raise ValueError("SLOT sidecar tensor must be uint8")
            if blob.device.type != "cpu":
                raise ValueError("SLOT sidecar tensor must be on CPU")
            if not blob.is_contiguous():
                raise ValueError("SLOT sidecar tensor must be contiguous")
            return memoryview(blob.reshape(-1).numpy())
        view = memoryview(blob)
        if not view.c_contiguous:
            raise ValueError("SLOT sidecar bytes must be contiguous")
        return view if view.format == "B" and view.ndim == 1 else view.cast("B")

    def _create_slot_stream(self):
        slot_codec, _, _ = self._require_slot_components()
        if slot_codec.device.type != "cuda":
            return None
        return torch.cuda.Stream(device=slot_codec.device)

    @staticmethod
    def _slot_stream_context(stream):
        return nullcontext() if stream is None else torch.cuda.stream(stream)

    def _copy_slot_staging_to_cpu(self, staging_id: int) -> torch.Tensor:
        slot_codec, _, _ = self._require_slot_components()
        row = self._slot_staging_view(staging_id)
        payload_bytes = self._slot_payload_bytes(slot_codec)
        host = torch.empty(
            (HEADER_BYTES + payload_bytes,),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=row.device.type != "cpu",
        )
        payload = host[HEADER_BYTES:]
        if row.device.type == "cpu":
            payload.copy_(row)
            return host
        stream = self._create_slot_stream()
        try:
            with self._slot_stream_context(stream):
                payload.copy_(row, non_blocking=True)
        finally:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception as exc:
                    raise _SlotStagingSyncError(
                        "SLOT D2H completion was not confirmed"
                    ) from exc
        return host

    def _restore_slot_payload(
        self,
        payload: torch.Tensor,
        staging_id: int,
        destination_group: int,
    ) -> None:
        slot_codec, _, _ = self._require_slot_components()
        payload_bytes = self._slot_payload_bytes(slot_codec)
        host = payload.reshape(-1)
        if host.dtype is not torch.uint8 or host.device.type != "cpu":
            raise ValueError("SLOT payload view must be a CPU uint8 tensor")
        if int(host.numel()) != payload_bytes:
            raise ValueError(
                "SLOT payload size changed after decode: "
                f"got {int(host.numel())}, expected {payload_bytes}"
            )
        row = self._slot_staging_view(staging_id)
        stream = self._create_slot_stream()
        try:
            with self._slot_stream_context(stream):
                row.copy_(
                    host,
                    non_blocking=slot_codec.device.type == "cuda",
                )
                slot_codec.scatter_slot(row, destination_group, stream=stream)
        finally:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception as exc:
                    raise _SlotStagingSyncError(
                        "SLOT restore completion was not confirmed"
                    ) from exc

    def _load_slot(
        self,
        req: LMCacheReqMeta,
        reservation: _SlotLoadBatchReservation | None,
    ) -> bool:
        spec = getattr(req, "slot_load_spec", None)
        if spec is None:
            return True
        if reservation is None:
            raise RuntimeError("SLOT load requires a pre-reserved staging row")
        staging_id = reservation.staging_id
        slot_codec, slot_store, _ = self._require_slot_components()
        payload_bytes = self._slot_payload_bytes(slot_codec)
        checkpoint_codec = self._require_checkpoint_codec()
        key = self._slot_key(spec.boundary_block_hash)
        try:
            with slot_store.borrow(key) as blob:
                if blob is None:
                    return False
                _, payload = checkpoint_codec.decode_tensor(
                    blob,
                    expected_boundary_tokens=spec.boundary_tokens,
                    expected_boundary_block_hash=spec.boundary_block_hash,
                    expected_payload_bytes=payload_bytes,
                )
                self._restore_slot_payload(
                    payload,
                    staging_id,
                    spec.destination_group,
                )
        except (DSV4CheckpointCorruptionError, DSV4CheckpointError):
            # Evict all stale copies. Otherwise LMCache's duplicate-key guard
            # would silently skip the recomputed replacement forever. The
            # store fences the key if removal fails, so a later put cannot be
            # mistaken for a committed replacement of the same corrupt bytes.
            try:
                invalidated = slot_store.invalidate(key)
            except Exception as exc:  # noqa: BLE001  # storage boundary
                logger.warning(
                    "LMCache SLOT sidecar invalidation failed error_type=%s",
                    type(exc).__name__,
                )
            else:
                if not invalidated:
                    logger.warning(
                        "LMCache SLOT sidecar corruption remains fenced by the store"
                    )
            return False
        return True

    def _do_load_req(
        self,
        req: LMCacheReqMeta,
        slot_reservation: _SlotLoadBatchReservation | None = None,
    ) -> None:
        loaded = False
        staging_reusable = True
        slot_spec = getattr(req, "slot_load_spec", None)
        load_failure_reason = "operation"
        load_error_type = "none"
        try:
            if slot_reservation is not None and not slot_reservation.reusable:
                load_failure_reason = "staging_unavailable"
                loaded = False
            else:
                loaded = self._load_page(req)
                if loaded:
                    loaded = self._load_slot(req, slot_reservation)
                    if not loaded:
                        load_failure_reason = "sidecar_unavailable"
                else:
                    load_failure_reason = "page_unavailable"
        except _SlotStagingSyncError as exc:
            load_failure_reason = "gpu_completion"
            load_error_type = type(exc).__name__
            if slot_spec is None:
                logger.warning(
                    "LMCache offload: PAGE load GPU completion failed req=%s "
                    "error_type=%s",
                    req.req_id,
                    load_error_type,
                )
            staging_reusable = False
            loaded = False
        except Exception as exc:  # noqa: BLE001
            load_error_type = type(exc).__name__
            if slot_spec is None:
                logger.warning(
                    "LMCache offload: PAGE load failed req=%s error_type=%s",
                    req.req_id,
                    load_error_type,
                )
            loaded = False
        finally:
            if not self._finish_slot_load_reservation(
                req.req_id,
                slot_reservation,
                reusable=staging_reusable,
            ):
                loaded = False
                load_failure_reason = "staging_cleanup"
            # A worker owns one lookup pin for the emitted composite load. It is
            # released after PAGE and SLOT reach a terminal state, exactly once.
            self._lookup_unpin(req.req_id)
            self._complete_load(req, succeeded=loaded)
            if slot_spec is not None:
                if loaded:
                    slot_codec, _, _ = self._require_slot_components()
                    logger.info(
                        "LMCache offload: SLOT sidecar load restored rank=%s "
                        "req=%s boundary=%d bytes=%d",
                        getattr(self, "_rank", "?"),
                        req.req_id,
                        slot_spec.boundary_tokens,
                        self._slot_payload_bytes(slot_codec),
                    )
                else:
                    logger.warning(
                        "LMCache offload: SLOT sidecar load failed rank=%s "
                        "req=%s boundary=%d reason=%s error_type=%s",
                        getattr(self, "_rank", "?"),
                        req.req_id,
                        slot_spec.boundary_tokens,
                        load_failure_reason,
                        load_error_type,
                    )

    def _page_save_plan(
        self,
        req: LMCacheReqMeta,
    ) -> tuple[list[int], int] | None:
        ss = getattr(req, "save_spec", None)
        if ss is None or not bool(getattr(ss, "can_save", True)):
            return None
        chunk_size = int(self.chunk_size or 256)
        toks = req.token_ids
        if not req.is_last_prefill:
            toks = toks[: (len(toks) // chunk_size) * chunk_size]
        skip = (int(ss.skip_leading_tokens) // chunk_size) * chunk_size
        if skip >= len(toks):
            return None
        return toks, skip

    def _wait_for_session_publication(self, probe) -> bool:
        return _wait_for_publication(
            probe,
            timeout_s=self._publication_timeout_s,
            poll_interval_s=self._publication_poll_interval_s,
            clock=self._publication_clock,
            sleep=self._publication_sleep,
        )

    def _do_save_req(
        self,
        req: LMCacheReqMeta,
        producer_event=None,
        slot_snapshot: _SlotSaveSnapshot | None = None,
    ) -> None:
        slot_spec = getattr(req, "slot_save_spec", None)
        sidecar_published = False
        save_failure_reason = "operation"
        save_error_type = "none"
        snapshot_synchronized = False
        staging_reusable = True
        staging_terminalized = False
        slot_blob = None
        slot_preparation_error: Exception | None = None
        page_plan = None
        page_stored = False
        t_total0 = time.perf_counter()
        store_ms = 0.0
        transfer_stats: dict[str, int | float] = {}
        toks: list[int] = []
        skip = 0
        try:
            # For composite saves this is the event recorded after the RPC
            # stream's SLOT snapshot; waiting it also covers PAGE producers.
            if producer_event is not None:
                try:
                    producer_event.synchronize()
                except Exception as exc:
                    if (
                        slot_snapshot is not None
                        and slot_snapshot.staging_id is not None
                        and producer_event is slot_snapshot.ready_event
                    ):
                        staging_reusable = False
                        raise _SlotStagingSyncError(
                            "SLOT snapshot completion was not confirmed"
                        ) from exc
                    raise
                snapshot_synchronized = (
                    slot_snapshot is not None
                    and producer_event is slot_snapshot.ready_event
                )

            if slot_spec is not None:
                try:
                    if (
                        slot_snapshot is None
                        or slot_snapshot.staging_id is None
                        or not slot_snapshot.snapshot_ok
                    ):
                        raise RuntimeError(
                            "SLOT snapshot was not acquired successfully"
                        )

                    # A caller may supply a PAGE producer event distinct from
                    # the SLOT-ready event. Fence SLOT explicitly before D2H.
                    if (
                        not snapshot_synchronized
                        and slot_snapshot.ready_event is not None
                    ):
                        try:
                            slot_snapshot.ready_event.synchronize()
                        except Exception as exc:
                            staging_reusable = False
                            raise _SlotStagingSyncError(
                                "SLOT snapshot completion was not confirmed"
                            ) from exc
                        snapshot_synchronized = True

                    slot_blob = self._copy_slot_staging_to_cpu(slot_snapshot.staging_id)
                    # D2H completion makes the CPU frame ownership-independent.
                    # Do not hold scarce GPU temp capacity while PAGE/store waits.
                    released = self._release_slot_staging(
                        req.req_id,
                        slot_snapshot.staging_id,
                        operation="save",
                    )
                    if released:
                        staging_terminalized = True
                    else:
                        # Ownership is uncertain after a failed admission
                        # transition. Keep the row out of circulation.
                        staging_reusable = False
                        staging_terminalized = self._quarantine_slot_staging(
                            req.req_id,
                            slot_snapshot.staging_id,
                            operation="save release failure",
                        )
                        save_failure_reason = "staging_cleanup"
                        raise RuntimeError("SLOT staging release failed after D2H")
                except Exception as exc:  # noqa: BLE001
                    # PAGE storage is independent and must still run when SLOT
                    # admission, gather, D2H, or cleanup fails.
                    if isinstance(exc, _SlotStagingSyncError):
                        staging_reusable = False
                    slot_preparation_error = exc

            page_plan = self._page_save_plan(req)
            if page_plan is not None:
                toks, skip = page_plan
                mask = torch.ones(len(toks), dtype=torch.bool)
                mask[:skip] = False
                t_store0 = time.perf_counter()
                self._reset_gpu_connector_transfer_stats()
                self._engine.store(
                    tokens_to_tensor(toks),
                    mask=mask,
                    block_ids=req.block_ids,
                    req_id=str(req.req_id),
                )
                store_ms = (time.perf_counter() - t_store0) * 1000
                transfer_stats = self._last_gpu_connector_transfer_stats()
            if slot_spec is None:
                # PAGE-only save. `store` returning is the only durability
                # signal that exists here, so it is the verdict; a raise above
                # leaves this False and the advance gets taken back.
                page_stored = True

            if slot_spec is not None:
                # A stateful save can check its own work, and this probe is the
                # only confirmation that the PAGEs behind the advance are
                # actually queryable. Run it before anything SLOT-related, so
                # the PAGE verdict never depends on staging, and claim the
                # advance only once it passes -- reporting success on a
                # visibility timeout would commit a watermark over a prefix
                # nobody can read, which is the hole this channel exists to
                # prevent.
                boundary_tokens = int(slot_spec.boundary_tokens)

                def _page_boundary_visible() -> bool:
                    page_hit = self._engine.lookup(
                        req.token_ids[:boundary_tokens],
                        pin=False,
                    )
                    return page_hit is not None and int(page_hit) >= boundary_tokens

                if not self._wait_for_session_publication(_page_boundary_visible):
                    save_failure_reason = "page_visibility_timeout"
                    raise RuntimeError(
                        "PAGE coverage did not become session-visible before timeout"
                    )
                # Confirmed. Everything below is sidecar-only, and a SLOT
                # failure from here on must not revoke a PAGE advance that did
                # land -- that is why the verdict is taken at this exact point
                # and not at the end of the block.
                page_stored = True

                if slot_preparation_error is not None:
                    raise slot_preparation_error
                _, slot_store, _ = self._require_slot_components()
                checkpoint_codec = self._require_checkpoint_codec()
                if slot_blob is None:
                    raise RuntimeError("SLOT CPU frame is unavailable after D2H")
                checkpoint_codec.finalize_tensor_(
                    slot_blob,
                    boundary_tokens=slot_spec.boundary_tokens,
                    boundary_block_hash=slot_spec.boundary_block_hash,
                )
                key = self._slot_key(slot_spec.boundary_block_hash)
                if not slot_store.put(key, slot_blob):
                    save_failure_reason = "sidecar_submission"
                    raise RuntimeError("SLOT sidecar store rejected put")
                if not self._wait_for_session_publication(
                    lambda: slot_store.contains(key)
                ):
                    save_failure_reason = "sidecar_visibility_timeout"
                    raise RuntimeError(
                        "submitted SLOT sidecar did not become session-visible "
                        "before timeout"
                    )
                sidecar_published = True
        except _SlotStagingSyncError as exc:
            staging_reusable = False
            save_failure_reason = "gpu_completion"
            save_error_type = type(exc).__name__
        except Exception as exc:  # noqa: BLE001
            save_error_type = type(exc).__name__
            if slot_spec is None:
                logger.warning(
                    "LMCache offload: PAGE save failed req=%s error_type=%s",
                    req.req_id,
                    save_error_type,
                )
        finally:
            if (
                slot_snapshot is not None
                and slot_snapshot.staging_id is not None
                and not staging_terminalized
            ):
                if not snapshot_synchronized and slot_snapshot.ready_event is not None:
                    try:
                        slot_snapshot.ready_event.synchronize()
                    except Exception:  # noqa: BLE001
                        staging_reusable = False
                        sidecar_published = False
                        save_failure_reason = "snapshot_fence"
                        save_error_type = "SlotSnapshotFenceError"
                if staging_reusable:
                    released = self._release_slot_staging(
                        req.req_id,
                        slot_snapshot.staging_id,
                        operation="save",
                    )
                    if released:
                        staging_terminalized = True
                    else:
                        staging_reusable = False
                        staging_terminalized = self._quarantine_slot_staging(
                            req.req_id,
                            slot_snapshot.staging_id,
                            operation="save release failure",
                        )
                else:
                    staging_terminalized = self._quarantine_slot_staging(
                        req.req_id,
                        slot_snapshot.staging_id,
                        operation="save",
                    )
                    released = False
                if not released:
                    sidecar_published = False
                    save_failure_reason = "staging_cleanup"

            with self._lock:
                self._report_page_save_locked(req, succeeded=page_stored)
                if slot_spec is not None:
                    if sidecar_published:
                        self._done_sidecar_save.add(self._save_completion_id(req))
                    else:
                        self._failed_sidecar_save.add(self._save_completion_id(req))
            if not page_stored and getattr(req, "save_spec", None) is not None:
                logger.warning(
                    "LMCache offload: PAGE save did not land rank=%s req=%s "
                    "skip=%d reason=%s error_type=%s",
                    getattr(self, "_rank", "?"),
                    req.req_id,
                    skip,
                    save_failure_reason,
                    save_error_type,
                )
            if slot_spec is not None:
                if sidecar_published:
                    logger.info(
                        "LMCache offload: SLOT sidecar save published rank=%s "
                        "req=%s boundary=%d",
                        getattr(self, "_rank", "?"),
                        req.req_id,
                        slot_spec.boundary_tokens,
                    )
                else:
                    logger.warning(
                        "LMCache offload: SLOT sidecar save failed rank=%s "
                        "req=%s boundary=%d reason=%s error_type=%s",
                        getattr(self, "_rank", "?"),
                        req.req_id,
                        slot_spec.boundary_tokens,
                        save_failure_reason,
                        save_error_type,
                    )

        if page_plan is not None and self._profile_enabled():
            total_ms = (time.perf_counter() - t_total0) * 1000
            logger.info(
                "[OFFLOAD-SAVE-PROF] rank=%s req=%s toks=%d skip=%d "
                "chunks=%d groups=%d max_chunk_bytes=%d max_group_bytes=%d "
                "gpu_staging_chunk_bytes=%d "
                "gpu_staging_buffer_chunks=%d gpu_staging_buffer_bytes=%d "
                "total_bytes=%d pack_ms=%.2f copy_ms=%.2f sync_ms=%.2f "
                "transfer_ms=%.2f effective_gbps=%.2f "
                "store_ms=%.2f total_ms=%.2f",
                getattr(self, "_rank", "?"),
                req.req_id,
                len(toks),
                skip,
                int(transfer_stats.get("chunks", 0)),
                int(transfer_stats.get("groups", 0)),
                int(transfer_stats.get("max_chunk_bytes", 0)),
                int(transfer_stats.get("max_group_bytes", 0)),
                int(transfer_stats.get("gpu_staging_chunk_bytes", 0)),
                int(transfer_stats.get("gpu_staging_buffer_chunks", 0)),
                int(transfer_stats.get("gpu_staging_buffer_bytes", 0)),
                int(transfer_stats.get("total_bytes", 0)),
                float(transfer_stats.get("pack_ms", 0.0)),
                float(transfer_stats.get("copy_ms", 0.0)),
                float(transfer_stats.get("sync_ms", 0.0)),
                float(transfer_stats.get("transfer_ms", 0.0)),
                float(transfer_stats.get("effective_gbps", 0.0)),
                store_ms,
                total_ms,
            )

    # -- per-step (RPC thread, post-forward): poll completions ------------
    def get_finished(self) -> KVConnectorOutput:
        # Offload uses extended completion states:
        # - finished_loading wakes successfully loaded requests.
        # - failed_loading wakes them for recompute using already allocated blocks.
        # - finished_saving releases blocks whose free was deferred during save.
        with self._lock:
            dl, fl, ds = self._drain_common_completions_locked()
            dss = set(self._done_sidecar_save)
            fss = set(self._failed_sidecar_save)
            self._done_sidecar_save.clear()
            self._failed_sidecar_save.clear()
            dps = set(self._done_page_save)
            fps = set(self._failed_page_save)
            self._done_page_save.clear()
            self._failed_page_save.clear()
        connector_completions = {
            ConnectorCompletion(
                channel=DSV4_CHECKPOINT_SAVE_CHANNEL,
                operation_id=completion_id,
                succeeded=True,
            )
            for completion_id in dss
        }
        connector_completions.update(
            ConnectorCompletion(
                channel=DSV4_CHECKPOINT_SAVE_CHANNEL,
                operation_id=completion_id,
                succeeded=False,
            )
            for completion_id in fss
        )
        connector_completions.update(
            ConnectorCompletion(
                channel=DSV4_PAGE_SAVE_CHANNEL,
                operation_id=completion_id,
                succeeded=True,
            )
            for completion_id in dps
        )
        connector_completions.update(
            ConnectorCompletion(
                channel=DSV4_PAGE_SAVE_CHANNEL,
                operation_id=completion_id,
                succeeded=False,
            )
            for completion_id in fps
        )
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=dl,
            failed_loading=fl,
            finished_saving=ds,
            connector_completions=connector_completions,
        )

    def get_finished_recv_blocks(self) -> list[int]:
        # Local CUDA copies are ordered by the copy stream + synchronize() before
        # we mark done; no RDMA-style GPU fence needed.
        return []


# =====================================================================
# Scheduler side
# =====================================================================
class DSV4OffloadScheduler(OffloadSchedulerMixin, KVConnectorSchedulerBase):
    # Consumer semantics: finished_recving wakes parked seqs (the engine asserts
    # `not is_producer` on that path). Offload never uses finished_sending.
    is_producer = False
    # Opt the scheduler into offload-wake (suffix prefill) instead of the P/D
    # decode-jump in Scheduler.schedule(); see Scheduler._is_offload_connector.
    is_offload = True

    def __init__(self, config) -> None:
        self._init_offload_statistics()
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        # LMCache storage and DSV4 geometry are required configuration.  Keep
        # them outside the optional lookup-client boundary, and pass raw values
        # to the strict profile builder so fractional geometry is not truncated.
        cfg = offcfg.build_lmcache_config(kvc)
        self.profile = build_dsv4_profile(
            config,
            chunk_size=cfg.chunk_size,
        )
        self.block_size = self.profile.block_size
        self.hash_block_size = self.profile.hash_block_size
        self.chunk_size = self.profile.chunk_size
        self.resume_alignment = self.profile.resume_alignment
        self.sidecar_interval = self.profile.sidecar_interval
        self._lookup_client = None

        # req_id -> LoadSpec (pending load decided at match time)
        self._load_specs: dict[str, LoadSpec] = {}
        # req_id -> Sequence (queued to recv this step)
        self._reqs_need_recv: dict[str, object] = {}
        # req_id -> HBM chunk frontier for an emitted load. If the load fails,
        # lower the save frontier to this value so recomputed chunks can be
        # stored again.
        self._load_save_floors: dict[str, int] = {}
        # req_id -> LMCache chunk frontier observed by lookup. The scheduler
        # should not re-save this already-persisted prefix unless a later load
        # actually fails.
        self._hit_save_floors: dict[str, int] = {}
        # Persistent save tracker: sid -> [seq, saved_offset]. A seq's prompt
        # prefix is stored to LMCache once prefill computes it
        # (seq.prefix_hashes_published flips True), chunk by chunk.
        self._save_tracker: dict[str, list] = {}
        # The worker has the same bound as a last-resort guard, but admission
        # belongs here: rejecting after dispatch lets TP ranks make different
        # decisions and forces the scheduler to roll the PAGE watermark back.
        # Keep one shared limit for PAGE-only, SLOT-only, and PAGE+SLOT saves.
        self._max_pending_saves = max_pending_saves(
            kvc,
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1") or 1)
        )
        # Save admission separates a lightweight candidate from a capacity-
        # holding reservation.  This matters at request finish: connector
        # metadata for the current step has already been built, so the request
        # cannot be dispatched immediately.  A committed entry reserves one
        # worker slot until the next metadata build emits it.
        save_admission = load_save_admission_config()
        self._save_max_pinned_ratio = save_admission.max_pinned_ratio
        self._save_max_pinned_blocks = save_admission.max_pinned_blocks
        self._block_manager = None
        self._save_pin_total_blocks = 0
        self._save_pin_budget_blocks: int | None = None
        self._save_block_reservations: dict[
            str | SaveOperationId, SaveBlockReservation
        ] = {}
        self._save_min_observed_count = save_admission.min_observed_count
        self._save_aging_weight = save_admission.aging_weight
        self._save_release_weight = save_admission.release_weight
        self._prefix_demand = PrefixDemandTracker(
            block_tokens=save_admission.demand_block_tokens,
            max_entries=save_admission.demand_max_entries,
            ttl_seconds=save_admission.demand_ttl_seconds,
        )
        self._save_demand_keys: dict[
            str, tuple[object, tuple[PrefixDemandKey, ...]]
        ] = {}
        self._save_candidate_since: dict[str, tuple[object, float]] = {}
        self._save_candidate_generation: dict[str, tuple[object, int]] = {}
        self._save_candidate_nonce = 0
        self._save_committed: dict[str, object] = {}
        self._finished_save_requests: dict[str, object] = {}
        self._finished_save_failed: set[str] = set()
        self._save_inflight_since: dict[SaveOperationId, float] = {}
        self._save_operation_owners: dict[SaveOperationId, object] = {}
        self.total_save_admitted = 0
        self.total_save_budget_rejected = 0
        self.total_save_budget_rejected_blocks = 0
        self.total_save_budget_evicted = 0
        self.total_save_budget_evicted_blocks = 0
        self.total_save_oversized = 0
        self._save_drop_totals = {
            "capacity": 0,
            "low_value": 0,
            "terminal_failure": 0,
            "stale": 0,
        }
        self._save_drop_token_totals = dict.fromkeys(self._save_drop_totals, 0)
        # Scheduler-lifetime completion generation: every emitted save gets a
        # distinct SaveOperationId, so late TP notifications cannot complete a
        # later PAGE/SLOT save after request cleanup or request-ID reuse.
        self._save_nonce = 0
        self._load_nonce = 0
        self._load_lifecycles: dict[str, object] = {}
        self._active_load_operations: dict[str, tuple[object, LoadOperationId]] = {}
        self._save_inflight: dict[str, set[SaveOperationId]] = {}
        # sid -> {save operation: the watermark before that operation's advance}.
        # ``build_connector_meta`` advances the saved watermark where it *emits*
        # a save, so the advance is a prediction, not a fact. Keep what each one
        # was contingent on so a save that never lands can take its advance back.
        self._save_watermark_rollback: dict[
            str, dict[SaveOperationId, tuple[object, int]]
        ] = {}
        # Finished requests get a bounded window without successful PAGE saves.
        # Dispatch refreshes the copy-retention clock, not this progress clock.
        self._finished_save_progress_at: dict[str, float] = {}
        # Stateful PAGE/SLOT protocol. Sidecar commits are session-local because
        # worker-side sidecar storage is not queried by the scheduler.
        self._committed_sidecar_hashes = _BoundedLRUSet(
            _committed_sidecar_capacity(kvc)
        )
        self._sidecar_save_inflight: dict[str, tuple[SaveOperationId, int, int]] = {}
        self._failed_sidecar_saves: dict[str, set[tuple[int, int]]] = {}
        self._pending_slot_loads: dict[str, tuple[int, int]] = {}
        self._active_slot_loads: dict[str, tuple[int, int]] = {}
        # sid -> (concrete Sequence object, chained hashes, boundary records).
        # Object identity prevents a reused request ID from inheriting hashes.
        self._sidecar_hash_cache: dict[
            str, tuple[object, dict[int, int], list[tuple[int, int]]]
        ] = {}
        self._lookup_in_step: list[str] = []
        self._handoff_loads: set[str] = set()
        # Unaligned handoff is always on: when the HBM prefix-cache hit is not
        # chunk-aligned, recompute the misaligned head up to the next chunk
        # boundary, then load the aligned remainder from CPU. (Previously gated
        # by the OFFLOAD_UNALIGNED_HANDOFF env var; now unconditional.)
        try:
            self._min_load_tokens = max(
                0, int(os.environ.get("OFFLOAD_MIN_LOAD_TOKENS", "8192"))
            )
        except ValueError:
            logger.warning(
                "LMCache offload scheduler: invalid OFFLOAD_MIN_LOAD_TOKENS=%r; "
                "using 8192",
                os.environ.get("OFFLOAD_MIN_LOAD_TOKENS"),
            )
            self._min_load_tokens = 8192

        world = offcfg.lmcache_replica_world_size(config)
        meta = offcfg.build_lmcache_metadata(config, cfg, world, 0)
        try:
            from lmcache.v1.lookup_client.factory import LookupClientFactory

            self._lookup_client = LookupClientFactory.create_lookup_client(cfg, meta)
            logger.info(
                "LMCache offload scheduler: lookup client on %s (world=%d)",
                meta.engine_id,
                world,
            )
        except Exception as e:  # noqa: BLE001  # optional third-party client
            logger.warning(
                "LMCache offload scheduler: lookup client unavailable: %s", e
            )

    def bind_block_manager(self, block_manager) -> None:
        if self._block_manager is not None and self._block_manager is not block_manager:
            raise RuntimeError("offload scheduler is already bound to a block manager")
        self._block_manager = block_manager
        total_blocks = int(block_manager.total_allocatable_kv_blocks)
        if total_blocks <= 0:
            raise ValueError("BlockManager must expose a positive KV block capacity")
        ratio_budget = int(total_blocks * self._save_max_pinned_ratio)
        absolute = self._save_max_pinned_blocks
        self._save_pin_total_blocks = total_blocks
        self._save_pin_budget_blocks = (
            ratio_budget if absolute is None else min(ratio_budget, absolute)
        )

    # -- match: how many extra tokens can come from CPU/NVMe -------------
    def _begin_load_lifecycle(self, seq) -> None:
        sid = str(seq.id)
        previous = self._load_lifecycles.get(sid)
        if previous is not None and previous is not seq:
            self._load_specs.pop(sid, None)
            self._reqs_need_recv.pop(sid, None)
            self._load_save_floors.pop(sid, None)
            self._hit_save_floors.pop(sid, None)
            self._pending_slot_loads.pop(sid, None)
            self._active_slot_loads.pop(sid, None)
            self._active_load_operations.pop(sid, None)
            self._handoff_loads.discard(sid)
            self._load_failed_seqs.pop(sid, None)
        self._load_lifecycles[sid] = seq

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        if not self._do_load or self._lookup_client is None:
            return 0, False
        self._begin_load_lifecycle(seq)
        sid = str(seq.id)
        if self._repeat_load_suppressed(seq, sid):
            return 0, False
        num_prompt = seq.num_prompt_tokens
        token_ids = list(seq.token_ids[:num_prompt])
        if sid not in self._lookup_in_step:
            self._lookup_in_step.append(sid)
        try:
            hit = self._lookup_client.lookup(token_ids, lookup_id=sid)
        except Exception:
            logger.exception("LMCache offload lookup failed for seq %s", seq.id)
            self._clear_lookup_retry_state(sid)
            return 0, False
        if logger.isEnabledFor(logging.DEBUG):
            _lh = None
            try:
                tdb = getattr(self._lookup_client, "token_database", None)
                if tdb is not None:
                    _lh = [
                        k
                        for (_s, _e, k) in list(
                            tdb.process_tokens(token_ids, make_key=False)
                        )[:3]
                    ]
            except Exception as e:  # noqa: BLE001  # debug-only introspection
                _lh = f"err:{e}"
            logger.debug(
                "[OFFLOAD-LOOKUP] seq=%s num_prompt=%d hbm_cached=%d hit=%s lookuphash3=%s",
                seq.id,
                num_prompt,
                int(seq.num_cached_tokens),
                hit,
                _lh,
            )
        if not hit:
            self._clear_lookup_retry_state(sid)
            return 0, False
        hit = self._loadable_hit(hit, num_prompt)
        self._hit_save_floors[sid] = hit
        if bool(getattr(seq, "has_per_req_cache", False)):
            boundary_hashes, _ = self._sidecar_hash_data(seq)
            boundary = (hit // self.resume_alignment) * self.resume_alignment
            committed_hashes = self._committed_sidecar_hashes
            while boundary > 0:
                boundary_hash = boundary_hashes.get(boundary)
                if boundary_hash in committed_hashes:
                    break
                boundary -= self.resume_alignment
            else:
                boundary_hash = None

            if boundary_hash is None:
                self._clear_lookup_retry_state(sid)
                return 0, False

            hit = boundary
            self._pending_slot_loads[sid] = (boundary, boundary_hash)

        need = hit - int(seq.num_cached_tokens)
        if need <= 0:
            self._clear_lookup_retry_state(sid)
            return 0, False
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=int(seq.num_cached_tokens),
            lmcache_cached_tokens=hit,
            can_load=False,
        )
        return need, True  # True => park in WAITING_FOR_REMOTE_KVS

    def update_state_after_alloc(self, seq) -> None:
        sid = str(seq.id)
        self._begin_load_lifecycle(seq)
        ls = self._load_specs.get(sid) if self._do_load else None
        initial_saved = max(
            self._lmcache_hit_save_floor(ls),
            int(self._hit_save_floors.get(sid, 0)),
        )
        logger.debug(
            "[OFFLOAD-ALLOC] seq=%s ls_found=%s num_cached_now=%s",
            seq.id,
            ls is not None,
            int(getattr(seq, "num_cached_tokens", -1)),
        )
        pending_slot = self._pending_slot_loads.get(sid)
        destination_group = getattr(seq, "state_slot", -1)
        if pending_slot is not None and (
            not isinstance(destination_group, int) or destination_group < 0
        ):
            logger.warning(
                "LMCache offload: rejecting stateful load without SLOT group "
                "for seq %s",
                seq.id,
            )
            self._clear_pending_load(sid)
            ls = None
        if ls is not None:
            ls.can_load = True
            self._reqs_need_recv[sid] = seq
        # Track for save; build_connector_meta stores chunks once the scheduler's
        # computed frontier (seq.num_cached_tokens) has advanced past them.
        #
        # If LMCache lookup already found a prefix for this request, do not save
        # that prefix again. This covers both direct loads and the
        # hbm_satisfies_after_alloc case where HBM prefix cache already covers
        # the lookup hit. Only suffix chunks computed by this request should be
        # stored.
        if self._do_save:
            entry = self._save_tracker.get(sid)
            if entry is None or entry[0] is not seq:
                self._save_tracker[sid] = [seq, initial_saved]
                now = time.monotonic()
                demand_keys = self._prefix_demand.observe(
                    seq.token_ids,
                    int(seq.num_prompt_tokens),
                    now,
                )
                self._save_demand_keys[sid] = (seq, demand_keys)
                self._save_candidate_generation[sid] = (
                    seq,
                    self._save_candidate_nonce,
                )
                self._save_candidate_nonce += 1
                previous_committed = self._save_committed.pop(sid, None)
                if previous_committed is not None:
                    self._release_save_reservation(sid, owner=previous_committed)
                self._finished_save_requests.pop(sid, None)
                self._finished_save_failed.discard(sid)
                self._finished_save_progress_at.pop(sid, None)
                self._sidecar_hash_cache.pop(sid, None)
                self._failed_sidecar_saves.pop(sid, None)
            else:
                entry[1] = max(int(entry[1]), initial_saved)

    def _sidecar_hash_data(self, seq) -> tuple[dict[int, int], list[tuple[int, int]]]:
        sid = str(seq.id)
        cached = self._sidecar_hash_cache.get(sid)
        if cached is not None and cached[0] is seq:
            return cached[1], cached[2]

        if not bool(getattr(seq, "has_per_req_cache", False)):
            result = ({}, [])
            self._sidecar_hash_cache[sid] = (seq, *result)
            return result
        alignment = int(getattr(self, "resume_alignment", 0) or 0)
        if alignment <= 0:
            result = ({}, [])
            self._sidecar_hash_cache[sid] = (seq, *result)
            return result
        boundaries = sidecar_boundary_tokens(
            num_prompt_tokens=int(getattr(seq, "num_prompt_tokens", 0)),
            resume_alignment=alignment,
            sidecar_interval=int(getattr(self, "sidecar_interval", 0) or 0),
        )
        if not boundaries:
            result = ({}, [])
            self._sidecar_hash_cache[sid] = (seq, *result)
            return result

        hashes = _chained_prefix_hashes(
            seq.token_ids,
            self.hash_block_size,
        )
        records = [
            (boundary, hashes[boundary])
            for boundary in sorted(boundaries)
            if boundary in hashes
        ]
        self._sidecar_hash_cache[sid] = (seq, hashes, records)
        return hashes, records

    def _sidecar_boundary_records(self, seq) -> list[tuple[int, int]]:
        return self._sidecar_hash_data(seq)[1]

    def _sidecar_save_candidate(
        self,
        seq,
        computed: int,
    ) -> tuple[int, int] | None:
        sid = str(seq.id)
        source_group = getattr(seq, "state_slot", -1)
        if (
            not bool(getattr(seq, "_state_initialized_after_alloc", False))
            or not isinstance(source_group, int)
            or source_group < 0
            or sid in self._sidecar_save_inflight
        ):
            return None
        for boundary, boundary_hash in self._sidecar_boundary_records(seq):
            if boundary != computed:
                continue
            identity = (boundary, boundary_hash)
            if boundary_hash in self._committed_sidecar_hashes:
                return None
            if identity in self._failed_sidecar_saves.get(sid, set()):
                return None
            return identity
        return None

    def _next_pending_sidecar_boundary(
        self,
        seq,
        start: int,
        end: int,
    ) -> tuple[int, int] | None:
        source_group = getattr(seq, "state_slot", -1)
        if (
            not bool(getattr(seq, "_state_initialized_after_alloc", False))
            or not isinstance(source_group, int)
            or source_group < 0
        ):
            return None
        sid = str(seq.id)
        return select_pending_sidecar_boundary(
            self._sidecar_boundary_records(seq),
            start=start,
            end=end,
            committed_hashes=self._committed_sidecar_hashes,
            inflight=self._sidecar_save_inflight.get(sid),
            failed=self._failed_sidecar_saves.get(sid, set()),
        )

    def _next_save_operation(self, seq) -> SaveOperationId:
        """Issue an exact, scheduler-lifetime identity for one save generation."""
        operation = SaveOperationId(seq.id, self._save_nonce)
        self._save_nonce += 1
        return operation

    def _inflight_save_operations(self) -> set[SaveOperationId]:
        operations = {
            operation
            for inflight in self._save_inflight.values()
            for operation in inflight
        }
        operations.update(
            inflight[0] for inflight in self._sidecar_save_inflight.values()
        )
        return operations

    def _retire_save_operation_owner(self, operation: SaveOperationId) -> None:
        if any(operation in inflight for inflight in self._save_inflight.values()):
            return
        if any(
            inflight[0] == operation
            for inflight in self._sidecar_save_inflight.values()
        ):
            return
        self._save_inflight_since.pop(operation, None)
        owner = self._save_operation_owners.pop(operation, None)
        self._release_save_reservation(operation, owner=owner)

    def _may_emit_save(self) -> bool:
        """Return whether the scheduler may dispatch one more save operation.

        A PAGE+SLOT request is one worker operation even though both completion
        channels retain its identity. Counting distinct operation IDs keeps the
        scheduler's view equal to the worker semaphore while old multi-inflight
        state, if any, drains safely. A committed finished save also consumes a
        slot: it is guaranteed dispatch on the next metadata build and must not
        be overbooked by an active request in the meantime.
        """

        return (
            len(self._inflight_save_operations()) + len(self._save_committed)
            < self._max_pending_saves
        )

    def _save_count_allows(self, committed_after: int) -> bool:
        return (
            len(self._inflight_save_operations()) + committed_after
            <= self._max_pending_saves
        )

    def _pinned_save_block_ids(
        self,
        reservations: dict[str | SaveOperationId, SaveBlockReservation],
    ) -> set[int]:
        pinned: set[int] = set()
        for reservation in reservations.values():
            if self._finished_save_requests.get(reservation.sid) is reservation.seq:
                pinned.update(reservation.block_ids)
        return pinned

    def _save_block_usage(
        self,
        reservations: dict[str | SaveOperationId, SaveBlockReservation] | None = None,
    ) -> tuple[int, int, int]:
        """Return ``(reserved, pinned, total_budget_usage)`` without double count."""

        current = (
            self._save_block_reservations if reservations is None else reservations
        )
        pinned_ids = self._pinned_save_block_ids(current)
        reserved_ids: set[int] = set()
        pinned_estimated = 0
        reserved_estimated = 0
        for reservation in current.values():
            if self._finished_save_requests.get(reservation.sid) is reservation.seq:
                pinned_estimated += int(reservation.estimated_blocks)
            else:
                reserved_ids.update(reservation.block_ids)
                reserved_estimated += int(reservation.estimated_blocks)
        reserved = len(reserved_ids - pinned_ids) + reserved_estimated
        pinned = len(pinned_ids) + pinned_estimated
        total = len(pinned_ids | reserved_ids) + pinned_estimated + reserved_estimated
        return reserved, pinned, total

    def _candidate_block_reservation(
        self,
        candidate: _SaveCandidate,
        *,
        priority_score: float,
    ) -> SaveBlockReservation:
        # DSV4 in this branch cannot release PAGE source ranges independently.
        # If the request finishes while its PAGE or SLOT save is outstanding,
        # teardown retains the complete block table. Reserve that exact local
        # physical footprint up front so admission cannot pretend only the
        # dirty PAGE interval is pinned.
        table = list(getattr(candidate.seq, "block_table", ()))
        physical_ids = [int(block_id) for block_id in table if int(block_id) >= 0]
        block_ids = frozenset(physical_ids)
        required_tokens = max(
            candidate.aligned if candidate.page_save_due else 0,
            (
                candidate.sidecar_candidate[0]
                if candidate.sidecar_candidate is not None
                else 0
            ),
        )
        required_blocks = ceil(required_tokens / max(1, int(self.block_size)))
        estimated = max(0, required_blocks - len(physical_ids))
        return SaveBlockReservation(
            sid=candidate.sid,
            seq=candidate.seq,
            generation=candidate.generation,
            block_ids=block_ids,
            estimated_blocks=estimated,
            priority_score=float(priority_score),
            committed=True,
        )

    @staticmethod
    def _reservation_footprint(reservation: SaveBlockReservation) -> int:
        return len(reservation.block_ids) + int(reservation.estimated_blocks)

    def _reservation_fits(
        self,
        reservation: SaveBlockReservation,
        *,
        victim_keys: tuple[str, ...] = (),
    ) -> bool:
        projected = dict(self._save_block_reservations)
        for key in victim_keys:
            projected.pop(key, None)
        projected[reservation.sid] = reservation
        committed_after = sum(item.committed for item in projected.values())
        if not self._save_count_allows(committed_after):
            return False
        budget = self._save_pin_budget_blocks
        return budget is None or self._save_block_usage(projected)[2] <= budget

    def _try_reserve_save_candidate(
        self,
        candidate: _SaveCandidate,
        *,
        allow_eviction: bool,
        now: float | None = None,
    ) -> bool:
        """Atomically reserve one DSV4 PAGE+SLOT save operation."""

        existing = self._save_block_reservations.get(candidate.sid)
        if (
            existing is not None
            and existing.seq is candidate.seq
            and existing.generation == candidate.generation
            and existing.committed
        ):
            return True

        score = self._candidate_priority(
            candidate,
            time.monotonic() if now is None else now,
        )
        reservation = self._candidate_block_reservation(candidate, priority_score=score)
        victims: list[SaveBlockReservation] = []
        if not self._reservation_fits(reservation):
            if allow_eviction:
                eligible = sorted(
                    (
                        item
                        for key, item in self._save_block_reservations.items()
                        if isinstance(key, str)
                        and item.committed
                        and item.priority_score < reservation.priority_score
                        and self._save_committed.get(key) is item.seq
                        and key not in self._save_inflight
                        and key not in self._sidecar_save_inflight
                        and (entry := self._save_tracker.get(key)) is not None
                        and entry[0] is item.seq
                    ),
                    key=lambda item: (
                        item.priority_score,
                        item.sid,
                        item.generation,
                    ),
                )
                for victim in eligible:
                    victims.append(victim)
                    if self._reservation_fits(
                        reservation,
                        victim_keys=tuple(item.sid for item in victims),
                    ):
                        break
                else:
                    victims = []
            if not victims and not self._reservation_fits(reservation):
                footprint = self._reservation_footprint(reservation)
                self.total_save_budget_rejected += 1
                self.total_save_budget_rejected_blocks += footprint
                budget = self._save_pin_budget_blocks
                if budget is not None and footprint > budget:
                    self.total_save_oversized += 1
                return False

        # Simulate the whole victim set before mutating anything. PAGE and SLOT
        # share one reservation and therefore can only be replaced together.
        for victim in victims:
            self.total_save_budget_evicted += 1
            self.total_save_budget_evicted_blocks += self._reservation_footprint(victim)
            self.drop_unadmitted_save(victim.seq, reason="capacity")
        self._save_committed[candidate.sid] = candidate.seq
        self._save_block_reservations[candidate.sid] = reservation
        self.total_save_admitted += 1
        return True

    def _release_save_reservation(
        self,
        key: str | SaveOperationId,
        *,
        owner=None,
    ) -> SaveBlockReservation | None:
        reservation = self._save_block_reservations.get(key)
        if reservation is None or (owner is not None and reservation.seq is not owner):
            return None
        return self._save_block_reservations.pop(key)

    def _refresh_finished_save_reservations(self, seq) -> None:
        """Charge every physical block retained by a now-finished request."""

        block_ids = frozenset(
            physical_id
            for block_id in getattr(seq, "block_table", ())
            if (physical_id := int(block_id)) >= 0
        )
        for key, reservation in list(self._save_block_reservations.items()):
            if reservation.seq is seq:
                self._save_block_reservations[key] = replace(
                    reservation,
                    block_ids=block_ids,
                    estimated_blocks=0,
                )

    def _candidate_generation_for(self, sid: str, seq) -> int:
        record = self._save_candidate_generation.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        generation = self._save_candidate_nonce
        self._save_candidate_nonce += 1
        self._save_candidate_generation[sid] = (seq, generation)
        return generation

    def _candidate_since_for(self, sid: str, seq, now: float) -> float:
        record = self._save_candidate_since.get(sid)
        if record is not None and record[0] is seq:
            return record[1]
        self._save_candidate_since[sid] = (seq, now)
        return now

    def _candidate_demand(
        self,
        sid: str,
        seq,
        *,
        aligned: int,
        now: float,
    ) -> tuple[int, int]:
        record = self._save_demand_keys.get(sid)
        if record is None or record[0] is not seq:
            return 0, 0
        return self._prefix_demand.heat(
            record[1],
            max_tokens=aligned,
            now=now,
        )

    def _build_save_candidate(
        self, sid: str, *, now: float | None = None
    ) -> _SaveCandidate | None:
        entry = self._save_tracker.get(sid)
        if entry is None:
            return None
        seq, saved = entry
        computed = min(
            int(getattr(seq, "num_cached_tokens", 0)),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        chunk = int(self.chunk_size or 256)
        aligned = (computed // chunk) * chunk
        sidecar_candidate = self._sidecar_save_candidate(seq, computed)
        if aligned <= int(saved) and sidecar_candidate is None:
            return None
        candidate_now = time.monotonic() if now is None else float(now)
        observed_count, reusable_tokens = self._candidate_demand(
            sid,
            seq,
            aligned=aligned,
            now=candidate_now,
        )
        finished = self._finished_save_requests.get(sid) is seq
        return _SaveCandidate(
            sid=sid,
            seq=seq,
            generation=self._candidate_generation_for(sid, seq),
            saved=int(saved),
            computed=computed,
            aligned=aligned,
            sidecar_candidate=sidecar_candidate,
            observed_count=observed_count,
            reusable_tokens=reusable_tokens,
            enqueued_at=self._candidate_since_for(sid, seq, candidate_now),
            finished=finished,
            held_blocks=len(getattr(seq, "block_table", ())) if finished else 0,
        )

    def _candidate_priority(self, candidate: _SaveCandidate, now: float) -> float:
        chunk = max(1, int(self.chunk_size or 256))
        cost_units = max(1, ceil(candidate.dirty_tokens / chunk))
        reusable = max(candidate.reusable_tokens, chunk)
        benefit = candidate.observed_count * reusable
        release_bonus = (
            self._save_release_weight
            * candidate.held_blocks
            * max(1, int(self.block_size))
        )
        age = max(0.0, now - candidate.enqueued_at)
        return (benefit + release_bonus) / cost_units + self._save_aging_weight * age

    def _candidate_sort_key(
        self, candidate: _SaveCandidate, now: float
    ) -> tuple[float, int, float, str, int]:
        return (
            -self._candidate_priority(candidate, now),
            candidate.dirty_tokens,
            candidate.enqueued_at,
            candidate.sid,
            candidate.generation,
        )

    def _meets_save_value_threshold(self, candidate: _SaveCandidate) -> bool:
        return candidate.observed_count >= self._save_min_observed_count

    def _commit_finished_save(self, candidate: _SaveCandidate) -> bool:
        if not candidate.finished or not self._meets_save_value_threshold(candidate):
            return False
        entry = self._save_tracker.get(candidate.sid)
        if entry is None or entry[0] is not candidate.seq:
            return False
        return self._try_reserve_save_candidate(
            candidate,
            allow_eviction=True,
        )

    def _forget_save_candidate(self, sid: str, seq) -> None:
        for mapping in (
            self._save_demand_keys,
            self._save_candidate_since,
            self._save_candidate_generation,
        ):
            record = mapping.get(sid)
            if record is not None and record[0] is seq:
                mapping.pop(sid, None)

    def drop_unadmitted_save(self, seq, *, reason: str) -> bool:
        """Drop save work that has never been handed to a worker.

        PAGE and SLOT form one checkpoint operation, so cleanup is all-or-none.
        An inflight operation is never cancelled here; its source may already
        be in use by a worker.
        """

        sid = str(seq.id)
        if sid in self._save_inflight or sid in self._sidecar_save_inflight:
            return False
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return False
        candidate = self._build_save_candidate(sid)
        dropped_tokens = 0 if candidate is None else candidate.dirty_tokens
        self._save_tracker.pop(sid, None)
        if self._save_committed.get(sid) is seq:
            self._save_committed.pop(sid, None)
        self._release_save_reservation(sid, owner=seq)
        if self._finished_save_requests.get(sid) is seq:
            self._finished_save_requests.pop(sid, None)
        self._finished_save_failed.discard(sid)
        self._failed_sidecar_saves.pop(sid, None)
        self._save_watermark_rollback.pop(sid, None)
        cached = self._sidecar_hash_cache.get(sid)
        if cached is not None and cached[0] is seq:
            self._sidecar_hash_cache.pop(sid, None)
        self._forget_save_candidate(sid, seq)
        if reason not in self._save_drop_totals:
            reason = "stale"
        self._save_drop_totals[reason] += 1
        self._save_drop_token_totals[reason] += dropped_tokens
        logger.info(
            "LMCache offload: dropped unadmitted save req=%s reason=%s tokens=%d",
            sid,
            reason,
            dropped_tokens,
        )
        return True

    def _settle_finished_save(self, sid: str) -> None:
        seq = self._finished_save_requests.get(sid)
        if seq is None:
            return
        if (
            sid in self._save_inflight
            or sid in self._sidecar_save_inflight
            or sid in self._save_committed
        ):
            return
        if sid in self._finished_save_failed:
            self.drop_unadmitted_save(seq, reason="terminal_failure")
            return
        candidate = self._build_save_candidate(sid)
        if candidate is None:
            entry = self._save_tracker.get(sid)
            if entry is not None and entry[0] is seq:
                self._save_tracker.pop(sid, None)
            self._finished_save_requests.pop(sid, None)
            self._finished_save_failed.discard(sid)
            self._failed_sidecar_saves.pop(sid, None)
            self._save_watermark_rollback.pop(sid, None)
            self._forget_save_candidate(sid, seq)
            return
        if self._commit_finished_save(candidate):
            return
        reason = (
            "low_value"
            if not self._meets_save_value_threshold(candidate)
            else "capacity"
        )
        self.drop_unadmitted_save(seq, reason=reason)

    def _emit_save_candidate(
        self,
        meta: LMCacheOffloadMetadata,
        candidate: _SaveCandidate,
    ) -> None:
        sid = candidate.sid
        seq = candidate.seq
        slot_save_spec = None
        if candidate.sidecar_candidate is not None:
            boundary, boundary_hash = candidate.sidecar_candidate
            slot_save_spec = SlotSaveSpec(
                boundary_tokens=boundary,
                boundary_block_hash=boundary_hash,
                source_group=int(seq.state_slot),
            )
        token_end = max(
            candidate.aligned if candidate.page_save_due else 0,
            (
                candidate.sidecar_candidate[0]
                if candidate.sidecar_candidate is not None
                else 0
            ),
        )
        logger.debug(
            "[OFFLOAD-SAVE-EMIT] seq=%s computed=%d num_prompt=%d "
            "aligned=%d saved=%d sidecar=%s observed=%d",
            seq.id,
            candidate.computed,
            int(seq.num_prompt_tokens),
            candidate.aligned,
            candidate.saved,
            candidate.sidecar_candidate,
            candidate.observed_count,
        )
        save_operation = self._next_save_operation(seq)
        committed_reservation = self._save_block_reservations.get(sid)
        if (
            committed_reservation is None
            or committed_reservation.seq is not seq
            or committed_reservation.generation != candidate.generation
            or not committed_reservation.committed
        ):
            return
        self._track_save_statistics(save_operation, candidate.dirty_tokens)
        self._refresh_save_reclaim_clock(seq)
        self._save_inflight_since[save_operation] = time.monotonic()
        self._save_operation_owners[save_operation] = seq
        meta.add_request(
            LMCacheReqMeta(
                req_id=seq.id,
                token_ids=list(seq.token_ids[:token_end]),
                block_ids=list(seq.block_table),
                save_spec=(
                    SaveSpec(skip_leading_tokens=candidate.saved, can_save=True)
                    if candidate.page_save_due
                    else None
                ),
                is_last_prefill=(candidate.computed >= int(seq.num_prompt_tokens)),
                slot_save_spec=slot_save_spec,
                save_operation=save_operation,
            )
        )
        entry = self._save_tracker.get(sid)
        if candidate.page_save_due and entry is not None and entry[0] is seq:
            self._save_watermark_rollback.setdefault(sid, {})[save_operation] = (
                seq,
                int(entry[1]),
            )
            entry[1] = candidate.aligned
            self._save_inflight.setdefault(sid, set()).add(save_operation)
        if candidate.sidecar_candidate is not None:
            self._sidecar_save_inflight[sid] = (
                save_operation,
                candidate.sidecar_candidate[0],
                candidate.sidecar_candidate[1],
            )
        self._save_committed.pop(sid, None)
        self._release_save_reservation(sid, owner=seq)
        self._save_block_reservations[save_operation] = replace(
            committed_reservation,
            committed=False,
        )

    def _clear_lookup_status(self, sid: str) -> None:
        if self._lookup_client is None:
            return
        try:
            self._lookup_client.clear_lookup_status(sid)
        except Exception:  # best-effort cleanup
            logger.debug(
                "LMCache offload: clear lookup status failed for %s",
                sid,
                exc_info=True,
            )

    def _clear_lookup_retry_state(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._pending_slot_loads.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        self._clear_lookup_status(sid)

    def _clear_pending_load(self, sid: str) -> None:
        self._load_specs.pop(sid, None)
        self._reqs_need_recv.pop(sid, None)
        self._handoff_loads.discard(sid)
        self._pending_slot_loads.pop(sid, None)
        self._load_save_floors.pop(sid, None)
        self._hit_save_floors.pop(sid, None)
        self._clear_lookup_status(sid)

    def _decide_load_after_alloc(
        self, seq, ls: LoadSpec
    ) -> tuple[bool, str, int, int, int, int]:
        """Choose whether the post-allocation LMCache PAGE load is safe.

        Version 1 cannot merge a newly restored full SLOT at boundary ``lmc``
        with a stateful request that already owns an HBM prefix below that
        boundary. Such a merge would pair PAGE and SLOT state from different
        logical checkpoints, so any nonzero stateful HBM floor skips LMCache
        loading and recomputes instead.
        """
        hbm = int(getattr(seq, "num_cached_tokens", ls.hbm_cached_tokens))
        lmc = int(ls.lmcache_cached_tokens)
        ls.hbm_cached_tokens = hbm
        chunk = int(self.chunk_size or 256)
        need = lmc - hbm
        if lmc <= hbm:
            return False, "hbm_satisfies_after_alloc", hbm, lmc, need, chunk
        # Fail closed: PAGE above a nonzero HBM floor cannot be combined with
        # one request-level SLOT snapshot in the version-1 stateful protocol.
        if bool(getattr(seq, "has_per_req_cache", False)) and hbm > 0:
            return False, "stateful_nonzero_hbm_floor", hbm, lmc, need, chunk
        if hbm % chunk != 0:
            return False, "unaligned_hbm_prefill", hbm, lmc, need, chunk
        min_load = int(getattr(self, "_min_load_tokens", 8192))
        if need < min_load:
            return False, "too_small", hbm, lmc, need, chunk
        return True, "aligned_large_hit", hbm, lmc, need, chunk

    def adjust_prefill_chunk_after_alloc(self, seq, chunk: int) -> int:
        sid = str(seq.id)
        start = int(getattr(seq, "num_cached_tokens", 0))
        adjusted = int(chunk)

        if sid in self._handoff_loads:
            handoff = getattr(seq, "offload_handoff_boundary_tokens", None)
            if handoff is not None:
                handoff_limit = int(handoff) - start
                if handoff_limit > 0:
                    adjusted = min(adjusted, handoff_limit)

        sidecar = self._next_pending_sidecar_boundary(
            seq,
            start,
            start + adjusted,
        )
        if sidecar is not None:
            adjusted = min(adjusted, sidecar[0] - start)
        return max(1, adjusted)

    def cancel_pending_load(self, seq) -> None:
        """Cancel load-only state for one concrete request lifecycle."""
        sid = str(seq.id)
        if self._load_lifecycles.get(sid) is not seq:
            return
        self._clear_pending_load(sid)
        active = self._active_load_operations.get(sid)
        if active is not None and active[0] is seq:
            self._active_load_operations.pop(sid, None)
            operation = active[1]
            self._cancel_load_statistics(operation)
            if getattr(seq, "_load_operation", None) == operation:
                delattr(seq, "_load_operation")
        self._active_slot_loads.pop(sid, None)

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = LMCacheOffloadMetadata()

        # Loads
        logger.debug("[OFFLOAD-BUILD] reqs_need_recv=%d", len(self._reqs_need_recv))
        loading_sids: set[str] = set()
        load_items = list(self._reqs_need_recv.items()) if self._do_load else []
        for sid, seq in load_items:
            ls = self._load_specs.pop(sid, None)
            if ls is None or not ls.can_load:
                logger.debug(
                    "[OFFLOAD-LOAD-SKIP] seq=%s ls=%s can_load=%s",
                    sid,
                    ls is not None,
                    getattr(ls, "can_load", None),
                )
                continue
            # ★ Use the REAL HBM-cached count as the load floor.
            # get_num_new_matched_tokens runs BEFORE the prefix-cache match in
            # block_manager.allocate, so seq.num_cached_tokens was stale (often
            # 0) when the LoadSpec was recorded. By now (post-allocate) it is the
            # true HBM hit. Loading below this floor would overwrite HBM
            # prefix-cache blocks (possibly shared with other seqs) -> output
            # corruption. So load only [hbm_cached, offload_hit).
            should_load, reason, hbm, lmc, need, chunk = self._decide_load_after_alloc(
                seq, ls
            )
            if not should_load:
                self._mark_load_skip(seq, reason, hbm, lmc, need, chunk)
                self._clear_pending_load(sid)
                continue
            slot_load_spec = None
            if bool(getattr(seq, "has_per_req_cache", False)):
                pending_slot = self._pending_slot_loads.get(sid)
                destination_group = getattr(seq, "state_slot", -1)
                if (
                    pending_slot is None
                    or not isinstance(destination_group, int)
                    or destination_group < 0
                    or pending_slot[0] != lmc
                ):
                    logger.warning(
                        "LMCache offload: rejecting incomplete stateful load "
                        "metadata for seq %s",
                        seq.id,
                    )
                    self._clear_pending_load(sid)
                    continue
                boundary, boundary_hash = pending_slot
                slot_load_spec = SlotLoadSpec(
                    boundary_tokens=boundary,
                    boundary_block_hash=boundary_hash,
                    destination_group=destination_group,
                )
                self._active_slot_loads[sid] = pending_slot
            # num_cached after load = max(HBM, offload); never drop below HBM.
            # Persist the physical load start on the sequence. The scheduler
            # combines it with offload_loaded_tokens after all TP workers
            # succeed to publish the restored GPU prefix.
            seq.offload_load_start_tokens = hbm
            seq.offload_loaded_tokens = self._claim_after_load(seq, hbm, lmc)
            # req_id MUST be the raw seq.id (the type the scheduler compares
            # against in _update_waiting_for_remote_kv); str(seq.id) is only for
            # LMCache's lookup/pin API. A str here silently never wakes the seq.
            logger.debug(
                "[OFFLOAD-LOAD-EMIT] seq=%s hbm_cached=%d lmc_cached=%d "
                "offload_loaded=%d need=%d min_load=%d nblocks=%d reason=aligned_large_hit",
                seq.id,
                hbm,
                lmc,
                seq.offload_loaded_tokens,
                need,
                int(getattr(self, "_min_load_tokens", 8192)),
                len(list(seq.block_table)),
            )
            loading_sids.add(sid)
            self._load_save_floors[sid] = self._chunk_floor(hbm)
            load_operation = LoadOperationId(seq.id, self._load_nonce)
            self._load_nonce += 1
            seq.offload_loaded = False
            seq.offload_load_failed = False
            seq._load_operation = load_operation
            self._active_load_operations[sid] = (seq, load_operation)
            self._track_load_statistics(load_operation, lmc - hbm)
            meta.add_request(
                LMCacheReqMeta(
                    req_id=seq.id,
                    token_ids=list(seq.token_ids[:lmc]),
                    block_ids=list(seq.block_table),
                    load_spec=ls,
                    slot_load_spec=slot_load_spec,
                    load_operation=load_operation,
                )
            )
            self._pending_slot_loads.pop(sid, None)
        meta.lookup_requests_in_step = [
            sid for sid in self._lookup_in_step if sid not in self._handoff_loads
        ]
        # Saves. Admit/replace the complete candidate set before
        # dispatch so a newly hotter candidate can still evict lower-value
        # committed work. Dispatching first would make that work inflight and
        # therefore correctly, but prematurely, non-evictable.
        if self._do_save:
            now = time.monotonic()
            for sid, seq in list(self._save_committed.items()):
                entry = self._save_tracker.get(sid)
                if entry is None or entry[0] is not seq:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    continue
                candidate = self._build_save_candidate(sid, now=now)
                if candidate is None:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    self._settle_finished_save(sid)

            candidates = [
                candidate
                for sid in self._save_tracker
                if (candidate := self._build_save_candidate(sid, now=now)) is not None
                and self._meets_save_value_threshold(candidate)
            ]
            candidates.sort(
                key=lambda candidate: self._candidate_sort_key(candidate, now)
            )

            for candidate in candidates:
                sid = candidate.sid
                if sid in self._save_committed:
                    continue
                if sid in self._reqs_need_recv or sid in loading_sids:
                    continue
                if sid in self._save_inflight or sid in self._sidecar_save_inflight:
                    continue
                if not self._try_reserve_save_candidate(
                    candidate,
                    allow_eviction=True,
                    now=now,
                ):
                    self.drop_unadmitted_save(candidate.seq, reason="capacity")

            committed = []
            for sid, seq in list(self._save_committed.items()):
                candidate = self._build_save_candidate(sid, now=now)
                if candidate is None or candidate.seq is not seq:
                    self._save_committed.pop(sid, None)
                    self._release_save_reservation(sid, owner=seq)
                    continue
                committed.append(candidate)
            committed.sort(
                key=lambda candidate: self._candidate_sort_key(candidate, now)
            )
            for candidate in committed:
                sid = candidate.sid
                if sid in self._reqs_need_recv or sid in loading_sids:
                    continue
                if sid in self._save_inflight or sid in self._sidecar_save_inflight:
                    continue
                self._emit_save_candidate(meta, candidate)
        dispatched = set(meta.lookup_requests_in_step)
        self._lookup_in_step = [
            sid for sid in self._lookup_in_step if sid not in dispatched
        ]
        self._reqs_need_recv.clear()
        return meta

    def _has_pending_sidecar_save(self, seq) -> bool:
        sid = str(seq.id)
        if sid not in self._save_tracker:
            return False
        computed = min(
            int(getattr(seq, "num_cached_tokens", 0)),
            int(getattr(seq, "num_prompt_tokens", 0)),
        )
        return self._sidecar_save_candidate(seq, computed) is not None

    def _save_retries_expired(self, sid: str) -> bool:
        """Stop new work after a finished request makes no save progress.

        Use the existing abandon window, independently of the per-dispatch
        source-retention clock. In-flight PAGE/SLOT operations still own the
        blocks until their completion or the normal stalled-copy reclaim.
        """
        progress_at = self._finished_save_progress_at.get(sid)
        if progress_at is None:
            return False
        timeout = self.save_abandon_timeout_s()
        return timeout > 0 and time.monotonic() - progress_at >= timeout

    def should_defer_free(self, seq) -> bool:
        if self._has_active_load(seq):
            return True
        if not self._do_save:
            return False
        sid = str(seq.id)
        return (
            sid in self._save_inflight
            or sid in self._sidecar_save_inflight
            or self._save_committed.get(sid) is seq
            or (
                not self._save_retries_expired(sid)
                and (self._has_pending_save(seq) or self._has_pending_sidecar_save(seq))
            )
        )

    def has_pending_work(self) -> bool:
        """True while a load still needs dispatch or a save is unreported.

        Feeds ``EngineCore.has_pending_kv_work()``, so it reads only state
        that clears itself: ``_reqs_need_recv`` is emptied by every
        ``build_connector_meta`` and ``_save_inflight`` by ``save_finished``.
        A priority save committed after its request finished is included even
        before its next-step dispatch, so an idle engine keeps producing
        connector metadata until that reservation becomes an operation.
        """
        return bool(
            self._reqs_need_recv
            or self._save_committed
            or self._save_inflight
            or self._sidecar_save_inflight
        )

    @staticmethod
    def _watermark_sid(operation) -> str:
        return str(getattr(operation, "req_id", operation))

    def _forget_save_watermark(self, operation) -> tuple[object, int] | None:
        """Drop one operation's contingency record, returning what it held."""
        sid = self._watermark_sid(operation)
        pending = self._save_watermark_rollback.get(sid)
        if pending is None:
            return None
        previous = pending.pop(operation, None)
        if not pending:
            self._save_watermark_rollback.pop(sid, None)
        return previous

    def _release_save_watermark(self, operation) -> None:
        """Retire the record of a save that landed; its advance was correct."""
        previous = self._forget_save_watermark(operation)
        sid = self._watermark_sid(operation)
        if previous is not None and sid in self._finished_save_progress_at:
            self._finished_save_progress_at[sid] = time.monotonic()

    def _rollback_save_watermark(self, operation) -> None:
        """Take back the watermark advance of a save that never landed.

        The advance told every later incremental save to skip that range. With
        nothing persisting it the range is a permanent hole in the prefix:
        ``_page_boundary_visible`` can never see a PAGE hit reach
        ``boundary_tokens``, so SLOT publication times out forever and no prefix
        is ever complete -- which is why raising CPU capacity cannot help, one
        saturation poisons the sequence. Lowering the watermark makes the next
        step re-emit the range. Lowering past a *later* save that did land costs
        one redundant write and nothing else: stores are idempotent, and the
        prefix has to be contiguous anyway, so everything beyond the hole was
        unusable regardless.
        """
        rollback = self._forget_save_watermark(operation)
        if rollback is None:
            return
        seq, previous = rollback
        sid = self._watermark_sid(operation)
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return
        current = int(entry[1])
        if current <= previous:
            return
        entry[1] = previous
        logger.warning(
            "LMCache offload: save watermark rolled back req=%s %d -> %d "
            "(save did not land); those PAGEs will be saved again",
            sid,
            current,
            previous,
        )

    def save_finished(self, req_id) -> None:
        if isinstance(req_id, SaveOperationId):
            sid = str(req_id.req_id)
            owner = self._save_operation_owners.get(req_id)
            inflight = self._save_inflight.get(sid)
            if inflight is not None:
                inflight.discard(req_id)
                if not inflight:
                    self._save_inflight.pop(sid, None)
            self._save_inflight_since.pop(req_id, None)
            self._finish_save_statistics(req_id)
            self._retire_save_operation_owner(req_id)
            if owner is not None and self._finished_save_requests.get(sid) is owner:
                self._settle_finished_save(sid)
            return

        sid = str(req_id)
        inflight = self._save_inflight.get(sid)
        if inflight and any(
            isinstance(operation, SaveOperationId) for operation in inflight
        ):
            return
        if sid in self._sidecar_save_inflight:
            return
        retired = self._save_inflight.pop(sid, None)
        if retired is not None:
            for operation in retired:
                self._save_inflight_since.pop(operation, None)
                self._retire_save_operation_owner(operation)
        self._finish_save_statistics(req_id)
        self._settle_finished_save(sid)

    def abandon_save(self, req_id) -> None:
        """Force-drop a save the scheduler reclaimed after it stalled.

        The completion path (`save_finished`) is precise: it discards one exact
        `SaveOperationId` and leaves the rest of the request's inflight set. This
        is the opposite need. `_reconcile_stalled_deferred_saves` has already
        freed the blocks of a save the backend never reported, and holds only
        the raw request id, so drop the *whole* request unconditionally -- the
        page set and the SLOT sidecar. Without this the entries linger,
        `should_defer_free` stays True and `has_pending_work` never clears, and
        the engine busy-loops with every GPU idle (the DSV4 twin of the dense
        stall). Not a completion: the bytes were never persisted, so every
        operation's statistics are *cancelled*, not finished, and the tracker
        entry is dropped so the save loop cannot re-emit against freed blocks.
        """
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        retired_operations: set[SaveOperationId] = set()
        inflight = self._save_inflight.pop(sid, None)
        if inflight is not None:
            for operation in inflight:
                retired_operations.add(operation)
                self._save_inflight_since.pop(operation, None)
                self._save_operation_owners.pop(operation, None)
                self._cancel_save_statistics(operation)
        sidecar = self._sidecar_save_inflight.pop(sid, None)
        if sidecar is not None:
            retired_operations.add(sidecar[0])
            self._save_inflight_since.pop(sidecar[0], None)
            self._save_operation_owners.pop(sidecar[0], None)
            self._cancel_save_statistics(sidecar[0])
        for operation in retired_operations:
            self._release_save_reservation(operation)
        committed = self._save_committed.pop(sid, None)
        if committed is not None:
            self._release_save_reservation(sid, owner=committed)
        finished = self._finished_save_requests.pop(sid, None)
        seq = finished if finished is not None else committed
        if seq is None:
            entry = self._save_tracker.get(sid)
            seq = None if entry is None else entry[0]
        self._save_tracker.pop(sid, None)
        self._finished_save_failed.discard(sid)
        if seq is not None:
            self._forget_save_candidate(sid, seq)
        self._finished_save_progress_at.pop(sid, None)
        # The tracker entry is gone, so there is no longer a watermark to take
        # back; keeping the records would only leak.
        self._save_watermark_rollback.pop(sid, None)

    def release_stalled_save(self, seq) -> None:
        """Drop bookkeeping for a stall-escaped save the scheduler is freeing.

        No-op on DSV4: like dense, its `should_defer_free` has no stall escape,
        so a request with a pending page or sidecar save always defers and is
        never preemptable. Defined so every offload impl answers the scheduler's
        `release_stalled_save` forward uniformly (the mixin declares it abstract).
        """

    def connector_completion(self, completion: ConnectorCompletion) -> bool:
        """Apply one TP-aggregated completion owned by the DSV4 scheduler."""

        if completion.channel == DSV4_PAGE_SAVE_CHANNEL:
            # `process_completions` runs this before `save_finished`, so the
            # watermark is corrected while the operation is still in
            # `_save_inflight`; clearing it there then lets the very next
            # `build_connector_meta` re-emit the range.
            if completion.succeeded:
                self._release_save_watermark(completion.operation_id)
            else:
                sid = self._watermark_sid(completion.operation_id)
                inflight = self._save_inflight.get(sid, set())
                sidecar = self._sidecar_save_inflight.get(sid)
                owns_operation = completion.operation_id in inflight or (
                    sidecar is not None and sidecar[0] == completion.operation_id
                )
                owner = self._save_operation_owners.get(completion.operation_id)
                if (
                    owns_operation
                    and owner is not None
                    and self._finished_save_requests.get(sid) is owner
                ):
                    self._finished_save_failed.add(sid)
                self._rollback_save_watermark(completion.operation_id)
                # Nothing was persisted, so these bytes must not land in
                # `total_saved_tokens`; cancelling first makes the
                # `_finish_save_statistics` in `save_finished` a no-op.
                self._cancel_save_statistics(completion.operation_id)
            return True
        if completion.channel != DSV4_CHECKPOINT_SAVE_CHANNEL:
            return False
        if completion.succeeded:
            self.sidecar_save_finished(completion.operation_id)
        else:
            self.sidecar_save_failed(completion.operation_id)
        return True

    def sidecar_save_finished(self, req_id) -> None:
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        inflight = self._sidecar_save_inflight.get(sid)
        if inflight is None:
            return
        if isinstance(req_id, SaveOperationId) and inflight[0] != req_id:
            return
        if not isinstance(req_id, SaveOperationId) and isinstance(
            inflight[0], SaveOperationId
        ):
            return
        operation = inflight[0]
        owner = self._save_operation_owners.get(operation)
        self._sidecar_save_inflight.pop(sid, None)
        identity = (inflight[1], inflight[2])
        self._committed_sidecar_hashes.add(identity[1])
        failed = self._failed_sidecar_saves.get(sid)
        if failed is not None:
            failed.discard(identity)
            if not failed:
                self._failed_sidecar_saves.pop(sid, None)
        self._retire_save_operation_owner(operation)
        if owner is not None and self._finished_save_requests.get(sid) is owner:
            self._settle_finished_save(sid)

    def sidecar_save_failed(self, req_id) -> None:
        sid = str(req_id.req_id if isinstance(req_id, SaveOperationId) else req_id)
        inflight = self._sidecar_save_inflight.get(sid)
        if inflight is None:
            return
        if isinstance(req_id, SaveOperationId) and inflight[0] != req_id:
            return
        if not isinstance(req_id, SaveOperationId) and isinstance(
            inflight[0], SaveOperationId
        ):
            return
        operation = inflight[0]
        owner = self._save_operation_owners.get(operation)
        if owner is not None and self._finished_save_requests.get(sid) is owner:
            self._finished_save_failed.add(sid)
        self._sidecar_save_inflight.pop(sid, None)
        identity = (inflight[1], inflight[2])
        self._failed_sidecar_saves.setdefault(sid, set()).add(identity)
        self._retire_save_operation_owner(operation)
        if owner is not None and self._finished_save_requests.get(sid) is owner:
            self._settle_finished_save(sid)

    def load_failed(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            return False
        self._finish_load_statistics(req_id, succeeded=False)
        active_slot = self._active_slot_loads.pop(sid, None)
        if active_slot is not None:
            self._committed_sidecar_hashes.discard(active_slot[1])
        floor = self._load_save_floors.get(sid)
        entry = self._save_tracker.get(sid)
        if floor is not None and entry is not None:
            # The LMCache hit was not actually loaded. Let the recomputed
            # [HBM, LMC) chunks be saved again instead of permanently treating
            # them as already persisted.
            entry[1] = self._chunk_floor(floor)
        self._record_failed_load_attempt(sid)
        self._clear_pending_load(sid)
        return True

    def load_finished(self, req_id) -> bool:
        sid = str(req_id.req_id if isinstance(req_id, LoadOperationId) else req_id)
        active = self._active_load_operations.get(sid)
        if isinstance(req_id, LoadOperationId):
            if active is None or active[1] != req_id:
                return False
            self._active_load_operations.pop(sid, None)
        elif active is not None:
            return False
        self._finish_load_statistics(req_id, succeeded=True)
        self._active_slot_loads.pop(sid, None)
        self._load_save_floors.pop(sid, None)
        return True

    def get_statistics(self) -> dict[str, int | float]:
        statistics = super().get_statistics()
        now = time.monotonic()
        candidates = []
        for sid in self._save_tracker:
            if (
                sid in self._save_inflight
                or sid in self._sidecar_save_inflight
                or sid in self._save_committed
            ):
                continue
            candidate = self._build_save_candidate(sid, now=now)
            if candidate is not None:
                candidates.append(candidate)

        finished = [
            candidate
            for candidate in candidates
            if self._finished_save_requests.get(candidate.sid) is candidate.seq
        ]
        candidate_scores = [
            self._candidate_priority(candidate, now) for candidate in candidates
        ]
        reserved_blocks, pinned_blocks, budget_used = self._save_block_usage()
        budget = self._save_pin_budget_blocks
        total_blocks = self._save_pin_total_blocks
        statistics.update(
            save_candidates=len(candidates),
            save_candidates_finished=len(finished),
            save_committed=len(self._save_committed),
            save_admitted=self.total_save_admitted,
            save_candidate_wait_seconds=max(
                (max(0.0, now - candidate.enqueued_at) for candidate in candidates),
                default=0.0,
            ),
            save_priority_score=max(candidate_scores, default=0.0),
            save_inflight_wait_seconds=max(
                (
                    max(0.0, now - started)
                    for started in self._save_inflight_since.values()
                ),
                default=0.0,
            ),
            save_pinned_blocks=pinned_blocks,
            save_pinned_tokens=pinned_blocks * int(self.block_size),
            save_pin_budget_blocks=0 if budget is None else budget,
            save_reserved_blocks=reserved_blocks,
            save_pinned_ratio=(budget_used / total_blocks if total_blocks > 0 else 0.0),
            save_budget_available_blocks=(
                0 if budget is None else max(0, budget - budget_used)
            ),
            save_budget_rejected=self.total_save_budget_rejected,
            save_budget_rejected_blocks=self.total_save_budget_rejected_blocks,
            save_budget_evicted=self.total_save_budget_evicted,
            save_budget_evicted_blocks=self.total_save_budget_evicted_blocks,
            save_oversized=self.total_save_oversized,
            deferred_free_requests=len(
                {
                    reservation.sid
                    for reservation in self._save_block_reservations.values()
                    if self._finished_save_requests.get(reservation.sid)
                    is reservation.seq
                }
            ),
        )
        for reason, total in self._save_drop_totals.items():
            statistics[f"save_dropped_{reason}"] = total
            statistics[f"save_dropped_tokens_{reason}"] = self._save_drop_token_totals[
                reason
            ]
        return statistics

    def _drop_finished_save_state(self, sid: str, seq) -> None:
        """Forget a finished request's save bookkeeping.

        Keyed on the *blocks*, not the request: `_save_tracker` holds the
        `Sequence` and is what `build_connector_meta`'s save loop iterates
        (reading `seq.block_table` straight out of it), so it may only be
        dropped once nothing can still read that block table. The sidecar
        failure set and the watermark records are scoped to the same tracker
        entry and go with it. Both terminals route here -- `request_finished`
        when the free was not deferred at all, `source_blocks_released` when
        it was.
        """
        entry = self._save_tracker.get(sid)
        if entry is None or entry[0] is not seq:
            return
        self._save_tracker.pop(sid, None)
        self._finished_save_progress_at.pop(sid, None)
        self._failed_sidecar_saves.pop(sid, None)
        self._save_watermark_rollback.pop(sid, None)
        if self._finished_save_requests.get(sid) is seq:
            self._finished_save_requests.pop(sid, None)
        self._finished_save_failed.discard(sid)
        committed = self._save_committed.get(sid)
        if committed is seq:
            self._save_committed.pop(sid, None)
            self._release_save_reservation(sid, owner=seq)
        self._forget_save_candidate(sid, seq)

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        self._release_failed_load_attempt(sid, seq)
        if self._load_lifecycles.get(sid) is seq:
            self._clear_pending_load(sid)
            self._active_slot_loads.pop(sid, None)
            active = self._active_load_operations.get(sid)
            if active is not None and active[0] is seq:
                self._active_load_operations.pop(sid, None)
                self._cancel_load_statistics(active[1])
            self._load_lifecycles.pop(sid, None)
        cached = self._sidecar_hash_cache.get(sid)
        if cached is not None and cached[0] is seq:
            self._sidecar_hash_cache.pop(sid, None)
        entry = self._save_tracker.get(sid)
        if entry is not None and entry[0] is seq:
            self._finished_save_progress_at.setdefault(sid, time.monotonic())
            self._finished_save_requests[sid] = seq
            self._refresh_finished_save_reservations(seq)
            if (
                sid in self._save_inflight
                or sid in self._sidecar_save_inflight
                or self._save_committed.get(sid) is seq
            ):
                # Already committed/dispatched. Its terminal callbacks will
                # either reserve one residual tail or drop it.
                pass
            else:
                candidate = self._build_save_candidate(sid)
                if candidate is None:
                    self._settle_finished_save(sid)
                elif not self._meets_save_value_threshold(candidate):
                    self.drop_unadmitted_save(seq, reason="low_value")
                elif not self._commit_finished_save(candidate):
                    self.drop_unadmitted_save(seq, reason="capacity")
        if not self.should_defer_free(seq):
            self._drop_finished_save_state(sid, seq)
        if hasattr(seq, "_load_operation"):
            delattr(seq, "_load_operation")

    def source_blocks_released(self, seq) -> None:
        """Terminal for the deferred path: the blocks are back in the pool.

        DSV4 always defers -- it has no early-release exit and so no
        `_finish_retired_request` -- which makes this the *only* terminal its
        normal completion path ever reaches. Without it every request leaks a
        tracker entry (and the `Sequence` it pins), and a request whose
        watermark was rolled back by a failed save keeps `page_save_due` True
        forever, re-emitting a save against a freed, reusable block table on
        every step.
        """
        self._drop_finished_save_state(str(seq.id), seq)


__all__ = [
    "DSV4OffloadConnector",
    "DSV4OffloadScheduler",
]
