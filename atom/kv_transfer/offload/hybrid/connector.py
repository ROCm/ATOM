# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The ``hybrid`` layout family connector — profile-driven terminal offload bundles.

Selected by ``--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":
"offload"}'`` when the model resolves to the ``hybrid`` family (see
``offload/dispatch.py``), plus LMCache env (``LMCACHE_LOCAL_CPU=True`` etc.).
Unlike the ``dense`` family (token-chunked MHA/MLA), this stores one opaque
N-component *offload bundle* per aligned terminal boundary. The component set +
geometry + cadence come from a :class:`HybridProfile` (``offload/profiles/``);
DSV4 is ``profiles/dsv4.py``. See ``dsv4-lmcache-bundle-plan.md``.

SAVE: capture at prefill end (``B % 128 == 0``). The CSA-state snapshot
(``gather_slot``) runs synchronously on the RPC thread (before the next forward
overwrites the compute slot); compressed/SWA gather + D2H + store run on a
background executor.

LOAD: on a prefix hit for the largest stored 128-aligned ``B < prompt_len``, the
scheduler parks the seq, the worker validates + scatters compressed KV + SWA tail
into the freshly-allocated pages and the CSA state into a pool slot then
``scatter_slot`` into the compute slot, and the seq resumes suffix prefill
``[B, prompt_len)``. HCA ring is neither saved nor restored (compressed-only).
"""

from __future__ import annotations

import logging
import os

import torch

from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata
from atom.kv_transfer.offload._offload_common import (
    OffloadWorkerMixin,
    build_offload_engine,
)
from atom.kv_transfer.offload.hybrid.gpu_connector import (
    BundleGPUConnector,
)
from atom.kv_transfer.offload.hybrid.admission import CheckpointAdmission
from atom.kv_transfer.offload.hybrid.policy import (
    candidate_boundaries,
    checkpoint_key,
    select_resume_boundary,
    should_save_at,
)
from atom.kv_transfer.offload.hybrid.store import LMCacheBundleStore
from atom.kv_transfer.offload.hybrid.kv_bundle_codec import (
    BundleCodec,
    BundleError,
)
from atom.kv_transfer.offload.hybrid.profiles import select_profile
from atom.kv_transfer.offload.hybrid.profiles.base import RegionKind, BundleSources

logger = logging.getLogger("atom")


# =====================================================================
# Worker side
# =====================================================================
class HybridOffloadConnector(OffloadWorkerMixin, KVConnectorBase):
    def __init__(self, config) -> None:
        self._config = config
        self._init_worker_common(config)  # executors, lock, done/failed tallies
        self._engine = None
        self._store: LMCacheBundleStore | None = None
        self._profile = None
        self._sources: BundleSources | None = None
        self._gpu: BundleGPUConnector | None = None
        self._codec: BundleCodec | None = None
        self._admission: CheckpointAdmission | None = None
        self._has_staging = False
        self._rank = 0

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        if transfer_tensors is None:
            logger.warning("offload[hybrid]: no transfer_tensors; connector disabled")
            return

        tp = get_tp_group()
        rank, world = tp.rank_in_group, tp.world_size
        self._rank = rank

        # Profile-driven: select_profile picks the HybridProfile from config (same
        # signal the scheduler uses). A profile declares the ordered unit
        # components + their region kinds (BLOCK/SWA/SLOT/STAGING) + geometry +
        # cadence. Adding a hybrid-KV target = a new profile, not a new connector.
        self._profile = select_profile(self._config, world=world, rank=rank)
        geom = self._profile.build_geometry()
        self._has_staging = any(
            spec.kind is RegionKind.STAGING for spec in self._profile.components
        )

        # Reuse the shared opaque-uint8 engine build; the unit store uses
        # storage_manager.allocate with explicit unit sizes, so the nominal
        # bytes_per_block here only affects unused bookkeeping.
        self._engine, _cfg, _meta = build_offload_engine(
            self._config,
            engine_id=f"atom-offload-hybrid-{rank}",
            block_size=self._profile.block_size,
            bytes_per_block=256,
            gpu_connector_factory=lambda cfg, meta: _NoopGPUConnector(),
            world=world,
            rank=rank,
        )

        self._codec = BundleCodec(geom)
        device = torch.device("cuda", torch.cuda.current_device())
        self._gpu = BundleGPUConnector(self._codec, device=device)
        self._sources = BundleSources(self._profile, transfer_tensors)
        self._store = LMCacheBundleStore(
            self._engine,
            model_name=self._profile.model_tag,
            world_size=world,
            worker_id=rank,
        )
        max_inflight = int(os.environ.get("OFFLOAD_MAX_INFLIGHT_SAVES", "4"))
        self._admission = CheckpointAdmission(
            state_pool_size=max(1, self._sources.staging_pool_size),
            max_inflight_saves=max_inflight,
        )
        _ex_nblocks = -(-16384 // self._profile.block_size)
        example_unit = self._sources.unit_bytes(
            block_table=list(range(_ex_nblocks)),
            swa_block_table=[0],
            B=16384,
            slot_id=0,
            pool_idx=0 if self._has_staging else -1,
        )
        logger.info(
            "offload[hybrid:%s] worker rank=%d ready: components=%s state_pool=%d "
            "has_staging=%s example_unit@B16384=%d max_inflight=%d save=%s load=%s",
            self._profile.model_tag, rank, self._profile.component_names(),
            self._sources.staging_pool_size, self._has_staging, example_unit,
            max_inflight, self._do_save, self._do_load,
        )

    # -- per-step (RPC thread) -------------------------------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, HybridOffloadMetadata) or self._sources is None:
            return
        for req in metadata.requests:
            if req.op == "save" and self._do_save:
                self._begin_and_submit_save(req)
            elif req.op == "load" and self._do_load:
                self._load_executor.submit(self._do_load_req, req)

    # -- SAVE ------------------------------------------------------------
    def _begin_and_submit_save(self, req: "HybridReqMeta") -> None:
        s = self._sources
        assert s is not None and self._admission is not None and self._profile is not None
        if not self._profile.is_checkpoint_boundary(req.B):
            return
        if self._store is not None and self._store.contains(req.key):
            with self._lock:
                self._done_save.add(req.req_id)
            return
        pool_idx = self._admission.try_admit()
        if pool_idx is None:
            logger.debug("offload[hybrid]: admission skip req=%s B=%d", req.req_id, req.B)
            return
        live_swa = [int(b) for b in req.swa_block_ids if int(b) >= 0]
        try:
            # STAGING component (e.g. DSV4 CSA overlap state): snapshot the compute
            # slot into the pool slot on the RPC thread before the next forward
            # overwrites it. Profiles without staging have gather_slot=None.
            if self._has_staging and s.gather_slot is not None:
                s.gather_slot(int(req.compute_slot), int(pool_idx))
                torch.cuda.current_stream().synchronize()
        except Exception:
            logger.exception("offload[hybrid]: gather_slot failed req=%s", req.req_id)
            self._admission.complete(pool_idx)
            return
        self._save_executor.submit(self._finish_save, req, pool_idx, live_swa)

    def _finish_save(self, req: "HybridReqMeta", pool_idx: int, live_swa: list[int]) -> None:
        s, gpu, store = self._sources, self._gpu, self._store
        try:
            components = s.build_components(
                block_table=req.block_ids, swa_block_table=req.swa_block_ids,
                B=req.B, slot_id=req.compute_slot, pool_idx=pool_idx,
            )
            nbytes = s.unit_bytes(
                block_table=req.block_ids, swa_block_table=req.swa_block_ids,
                B=req.B, slot_id=req.compute_slot, pool_idx=pool_idx,
            )
            host = torch.empty(nbytes, dtype=torch.uint8)
            gpu.save_bundle(memory_obj=host, components=components, boundary_B=req.B)
            ok = store.put(req.key, host)
            logger.info(
                "offload[hybrid] SAVE %s rank=%d req=%s B=%d unit_bytes=%d key=%s",
                "ok" if ok else "STORE-FAILED",
                self._rank, req.req_id, req.B, nbytes, req.key,
            )
        except Exception:
            logger.exception("offload[hybrid]: save failed req=%s B=%d", req.req_id, req.B)
        finally:
            self._admission.complete(pool_idx)
            with self._lock:
                self._done_save.add(req.req_id)

    # -- LOAD ------------------------------------------------------------
    def _do_load_req(self, req: "HybridReqMeta") -> None:
        s, gpu, store = self._sources, self._gpu, self._store
        assert s is not None and self._admission is not None
        ok = False
        pool_idx = self._admission.try_admit()
        if pool_idx is None:
            logger.warning("offload[hybrid]: load slot exhausted req=%s => recompute", req.req_id)
            with self._lock:
                self._failed_load.add(req.req_id)
            return
        try:
            host = store.get(req.key)
            if host is None:
                logger.warning("offload[hybrid]: LOAD miss key=%s => recompute", req.key)
            else:
                components = s.build_components(
                    block_table=req.block_ids, swa_block_table=req.swa_block_ids,
                    B=req.B, slot_id=req.compute_slot, pool_idx=pool_idx,
                )
                gpu.load_bundle(
                    memory_obj=host, components=components, expect_boundary_B=req.B,
                )
                # STAGING component: drain the loaded pool slot into the compute
                # slot. Profiles without staging have scatter_slot=None.
                if self._has_staging and s.scatter_slot is not None:
                    s.scatter_slot(int(req.compute_slot), int(pool_idx))
                    torch.cuda.current_stream().synchronize()
                ok = True
                logger.info(
                    "offload[hybrid] LOAD ok rank=%d req=%s B=%d key=%s",
                    self._rank, req.req_id, req.B, req.key,
                )
        except BundleError:
            logger.warning("offload[hybrid]: LOAD fail-closed req=%s key=%s", req.req_id, req.key)
        except Exception:
            logger.exception("offload[hybrid]: load errored req=%s", req.req_id)
        finally:
            self._admission.complete(pool_idx)
            with self._lock:
                (self._done_load if ok else self._failed_load).add(req.req_id)

    # get_finished / get_finished_recv_blocks inherited from OffloadWorkerMixin.


class _NoopGPUConnector:
    def from_gpu(self, *a, **k) -> None: ...
    def to_gpu(self, *a, **k) -> None: ...
    def batched_from_gpu(self, *a, **k) -> None: ...
    def batched_to_gpu(self, *a, **k) -> None: ...


# =====================================================================
# Metadata
# =====================================================================
class HybridReqMeta:
    __slots__ = ("req_id", "op", "block_ids", "swa_block_ids", "compute_slot", "B", "key")

    def __init__(self, *, req_id, op, block_ids, swa_block_ids, compute_slot, B, key):
        self.req_id = req_id
        self.op = op
        self.block_ids = block_ids
        self.swa_block_ids = swa_block_ids
        self.compute_slot = int(compute_slot)
        self.B = int(B)
        self.key = key


class HybridOffloadMetadata(ConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[HybridReqMeta] = []

    def add_request(self, meta: HybridReqMeta) -> None:
        self.requests.append(meta)


# =====================================================================
# Scheduler side
# =====================================================================
class HybridOffloadScheduler(KVConnectorSchedulerBase):
    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._config = config
        world = int(getattr(config, "tensor_parallel_size", 1) or 1)
        # Profile-driven: the same select_profile the worker uses (config-only, so
        # scheduler + worker agree). cadence.align/min_len replace the hardcoded
        # DSV4 128 so a profile (e.g. Qwen3-Next) can checkpoint on its own grid.
        self._profile = select_profile(config, world=world, rank=0)
        self._geom = self._profile.build_geometry()
        self._fingerprint = self._geom.fingerprint()
        self._align = int(self._profile.cadence.align)
        self._min_len = int(self._profile.cadence.min_len)
        self._max_probes = int(os.environ.get("OFFLOAD_MAX_HIT_PROBES", "128"))
        # Session-local record of stored checkpoint keys (rank-0 authoritative).
        self._saved: set[str] = set()
        self._save_tracker: dict[str, object] = {}
        self._save_inflight: set[str] = set()
        self._req_key: dict[str, str] = {}
        # Loads: sid -> (B, key)
        self._load_specs: dict[str, tuple[int, str]] = {}
        self._reqs_need_load: dict[str, object] = {}

    # -- match / load ----------------------------------------------------
    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        num_prompt = int(seq.num_prompt_tokens)
        cands = candidate_boundaries(
            num_prompt, align=self._align, max_probes=self._max_probes
        )
        hit_boundaries: list[int] = []
        for B in cands:
            toks = list(seq.token_ids[:B])
            if len(toks) < B:
                continue
            if (
                checkpoint_key(toks, B, fingerprint=self._fingerprint, align=self._align)
                in self._saved
            ):
                hit_boundaries.append(B)
                break  # candidates already descending; first hit is the largest
        B = select_resume_boundary(hit_boundaries, num_prompt, align=self._align)
        if B is None:
            return 0, False
        num_cached = int(getattr(seq, "num_cached_tokens", 0))
        if B <= num_cached:  # HBM prefix already covers it
            return 0, False
        key = checkpoint_key(
            list(seq.token_ids[:B]), B, fingerprint=self._fingerprint, align=self._align
        )
        self._load_specs[str(seq.id)] = (B, key)
        logger.info("offload[hybrid] LOAD-HIT seq=%s B=%d num_cached=%d num_prompt=%d",
                    seq.id, B, num_cached, num_prompt)
        return B - num_cached, True

    def update_state_after_alloc(self, seq) -> None:
        self._save_tracker[str(seq.id)] = seq

    def should_park_for_load_after_alloc(self, seq) -> bool:
        sid = str(seq.id)
        spec = self._load_specs.get(sid)
        if spec is None:
            return False
        B, _key = spec
        num_cached = int(getattr(seq, "num_cached_tokens", 0))
        if B <= num_cached:  # HBM caught up post-alloc
            self._load_specs.pop(sid, None)
            return False
        seq.offload_loaded_tokens = B  # scheduler materializes SWA at this boundary
        self._reqs_need_load[sid] = seq
        return True

    def build_connector_meta(self) -> HybridOffloadMetadata:
        meta = HybridOffloadMetadata()
        # Loads first (TTFT-critical).
        for sid, seq in list(self._reqs_need_load.items()):
            spec = self._load_specs.pop(sid, None)
            if spec is None:
                continue
            B, key = spec
            meta.add_request(HybridReqMeta(
                req_id=seq.id, op="load",
                block_ids=list(seq.block_table),
                swa_block_ids=list(getattr(seq, "swa_block_table", []) or []),
                compute_slot=int(getattr(seq, "per_req_cache_group", -1)),
                B=B, key=key,
            ))
            logger.info("offload[hybrid] LOAD-EMIT seq=%s B=%d compute_slot=%d",
                        seq.id, B, int(getattr(seq, "per_req_cache_group", -1)))
        self._reqs_need_load.clear()
        # Saves.
        for sid, seq in list(self._save_tracker.items()):
            if sid in self._save_inflight:
                continue
            B = int(seq.num_prompt_tokens)
            computed = int(getattr(seq, "num_cached_tokens", 0))
            if not should_save_at(B, computed, align=self._align, min_len=self._min_len):
                continue
            toks = list(seq.token_ids[:B])
            if len(toks) < B:
                continue
            key = checkpoint_key(toks, B, fingerprint=self._fingerprint, align=self._align)
            if key in self._saved:
                continue
            meta.add_request(HybridReqMeta(
                req_id=seq.id, op="save",
                block_ids=list(seq.block_table),
                swa_block_ids=list(getattr(seq, "swa_block_table", []) or []),
                compute_slot=int(getattr(seq, "per_req_cache_group", -1)),
                B=B, key=key,
            ))
            self._save_inflight.add(sid)
            self._req_key[str(seq.id)] = key
            logger.info(
                "offload[hybrid] SAVE-EMIT seq=%s B=%d compute_slot=%d nblocks=%d nswa=%d",
                seq.id, B, int(getattr(seq, "per_req_cache_group", -1)),
                len(list(seq.block_table)),
                len([b for b in (getattr(seq, "swa_block_table", []) or []) if b >= 0]),
            )
        return meta

    def save_finished(self, req_id) -> None:
        sid = str(req_id)
        self._save_inflight.discard(sid)
        key = self._req_key.pop(sid, None)
        if key is not None:
            self._saved.add(key)
            logger.info("offload[hybrid] SAVE-DONE seq=%s total_saved=%d", req_id, len(self._saved))

    def should_defer_free(self, seq) -> bool:
        return str(seq.id) in self._save_inflight

    def load_failed(self, req_id) -> None:
        sid = str(req_id)
        self._load_specs.pop(sid, None)
        self._reqs_need_load.pop(sid, None)

    def request_finished(self, seq) -> None:
        sid = str(seq.id)
        self._save_tracker.pop(sid, None)
        self._load_specs.pop(sid, None)
        self._reqs_need_load.pop(sid, None)
