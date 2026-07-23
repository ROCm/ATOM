# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM standalone DSV4 terminal-checkpoint offload connector.

Enable via ``--kv-transfer-config '{"kv_connector":"dsv4_offload","kv_role":"offload"}'``
plus LMCache env (``LMCACHE_LOCAL_CPU=True`` etc.). Unlike ``lmcache_offload``
(token-chunked MHA/MLA), this stores one opaque *offload unit* per 128-aligned
terminal boundary — see ``dsv4-lmcache-bundle-plan.md``.

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
from atom.kv_transfer.offload.dsv4.gpu_connector import (
    DSV4OffloadUnitGPUConnector,
)
from atom.kv_transfer.offload.dsv4.admission import DSV4CheckpointAdmission
from atom.kv_transfer.offload.dsv4.policy import (
    CHECKPOINT_ALIGN,
    candidate_boundaries,
    checkpoint_key,
    select_resume_boundary,
    should_save_at,
)
from atom.kv_transfer.offload.dsv4.sources import DSV4OffloadSources
from atom.kv_transfer.offload.dsv4.store import LMCacheUnitStore
from atom.kv_transfer.offload.dsv4.unit import DSV4OffloadUnitGeometry
from atom.kv_transfer.offload.dsv4.unit_codec import (
    DSV4OffloadUnitCodec,
    DSV4OffloadUnitError,
)

logger = logging.getLogger("atom")

DSV4_BLOCK_SIZE = 128  # DSV4 attention block_size (deepseek_v4_attn.py); constant


def _geometry(config, *, world: int, rank: int) -> DSV4OffloadUnitGeometry:
    hf = config.hf_config
    ratios = tuple(int(r) for r in getattr(hf, "compress_ratios", ()) or ())
    head_dim = int(getattr(hf, "head_dim", 0) or 0)
    window = int(getattr(hf, "sliding_window", CHECKPOINT_ALIGN) or CHECKPOINT_ALIGN)
    return DSV4OffloadUnitGeometry(
        model_name=str(getattr(config, "model", "deepseek-v4")),
        layout_version=1,
        num_layers=int(getattr(hf, "num_hidden_layers", 0) or 0),
        head_dim=head_dim,
        window_size=window,
        block_size=DSV4_BLOCK_SIZE,
        swa_block_size=DSV4_BLOCK_SIZE,
        k1_csa=DSV4_BLOCK_SIZE // 4,
        k2_hca=max(1, DSV4_BLOCK_SIZE // 128),
        kv_dtype=str(getattr(config, "kv_cache_dtype", "auto")),
        compress_ratios=ratios,
        tp_size=int(world),
        tp_rank=int(rank),
    )


# =====================================================================
# Worker side
# =====================================================================
class DSV4OffloadConnector(OffloadWorkerMixin, KVConnectorBase):
    def __init__(self, config) -> None:
        self._config = config
        self._init_worker_common(config)  # executors, lock, done/failed tallies
        self._engine = None
        self._store: LMCacheUnitStore | None = None
        self._sources: DSV4OffloadSources | None = None
        self._gpu: DSV4OffloadUnitGPUConnector | None = None
        self._codec: DSV4OffloadUnitCodec | None = None
        self._admission: DSV4CheckpointAdmission | None = None
        self._rank = 0

    # -- lifecycle --------------------------------------------------------
    def register_kv_caches(
        self, kv_caches: dict, transfer_tensors=None, num_blocks: int | None = None
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        if transfer_tensors is None:
            logger.warning("DSV4 offload: no transfer_tensors; connector disabled")
            return

        tp = get_tp_group()
        rank, world = tp.rank_in_group, tp.world_size
        self._rank = rank

        # Reuse the shared opaque-uint8 engine build; the DSV4 store uses
        # storage_manager.allocate with explicit unit sizes, so the nominal
        # bytes_per_block here only affects unused bookkeeping.
        self._engine, _cfg, _meta = build_offload_engine(
            self._config,
            engine_id=f"atom-dsv4-offload-{rank}",
            block_size=DSV4_BLOCK_SIZE,
            bytes_per_block=256,
            gpu_connector_factory=lambda cfg, meta: _NoopGPUConnector(),
            world=world,
            rank=rank,
        )

        geom = _geometry(self._config, world=world, rank=rank)
        self._codec = DSV4OffloadUnitCodec(geom)
        device = torch.device("cuda", torch.cuda.current_device())
        self._gpu = DSV4OffloadUnitGPUConnector(self._codec, device=device)
        self._sources = DSV4OffloadSources(
            transfer_tensors, block_size=DSV4_BLOCK_SIZE, window_size=geom.window_size
        )
        self._store = LMCacheUnitStore(
            self._engine, model_name=geom.model_name, world_size=world, worker_id=rank
        )
        max_inflight = int(os.environ.get("DSV4_MAX_INFLIGHT_SAVES", "4"))
        self._admission = DSV4CheckpointAdmission(
            state_pool_size=self._sources.staging_pool_size,
            max_inflight_saves=max_inflight,
        )
        self._sources.log_sizing(rank=rank, example_B=16384, example_swa_blocks=1)
        logger.info(
            "DSV4 offload worker rank=%d ready: state_pool=%d max_inflight=%d "
            "save=%s load=%s",
            rank, self._sources.staging_pool_size, max_inflight,
            self._do_save, self._do_load,
        )

    # -- per-step (RPC thread) -------------------------------------------
    def start_load_kv(self, metadata) -> None:
        if not isinstance(metadata, DSV4OffloadMetadata) or self._sources is None:
            return
        for req in metadata.requests:
            if req.op == "save" and self._do_save:
                self._begin_and_submit_save(req)
            elif req.op == "load" and self._do_load:
                self._load_executor.submit(self._do_load_req, req)

    # -- SAVE ------------------------------------------------------------
    def _begin_and_submit_save(self, req: "DSV4ReqMeta") -> None:
        s = self._sources
        assert s is not None and self._admission is not None
        if not s.is_checkpoint_boundary(req.B):
            return
        if self._store is not None and self._store.contains(req.key):
            with self._lock:
                self._done_save.add(req.req_id)
            return
        pool_idx = self._admission.try_admit()
        if pool_idx is None:
            logger.debug("DSV4 offload: admission skip req=%s B=%d", req.req_id, req.B)
            return
        live_swa = [int(b) for b in req.swa_block_ids if int(b) >= 0]
        try:
            if s.gather_slot is not None:
                s.gather_slot(int(req.compute_slot), int(pool_idx))
                torch.cuda.current_stream().synchronize()
        except Exception:
            logger.exception("DSV4 offload: gather_slot failed req=%s", req.req_id)
            self._admission.complete(pool_idx)
            return
        self._save_executor.submit(self._finish_save, req, pool_idx, live_swa)

    def _finish_save(self, req: "DSV4ReqMeta", pool_idx: int, live_swa: list[int]) -> None:
        s, gpu, store = self._sources, self._gpu, self._store
        try:
            compressed, swa, csa_state = s.build_save_sources(
                block_table=req.block_ids, swa_block_table=req.swa_block_ids,
                state_pool_idx=pool_idx, B=req.B,
            )
            nbytes = s.unit_bytes(
                block_table=req.block_ids, swa_block_table=req.swa_block_ids, B=req.B
            )
            host = torch.empty(nbytes, dtype=torch.uint8)
            gpu.save_unit(
                memory_obj=host, compressed=compressed, swa=swa,
                csa_state=csa_state, boundary_B=req.B,
            )
            ok = store.put(req.key, host)
            logger.info(
                "DSV4 offload SAVE %s rank=%d req=%s B=%d unit_bytes=%d key=%s",
                "ok" if ok else "STORE-FAILED",
                self._rank, req.req_id, req.B, nbytes, req.key,
            )
        except Exception:
            logger.exception("DSV4 offload: save failed req=%s B=%d", req.req_id, req.B)
        finally:
            self._admission.complete(pool_idx)
            with self._lock:
                self._done_save.add(req.req_id)

    # -- LOAD ------------------------------------------------------------
    def _do_load_req(self, req: "DSV4ReqMeta") -> None:
        s, gpu, store = self._sources, self._gpu, self._store
        assert s is not None and self._admission is not None
        ok = False
        pool_idx = self._admission.try_admit()
        if pool_idx is None:
            logger.warning("DSV4 offload: load slot exhausted req=%s => recompute", req.req_id)
            with self._lock:
                self._failed_load.add(req.req_id)
            return
        try:
            host = store.get(req.key)
            if host is None:
                logger.warning("DSV4 offload: LOAD miss key=%s => recompute", req.key)
            else:
                compressed = s.compressed_sources(req.block_ids, req.B)
                swa = s.swa_sources(req.swa_block_ids)
                csa_state = s.csa_state_sources(pool_idx)
                gpu.load_unit(
                    memory_obj=host, compressed=compressed, swa=swa,
                    csa_state=csa_state, expect_boundary_B=req.B,
                )
                if csa_state and s.scatter_slot is not None:
                    s.scatter_slot(int(req.compute_slot), int(pool_idx))
                    torch.cuda.current_stream().synchronize()
                ok = True
                logger.info(
                    "DSV4 offload LOAD ok rank=%d req=%s B=%d key=%s",
                    self._rank, req.req_id, req.B, req.key,
                )
        except DSV4OffloadUnitError:
            logger.warning("DSV4 offload: LOAD fail-closed req=%s key=%s", req.req_id, req.key)
        except Exception:
            logger.exception("DSV4 offload: load errored req=%s", req.req_id)
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
class DSV4ReqMeta:
    __slots__ = ("req_id", "op", "block_ids", "swa_block_ids", "compute_slot", "B", "key")

    def __init__(self, *, req_id, op, block_ids, swa_block_ids, compute_slot, B, key):
        self.req_id = req_id
        self.op = op
        self.block_ids = block_ids
        self.swa_block_ids = swa_block_ids
        self.compute_slot = int(compute_slot)
        self.B = int(B)
        self.key = key


class DSV4OffloadMetadata(ConnectorMetadata):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[DSV4ReqMeta] = []

    def add_request(self, meta: DSV4ReqMeta) -> None:
        self.requests.append(meta)


# =====================================================================
# Scheduler side
# =====================================================================
class DSV4OffloadConnectorScheduler(KVConnectorSchedulerBase):
    is_producer = False
    is_offload = True

    def __init__(self, config) -> None:
        self._config = config
        world = int(getattr(config, "tensor_parallel_size", 1) or 1)
        self._geom = _geometry(config, world=world, rank=0)
        self._fingerprint = self._geom.fingerprint()
        self._max_probes = int(os.environ.get("DSV4_MAX_HIT_PROBES", "128"))
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
        cands = candidate_boundaries(num_prompt, max_probes=self._max_probes)
        hit_boundaries: list[int] = []
        for B in cands:
            toks = list(seq.token_ids[:B])
            if len(toks) < B:
                continue
            if checkpoint_key(toks, B, fingerprint=self._fingerprint) in self._saved:
                hit_boundaries.append(B)
                break  # candidates already descending; first hit is the largest
        B = select_resume_boundary(hit_boundaries, num_prompt)
        if B is None:
            return 0, False
        num_cached = int(getattr(seq, "num_cached_tokens", 0))
        if B <= num_cached:  # HBM prefix already covers it
            return 0, False
        key = checkpoint_key(list(seq.token_ids[:B]), B, fingerprint=self._fingerprint)
        self._load_specs[str(seq.id)] = (B, key)
        logger.info("DSV4 offload LOAD-HIT seq=%s B=%d num_cached=%d num_prompt=%d",
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

    def build_connector_meta(self) -> DSV4OffloadMetadata:
        meta = DSV4OffloadMetadata()
        # Loads first (TTFT-critical).
        for sid, seq in list(self._reqs_need_load.items()):
            spec = self._load_specs.pop(sid, None)
            if spec is None:
                continue
            B, key = spec
            meta.add_request(DSV4ReqMeta(
                req_id=seq.id, op="load",
                block_ids=list(seq.block_table),
                swa_block_ids=list(getattr(seq, "swa_block_table", []) or []),
                compute_slot=int(getattr(seq, "per_req_cache_group", -1)),
                B=B, key=key,
            ))
            logger.info("DSV4 offload LOAD-EMIT seq=%s B=%d compute_slot=%d",
                        seq.id, B, int(getattr(seq, "per_req_cache_group", -1)))
        self._reqs_need_load.clear()
        # Saves.
        for sid, seq in list(self._save_tracker.items()):
            if sid in self._save_inflight:
                continue
            B = int(seq.num_prompt_tokens)
            computed = int(getattr(seq, "num_cached_tokens", 0))
            if not should_save_at(B, computed):
                continue
            toks = list(seq.token_ids[:B])
            if len(toks) < B:
                continue
            key = checkpoint_key(toks, B, fingerprint=self._fingerprint)
            if key in self._saved:
                continue
            meta.add_request(DSV4ReqMeta(
                req_id=seq.id, op="save",
                block_ids=list(seq.block_table),
                swa_block_ids=list(getattr(seq, "swa_block_table", []) or []),
                compute_slot=int(getattr(seq, "per_req_cache_group", -1)),
                B=B, key=key,
            ))
            self._save_inflight.add(sid)
            self._req_key[str(seq.id)] = key
            logger.info(
                "DSV4 offload SAVE-EMIT seq=%s B=%d compute_slot=%d nblocks=%d nswa=%d",
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
            logger.info("DSV4 offload SAVE-DONE seq=%s total_saved=%d", req_id, len(self._saved))

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
