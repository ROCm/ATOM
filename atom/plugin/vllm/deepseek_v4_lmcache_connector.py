# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM adapter for ATOM's DeepSeek-V4 LMCache PAGE/SLOT data path.

DeepSeek-V4 exposes one packed uint8 proxy allocation to vLLM.  The generic
LMCache connector mistakes that allocation for an ordinary ``[K, V]`` cache
and repeats its single base pointer for every model layer.  This adapter keeps
vLLM's connector lifecycle while delegating PAGE/SLOT movement and lookup to
ATOM's layout-aware DSV4 offload implementation.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)

logger = logging.getLogger("atom")


def _is_deepseek_v4(vllm_config) -> bool:
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", ())
    return any(str(name) == "DeepseekV4ForCausalLM" for name in architectures or ())


def _atom_config(vllm_config):
    """Present the small native-ATOM config surface used by DSV4 offload."""
    transfer = vllm_config.kv_transfer_config
    if hasattr(transfer, "model_dump"):
        transfer_dict = transfer.model_dump(mode="python")
    elif dataclasses.is_dataclass(transfer):
        transfer_dict = dataclasses.asdict(transfer)
    else:
        transfer_dict = dict(transfer)
    transfer_dict["kv_connector"] = "lmcache_offload"
    transfer_dict["kv_role"] = "offload"
    extra = transfer_dict.get("kv_connector_extra_config") or {}

    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    return SimpleNamespace(
        kv_transfer_config=transfer_dict,
        kv_cache_block_size=int(cache.block_size),
        kv_cache_dtype=str(cache.cache_dtype),
        index_cache_dtype="fp8",
        hf_config=model.hf_config,
        model=model.model,
        model_tag=model.model,
        tensor_parallel_size=int(parallel.tensor_parallel_size),
        pipeline_parallel_size=int(parallel.pipeline_parallel_size),
        parallel_config=parallel,
        decode_context_parallel_size=1,
        state_checkpoint_interval_tokens=int(
            extra.get("state_checkpoint_interval_tokens", -1)
        ),
        speculative_config=vllm_config.speculative_config,
    )


def _page_transfer_tensors(proxy: torch.Tensor, vllm_config):
    """Describe V4 PAGE and per-request SLOT bytes in the proxy allocation."""
    from atom.kv_transfer.disaggregation.types import (
        KVTransferRegion,
        KVTransferTensors,
    )
    from atom.plugin.vllm.deepseek_v4_bridge import (
        _index_row_bytes,
        _layer_counts,
        _v4_kv_fp8,
        _v4_rope_head_dim,
        _v4_state_layout,
        _v4_win_with_spec,
        slice_deepseek_v4_proxy_cache_views,
    )

    hf = vllm_config.model_config.hf_config
    ratios, _dense, _csa, _hca = _layer_counts(hf)
    num_slots = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    window = int(getattr(hf, "sliding_window", 128) or 128)
    kv_fp8 = _v4_kv_fp8(vllm_config)
    if not kv_fp8:
        raise ValueError("DSV4 vLLM LMCache adapter currently requires FP8 KV cache")
    head_dim = int(getattr(hf, "head_dim", 512))
    index_head_dim = int(getattr(hf, "index_head_dim", 128))
    rope_head_dim = _v4_rope_head_dim(hf)
    arena_planes, arena_rows, row_widths = _v4_state_layout(vllm_config, kv_fp8)
    views = slice_deepseek_v4_proxy_cache_views(
        proxy,
        compress_ratios=ratios,
        num_slots=num_slots,
        window_size=_v4_win_with_spec(vllm_config, window),
        head_dim=head_dim,
        index_head_dim=index_head_dim,
        kv_fp8=True,
        rope_head_dim=rope_head_dim,
        arena_planes=arena_planes,
        arena_rows=arena_rows,
        row_widths=row_widths,
    )
    geometry = views["geometry"]
    num_blocks = int(proxy.shape[1])
    regions: list[KVTransferRegion] = []

    plane_specs = [
        (views["unified"], head_dim, "dsv4.main.nope"),
        (views["unified_rope"], rope_head_dim * 2, "dsv4.main.rope"),
    ]
    for layer_views, row_bytes, role in plane_specs:
        if not layer_views:
            continue
        plane = layer_views[0]
        unit_bytes = int(geometry.block_bytes(row_bytes))
        regions.append(
            KVTransferRegion(
                plane.data_ptr(),
                num_blocks * unit_bytes,
                unit_bytes,
                semantic_role=role,
            )
        )

    csa_layer_ids = [i for i, ratio in enumerate(ratios) if ratio == 4]
    for layer_id, view in zip(csa_layer_ids, views["csa_indexer"], strict=True):
        if not view.is_contiguous():
            raise RuntimeError("DSV4 CSA indexer PAGE view must be contiguous")
        unit_bytes = int(view.stride(0) * view.element_size())
        regions.append(
            KVTransferRegion(
                view.data_ptr(),
                view.numel() * view.element_size(),
                unit_bytes,
                semantic_role=f"dsv4.indexer.layer_{layer_id}",
            )
        )

    slot_regions: list[KVTransferRegion] = []
    slot_start, _ = geometry.slot_span(geometry.physical_slot(num_slots - 1))
    for layer_views, row_bytes, role in plane_specs:
        if not layer_views:
            continue
        plane = layer_views[0]
        unit_bytes = int(geometry.slot_bytes(row_bytes))
        slot_regions.append(
            KVTransferRegion(
                plane.data_ptr() + slot_start * row_bytes,
                num_slots * unit_bytes,
                unit_bytes,
                reverse_indexed=True,
                semantic_role=role,
            )
        )

    transfer = KVTransferTensors(
        block_regions=regions,
        slot_regions=[],
        num_slots=num_slots,
        swa_block_regions=slot_regions,
        expected_full_slot_region_count=len(slot_regions),
    )
    transfer.set_block_count(num_blocks)
    return transfer, num_blocks


class _SequenceView:
    """Mutable native-ATOM sequence view backed by a vLLM request."""

    def __init__(self, request, num_cached_tokens: int) -> None:
        self.request = request
        self.id = str(request.request_id)
        self.block_table: list[int] = []
        self.has_per_req_cache = True
        self.state_slot = -1
        self.offload_loaded = False
        self.offload_loaded_tokens = 0
        self.offload_load_failed = False
        self.offload_load_start_tokens = 0
        self.offload_handoff_boundary_tokens = 0
        self.hbm_floor = int(num_cached_tokens)
        self.refresh(num_cached_tokens)

    def refresh(self, num_cached_tokens: int | None = None) -> None:
        token_ids = getattr(self.request, "all_token_ids", None)
        if token_ids is None:
            token_ids = getattr(self.request, "prompt_token_ids", None) or []
        self.token_ids = list(token_ids)
        prompt = getattr(self.request, "prompt_token_ids", None) or self.token_ids
        self.num_prompt_tokens = len(prompt)
        if num_cached_tokens is None:
            num_cached_tokens = int(
                getattr(self.request, "num_computed_tokens", 0) or 0
            )
        self.num_cached_tokens = int(num_cached_tokens)

    def extend_blocks(self, block_ids) -> None:
        for block_id in block_ids or ():
            bid = int(block_id)
            if bid not in self.block_table:
                self.block_table.append(bid)


@dataclass
class DSV4LMCacheConnectorMetadata(KVConnectorMetadata):
    native: Any


@dataclass
class DSV4WorkerMeta(KVConnectorWorkerMetadata):
    """Rank-ordered ATOM outputs for its existing TP completion aggregator."""

    worker_outputs: list[Any] = dataclasses.field(default_factory=list)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        if not isinstance(other, DSV4WorkerMeta):
            return self
        return DSV4WorkerMeta(
            worker_outputs=[*self.worker_outputs, *other.worker_outputs],
        )


class DSV4LMCacheConnector(KVConnectorBase_V1):
    """Role-split vLLM connector backed by ATOM's DSV4 offload classes."""

    def __init__(self, vllm_config, role, kv_cache_config) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._vllm_config = vllm_config
        self._role = role
        self._config = _atom_config(vllm_config)
        self._sequences: dict[str, _SequenceView] = {}
        self._deferred_saves: set[str] = set()
        self._failed_block_ids: set[int] = set()
        self._completion_world_size = int(
            getattr(vllm_config.parallel_config, "world_size", 0)
            or (
                int(vllm_config.parallel_config.tensor_parallel_size)
                * int(vllm_config.parallel_config.pipeline_parallel_size)
            )
        )
        self._worker = None
        self._proxy_data_ptr = None
        self._scheduler = None
        self._state_slot_allocator = None
        self._output_aggregator = None
        self._worker_meta: DSV4WorkerMeta | None = None
        if role == KVConnectorRole.SCHEDULER:
            from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
            from atom.kv_transfer.offload.hybrid.dsv4.connector import (
                DSV4OffloadScheduler,
            )
            from atom.plugin.vllm.deepseek_v4_bridge import _V4StateSlotAllocator

            self._scheduler = DSV4OffloadScheduler(self._config)
            self._state_slot_allocator = _V4StateSlotAllocator(
                int(vllm_config.scheduler_config.max_num_seqs)
            )
            self._output_aggregator = KVOutputAggregator(self._completion_world_size)
        else:
            from atom.kv_transfer.offload.hybrid.dsv4.connector import (
                DSV4OffloadConnector,
            )

            self._worker = DSV4OffloadConnector(self._config)

    @property
    def role(self):
        return self._role

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self._worker is None:
            return
        if len(kv_caches) != 1:
            raise ValueError(
                f"DSV4 vLLM adapter expected one proxy KV tensor, got {len(kv_caches)}"
            )
        proxy = next(iter(kv_caches.values()))
        self._proxy_data_ptr = proxy.untyped_storage().data_ptr()
        transfer, num_blocks = _page_transfer_tensors(proxy, self._vllm_config)
        self._worker.register_kv_caches(
            {}, transfer_tensors=transfer, num_blocks=num_blocks
        )
        logger.info(
            "ATOM DSV4 vLLM LMCache adapter registered %d PAGE and %d SLOT regions",
            len(transfer.block_regions),
            len(transfer.swa_block_regions),
        )

    def start_load_kv(self, forward_context, **kwargs: Any) -> None:
        if self._worker is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DSV4LMCacheConnectorMetadata):
            return
        if metadata.native is not None:
            from atom.plugin.vllm.deepseek_v4_bridge import (
                reserve_deepseek_v4_state_slot,
            )

            for request_meta in metadata.native.requests:
                slot_spec = request_meta.slot_load_spec
                if slot_spec is not None:
                    reserve_deepseek_v4_state_slot(
                        self._proxy_data_ptr,
                        str(request_meta.req_id),
                        slot_spec.destination_group,
                    )
        self._worker.start_load_kv(metadata.native)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        return

    @staticmethod
    def _request_ids(completions) -> set[str]:
        return {
            str(getattr(completion, "req_id", completion))
            for completion in completions
        }

    def get_finished(self, finished_req_ids: set[str]):
        if self._worker is None:
            return None, None
        output = self._worker.get_finished()
        self._worker_meta = DSV4WorkerMeta(worker_outputs=[output])
        receiving = self._request_ids(output.finished_loading)
        receiving.update(self._request_ids(output.failed_loading))
        # Failed loads must not look like successful receives. The scheduler
        # marks their blocks invalid from worker_meta.
        return None, receiving or None

    def build_connector_worker_meta(self):
        meta, self._worker_meta = self._worker_meta, None
        return meta

    def has_pending_push_work(self) -> bool:
        if self._scheduler is None:
            return False
        # A WAITING_FOR_REMOTE_KVS request lives outside vLLM's runnable
        # queues. Keep zero-token connector steps running until worker
        # finished_recving reports have been aggregated across every rank.
        # Otherwise the workers finish the LMCache retrieve, but EngineCore
        # quiesces before polling the terminal completion.
        active_loads = getattr(self._scheduler, "_active_load_operations", {})
        return bool(
            self._deferred_saves
            or active_loads
            or self._scheduler.has_pending_work()
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed, self._failed_block_ids = self._failed_block_ids, set()
        return failed

    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        if self._scheduler is None:
            return 0, False
        sid = str(request.request_id)
        seq = self._sequences.get(sid)
        if seq is None or seq.request is not request:
            seq = self._sequences[sid] = _SequenceView(request, num_computed_tokens)
        else:
            seq.refresh(num_computed_tokens)
        slots, _ = self._state_slot_allocator.assign([sid], [num_computed_tokens])
        seq.state_slot = int(slots[0])
        seq.hbm_floor = int(num_computed_tokens)
        need, asynchronous = self._scheduler.get_num_new_matched_tokens(seq)
        if need <= 0:
            return need, False
        return need, asynchronous

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        if self._scheduler is None:
            return
        sid = str(request.request_id)
        seq = self._sequences.get(sid)
        if seq is None:
            seq = self._sequences[sid] = _SequenceView(request, 0)
        # vLLM may already include the promised external tokens in
        # request.num_computed_tokens here. Keep the real pre-lookup HBM floor
        # captured by get_num_new_matched_tokens; otherwise ATOM decides that
        # HBM satisfies the load and vLLM waits forever for a completion.
        seq.refresh(seq.hbm_floor)
        block_groups = blocks.get_block_ids()
        if block_groups:
            seq.extend_blocks(block_groups[0])
        self._scheduler.update_state_after_alloc(seq)
        seq._state_initialized_after_alloc = True

    def _sync_from_scheduler_output(self, scheduler_output) -> None:
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = set(getattr(cached, "resumed_req_ids", ()) or ())
        for new_request in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            seq = self._sequences.get(str(new_request.req_id))
            if seq is None:
                continue
            block_groups = new_request.block_ids
            if block_groups:
                seq.extend_blocks(block_groups[0])
            seq.refresh(int(new_request.num_computed_tokens))
        if cached is not None:
            for i, req_id in enumerate(getattr(cached, "req_ids", ()) or ()):
                seq = self._sequences.get(str(req_id))
                if seq is None:
                    continue
                new_blocks = cached.new_block_ids[i] if cached.new_block_ids else None
                group = new_blocks[0] if new_blocks else None
                if str(req_id) in resumed:
                    seq.block_table = [int(b) for b in (group or ())]
                else:
                    seq.extend_blocks(group)
                # This is the completed frontier before the current forward.
                # num_scheduled_tokens names work that has not run yet.
                seq.refresh(int(cached.num_computed_tokens[i]))

    def build_connector_meta(self, scheduler_output):
        if self._scheduler is None:
            return DSV4LMCacheConnectorMetadata(None)
        self._sync_from_scheduler_output(scheduler_output)
        # Loads are decided from the pre-lookup HBM floor. Do not let this
        # step's scheduled/computed tokens make ATOM skip the retrieve while
        # vLLM is still parked on finished_recving.
        for sid in list(getattr(self._scheduler, "_reqs_need_recv", {}) or {}):
            seq = self._sequences.get(str(sid))
            if seq is not None:
                seq.num_cached_tokens = int(seq.hbm_floor)
        return DSV4LMCacheConnectorMetadata(self._scheduler.build_connector_meta())

    def update_connector_output(self, connector_output) -> None:
        if self._scheduler is None:
            return
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(meta, DSV4WorkerMeta):
            processed = self._scheduler.process_completions(
                self._output_aggregator.aggregate(meta.worker_outputs)
            )
            for req_id in processed.failed_loading or ():
                seq = self._sequences.get(str(req_id))
                if seq is not None:
                    self._failed_block_ids.update(seq.block_table)
        released = set(connector_output.finished_sending or ())
        for sid in tuple(self._deferred_saves):
            seq = self._sequences.get(sid)
            if seq is not None and not self._scheduler.should_defer_free(seq):
                released.add(sid)
        connector_output.finished_sending = released or None
        for req_id in connector_output.finished_sending or ():
            sid = str(req_id)
            self._deferred_saves.discard(sid)
            self._sequences.pop(sid, None)

    def _request_aborted(self, request) -> bool:
        status = getattr(request, "status", None)
        name = getattr(status, "name", str(status or ""))
        return "ABORT" in name.upper()

    def request_finished(self, request, block_ids: list[int]):
        if self._scheduler is None:
            return False, None
        sid = str(request.request_id)
        seq = self._sequences.get(sid)
        if seq is None:
            return False, None
        if self._request_aborted(request):
            self._sequences.pop(sid, None)
            self._deferred_saves.discard(sid)
            self._scheduler.abandon_save(seq.id)
            self._scheduler.request_finished(seq)
            return False, None
        self._scheduler.request_finished(seq)
        if self._scheduler.should_defer_free(seq):
            self._deferred_saves.add(sid)
            return True, None
        self._sequences.pop(sid, None)
        return False, None

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.close()


def apply_vllm_dsv4_lmcache_connector_patch() -> None:
    """Route only DeepSeek-V4 LMCacheConnectorV1 creation to this adapter."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    original = KVConnectorFactory.create_connector.__func__
    if getattr(original, "_atom_dsv4_lmcache_patched", False):
        return

    @functools.wraps(original)
    def create_connector(cls, config, role, kv_cache_config):
        transfer = config.kv_transfer_config
        if (
            transfer is not None
            and transfer.kv_connector == "LMCacheConnectorV1"
            and _is_deepseek_v4(config)
        ):
            logger.info("Creating ATOM DSV4-aware vLLM LMCache connector")
            return DSV4LMCacheConnector(config, role, kv_cache_config)
        return original(cls, config, role, kv_cache_config)

    create_connector._atom_dsv4_lmcache_patched = True
    KVConnectorFactory.create_connector = classmethod(create_connector)


__all__ = [
    "DSV4LMCacheConnector",
    "DSV4LMCacheConnectorMetadata",
    "apply_vllm_dsv4_lmcache_connector_patch",
]
