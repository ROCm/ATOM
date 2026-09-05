# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic PAGE adapter for LMCache's standalone multiprocess server.

Attention backends publish block-major tensor views through
``KVTransferTensors``. LMCache MP can attach several physical kernel groups to
the same engine block-id space, so this adapter registers those opaque views
without knowing which model or attention implementation produced them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import torch

from atom.kv_transfer.disaggregation.base import KVConnectorBase
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadCompletionId,
    LoadOperationId,
    SaveCompletionId,
)
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import validated_kv_role
from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta

logger = logging.getLogger("atom")

_MP_LAYOUT_VERSION = 2
_OPERATION_TOMBSTONE_LIMIT = 4096


def _extra_config(config: Any) -> dict[str, Any]:
    kvc = getattr(config, "kv_transfer_config", {}) or {}
    extra = kvc.get("kv_connector_extra_config", {}) or {}
    if not isinstance(extra, dict):
        raise TypeError("kv_connector_extra_config must be a dictionary")
    return extra


def _storage_kv_transfer_config(config: Any) -> dict[str, Any]:
    """Remove MP transport-only options before LMCache storage parsing."""

    kvc = dict(getattr(config, "kv_transfer_config", {}) or {})
    extra = kvc.get("kv_connector_extra_config")
    if isinstance(extra, dict):
        kvc["kv_connector_extra_config"] = {
            key: value
            for key, value in extra.items()
            if not (isinstance(key, str) and key.startswith("lmcache.mp."))
        }
    return kvc


def _transfer_mode(config: Any) -> str:
    extra = _extra_config(config)
    configured_mode = extra.get("lmcache.mp.mp_transfer_mode")
    if configured_mode is None:
        configured_mode = os.environ.get("LMCACHE_MP_TRANSFER_MODE", "auto")
    transfer_mode = str(configured_mode).strip().lower()
    if transfer_mode not in ("auto", "lmcache_driven", "engine_driven"):
        raise ValueError(
            "LMCache MP transfer mode must be 'auto', 'lmcache_driven', or "
            f"'engine_driven', got {configured_mode!r}"
        )
    if transfer_mode == "engine_driven":
        raise NotImplementedError(
            "ATOM lmcache_mp requires LMCache's lmcache_driven transfer path "
            "because engine_driven does not support multiple physical "
            "cache groups"
        )
    return transfer_mode


def _validate_mp_config(config: Any) -> tuple[int, int]:
    """Validate topology constraints shared by generic PAGE layouts."""
    tp_size = offcfg._strict_integer(
        "tensor_parallel_size",
        getattr(config, "tensor_parallel_size", 1) or 1,
        minimum=1,
    )
    pp_size = offcfg._strict_integer(
        "pipeline_parallel_size",
        getattr(config, "pipeline_parallel_size", 1) or 1,
        minimum=1,
    )
    dcp_size = offcfg._strict_integer(
        "decode_context_parallel_size",
        getattr(config, "decode_context_parallel_size", 1) or 1,
        minimum=1,
    )
    pcp_size = offcfg._strict_integer(
        "prefill_context_parallel_size",
        getattr(config, "prefill_context_parallel_size", 1) or 1,
        minimum=1,
    )
    parallel_config = getattr(config, "parallel_config", None)
    dp_size = offcfg._strict_integer(
        "data_parallel_size",
        getattr(
            parallel_config,
            "data_parallel_size",
            getattr(config, "data_parallel_size", 1),
        )
        or 1,
        minimum=1,
    )
    if pp_size != 1:
        raise NotImplementedError("lmcache_mp does not support PP yet")
    if dcp_size != 1:
        raise NotImplementedError("lmcache_mp does not support DCP yet")
    if pcp_size != 1:
        raise NotImplementedError("lmcache_mp does not support PCP yet")
    if dp_size != 1 or bool(getattr(config, "enable_dp_attention", False)):
        raise NotImplementedError("lmcache_mp currently supports TP-only deployments")

    _transfer_mode(config)
    return tp_size, pp_size


def _config_has_fully_replicated_tp_pages(config: Any) -> bool:
    """Conservatively identify configs whose complete PAGE cache is TP-replicated.

    MLA caches the shared latent before the TP-sharded KV-B projection. ATOM's
    sparse MLA index-key projection is replicated too, so its auxiliary PAGE
    plane has the same property.

    The worker validates this config-time prediction against the attention
    backend's ``KVTransferTensors.tp_replication_factor`` declaration before
    registering any cache memory.
    """

    hf_config = getattr(config, "hf_config", None)
    hf_config = getattr(hf_config, "text_config", hf_config)
    return getattr(hf_config, "kv_lora_rank", None) is not None


def _tp_replication_factor(config: Any) -> int:
    """Return the PAGE rank-collapse factor selected before workers start.

    ``auto`` uses only structural cache information available in the shared
    engine config. An explicit boolean is useful for a new attention backend:
    ``True`` requests full TP collapse, but worker registration still fails
    closed unless that backend declares every PAGE region byte-identical.
    """

    tp_size, _ = _validate_mp_config(config)
    configured = _extra_config(config).get("lmcache.mp.tp_rank_collapse", "auto")
    if isinstance(configured, str) and configured.strip().lower() == "auto":
        collapse = _config_has_fully_replicated_tp_pages(config)
    elif type(configured) is bool:
        collapse = configured
    else:
        raise TypeError("lmcache.mp.tp_rank_collapse must be true, false, or 'auto'")
    return tp_size if collapse else 1


def _published_tp_replication_factor(
    transfer_tensors: Any,
    *,
    tp_size: int,
) -> int:
    """Validate the attention backend's whole-PAGE replication declaration."""

    factor = offcfg._strict_integer(
        "KV PAGE TP replication factor",
        getattr(transfer_tensors, "tp_replication_factor", 1),
        minimum=1,
    )
    if tp_size % factor:
        raise ValueError(
            f"KV PAGE TP replication factor {factor} must divide TP size {tp_size}"
        )
    if factor not in (1, tp_size):
        raise NotImplementedError(
            "lmcache_mp currently supports only sharded or fully TP-replicated "
            f"PAGE layouts, got replication factor {factor} for TP size {tp_size}"
        )
    return factor


def _server_urls(config: Any) -> list[str]:
    extra = _extra_config(config)
    configured = extra.get("lmcache.mp.server_urls")
    if configured is not None:
        if isinstance(configured, (list, tuple)):
            urls = [str(value).strip() for value in configured if str(value).strip()]
        else:
            urls = [
                value.strip() for value in str(configured).split(",") if value.strip()
            ]
    else:
        host = str(extra.get("lmcache.mp.host", "tcp://localhost")).strip()
        if not host:
            raise ValueError("lmcache.mp.host must be non-empty")
        port = offcfg._strict_integer(
            "lmcache.mp.port",
            extra.get("lmcache.mp.port", 5555),
            minimum=1,
        )
        if not 1 <= port <= 65535:
            raise ValueError("lmcache.mp.port must be in [1, 65535]")
        urls = [f"{host}:{port}"]
    urls = [url if "://" in url else f"tcp://{url}" for url in urls]
    if len(urls) != 1:
        raise NotImplementedError(
            "lmcache_mp currently supports exactly one LMCache server"
        )
    return urls


def _model_namespace(config: Any) -> str:
    """Build a model/layout namespace shared by scheduler and workers."""

    cfg = offcfg.build_lmcache_config(_storage_kv_transfer_config(config))
    world_size = offcfg.lmcache_replica_world_size(config)
    page_namespace = offcfg.build_page_namespace(
        config,
        cfg,
        world_size,
    )
    return f"{page_namespace}::lmcache-mp-v{_MP_LAYOUT_VERSION}"


def _parallel_strategy(config: Any, worker_id: int) -> Any:
    from lmcache.integration.atom import AtomMPParallelConfig

    tp_size, pp_size = _validate_mp_config(config)
    if worker_id < 0 or worker_id >= tp_size:
        raise ValueError(
            f"LMCache MP worker rank {worker_id} is outside [0, {tp_size})"
        )
    replication_factor = _tp_replication_factor(config)
    return AtomMPParallelConfig(
        world_size=tp_size * pp_size // replication_factor,
        worker_id=worker_id // replication_factor,
        tp_size=tp_size,
    )


def _make_scheduler_adapter(config: Any) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPSchedulerAdapter

    num_kv_readers = _tp_replication_factor(config)

    class _ReaderAwareSchedulerAdapter(AtomMPSchedulerAdapter):
        """Reserve one LMCache read lock for every collapsed TP consumer."""

        def _create_key(self, *args: Any, **kwargs: Any) -> Any:
            key = super()._create_key(*args, **kwargs)
            return replace(key, num_kv_readers=num_kv_readers)

    extra = _extra_config(config)
    return _ReaderAwareSchedulerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, 0),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
    )


def _make_worker_adapter(config: Any, rank: int) -> Any:
    import zmq
    from lmcache.integration.atom import AtomMPWorkerAdapter

    extra = _extra_config(config)
    return AtomMPWorkerAdapter(
        server_url=_server_urls(config)[0],
        context=zmq.Context.instance(),
        model_name=_model_namespace(config),
        block_size=int(config.kv_cache_block_size),
        parallel_config=_parallel_strategy(config, rank),
        mq_timeout=float(extra.get("lmcache.mp.mq_timeout", 300.0)),
        heartbeat_interval=float(extra.get("lmcache.mp.heartbeat_interval", 10.0)),
        transfer_mode=_transfer_mode(config),
    )


@dataclass(frozen=True)
class _CacheViews:
    """Validated block-major tensors and their copy-kernel group indices."""

    tensors: dict[str, torch.Tensor]
    layer_groups: tuple[tuple[int, ...], ...]
    bytes_per_block: int


def _build_cache_views(
    transfer_tensors: Any,
    *,
    num_blocks: int,
) -> _CacheViews:
    """Validate backend-published PAGE views without inspecting model internals."""

    if transfer_tensors is None:
        raise ValueError("lmcache_mp requires KVTransferTensors")
    if num_blocks <= 0:
        raise ValueError("lmcache_mp num_blocks must be positive")

    stateful_fields = {
        "num_slots": getattr(transfer_tensors, "num_slots", 0),
        "slot_regions": getattr(transfer_tensors, "slot_regions", None),
        "swa_block_regions": getattr(transfer_tensors, "swa_block_regions", None),
        "staging_region": getattr(transfer_tensors, "staging_region", None),
        "gather_slot": getattr(transfer_tensors, "gather_slot", None),
        "scatter_slot": getattr(transfer_tensors, "scatter_slot", None),
        "expected_full_slot_region_count": getattr(
            transfer_tensors, "expected_full_slot_region_count", None
        ),
    }
    populated_stateful_fields = [
        name for name, value in stateful_fields.items() if value
    ]
    if populated_stateful_fields:
        raise NotImplementedError(
            "lmcache_mp supports PAGE-only layouts; stateful SLOT data was "
            f"published through {', '.join(populated_stateful_fields)}"
        )

    regions = list(getattr(transfer_tensors, "block_regions", None) or [])
    views = list(getattr(transfer_tensors, "block_tensor_views", None) or [])
    if not regions or len(views) != len(regions):
        raise ValueError(
            "lmcache_mp requires one block_tensor_view per block region: "
            f"views={len(views)} regions={len(regions)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    indices_by_layout: dict[tuple[torch.dtype, tuple[int, ...]], list[int]] = {}
    devices: set[torch.device] = set()
    bytes_per_block = 0
    for index, (view, region) in enumerate(zip(views, regions, strict=True)):
        if not isinstance(view, torch.Tensor):
            raise TypeError(f"lmcache_mp block tensor view {index} is not a Tensor")
        if view.numel() == 0 or not view.is_contiguous():
            raise ValueError(
                f"lmcache_mp block tensor view {index} must be non-empty and contiguous"
            )
        if view.ndim != 3 or int(view.shape[0]) != num_blocks:
            raise ValueError(
                f"lmcache_mp block tensor view {index} has shape={tuple(view.shape)}, "
                f"expected [{num_blocks}, physical_slots, opaque_width]"
            )

        unit_bytes = int(region.unit_bytes)
        total_bytes = int(region.total_bytes)
        actual_unit_bytes = view[0].numel() * view.element_size()
        actual_total_bytes = view.numel() * view.element_size()
        if actual_unit_bytes != unit_bytes or actual_total_bytes != total_bytes:
            raise ValueError(
                f"lmcache_mp block tensor view {index} byte geometry mismatch: "
                f"unit={actual_unit_bytes}/{unit_bytes} "
                f"total={actual_total_bytes}/{total_bytes}"
            )
        if view.data_ptr() != int(region.base_addr):
            raise ValueError(
                f"lmcache_mp block tensor view {index} does not start at its "
                "declared region address"
            )
        if bool(getattr(region, "reverse_indexed", False)):
            raise ValueError("lmcache_mp PAGE regions cannot be reverse-indexed")

        role = str(getattr(region, "semantic_role", None) or f"plane_{index}")
        tensors[f"page.{index}.{role}"] = view
        layout = (view.dtype, tuple(int(dim) for dim in view.shape[1:]))
        indices_by_layout.setdefault(layout, []).append(index)
        devices.add(view.device)
        bytes_per_block += unit_bytes

    if len(devices) != 1:
        raise ValueError("lmcache_mp block tensor views must share one device")
    return _CacheViews(
        tensors=tensors,
        layer_groups=tuple(tuple(indices) for indices in indices_by_layout.values()),
        bytes_per_block=bytes_per_block,
    )


@dataclass
class _LookupState:
    token_ids: list[int]
    hit: int | None = None
    retrieve_start: int | None = None
    retrieve_end: int | None = None


@dataclass
class _PendingLoad:
    completion: LoadCompletionId
    future: Any | None


@dataclass
class _PendingSave:
    completion: SaveCompletionId
    future: Any | None


def _terminal_future_result(future: Any | None) -> tuple[bool, Any]:
    """Return ``(terminal, result)`` without blocking on a device future."""

    if future is None:
        return True, None
    try:
        if not future.query():
            return False, None
        return True, future.result(timeout=0)
    except Exception:
        # A query/result exception is not proof that the remote GPU stream has
        # quiesced. Keep the operation pending rather than risk block reuse.
        logger.warning(
            "LMCache MP transfer future could not prove terminal state",
            exc_info=True,
        )
        return False, None


def _transfer_operation_id(kind: str, completion: Any) -> str:
    """Return a stable LMCache handle ID for one ATOM completion generation."""

    request_id = getattr(completion, "req_id", completion)
    generation = getattr(completion, "generation", None)
    if generation is None:
        return f"{kind}:{request_id}"
    return f"{kind}:{request_id}:{generation}"


def _remember_operation_tombstone(
    operation_id: str,
    tombstones: set[str],
    tombstone_order: deque[str],
) -> None:
    if operation_id in tombstones:
        return
    tombstones.add(operation_id)
    tombstone_order.append(operation_id)
    while len(tombstone_order) > _OPERATION_TOMBSTONE_LIMIT:
        tombstones.discard(tombstone_order.popleft())


class _MPLookupClient:
    """Synchronous lookup facade used by ATOM's existing scheduler policy."""

    token_database = None

    def __init__(self, adapter: Any, *, timeout: float, poll_interval: float) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("LMCache MP lookup timeout and poll interval must be > 0")
        self._adapter = adapter
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._lookups: dict[str, _LookupState] = {}

    def lookup(self, token_ids: list[int], lookup_id: str) -> int:
        state = _LookupState(token_ids=list(token_ids))
        self._lookups[lookup_id] = state
        self._adapter.maybe_submit_lookup_request(lookup_id, token_ids)
        deadline = time.monotonic() + self._timeout
        while True:
            result = self._adapter.check_lookup_result(lookup_id)
            if result is not None:
                hit = int(result)
                state.hit = hit
                return hit
            if time.monotonic() >= deadline:
                logger.warning(
                    "LMCache MP lookup timed out after %.1fs for request %s",
                    self._timeout,
                    lookup_id,
                )
                # The MP API has no cancel-prefetch call. Keep the adapter job
                # intact so request_finished() can release locks if the result
                # becomes available; eagerly cleaning it here would orphan the
                # server-side lookup and its locks.
                return 0
            time.sleep(self._poll_interval)

    def prepare_retrieve(
        self,
        lookup_id: str,
        start: int,
        end: int | None = None,
    ) -> None:
        """Hand one subrange of a lookup hit to the worker retrieve.

        The worker owns and releases locks in ``[start, end)``. Any hit prefix
        already resident in HBM and any hit suffix beyond ``end`` will not be
        consumed by that retrieve, so release those locks here.
        """
        if start < 0 or (end is not None and end < start):
            raise ValueError(
                f"invalid retrieve range for {lookup_id}: start={start}, end={end}"
            )
        state = self._lookups.get(lookup_id)
        if (
            state is not None
            and state.hit is not None
            and end is not None
            and end > state.hit
        ):
            raise ValueError(
                f"retrieve end {end} exceeds lookup hit {state.hit} for {lookup_id}"
            )
        if state is not None and state.hit is not None and start > 0:
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=min(start, state.hit),
                request_id=lookup_id,
            )
        if (
            state is not None
            and state.hit is not None
            and end is not None
            and end < state.hit
        ):
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=end,
                end=state.hit,
                request_id=lookup_id,
            )
        if state is not None:
            state.retrieve_start = start
            state.retrieve_end = end
        self._adapter.cleanup_lookup_result(lookup_id)

    def complete_retrieve(self, lookup_id: str, *, succeeded: bool) -> None:
        # Once a retrieve is submitted, LMCache owns the remaining lookup
        # locks. Its lmcache-driven transfer releases them on both success and
        # failure (including partial failures). Releasing the range again from
        # the scheduler would decrement every TP rank's read locks twice. The
        # scheduler cannot distinguish a pre-submit failure; that rarer case is
        # left to request end_session/server TTL cleanup instead.
        self._lookups.pop(lookup_id, None)
        self._adapter.cleanup_lookup_result(lookup_id)

    def hit_tokens(self, lookup_id: str) -> int | None:
        state = self._lookups.get(lookup_id)
        return None if state is None else state.hit

    def clear_lookup_status(self, lookup_id: str) -> None:
        state = self._lookups.pop(lookup_id, None)
        if state is not None and state.hit is None:
            result = self._adapter.check_lookup_result(lookup_id)
            if result is None:
                logger.warning(
                    "LMCache MP lookup for request %s is still pending during "
                    "cleanup; dropping local state while server TTL/session "
                    "cleanup releases any eventual locks",
                    lookup_id,
                )
                self._adapter.cleanup_lookup_result(lookup_id)
                return
            state.hit = int(result)
        # Once retrieve has started, the transfer owns the remaining read
        # locks. Failed terminal loads call complete_retrieve(False) first.
        if (
            state is not None
            and state.retrieve_start is None
            and state.hit
            and state.hit > 0
        ):
            self._adapter.free_lookup_locks(
                token_ids=state.token_ids,
                start=0,
                end=state.hit,
                request_id=lookup_id,
            )
        self._adapter.cleanup_lookup_result(lookup_id)


class LMCacheMPConnector(KVConnectorBase):
    """Worker-side LMCache MP connector for backend-published PAGE views."""

    is_producer = False

    def __init__(self, config: Any) -> None:
        _validate_mp_config(config)
        self._config = config
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        self.kv_role = validated_kv_role(kvc)
        self._do_save = self.kv_role in ("offload", "kv_both", "kv_producer")
        self._do_load = self.kv_role in ("offload", "kv_both", "kv_consumer")
        self.block_size = offcfg._strict_integer(
            "LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        self.chunk_size: int | None = None
        self._adapter: Any = None
        self._is_kv_writer = True
        self._pending_saves: dict[str, _PendingSave] = {}
        self._pending_loads: dict[str, _PendingLoad] = {}
        self._submitting_saves: set[str] = set()
        self._submitting_loads: set[str] = set()
        # Keep in-flight IDs in the unbounded pending/submitting collections.
        # Only completed IDs enter these bounded replay tombstones.
        self._completed_save_operations: set[str] = set()
        self._completed_load_operations: set[str] = set()
        self._completed_save_operation_order: deque[str] = deque()
        self._completed_load_operation_order: deque[str] = deque()
        self._immediate_saves: set[SaveCompletionId] = set()
        self._immediate_load_failures: set[LoadCompletionId] = set()
        self._lock = threading.Lock()

    def register_kv_caches(
        self,
        _kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        if num_blocks is None:
            num_blocks = getattr(transfer_tensors, "num_blocks", None)
        if num_blocks is None:
            raise ValueError("lmcache_mp requires the scheduler block count")
        normalized_num_blocks = offcfg._strict_integer(
            "LMCache MP block count",
            num_blocks,
            minimum=1,
        )

        from aiter.dist.parallel_state import get_tp_group
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo

        tp = get_tp_group()
        rank = int(tp.rank_in_group)
        tp_size, _ = _validate_mp_config(self._config)
        requested_replication = _tp_replication_factor(self._config)
        published_replication = _published_tp_replication_factor(
            transfer_tensors,
            tp_size=tp_size,
        )
        if requested_replication > published_replication:
            raise ValueError(
                "LMCache MP TP rank collapse was requested, but the attention "
                "backend did not declare the complete PAGE layout fully "
                f"replicated (published factor={published_replication}, "
                f"TP size={tp_size})"
            )
        self._is_kv_writer = rank % requested_replication == 0
        views = _build_cache_views(
            transfer_tensors,
            num_blocks=normalized_num_blocks,
        )
        block_regions = getattr(transfer_tensors, "block_regions", None) or []
        expected = sum(int(region.unit_bytes) for region in block_regions)
        if expected != views.bytes_per_block:
            raise ValueError(
                "lmcache_mp block geometry mismatch: "
                f"views={views.bytes_per_block} transfer_regions={expected}"
            )

        adapter = _make_worker_adapter(self._config, rank)
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=layer_indices,
                tokens_per_block=self.block_size,
            )
            for layer_indices in views.layer_groups
        ]
        try:
            adapter.register_kv_caches(views.tensors, engine_group_infos=groups)
            chunk_size = offcfg._strict_integer(
                "LMCache MP chunk size",
                adapter.lmcache_tokens_per_chunk,
                minimum=1,
            )
            if chunk_size % self.block_size:
                raise ValueError(
                    f"LMCache MP chunk size {chunk_size} must be divisible by "
                    f"ATOM block size {self.block_size}"
                )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise
        self._adapter = adapter
        self.chunk_size = chunk_size
        logger.info(
            "LMCache MP registered rank=%d tensors=%d groups=%d "
            "bytes_per_block=%d chunk=%d tp_replication=%d writer=%s "
            "save=%s load=%s",
            rank,
            len(views.tensors),
            len(views.layer_groups),
            views.bytes_per_block,
            self.chunk_size,
            requested_replication,
            self._is_kv_writer,
            self._do_save,
            self._do_load,
        )

    def start_load_kv(self, metadata: Any) -> None:
        if not isinstance(metadata, LMCacheOffloadMetadata):
            return
        if self._adapter is None or self.chunk_size is None:
            raise RuntimeError("lmcache_mp KV caches are not registered")

        requests = [
            req
            for req in metadata.requests
            if (req.load_spec is not None and self._do_load)
            or (req.save_spec is not None and self._do_save)
        ]
        if not requests:
            return
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        for req in requests:
            if req.load_spec is not None and self._do_load:
                self._submit_load(req, event)
            if req.save_spec is not None and self._do_save:
                self._submit_save(req, event)

    def _block_slice(self, req: LMCacheReqMeta, start: int, end: int) -> list[int]:
        if start < 0 or end < start:
            raise ValueError(f"invalid LMCache MP token range [{start}, {end})")
        if start % self.block_size or end % self.block_size:
            raise ValueError(
                f"LMCache MP token range [{start}, {end}) must align to "
                f"block size {self.block_size}"
            )
        block_ids = list(
            req.block_ids[start // self.block_size : end // self.block_size]
        )
        expected = (end - start) // self.block_size
        if len(block_ids) != expected:
            raise ValueError(
                f"LMCache MP request {req.req_id} needs {expected} blocks for "
                f"[{start}, {end}), got {len(block_ids)}"
            )
        return block_ids

    def _submit_load(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        assert req.load_spec is not None
        completion = req.load_operation or req.req_id
        request_id = str(req.req_id)
        operation_id = _transfer_operation_id("load", completion)
        with self._lock:
            if (
                operation_id in self._completed_load_operations
                or operation_id in self._pending_loads
                or operation_id in self._submitting_loads
            ):
                raise RuntimeError(
                    f"duplicate LMCache MP load operation {operation_id!r}"
                )
            self._submitting_loads.add(operation_id)
        start = int(req.load_spec.hbm_cached_tokens)
        end = (
            int(req.load_spec.lmcache_cached_tokens)
            if req.load_spec.transfer_end_tokens is None
            else int(req.load_spec.transfer_end_tokens)
        )
        try:
            if start % self.chunk_size or end % self.chunk_size:
                raise ValueError(
                    f"load range [{start}, {end}) is not LMCache chunk aligned "
                    f"({self.chunk_size})"
                )
            block_ids = self._block_slice(req, start, end)
            op = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=[block_ids],
                start=start,
                end=end,
            )
            transfer = self._adapter.submit_retrieve_request(
                request_id,
                op,
                event,
            )
        except Exception:
            logger.exception(
                "LMCache MP load submission failed for %s",
                req.req_id,
            )
            with self._lock:
                self._submitting_loads.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                self._immediate_load_failures.add(completion)
            return
        with self._lock:
            self._submitting_loads.discard(operation_id)
            self._pending_loads[operation_id] = _PendingLoad(
                completion=completion,
                future=transfer,
            )

    def _submit_save(self, req: LMCacheReqMeta, event: Any) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        assert req.save_spec is not None
        completion = req.save_operation or req.req_id
        request_id = str(req.req_id)
        operation_id = _transfer_operation_id("save", completion)
        with self._lock:
            if (
                operation_id in self._completed_save_operations
                or operation_id in self._pending_saves
                or operation_id in self._submitting_saves
            ):
                raise RuntimeError(
                    f"duplicate LMCache MP save operation {operation_id!r}"
                )
            self._submitting_saves.add(operation_id)
        if not self._is_kv_writer:
            with self._lock:
                self._submitting_saves.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                self._immediate_saves.add(completion)
            return
        end = (len(req.token_ids) // self.chunk_size) * self.chunk_size
        start = (
            int(req.save_spec.skip_leading_tokens) // self.chunk_size
        ) * self.chunk_size
        if start >= end:
            with self._lock:
                self._submitting_saves.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                self._immediate_saves.add(completion)
            return
        try:
            block_ids = self._block_slice(req, start, end)
            op = AtomMPTransferSpec(
                token_ids=list(req.token_ids),
                block_ids=[block_ids],
                start=start,
                end=end,
            )
            transfer = self._adapter.submit_store_request(
                request_id,
                op,
                event,
            )
        except Exception:
            logger.exception(
                "LMCache MP save submission failed for %s",
                req.req_id,
            )
            with self._lock:
                self._submitting_saves.discard(operation_id)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
                self._immediate_saves.add(completion)
            return
        with self._lock:
            self._submitting_saves.discard(operation_id)
            self._pending_saves[operation_id] = _PendingSave(
                completion=completion,
                future=transfer,
            )

    def get_finished(self) -> KVConnectorOutput:
        if self._adapter is None:
            return KVConnectorOutput()
        done_load: set[LoadCompletionId] = set()
        failed_load: set[LoadCompletionId] = set()
        done_save: set[SaveCompletionId] = set()
        with self._lock:
            # Heartbeat health is a control-plane signal, not proof that GPU
            # work submitted before the failure has quiesced. Keep every real
            # future until its device event is terminal. A pre-submit drop is
            # represented by None, while LMCache's missing-registration path
            # returns an event-free terminal False future.
            for operation_id, pending in list(self._pending_saves.items()):
                terminal, _result = _terminal_future_result(pending.future)
                if terminal:
                    self._pending_saves.pop(operation_id, None)
                    _remember_operation_tombstone(
                        operation_id,
                        self._completed_save_operations,
                        self._completed_save_operation_order,
                    )
                    done_save.add(pending.completion)

            for operation_id, pending in list(self._pending_loads.items()):
                terminal, result = _terminal_future_result(pending.future)
                if not terminal:
                    continue
                self._pending_loads.pop(operation_id, None)
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
                if result is not True:
                    failed_load.add(pending.completion)
                else:
                    done_load.add(pending.completion)
            done_save.update(self._immediate_saves)
            failed_load.update(self._immediate_load_failures)
            self._immediate_saves.clear()
            self._immediate_load_failures.clear()
        return KVConnectorOutput(
            finished_loading=done_load,
            failed_loading=failed_load,
            finished_saving=done_save,
        )


class LMCacheMPConnectorScheduler(ChunkedOffloadSchedulerBase):
    """Scheduler-side LMCache MP connector for generic PAGE offload."""

    def __init__(self, config: Any) -> None:
        _validate_mp_config(config)
        kvc = getattr(config, "kv_transfer_config", {}) or {}
        validated_kv_role(kvc)
        offcfg._strict_integer(
            "LMCache MP block size",
            config.kv_cache_block_size,
            minimum=1,
        )
        adapter = _make_scheduler_adapter(config)
        try:
            extra = _extra_config(config)
            timeout = float(extra.get("lmcache.mp.lookup_timeout", 30.0))
            poll_interval = float(extra.get("lmcache.mp.lookup_poll_interval", 0.01))
            lookup_client = _MPLookupClient(
                adapter,
                timeout=timeout,
                poll_interval=poll_interval,
            )
            self._mp_adapter = adapter
            super().__init__(
                config,
                chunk_size=int(adapter.lmcache_tokens_per_chunk),
                lookup_client=lookup_client,
            )
        except Exception:
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise

    def get_num_new_matched_tokens(self, seq: Any) -> tuple[int, bool]:
        matched = super().get_num_new_matched_tokens(seq)
        sid = str(seq.id)
        hit = self._lookup_client.hit_tokens(sid)
        num_prompt = int(seq.num_prompt_tokens)
        if hit != num_prompt or num_prompt % self.chunk_size:
            return matched

        load_spec = self._load_specs.get(sid)
        if load_spec is None:
            return matched
        if self.block_size == 1:
            # Allocating prompt-1 one-token blocks cannot provide a destination
            # for the complete final LMCache chunk. Recompute instead.
            self._clear_pending_load(sid)
            return 0, False

        load_spec.transfer_end_tokens = num_prompt
        self._hit_save_floors[sid] = num_prompt
        return matched

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        metadata = super().build_connector_meta()
        for req in metadata.requests:
            if req.load_spec is not None:
                end = req.load_spec.transfer_end_tokens
                if end is None:
                    end = req.load_spec.lmcache_cached_tokens
                self._lookup_client.prepare_retrieve(
                    str(req.req_id),
                    int(req.load_spec.hbm_cached_tokens),
                    int(end),
                )
        return metadata

    def load_finished(self, req_id: Any) -> bool:
        finished = super().load_finished(req_id)
        if finished:
            raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=True,
                )
            except Exception:
                logger.warning(
                    "LMCache MP successful-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return finished

    def load_failed(self, req_id: Any) -> bool:
        raw_id = req_id.req_id if hasattr(req_id, "req_id") else req_id
        active = self._active_load_operations.get(str(raw_id))
        if isinstance(req_id, LoadOperationId):
            is_current = active is not None and active[1] == req_id
        else:
            is_current = active is None
        if is_current:
            try:
                self._lookup_client.complete_retrieve(
                    str(raw_id),
                    succeeded=False,
                )
            except Exception:
                logger.warning(
                    "LMCache MP failed-load cleanup failed for request %s",
                    raw_id,
                    exc_info=True,
                )
        return super().load_failed(req_id)

    def request_finished(self, seq: Any) -> None:
        super().request_finished(seq)
        try:
            self._mp_adapter.end_session(str(seq.id))
        except Exception:
            logger.warning(
                "LMCache MP end_session failed for request %s",
                seq.id,
                exc_info=True,
            )


__all__ = [
    "LMCacheMPConnector",
    "LMCacheMPConnectorScheduler",
]
