# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Expose ATOM's native DSV4 LMCache offload through vLLM's connector interface.

An adapter and nothing more: no transfer policy, no byte layout, no LMCache
calls of its own. Those stay in the native handler
(``atom.kv_transfer.offload.hybrid.dsv4``). LMCache's own ``LMCacheConnectorV1``
cannot be used because it assumes block ``N``'s bytes live at
``N * page_size_bytes``, while ATOM re-carves one ``FullAttentionSpec``
allocation into planes plus per-request slot state.

Two surfaces are translated: ``_AtomConfig`` presents an ATOM engine config
over ``VllmConfig``, and ``_Seq`` presents an ATOM ``Sequence`` over a vLLM
``Request`` -- one long-lived adapter per request, read live across steps.

Only PAGE is offloaded. The per-request SWA ring is never stored or fetched;
a reused prefix rebuilds it by re-forwarding its tail, matching what
``deepseek_v4_prefix_patch`` does to a local HBM hit. See ``_load_prefix_cap``.

Select it with::

    --kv-transfer-config '{"kv_connector":"DSV4LMCacheConnector",
      "kv_connector_module_path":"atom.plugin.vllm.deepseek_v4_lmcache_connector",
      "kv_role":"offload", ...}'
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)

from atom.kv_transfer.disaggregation.types import KVConnectorOutput as AtomOutput
from atom.plugin.vllm.deepseek_v4_bridge import (
    _v4_kv_fp8,
    v4_prefix_warmup_tokens,
)
from atom.plugin.vllm.deepseek_v4_kv_transfer import (
    build_deepseek_v4_transfer_tensors,
)

# A model wrapper registers the proxy under the draft name when it wraps
# `DeepseekV4MTPModel` and under the target name otherwise, so the layer is
# matched by marker rather than by one fixed name. Same pair
# `deepseek_v4_prefix_patch` uses.
_PROXY_LAYER_MARKERS = (
    ".atom_deepseek_v4_proxy",
    ".atom_deepseek_v4_draft_proxy",
)

# Only the non-fp8 fallbacks; fp8 is decided by `_v4_kv_fp8`, not by name.
_TORCH_TO_ATOM_DTYPE = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


def _apply_atom_log_level() -> None:
    """Lower the ``atom`` logger to ``ATOM_LOG_LEVEL``, if that asks for one."""
    name = os.getenv("ATOM_LOG_LEVEL", "").strip().upper()
    if not name:
        return
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        logger.warning("ignoring ATOM_LOG_LEVEL=%r: unknown level", name)
        return
    atom_logger = logging.getLogger("atom")
    atom_logger.setLevel(level)
    for handler in atom_logger.handlers:
        handler.setLevel(min(handler.level, level))


def _install_idempotent_lookup_unpin() -> None:
    """Make LMCache's lookup-pin release skip a chunk ``retrieve`` already unpinned."""
    try:
        from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
    except Exception:  # optional third-party patch boundary
        logger.warning(
            "ATOM V4 LMCache offload: could not reach LMCache's CPU backend to "
            "make its lookup-pin release idempotent; expect one 'Double unpin' "
            "warning per restored block",
            exc_info=True,
        )
        return

    original = LocalCPUBackend.unpin
    if getattr(original, "_atom_idempotent_unpin", False):
        return

    def unpin(self, key) -> bool:
        cpu_lock = getattr(self, "cpu_lock", None)
        hot_cache = getattr(self, "hot_cache", None)
        if cpu_lock is None or hot_cache is None:
            return original(self, key)
        with cpu_lock:
            # `not in` then `[key]`, as upstream: indexing records a touch.
            if key not in hot_cache:
                return False
            memory_obj = hot_cache[key]
            if not memory_obj.is_pinned:
                return True
            memory_obj.unpin()
            return True

    unpin._atom_idempotent_unpin = True
    LocalCPUBackend.unpin = unpin
    logger.info(
        "ATOM V4 LMCache offload: LMCache's lookup-pin release now leaves a "
        "chunk that retrieve already unpinned alone"
    )


class _AtomConfig:
    """ATOM's engine-config surface, over ``VllmConfig``."""

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        kvc = vllm_config.kv_transfer_config
        extra = dict(getattr(kvc, "kv_connector_extra_config", None) or {})
        # kv_role is on the object, the rest in extra config; native wants one
        # flat mapping.
        extra.setdefault("kv_role", getattr(kvc, "kv_role", None) or "offload")
        self.kv_transfer_config = extra

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        self.hf_config = model_config.hf_config
        # `build_page_namespace` keys the CacheEngineKey domain off these and
        # falls back to a constant when neither is present, which would put
        # every model in one namespace: a prefix stored by one model would then
        # satisfy another's token lookup and restore KV from the wrong weights.
        # The checkpoint path, not `served_model_name`: the served name is a
        # user-facing alias and two different checkpoints can be served under
        # the same one, which would put their PAGE bytes in one namespace.
        self.model_tag = getattr(model_config, "model", None)
        self.model = self.model_tag
        # Also fingerprinted by the namespace, and both change the bytes a
        # stored page holds: the indexer's dtype changes the CSA pools, and the
        # MTP token count changes the arena geometry through `_v4_spec_steps`.
        # Hashing either as a default would let a page written under one be
        # restored under the other.
        self.index_cache_dtype = getattr(
            cache_config, "index_cache_dtype", None
        ) or getattr(model_config.hf_config, "index_cache_dtype", "auto")
        self.speculative_config = vllm_config.speculative_config
        # Passed whole, not as scalars: `pp_aware_rank_and_world` and
        # `build_lmcache_metadata` read `parallel_config.pipeline_parallel_rank`
        # off it, and a copy that carries only the sizes leaves every stage
        # claiming rank 0 -- same layer slice, same engine id, wrong metadata.
        self.parallel_config = parallel_config
        self.kv_cache_block_size = cache_config.block_size
        # ATOM's dtype table has one `fp8`; vLLM's `fp8_ds_mla` would KeyError.
        # Ask the bridge so this cannot disagree with the registered geometry.
        self.kv_cache_dtype = (
            "fp8"
            if _v4_kv_fp8(vllm_config)
            else _TORCH_TO_ATOM_DTYPE.get(model_config.dtype, "bf16")
        )
        self.tensor_parallel_size = parallel_config.tensor_parallel_size
        self.pipeline_parallel_size = parallel_config.pipeline_parallel_size
        self.decode_context_parallel_size = getattr(
            parallel_config, "decode_context_parallel_size", 1
        )
        # Zero unconditionally: a nonzero value would give native a sidecar
        # interval to nominate snapshot boundaries on. Accepted, ignored.
        self.state_checkpoint_interval_tokens = 0
        # What a CPU load must leave for the request to forward itself, read
        # off the bridge so this and a local prefix hit withhold the same.
        self.state_rollback_tokens = v4_prefix_warmup_tokens(vllm_config)


class _Seq:
    """ATOM's ``Sequence`` surface, over a vLLM ``Request``."""

    def __init__(self, request: Request):
        self._request = request
        self.block_table: list[int] = []
        # PAGE alone: claiming PAGE+SLOT is what asks native for a snapshot on
        # save and a restore on load, and the ring is rebuilt instead.
        self.has_per_req_cache = False
        self.offload_loaded = False
        self.offload_loaded_tokens = 0
        self.offload_load_start_tokens = 0
        self.offload_load_failed = False
        self.offload_handoff_boundary_tokens = 0
        self._load_operation = None
        # Held only across the lookup; see num_cached_tokens.
        self.hbm_floor_override: int | None = None
        # Held over the same window; see token_ids.
        self.lookup_prompt_cap: int | None = None

    @property
    def id(self) -> str:
        # Native requires the raw id it will compare against later, not a copy.
        return self._request.request_id

    @property
    def num_prompt_tokens(self) -> int:
        return len(self._request.prompt_token_ids)

    @property
    def token_ids(self):
        """The request's tokens, truncated while a load is being decided.

        Truncate here, not ``num_prompt_tokens``: native drops a token from a
        hit that reaches the prompt's end, which would take the capped hit off
        the 128-token grid, and LMCache serves whole chunks -- 15872 turned
        into 15871 is answered with 15744 and reported as a failed load.
        """
        if self.lookup_prompt_cap is not None:
            return self._request.all_token_ids[: self.lookup_prompt_cap]
        return self._request.all_token_ids

    @property
    def num_cached_tokens(self) -> int:
        """ATOM's D2H-safe frontier: tokens whose KV is computed and resident.

        The override covers the two windows where vLLM's own counter is wrong:
        it is still 0 while vLLM asks how much we can supply, and it already
        includes the promised tokens once the request is parked -- a floor that
        includes the load makes the load look redundant and drops it.
        """
        if self.hbm_floor_override is not None:
            return self.hbm_floor_override
        return self._request.num_computed_tokens

    @property
    def _state_initialized_after_alloc(self) -> bool:
        """Whether a forward has written this request's slot state yet."""
        return self._request.num_computed_tokens > 0

    @property
    def per_req_cache_group(self) -> int:
        """No slot group is offered: -1 means "none"."""
        return -1


class _WorkerMeta(KVConnectorWorkerMetadata):
    """ATOM's completion report, travelling worker -> scheduler.

    A transfer is one operation per rank, and it is finished only when every
    rank has finished its own copy. ``world`` is how many reports to expect,
    so ``aggregate`` can hold an id back until all of them arrive.
    """

    def __init__(
        self, atom_output: AtomOutput, world: int = 1, seen: dict | None = None
    ):
        self.atom_output = atom_output
        self._world = max(1, int(world))
        # field name -> {id: how many ranks have reported it}
        self._seen = seen if seen is not None else self._count(atom_output)

    @staticmethod
    def _count(output: AtomOutput) -> dict:
        counts: dict = {}
        for field in dataclasses.fields(output):
            value = getattr(output, field.name)
            if isinstance(value, set):
                counts[field.name] = {item: 1 for item in value}
        return counts

    def aggregate(self, other: KVConnectorWorkerMetadata):
        """Combine two ranks' reports, exposing only what every rank finished.

        Unioning the sets would mark a transfer complete on the first rank to
        report: the scheduler would wake a load while another rank is still
        copying PAGE data, or free a save's blocks while another rank still
        reads them. So ranks are counted per id, and an id appears in the
        merged output only once ``world`` of them have reported it.
        """
        if not isinstance(other, _WorkerMeta):
            return self
        world = max(self._world, other._world)
        seen: dict = {}
        for name in set(self._seen) | set(other._seen):
            tally: dict = dict(self._seen.get(name, {}))
            for item, count in other._seen.get(name, {}).items():
                tally[item] = tally.get(item, 0) + count
            seen[name] = tally

        merged = dataclasses.replace(self.atom_output)
        for field in dataclasses.fields(merged):
            mine = getattr(merged, field.name)
            theirs = getattr(other.atom_output, field.name)
            if isinstance(mine, set):
                tally = seen.get(field.name, {})
                setattr(
                    merged,
                    field.name,
                    {item for item, count in tally.items() if count >= world},
                )
            elif isinstance(mine, int):
                setattr(merged, field.name, max(mine, theirs))
        return _WorkerMeta(merged, world=world, seen=seen)


class _Metadata(KVConnectorMetadata):
    """vLLM-typed envelope around ATOM's ``LMCacheOffloadMetadata``."""

    def __init__(self, atom_metadata: Any, finished_slot_keys: tuple = ()):
        self.atom_metadata = atom_metadata
        # The slot allocator is worker-local; only the scheduler knows a
        # request is done, so finished ids ride this hop to free their slots.
        self.finished_slot_keys = finished_slot_keys


class DSV4LMCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM-facing connector delegating to ATOM's native DSV4 offload."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        _apply_atom_log_level()
        self._atom_config = _AtomConfig(vllm_config)
        self._seqs: dict[str, _Seq] = {}
        self._slot_allocator = None
        self._proxy_group_index: int | None = None
        self._atom_output: AtomOutput | None = None
        # req_id -> when its deferred free began, for the reclaim window.
        self._deferred_since: dict[str, float] = {}
        self._finished_slot_keys: set[str] = set()
        # req_id -> (block ids, HBM floor tokens, promised tokens), kept from
        # the step a load starts to the step it is reported.
        self._load_blocks: dict[str, tuple[list[int], int, int]] = {}
        self._block_size = int(
            getattr(getattr(vllm_config, "cache_config", None), "block_size", 0) or 128
        )

        # Lazy: pulls in LMCache and aiter, which must not be hard imports
        # for configs that never select this connector.
        from atom.kv_transfer.offload.connector import (
            _build_scheduler,
            _build_worker,
        )

        if role == KVConnectorRole.SCHEDULER:
            self._impl = _build_scheduler(self._atom_config)
        else:
            _install_idempotent_lookup_unpin()
            self._impl = _build_worker(self._atom_config)
        logger.info(
            "ATOM V4 LMCache connector: role=%s impl=%s",
            role.name,
            type(self._impl).__name__,
        )

    def close(self) -> None:
        """Tear the native implementation down with us.

        The worker impl owns non-daemon save/load ``ThreadPoolExecutor``s.
        ``ModelRunner.exit()`` only calls ``close`` on the connector object, so
        without this the pools are never joined and the process either hangs on
        interpreter shutdown or has atexit tear them down under an in-flight
        copy. Idempotent, as native's own ``close`` is.
        """
        close = getattr(self._impl, "close", None)
        if callable(close):
            close()

    # -- shared ----------------------------------------------------------
    @property
    def _proxy_group(self) -> int:
        """Index of the proxy's KV cache group in vLLM's per-group block ids."""
        if self._proxy_group_index is None:
            groups = getattr(self._kv_cache_config, "kv_cache_groups", None) or ()
            for index, group in enumerate(groups):
                if any(
                    marker in name
                    for name in getattr(group, "layer_names", ())
                    for marker in _PROXY_LAYER_MARKERS
                ):
                    self._proxy_group_index = index
                    break
            else:
                raise ValueError(
                    f"no KV cache group contains a layer matching "
                    f"{_PROXY_LAYER_MARKERS}; cannot map vLLM block ids onto "
                    "the V4 arena"
                )
            if self._proxy_group_index != 0:
                # vLLM resolves the ids from `get_block_ids_with_load_errors`
                # against group 0 under a single-group unpack it has not
                # generalised, so off group 0 a failed load would invalidate
                # another group's blocks and the request would resume on KV
                # that was never restored. Refuse the configuration instead of
                # running one where failed loads cannot be recovered.
                raise ValueError(
                    "the V4 proxy is KV cache group "
                    f"{self._proxy_group_index}, but vLLM resolves invalid "
                    "block ids against group 0 only; failed offload loads "
                    "cannot be recovered on this configuration"
                )
        return self._proxy_group_index

    def _proxy_block_ids(self, per_group) -> list[int]:
        """Pull the proxy group's block ids out of a per-group structure."""
        if per_group is None:
            return []
        if not isinstance(per_group, (list, tuple)) or not per_group:
            return list(per_group or ())
        if not isinstance(per_group[0], (list, tuple)):
            # A single flat list: one group, so it is the proxy's.
            return list(per_group)
        return list(per_group[self._proxy_group])

    def _refresh_block_tables(self, scheduler_output: SchedulerOutput) -> None:
        """Track the blocks vLLM added this step.

        A request resumed from preemption replaces rather than extends;
        appending would leave the freed blocks in front of the real ones, so
        later saves would store the wrong pages under a correct prefix hash.
        """
        for new in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            seq = self._seqs.get(new.req_id)
            if seq is not None:
                seq.block_table = self._proxy_block_ids(new.block_ids)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None:
            return
        resumed = self._resumed_req_ids(cached)
        deltas = getattr(cached, "new_block_ids", None) or ()
        for req_id, delta in zip(getattr(cached, "req_ids", ()), deltas):
            if delta is None:
                continue
            seq = self._seqs.get(req_id)
            if seq is None:
                continue
            blocks = self._proxy_block_ids(delta)
            if req_id in resumed:
                seq.block_table = blocks
            else:
                seq.block_table.extend(blocks)

    @staticmethod
    def _resumed_req_ids(cached) -> frozenset:
        """Ids in this step's cached batch whose block ids replace, not extend.

        A ``set[str]`` in vLLM 0.27, a positional ``list[bool]`` before it.
        """
        ids = getattr(cached, "resumed_req_ids", None)
        if ids is not None:
            return frozenset(ids)
        flags = getattr(cached, "resumed_from_preemption", None)
        if flags is None:
            return frozenset()
        return frozenset(
            req_id
            for req_id, resumed in zip(getattr(cached, "req_ids", ()), flags)
            if resumed
        )

    def _seq(self, request: Request) -> _Seq:
        """Return this request's long-lived adapter, creating it once."""
        seq = self._seqs.get(request.request_id)
        if seq is None:
            seq = _Seq(request)
            self._seqs[request.request_id] = seq
        return seq

    # ==================================================================
    # Worker side
    # ==================================================================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Hand the native worker a region map instead of the raw allocation."""
        # The name is not fixed: a model wrapper registers its proxy under the
        # draft name instead of the target one when it wraps
        # `DeepseekV4MTPModel`, so one fixed name finds nothing on an MTP
        # engine. Match the markers instead, as
        # `deepseek_v4_prefix_patch` does. One arena is described here, so more
        # than one proxy is refused rather than half-described.
        proxies = sorted(
            name
            for name in kv_caches
            if any(marker in name for marker in _PROXY_LAYER_MARKERS)
        )
        if len(proxies) > 1:
            raise NotImplementedError(
                "ATOM V4 LMCache offload describes one proxy arena, but this "
                f"engine registered {len(proxies)}: {proxies}"
            )
        if not proxies:
            raise ValueError(
                "ATOM V4 LMCache offload found no proxy KV cache matching "
                f"{_PROXY_LAYER_MARKERS}; got {sorted(kv_caches)}"
            )
        proxy = kv_caches[proxies[0]]
        transfer_tensors = build_deepseek_v4_transfer_tensors(proxy, self._vllm_config)
        self._impl.register_kv_caches(
            {},
            transfer_tensors=transfer_tensors,
            num_blocks=transfer_tensors.num_blocks,
        )

    def _release_finished_slots(self, keys) -> None:
        """Hand finished requests' state slots back to the bridge allocator."""
        if not keys:
            return
        allocator = self._slot_allocator
        if allocator is None:
            from atom.plugin.vllm import deepseek_v4_bridge as bridge

            allocator = self._slot_allocator = getattr(
                bridge, "ATOM_V4_SLOT_ALLOCATOR", None
            )
            if allocator is None:
                return
        for key in keys:
            allocator.release(key)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, _Metadata):
            return
        self._release_finished_slots(metadata.finished_slot_keys)
        atom_metadata = metadata.atom_metadata
        if atom_metadata is None:
            return
        self._record_load_extents(atom_metadata)
        self._impl.start_load_kv(atom_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op: native transfers whole requests, not layers."""
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """No-op: the native scheduler emits saves in the load metadata."""
        return

    def wait_for_save(self):
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfers in vLLM's two-set form."""
        output = self._impl.get_finished()
        if not isinstance(output, AtomOutput):
            done_sending, done_recving = output
            return set(done_sending), set(done_recving)

        self._atom_output = output

        # Failures surface too, or the parked request waits forever.
        done_recving = {_req_id_of(x) for x in output.finished_loading}
        done_recving |= {_req_id_of(x) for x in output.finished_recving}
        done_recving |= {_req_id_of(x) for x in output.failed_loading}
        done_recving |= {_req_id_of(x) for x in output.failed_recving}
        return None, done_recving or None

    def _record_load_extents(self, atom_metadata: Any) -> None:
        """Remember which blocks each load this step is filling."""
        for req in getattr(atom_metadata, "requests", None) or ():
            load_spec = getattr(req, "load_spec", None)
            if load_spec is None:
                continue
            block_ids = list(getattr(req, "block_ids", None) or ())
            if not block_ids:
                continue
            if len(self._load_blocks) > 100000:
                self._load_blocks.clear()
            self._load_blocks[str(req.req_id)] = (
                block_ids,
                int(getattr(load_spec, "hbm_cached_tokens", 0) or 0),
                int(getattr(load_spec, "lmcache_cached_tokens", 0) or 0),
            )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Name the blocks a failed load left unusable.

        Only ``[hbm_floor, lmc)``. Must run after ``get_finished`` and before
        ``build_connector_worker_meta`` clears the cached output. The ids are
        the proxy group's, which `_proxy_group` has already refused unless it
        is group 0 -- the only group vLLM resolves these against.
        """
        output = self._atom_output
        if output is None:
            return set()
        invalid: set[int] = set()
        for field in ("failed_loading", "failed_recving"):
            for item in getattr(output, field, None) or ():
                entry = self._load_blocks.pop(_req_id_of(item), None)
                if entry is None:
                    continue
                block_ids, hbm_tokens, tokens = entry
                span = len(block_ids)
                if tokens > 0:
                    span = min(span, -(-tokens // self._block_size))
                # Blocks below the floor are vLLM's own cached prefix, shared
                # with other holders: naming one would have vLLM recompute it
                # in place and tear a co-holder's values.
                start = max(0, hbm_tokens) // self._block_size
                invalid.update(block_ids[start:span])
        for field in ("finished_loading", "finished_recving"):
            for item in getattr(output, field, None) or ():
                self._load_blocks.pop(_req_id_of(item), None)
        if invalid:
            logger.warning(
                "ATOM V4 LMCache offload: %d blocks reported invalid after a "
                "failed load; vLLM will recompute those prefixes",
                len(invalid),
            )
        return invalid

    def build_connector_worker_meta(self):
        """Carry ATOM's completions to the scheduler unflattened."""
        output = self._atom_output
        self._atom_output = None
        if output is None:
            return None
        # How many ranks have to report before `aggregate` exposes an id.
        return _WorkerMeta(output, world=self._atom_config.tensor_parallel_size)

    # ==================================================================
    # Scheduler side
    # ==================================================================
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """How many tokens beyond ``num_computed_tokens`` we can supply."""
        seq = self._seq(request)
        seq.hbm_floor_override = int(num_computed_tokens)
        seq.lookup_prompt_cap = self._load_prefix_cap(request)
        try:
            tokens, is_async = self._impl.get_num_new_matched_tokens(seq)
        finally:
            seq.lookup_prompt_cap = None
        if is_async and tokens and not self._load_survives_alloc(seq):
            tokens, is_async = 0, False
        # Stays pinned while the load is only promised; released in
        # update_connector_output once the transfer is reported.
        if not (is_async and tokens):
            seq.hbm_floor_override = None
        # Printed whatever the answer, so tier attribution never has to infer
        # from a missing line.
        logger.info(
            "[OFFLOAD-PROMISE] seq=%s hbm=%d promise=%d async=%d",
            seq.id,
            int(num_computed_tokens),
            int(tokens or 0),
            int(bool(is_async)),
        )
        return tokens, is_async

    def _load_prefix_cap(self, request: Request) -> int:
        """Highest prefix a load may supply, so the tail is re-forwarded.

        A cap, not a subtraction from the hit: whatever the lookup finds, the
        request forwards at least the warmup before its first new token.
        Floored to a whole block, or the ring's first tokens stay unforwarded.
        """
        warmup = int(getattr(self._atom_config, "state_rollback_tokens", 0) or 0)
        prompt = len(request.prompt_token_ids)
        if warmup <= 0:
            return prompt
        block = self._block_size or 128
        return max(0, ((prompt - warmup) // block) * block)

    def _load_survives_alloc(self, seq: _Seq) -> bool:
        """Whether native will still want this load once blocks are allocated.

        A promise parks the request until a load is reported. Native decides
        again after allocation, where a rejection clears only its own
        bookkeeping and leaves the request waiting forever, so its own decision
        function is asked here rather than a copy of its rules.
        """
        decide = getattr(self._impl, "_decide_load_after_alloc", None)
        spec = getattr(self._impl, "_load_specs", {}).get(str(seq.id))
        if decide is None or spec is None:
            return True
        try:
            should_load, reason = decide(seq, spec)[:2]
        except Exception:
            # A guard that fails must not withdraw a load native asked for.
            logger.exception(
                "ATOM V4 LMCache offload: pre-allocation load check failed "
                "for %s; keeping the promise",
                seq.id,
            )
            return True
        if should_load:
            return True
        logger.debug("[OFFLOAD-PROMISE-WITHDRAWN] seq=%s reason=%s", seq.id, reason)
        cancel = getattr(self._impl, "cancel_pending_load", None)
        if cancel is not None:
            cancel(seq)
        return False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        seq = self._seq(request)
        if blocks is not None:
            seq.block_table = self._proxy_block_ids(blocks.get_block_ids())
        self._impl.update_state_after_alloc(seq)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Wrap whatever the native scheduler decided this step."""
        self._refresh_block_tables(scheduler_output)
        finished_slot_keys = tuple(self._finished_slot_keys)
        self._finished_slot_keys.clear()
        return _Metadata(self._impl.build_connector_meta(), finished_slot_keys)

    def update_connector_output(self, connector_output: Any):
        """Feed worker completions back into native bookkeeping."""
        self._reap_freed_seqs(connector_output)
        self._feed_native_completions(connector_output)
        self._release_deferred_frees(connector_output)

    def _feed_native_completions(self, connector_output: Any) -> None:
        process = getattr(self._impl, "process_completions", None)
        if process is None:
            return
        meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(meta, _WorkerMeta):
            self._release_pinned_floors(meta.atom_output)
            process(meta.atom_output)
            return
        output = AtomOutput(
            finished_sending=set(connector_output.finished_sending or ()),
            finished_recving=set(connector_output.finished_recving or ()),
        )
        self._release_pinned_floors(output)
        process(output)

    def _release_deferred_frees(self, connector_output: Any) -> None:
        """Free the blocks vLLM is holding for saves that have since landed."""
        if not self._deferred_since:
            return
        now = time.monotonic()
        defer = getattr(self._impl, "should_defer_free", None)
        timeout = self._save_abandon_timeout_s()
        release: set[str] = set()
        for req_id, since in self._deferred_since.items():
            seq = self._seqs.get(req_id)
            if defer is None:
                # Nothing to wait on: this impl cannot hold blocks for a save,
                # so it should never have deferred one.
                release.add(req_id)
                continue
            # Without an adapter the predicate cannot be asked; hold, and let
            # the reclaim window be the way out.
            if seq is not None and not defer(seq):
                release.add(req_id)
                continue
            if timeout > 0 and now - since >= timeout:
                logger.warning(
                    "[OFFLOAD-DEFERRED-FREE-RECLAIM] seq=%s held=%.0fs: no save "
                    "completion ever arrived; freeing the blocks past LMCache's "
                    "pin timeout and dropping the save",
                    req_id,
                    now - since,
                )
                self._abandon_save(req_id)
                release.add(req_id)
        if not release:
            return
        for req_id in release:
            self._deferred_since.pop(req_id, None)
            self._seqs.pop(req_id, None)
        connector_output.finished_sending = (
            set(getattr(connector_output, "finished_sending", None) or ()) | release
        )

    def _save_abandon_timeout_s(self) -> float:
        """Native's reclaim window, or 0 when the operator disabled reclamation."""
        callback = getattr(self._impl, "save_abandon_timeout_s", None)
        return float(callback()) if callable(callback) else 0.0

    def _abandon_save(self, req_id: str) -> None:
        """Drop a save native still thinks is pending, so its bookkeeping clears."""
        callback = getattr(self._impl, "abandon_save", None)
        if callable(callback):
            callback(req_id)

    def _reap_freed_seqs(self, connector_output: Any) -> None:
        """Drop adapters for requests vLLM has now finished freeing."""
        for req_id in getattr(connector_output, "finished_sending", None) or ():
            self._seqs.pop(str(req_id), None)
            self._deferred_since.pop(str(req_id), None)

    def _release_pinned_floors(self, output: Any) -> None:
        """Stop overriding the floor for loads that are no longer outstanding."""
        for field in (
            "finished_loading",
            "finished_recving",
            "failed_loading",
            "failed_recving",
        ):
            for item in getattr(output, field, None) or ():
                seq = self._seqs.get(_req_id_of(item))
                if seq is not None:
                    seq.hbm_floor_override = None

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        # Before the early return: an untracked request still holds a slot
        # bound by an ordinary forward.
        self._finished_slot_keys.add(request.request_id)
        seq = self._seqs.get(request.request_id)
        if seq is None:
            return False, None
        seq.block_table = list(block_ids)
        self._impl.request_finished(seq)
        defer = getattr(self._impl, "should_defer_free", None)
        keep_blocks = bool(defer(seq)) if defer is not None else False
        if keep_blocks:
            # vLLM holds these blocks until we name the request and calls
            # back for nothing else, so start the reclaim clock.
            self._deferred_since[request.request_id] = time.monotonic()
        else:
            self._seqs.pop(request.request_id, None)
        return keep_blocks, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA entry point, and the one vLLM actually calls for this connector."""
        return self.request_finished(request, self._proxy_block_ids(block_ids))


def _req_id_of(completion: Any) -> str:
    """Reduce an ATOM completion to its request id."""
    return str(getattr(completion, "req_id", completion))
