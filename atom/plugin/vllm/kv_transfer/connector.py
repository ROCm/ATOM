# SPDX-License-Identifier: MIT
"""vLLM KV connector that drives ATOM's byte-level LMCache offload.

Why this exists rather than pointing vLLM at LMCache's own connector:

LMCache's GPU connectors accept only the clean NHD/HND family and pick ONE
format for the whole model. MiniMax-M3 registers three physical layouts at once
(dense K/V interleaved, sparse K/V in separate regions, plus a DSA index cache),
so no single format describes it -- and on ROCm the paths that could describe it
are unavailable anyway: the per-layer-format connector (V3) is off by default
and hangs on M3, and the multi-process path needs cupy, which LMCache's
``platform/rocm`` does not provide.

ATOM already solved this for its native engine by not asking LMCache to
understand the layout at all: ``DenseKVByteCodec`` gathers whole paged blocks
into a chunk-major uint8 blob, and LMCache only ever stores opaque bytes. That
codec is reused verbatim here; this module is the adapter that lets vLLM drive
it, so the plugin path gets the same guarantee the native path already tests
(byte-identical round-trip).

Layer-granular hooks are deliberately inert: ATOM moves a whole request's blocks
per transfer, not one layer at a time.

Hybrid models (Kimi-K3: MLA attention plus KDA recurrent layers) get a second
leg. vLLM gives them two KV cache groups, and a restored attention prefix is
only correct if the recurrent state at the same token boundary is restored with
it. The attention group keeps the byte-codec path described above; the recurrent
group is moved by ``kda_state``, whose module docstring explains why it cannot
share it. Everything joint lives here: the lookup cap that keeps a missing state
from ever producing a half-restored prefix, the block pins that hold a
handed-off boundary block across its store, and the two-leg join that only
reports a load finished once both halves have landed.
"""

import logging
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

from atom.kv_transfer.disaggregation.types import ConnectorCompletion
from atom.kv_transfer.offload._offload_common import offload_save_abandon_timeout_s
from atom.plugin.vllm.kv_transfer.kda_state import (
    KdaBoundaryPlanner,
    KdaPageViews,
    KdaStateTier,
    build_layout_id,
    find_mamba_groups,
    step_boundary_offloads,
    summarize_layout_id,
)
from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    gather_group_tensors,
    resolve_block_count,
    split_kv_caches_by_group,
)
from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config
from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = logging.getLogger("atom")

# How often the stale-save reconcile actually runs. `build_connector_meta` is
# called every step, but the window it enforces is minutes long, so matching the
# native engine's reconcile cadence (`model_engine.scheduler`) keeps the per-step
# cost off the scheduling path.
_SAVE_RECONCILE_INTERVAL_S = 5.0


class AtomOffloadMetadata(KVConnectorMetadata):
    """Carries ATOM's offload metadata through vLLM's connector plumbing.

    ATOM's ``LMCacheOffloadMetadata`` derives from ATOM's own ConnectorMetadata,
    not vLLM's, so it cannot be returned directly from ``build_connector_meta``.
    Wrapping keeps ATOM's descriptors intact instead of flattening them into a
    vLLM-shaped copy that would then have to be kept in sync.
    """

    def __init__(
        self,
        inner,
        preempted_req_ids=(),
        release_req_ids=(),
        kda_stores=(),
        kda_loads=(),
    ) -> None:
        super().__init__()
        self.inner = inner
        # Requests vLLM preempted during this same ``schedule()`` call. Their
        # blocks are already back in the free pool, so the worker has to fence
        # every transfer still reading them before the forward runs.
        self.preempted_req_ids: list[str] = list(preempted_req_ids)
        # Deferred-free requests whose save has now landed on every rank. The
        # worker echoes these as ``finished_sending``, which is the only thing
        # that frees blocks vLLM is holding on the connector's behalf.
        self.release_req_ids: list[str] = list(release_req_ids)
        # Hybrid models only: recurrent boundary states to persist, and the
        # boundary state each parked request needs back. Empty lists on every
        # single-group model, which is every model but K3 today.
        self.kda_stores = list(kda_stores)
        self.kda_loads = list(kda_loads)


class AtomOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Per-rank offload completions on their way back to the scheduler half.

    vLLM's ``KVConnectorOutput`` carries only two id sets and both are spoken
    for: ``finished_recving`` releases a parked request, ``finished_sending``
    frees its blocks. ATOM's scheduler needs two further facts that fit neither
    -- which saves landed (so the next chunk of the same request may be
    dispatched; at most one save per request is ever in flight) and which loads
    failed (so the un-loaded range is not recorded as persisted) -- and needs
    them for requests vLLM considers ordinary and running.

    Counts rather than sets, because a save has landed only once EVERY rank has
    written its shard: acting on the first report would drop
    ``should_defer_free`` while a slower rank is still reading the blocks.
    ``aggregate`` sums the ranks of one step; the scheduler half sums the steps.

    ``completions`` carries ATOM's connector-owned completion channels, which
    are a third thing vLLM has no slot for. They are what releases the save's
    block lease, and without them ``should_defer_free`` never goes false: the
    per-operation owner and source-safe maps only ever grow, every finished
    request keeps its blocks, and the pool deadlocks on capacity with nothing
    running. The events are forwarded verbatim rather than interpreted --
    channel semantics belong to the layout, not to this adapter.
    """

    def __init__(
        self,
        saved=None,
        load_failed=None,
        completions=None,
        state_stored=None,
        state_store_failed=None,
        state_load_failed=None,
    ) -> None:
        self.saved: dict[str, int] = dict(saved or {})
        self.load_failed: dict[str, int] = dict(load_failed or {})
        self.completions: list[ConnectorCompletion] = list(completions or ())
        # Recurrent-state reports. The store pair is keyed by the scheduler's
        # own op id rather than by request id, because a boundary outlives the
        # request that produced it -- it is keyed by prefix content, and the
        # pinned block has to be released on the last rank's report whether or
        # not that request still exists.
        self.state_stored: dict[int, int] = dict(state_stored or {})
        self.state_store_failed: dict[int, int] = dict(state_store_failed or {})
        self.state_load_failed: dict[str, int] = dict(state_load_failed or {})

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "AtomOffloadWorkerMetadata":
        for field in (
            "saved",
            "load_failed",
            "state_stored",
            "state_store_failed",
            "state_load_failed",
        ):
            mine = getattr(self, field)
            for req_id, count in (getattr(other, field, None) or {}).items():
                mine[req_id] = mine.get(req_id, 0) + int(count)
        self.completions.extend(getattr(other, "completions", None) or ())
        return self

    def __repr__(self) -> str:
        return (
            f"AtomOffloadWorkerMetadata(saved={self.saved}, "
            f"load_failed={self.load_failed}, "
            f"completions={self.completions}, "
            f"state_stored={self.state_stored}, "
            f"state_store_failed={self.state_store_failed}, "
            f"state_load_failed={self.state_load_failed})"
        )


class AtomLMCacheOffloadConnector(KVConnectorBase_V1, SupportsHMA):
    """Drives ``atom.kv_transfer.offload`` from vLLM's connector API.

    ``SupportsHMA`` is not optional for a hybrid model, and not cosmetic for the
    others. vLLM refuses to build a connector that lacks it when the hybrid
    memory allocator is on (``KVConnectorFactory.create_connector``), and
    silently turns HMA *off* when one is configured without it
    (``config/vllm.py``) -- and a hybrid SSM model then fails at startup. So
    without this base K3 does not mis-save, it does not boot.
    """

    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        # kv_cache_config is required of out-of-tree v1 connectors: the factory
        # rejects the 2-argument signature outright, and the base class stores
        # it for the group-aware paths.
        super().__init__(vllm_config, role, kv_cache_config)
        self._vllm_config = vllm_config
        self._config = build_offload_config(vllm_config)
        self._worker = None
        self._scheduler = None

        self._seqs = SeqViewRegistry()
        # Requests parked on a promised load, and how many steps ago. A promise
        # that never turns into a dispatched load is an unrecoverable hang, so
        # it is at least named in the log. See `_check_promised_loads`.
        self._promised_loads: dict[str, int] = {}

        # Worker half: the release list handed over by the scheduler for this
        # step, plus the completions owed back to it. Both are drained by the
        # hooks the model runner calls once per step.
        self._pending_release_ids: list[str] = []
        self._worker_saved: dict[str, int] = {}
        self._worker_load_failed: dict[str, int] = {}
        self._worker_completions: list[ConnectorCompletion] = []

        # Scheduler half: requests whose free vLLM is holding for us, split by
        # what they are still waiting for -- the save to land, or the worker to
        # echo the release back. See `_collect_releases`.
        self._deferred_frees: set[str] = set()
        # req_id -> monotonic time the deferral started, the clock
        # `_reconcile_stale_saves` bounds the wait against.
        self._deferred_free_at: dict[str, float] = {}
        self._releases_in_flight: set[str] = set()
        # Per-request rank tallies for the two facts that travel as worker
        # metadata rather than as one of vLLM's two id sets.
        self._save_reports: dict[str, int] = {}
        self._load_failure_reports: dict[str, int] = {}
        # channel/operation key -> [ranks reported, succeeded on all of them,
        # monotonic time the first rank reported]. The timestamp is carried so a
        # stale quorum can be identified in the log; the wait itself is bounded
        # by the deferral clock -- see `_reconcile_stale_saves`.
        self._completion_reports: dict[tuple, list] = {}
        self._next_save_reconcile_at = 0.0
        # vLLM's GPU block pool, handed over at `bind_gpu_block_pool`. Holding a
        # refcount on exactly the blocks a save is still reading is what lets
        # the rest of a finished request's table go back immediately.
        self._gpu_block_pool = None
        self._world_size = max(
            1, int(getattr(vllm_config.parallel_config, "world_size", 1) or 1)
        )

        # ---- hybrid (several KV cache groups) -------------------------------
        # Resolved once here so both halves agree on which group is which. A
        # model with no mamba group leaves every field below inert and takes
        # exactly the code path it takes today.
        self._mamba_groups = find_mamba_groups(
            getattr(kv_cache_config, "kv_cache_groups", None) or ()
        )
        self._attn_group_id = self._resolve_attention_group(kv_cache_config)
        # Worker half of the recurrent leg.
        self._kda_tier = None
        # Requests whose recurrent load is still outstanding, the dense halves
        # waiting on them, and the results waiting for a dense half. A load is
        # only reported to vLLM once both legs are in; see `_join_kda`.
        self._kda_expect: set[str] = set()
        self._kda_dense_done: set[str] = set()
        self._kda_results: dict[str, Any] = {}
        self._kda_error_blocks: set[int] = set()
        self._worker_state_stored: dict[int, int] = {}
        self._worker_state_store_failed: dict[int, int] = {}
        self._worker_state_load_failed: dict[str, int] = {}
        # KDA stores wait until `wait_for_save`. That hook runs after the
        # target forward and, when speculative decoding defers connector
        # finalize, after `postprocess_mamba` has folded the accepted tokens
        # into the boundary page. Recording the fence in `start_load_kv`
        # copies the page before that write.
        self._pending_kda_stores: list = []
        # No-forward steps never call `wait_for_save`. Nothing writes the
        # boundary page on those steps, so `get_finished` may flush instead.
        self._kda_flush_stores_in_get_finished = False
        # Scheduler half of the recurrent leg.
        self._kda_planner = None
        self._state_load_failure_reports: dict[str, int] = {}
        # vLLM Request objects for the requests this connector has seen. The
        # recurrent key is derived from `request.block_hashes`, and a boundary
        # hand-off names only a request id. These are the scheduler's own
        # objects, shared not copied, and dropped when the request finishes.
        self._requests: dict[str, Any] = {}

        if role == KVConnectorRole.WORKER:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

            self._worker = DenseOffloadConnector(self._config)
        else:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

            self._scheduler = DenseOffloadScheduler(self._config)
            self._init_kda_planner(vllm_config, kv_cache_config)

    def _resolve_attention_group(self, kv_cache_config) -> int:
        """Which group the existing byte-codec path owns.

        One, and it has to be one: ``DenseOffloadConnector`` builds a single
        codec over a single block geometry. Two attention groups would need two,
        and silently registering only the first would offload half the layers
        under keys that claim the whole prefix.
        """
        groups = getattr(kv_cache_config, "kv_cache_groups", None) or ()
        if len(groups) <= 1:
            return 0
        mamba_ids = {group_id for group_id, _ in self._mamba_groups}
        others = [gid for gid in range(len(groups)) if gid not in mamba_ids]
        if len(others) != 1:
            raise ValueError(
                "ATOM offload connector: expected exactly one non-recurrent KV "
                f"cache group, found {others}; the byte codec registers one "
                "block geometry and cannot span several"
            )
        return others[0]

    def _init_kda_planner(self, vllm_config, kv_cache_config) -> None:
        """Stand up the scheduler half of the recurrent leg, if there is one."""
        if not self._mamba_groups:
            return
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        group_ids = tuple(group_id for group_id, _ in self._mamba_groups)
        spec = self._mamba_groups[0][1]
        cache_mode = getattr(vllm_config.cache_config, "mamba_cache_mode", None)
        if cache_mode != "align":
            # Any other mode keeps the recurrent state somewhere this connector
            # has no hand-off for, so a boundary block id would be a guess. The
            # failure would be a wrong restored state, which nothing reports.
            raise ValueError(
                "ATOM offload connector: a hybrid model needs "
                f"--mamba-cache-mode align to offload KV, got {cache_mode!r}; "
                "vLLM hands off exact boundary state blocks only in that mode"
            )
        _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        self._kda_planner = KdaBoundaryPlanner(
            group_ids=group_ids,
            mamba_block_size=int(spec.block_size),
            hash_block_size=int(hash_block_size),
            chunk_size=int(self._scheduler.chunk_size),
            world_size=self._world_size,
        )
        self._scheduler.install_hit_cap_hook(self._kda_planner.cap_hit)
        logger.info(
            "ATOM LMCache offload: recurrent state leg on group(s) %s "
            "(mamba_block=%d, hash_block=%d, chunk=%d)",
            ",".join(str(group_id) for group_id in group_ids),
            int(spec.block_size),
            int(hash_block_size),
            int(self._scheduler.chunk_size),
        )

    # ---- worker side --------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Translate vLLM's flat registration and hand it to ATOM's codec.

        The layer modules go along with the tensors: M3 keeps its fp8 KV scales
        (one fp32 per token per head) on the layer, not in this dict, and a
        transfer that moves the mantissas without them silently dequantises a
        restored block against the previous occupant's scale.

        On a hybrid model the dict holds two groups' layers with two different
        page geometries. They are separated first, because the codec derives
        every segment's stride from one shared block count and mixing the groups
        would slice both at the wrong granularity.
        """
        groups = getattr(self._kv_cache_config, "kv_cache_groups", None) or ()
        per_group = split_kv_caches_by_group(kv_caches, list(groups))
        attention_caches = per_group[self._attn_group_id]
        tensors = build_kv_cache_tensors(attention_caches, self._attention_layers())
        if not tensors:
            raise ValueError("ATOM offload connector: vLLM registered no KV caches")

        # Every segment's per-block stride is derived from num_blocks, so it
        # has to be the count of blocks vLLM's block tables name, which is what
        # `resolve_block_count` takes from the config and reconciles against the
        # tensor. The leading dimension is NOT that count on ATOM's MLA backend:
        # it asks for a kernel block size of 1, so vLLM allocates one row per
        # token and dim 0 comes back 1536x too large on Kimi-K3.
        leading_dim = int(tensors[0].k_cache.shape[0])
        num_blocks = resolve_block_count(
            leading_dim,
            int(getattr(self._kv_cache_config, "num_blocks", 0)),
            int(self._config.kv_cache_block_size),
        )

        self._worker.register_kv_caches(
            {str(t.layer_num): t for t in tensors},
            num_blocks=num_blocks,
        )
        # After the dense registration, not before: the recurrent codec shares
        # the engine, its storage manager and its LMCache identity, and none of
        # those exist until the call above has run.
        self._init_kda_tier(per_group, list(groups))
        logger.info(
            "ATOM LMCache offload: registered %d layers, num_blocks=%d "
            "(leading dim %d, block_size=%d)",
            len(tensors),
            num_blocks,
            leading_dim,
            int(self._config.kv_cache_block_size),
        )
        # Layout/dtype census. The codec moves opaque bytes, so a wrong dtype
        # never surfaces here -- it surfaces much later inside an attention
        # kernel ("Both operands must be same dtype"), with nothing pointing
        # back at registration. One line here makes that diagnosable.
        census: dict[tuple, int] = {}
        for name, tensor in sorted(kv_caches.items()):
            key = (
                "index" if name.endswith(".index_cache") else "kv",
                tuple(tensor.shape[1:]),
                str(tensor.dtype),
            )
            census[key] = census.get(key, 0) + 1
        for (kind, shape, dtype), count in sorted(census.items(), key=str):
            logger.info(
                "ATOM LMCache offload:   %d x %s tail_shape=%s dtype=%s",
                count,
                kind,
                shape,
                dtype,
            )

    def _init_kda_tier(
        self, per_group: list[dict[str, "torch.Tensor"]], groups: list[Any]
    ) -> None:
        """Stand up the worker half of the recurrent leg, if there is one."""
        if not self._mamba_groups:
            return
        from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
        from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec

        # Group order, and within a group vLLM's own layer order: the store
        # gathers and the load scatters through this same list (see
        # `gather_group_tensors`).
        specs = [spec for _, spec in self._mamba_groups]
        tensors_by_group = gather_group_tensors(
            per_group, groups, [group_id for group_id, _ in self._mamba_groups]
        )
        views = KdaPageViews(
            tensors_by_group, layout_id=build_layout_id(specs, tensors_by_group)
        )

        # Sized to one whole state image. The KV staging buffer is sized in
        # LMCache chunks and is routinely an order of magnitude smaller, so
        # sharing it would fail every transfer at `ensure_buffer`.
        gpu_connector = self._worker._engine.gpu_connector
        staged = StagedTransfer(
            gpu_connector.device,
            staging_buffer_bytes=views.entry_bytes,
            release_after_transfer=gpu_connector.release_gpu_staging_after_transfer,
        )
        meta = self._worker._lmcache_metadata
        codec = StateByteCodec(
            views,
            staged,
            views.entry_bytes,
            model_name=meta.model_name,
            world_size=int(meta.world_size),
            worker_id=int(meta.worker_id),
            layout_id=views.layout_id,
        )
        # ONE pool, shared with the paged KV. A prefix's KV chunks and its
        # boundary states are written in the same window, so they enter the same
        # LRU and cool together -- which is exactly right, because a boundary
        # whose KV has been evicted is worth nothing on its own.
        codec.bind_storage_manager(self._worker._engine.storage_manager)
        self._kda_tier = KdaStateTier(codec)
        logger.info(
            "ATOM LMCache offload: recurrent state tier up, %d layers, "
            "entry=%.2f MiB, layout=%s",
            sum(len(tensors) for tensors in tensors_by_group),
            views.entry_bytes / (1 << 20),
            summarize_layout_id(views.layout_id),
        )

    def _dispatch_kda_loads(self, metadata) -> None:
        """Issue this step's recurrent loads. Stores are flushed later.

        Loads are on the TTFT path and write blocks of requests that are
        parked, not blocks this forward is updating. Stores are the opposite:
        the page they read is written by this step's forward and, with
        speculative decoding, by ``postprocess_mamba`` after ``get_finished``.
        ``wait_for_save`` is the hook that runs after both.
        """
        tier = self._kda_tier
        if tier is None:
            return
        for load in getattr(metadata, "kda_loads", None) or ():
            self._kda_expect.add(load.req_id)
            tier.submit_load(load)

    def _stash_kda_stores(self, metadata) -> None:
        stores = getattr(metadata, "kda_stores", None) or ()
        if stores:
            self._pending_kda_stores.extend(stores)

    def _flush_kda_stores(self) -> None:
        """D2H the stashed boundary pages, fenced to the compute stream now.

        One event for the whole step. Every producer kernel queued on the
        compute stream before this call -- the forward, and mamba postprocess
        when this runs from ``wait_for_save`` -- is visible to the gather.
        ``StagedTransfer`` reads on a private stream and does not wait on
        that producer itself.
        """
        tier = self._kda_tier
        stores = self._pending_kda_stores
        self._pending_kda_stores = []
        if tier is None or not stores:
            return
        ready_event = None
        if torch.cuda.is_available():
            ready_event = torch.cuda.Event()
            ready_event.record()
        for store in stores:
            tier.submit_store(store, ready_event)

    def _join_kda(self, dense_recving: set[str]) -> set[str]:
        """Hold a finished dense load until its recurrent half has landed.

        Reporting the request the moment the KV arrives is what would make a
        half-restored prefix reachable: vLLM unparks on that report and caches
        the whole external prefix. So a request with a recurrent leg is released
        only when both halves are in, and a failed recurrent half is released
        *with* the attention blocks it invalidates.
        """
        tier = self._kda_tier
        stored, store_failed = tier.take_store_reports()
        for op_id, count in stored.items():
            self._worker_state_stored[op_id] = (
                self._worker_state_stored.get(op_id, 0) + count
            )
        for op_id, count in store_failed.items():
            self._worker_state_store_failed[op_id] = (
                self._worker_state_store_failed.get(op_id, 0) + count
            )
        self._kda_results.update(tier.take_load_results())

        released = {r for r in dense_recving if r not in self._kda_expect}
        self._kda_dense_done |= dense_recving & self._kda_expect
        for req_id in sorted(self._kda_dense_done):
            result = self._kda_results.pop(req_id, None)
            if result is None:
                continue
            self._kda_dense_done.discard(req_id)
            self._kda_expect.discard(req_id)
            released.add(req_id)
            if not result.ok:
                self._kda_error_blocks.update(result.error_block_ids)
                self._worker_state_load_failed[req_id] = (
                    self._worker_state_load_failed.get(req_id, 0) + 1
                )
                if not result.error_block_ids:
                    # finished_recving with an empty invalid set is how vLLM
                    # caches an MLA prefix whose KDA state never arrived.
                    logger.error(
                        "ATOM LMCache offload: recurrent state missing for %s "
                        "and no attention block could be named; the external "
                        "prefix cannot be invalidated",
                        req_id,
                    )
                else:
                    logger.warning(
                        "ATOM LMCache offload: recurrent state missing for %s; "
                        "invalidating %d attention blocks so the prefix is recomputed",
                        req_id,
                        len(result.error_block_ids),
                    )
        return released

    def _attention_layers(self) -> dict[str, Any]:
        """The layer modules behind vLLM's registered KV cache names.

        vLLM keeps them in the static forward context, which is where its own
        attention-metadata builders read layers from; there is no per-layer
        handle in the connector API itself.
        """
        context = getattr(
            self._vllm_config.compilation_config, "static_forward_context", None
        )
        return dict(context) if context else {}

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        inner = getattr(metadata, "inner", None)
        if inner is not None:
            self._worker.start_load_kv(inner)
        self._dispatch_kda_loads(metadata)
        self._stash_kda_stores(metadata)
        # ``kv_connector_no_forward`` builds a context with no attention
        # metadata and does not call ``wait_for_save``. There is no forward
        # and no mamba postprocess on that step, so flushing from
        # ``get_finished`` cannot race a writer.
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        self._kda_flush_stores_in_get_finished = attn_metadata is None
        # The scheduler half's release list is picked up here rather than in
        # `get_finished` because this is the hook that runs on every step --
        # including the zero-token steps the engine is only turning the crank
        # for in order to deliver exactly these reports.
        self._pending_release_ids.extend(getattr(metadata, "release_req_ids", ()) or ())

    def handle_preemptions(self, kv_connector_metadata) -> None:
        """Fence transfers still reading the blocks of a just-preempted request.

        vLLM's `_preempt_request` frees the blocks inside `schedule()` -- no
        `request_finished`, no completion report, no chance to defer -- and the
        block pool hands them to the next allocation immediately. ATOM's saves
        run on a background executor, so a save issued for the preempted request
        goes on reading blocks that by then hold someone else's KV, and it
        stores those bytes under the preempted request's token ids.

        This hook is vLLM's answer: it runs inside `execute_model` before
        `_update_states` and before the forward, which is the last point at
        which the old occupant's bytes are still the ones in the blocks.
        Blocking here is therefore correct rather than merely convenient; it is
        also how upstream's own OffloadingConnector handles preemption.

        The in-flight save is left to finish rather than abandoned: once fenced
        it reads the right bytes, so its chunks are genuinely persisted and the
        request does not have to save them again after it is resumed.
        """
        req_ids = getattr(kv_connector_metadata, "preempted_req_ids", None) or ()
        if req_ids:
            logger.debug("ATOM LMCache offload: fencing preempted %s", list(req_ids))
            self._worker.wait_for_requests(req_ids)
            if self._kda_tier is not None:
                # Same hazard on the recurrent leg: a load in flight is writing
                # into the boundary block this request no longer owns.
                self._kda_tier.wait_for_requests(req_ids)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Inert: transfers are per-request, not per-layer.

        A load is published to the scheduler through ``get_finished`` only once
        every block of the request has landed, so there is no partially-loaded
        layer for a forward to wait on.
        """

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        """Inert: saves are issued per request from ``build_connector_meta``."""

    def wait_for_save(self) -> None:
        """Flush recurrent stores after the forward that writes their pages.

        Dense saves stay fire-and-forget; blocking the forward on the D2H
        would put offload on the critical path. This hook does not wait for
        the copy. It only records the fence and submits the job.

        vLLM calls it after the target forward. With speculative decoding it
        is deferred until after the draft model and after
        ``postprocess_mamba``, which is the copy that writes the accepted
        recurrent state into the boundary page. A fence taken in
        ``start_load_kv`` lands before that copy.
        """
        self._flush_kda_stores()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Translate ATOM's four completion sets into vLLM's two.

        ``finished_recving`` is the straightforward half: a load, successful or
        failed, is what releases a request parked in WAITING_FOR_REMOTE_KVS, and
        the alternative to waking a failed one is a hang. A failure additionally
        reports its unfilled blocks through ``get_block_ids_with_load_errors``,
        which is what actually truncates the request -- reporting the id alone
        has vLLM cache the whole external prefix as though it had arrived.

        ``finished_sending`` is NOT a save completion here. vLLM reads it as
        "free this request's blocks" and asserts the request is present and
        finished before doing so, while ATOM's saves are fire-and-forget and
        routinely land mid-decode. So saves travel back as worker metadata
        instead (see ``build_connector_worker_meta``), and what this returns is
        the scheduler half's own release list, echoed back once. That also makes
        ``finished_req_ids`` redundant: the scheduler half already knows which
        requests it deferred, and it is the side that owns the decision.

        ATOM's own worker deliberately reports an empty ``finished_sending``
        because ITS scheduler reads that as a P/D producer handoff. Same name,
        two contracts; the translation lives here rather than in either side.
        """
        out = self._worker.get_finished()

        finished_recving = {_req_id_of(c) for c in out.finished_loading}
        failed = {_req_id_of(c) for c in out.failed_loading}
        if failed:
            logger.warning(
                "ATOM LMCache offload: load failed for %s; recomputing", sorted(failed)
            )
            finished_recving |= failed
        for req_id in failed:
            self._worker_load_failed[req_id] = (
                self._worker_load_failed.get(req_id, 0) + 1
            )
        for completion in out.finished_saving:
            req_id = _req_id_of(completion)
            self._worker_saved[req_id] = self._worker_saved.get(req_id, 0) + 1
        # ATOM's connector-owned channels. `finished_saving` above is the
        # legacy terminal that clears `_save_inflight`; these are what release
        # the save's block lease, and dropping them deadlocks the pool.
        self._worker_completions.extend(out.connector_completions)

        if self._kda_tier is not None:
            if self._kda_flush_stores_in_get_finished:
                self._flush_kda_stores()
                self._kda_flush_stores_in_get_finished = False
            finished_recving = self._join_kda(finished_recving)

        finished_sending = set(self._pending_release_ids)
        self._pending_release_ids.clear()
        return finished_sending, finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """GPU blocks a failed load left unfilled.

        This is the half of a load failure that changes what the model reads.
        vLLM's scheduler truncates ``num_computed_tokens`` to the first block
        named here and recomputes from there; handed an empty set it takes the
        request's appearance in ``finished_recving`` at face value, caches the
        whole external prefix, and serves the never-written blocks as if they
        held real KV. Failure is not exotic on this path -- a hit whose HBM
        boundary is not chunk aligned, or a partial LMCache mask, is enough.

        A hybrid model adds a second source: the attention blocks are filled
        correctly but the recurrent state at their boundary never arrives. Those
        blocks are just as unusable, so the two sets are one set here.
        """
        blocks = self._worker.take_load_error_blocks()
        if self._kda_error_blocks:
            blocks = set(blocks) | self._kda_error_blocks
            self._kda_error_blocks = set()
        return blocks

    def build_connector_worker_meta(self) -> AtomOffloadWorkerMetadata | None:
        """Hand this rank's save and load-failure reports to the scheduler half.

        Called once per step, immediately after ``get_finished``, so it carries
        exactly the completions that call observed. ``None`` when there are
        none: the aggregator folds only non-``None`` metadata, and a step with
        no offload activity should not pickle an empty object per rank.
        """
        if not (
            self._worker_saved
            or self._worker_load_failed
            or self._worker_completions
            or self._worker_state_stored
            or self._worker_state_store_failed
            or self._worker_state_load_failed
        ):
            return None
        meta = AtomOffloadWorkerMetadata(
            saved=self._worker_saved,
            load_failed=self._worker_load_failed,
            completions=self._worker_completions,
            state_stored=self._worker_state_stored,
            state_store_failed=self._worker_state_store_failed,
            state_load_failed=self._worker_state_load_failed,
        )
        self._worker_saved = {}
        self._worker_load_failed = {}
        self._worker_completions = []
        self._worker_state_stored = {}
        self._worker_state_store_failed = {}
        self._worker_state_load_failed = {}
        return meta

    def shutdown(self) -> None:
        if self._kda_tier is not None:
            self._kda_tier.close()
            self._kda_tier = None
        for side in (self._worker, self._scheduler):
            close = getattr(side, "close", None) or getattr(side, "shutdown", None)
            if close is not None:
                close()

    # ---- scheduler side -----------------------------------------------

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        """Take the pool the recurrent leg pins handed-off boundary blocks in.

        A boundary block is handed over while it is still an ordinary block of a
        running request: nothing in vLLM holds it for the duration of an
        asynchronous store, so the request can free it, the pool can reissue it,
        and the store then reads the new occupant. The pin is how the block is
        kept until every rank reports; `KdaBoundaryPlanner` owns both ends of it.

        The pool is also kept on `self` for the exact-lease release path. vLLM
        calls this unconditionally on the scheduler half before any request
        runs, and only that half may touch the pool; the worker half never sees
        this call and its `_gpu_block_pool` stays None, which is what keeps the
        whole-request fallback correct on a vLLM too old to call this at all.
        """
        self._gpu_block_pool = gpu_block_pool
        if self._kda_planner is not None:
            self._kda_planner.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """How many extra prompt tokens the offload tier can supply.

        ``num_computed_tokens`` is vLLM's HBM-prefix-cache frontier; ATOM reads
        the same quantity off the seq to avoid re-loading what is already
        resident, so it has to be pushed in before the lookup runs.

        A lookup hit is not yet a decision to load. ATOM weighs that separately
        -- a hit already covered by HBM, one whose boundary is not chunk
        aligned, or one too small to be worth a transfer is dropped -- and its
        native scheduler asks `should_park_for_load_after_alloc` before parking
        anything.

        vLLM has no such second chance: returning True here parks the request in
        WAITING_FOR_REMOTE_KVS, and the ONLY thing that releases it is the
        worker reporting the id in `finished_recving`. A load that is never
        issued is never reported, so the request waits forever -- the engine
        spins in `schedule()` with every GPU idle and the server is dead. Two of
        the three drop reasons are routine: the default floor is 8192 tokens,
        which is every prompt in a chat-sized workload, and with HBM prefix
        caching on, vLLM's block-aligned frontier (128) is regularly not
        chunk-aligned (256).

        So the decision is taken here, before the promise. When ATOM declines it
        has already cleared its pending-load state, and reporting no external
        tokens leaves the request to prefill normally.

        This runs before `allocate_slots` and is gated only on
        `num_computed_tokens == 0`, so a request that fails allocation is asked
        the identical question again next step. What keeps that from becoming a
        per-step tier lookup lives in ATOM's scheduler; see
        `OffloadSchedulerMixin._init_tier_hit_memo`.
        """
        seq = self._seqs.get_or_create(request)
        seq.set_num_cached_tokens(num_computed_tokens)
        self._requests[str(request.request_id)] = request
        if self._kda_planner is not None:
            # ATOM's scheduler hands the cap hook a SeqView, which carries no
            # block hashes; this is how the hook reaches the vLLM Request whose
            # lookup is running. Armed per lookup and cleared unconditionally,
            # so a hook firing outside one has nothing stale to read.
            self._kda_planner.begin_lookup(request)
        try:
            need, _ = self._scheduler.get_num_new_matched_tokens(seq)
        finally:
            if self._kda_planner is not None:
                self._kda_planner.end_lookup()
        if need <= 0:
            return 0, False
        if not self._scheduler.should_park_for_load_after_alloc(seq):
            return 0, False
        self._promised_loads[request.request_id] = 0
        return need, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        """Record the allocation, and queue the recurrent half of any hit.

        The total computed frontier is reconstructed rather than read off the
        request: vLLM sets ``request.num_computed_tokens`` *after* this call, so
        reading it here yields the pre-hit value. Summing what the lookup was
        given with what it returned is also the stronger definition -- it is by
        construction the same boundary the attention leg is about to fill.
        """
        seq = self._seqs.get_or_create(request)
        self._requests[str(request.request_id)] = request
        groups = _group_block_ids(blocks)
        # ATOM's byte codec addresses the attention group only; the recurrent
        # group is addressed by the boundary hand-off, never positionally.
        attention_blocks = groups[self._attn_group_id] if groups else []
        num_total_computed = int(seq.num_cached_tokens) + int(num_external_tokens)
        seq.set_block_table(list(attention_blocks))
        self._scheduler.update_state_after_alloc(seq)
        if self._kda_planner is not None and num_external_tokens > 0:
            self._kda_planner.resolve_load(
                request,
                groups,
                num_total_computed,
                self._attn_group_id,
                int(num_external_tokens),
                int(self._config.kv_cache_block_size),
            )

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        """Snapshot this step's transfers.

        The frontier of every scheduled request is refreshed first: ATOM decides
        which chunks are safe to save by comparing against it, and a stale value
        would either skip chunks or offer up tokens that are not computed yet.

        The frontier is also capped at what the block table actually covers.
        On ATOM's native path these two quantities live on one Sequence and
        advance together; here they arrive through two independent vLLM
        callbacks -- ``num_computed_tokens`` rides on scheduler_output, the
        block table on ``update_state_after_alloc`` -- and under chunked
        prefill they separate by a block. ATOM then sizes a save from the
        frontier and hands the (shorter) block table to LMCache alongside it,
        which fails the transfer with "LMCache token range exceeds ATOM block
        table: needed_blocks=N+1, available_blocks=N".

        Capping is the correct semantics, not just a guard: KV for tokens past
        the block table is not in any block this connector knows about, so
        offering it up was never right. The remainder is saved on a later step,
        once the allocation catches up -- which is why the block table has to be
        grown here as well. vLLM calls ``update_state_after_alloc`` exactly once
        per admission and hands it only the blocks allocated by then, so a
        prompt longer than one prefill budget leaves a permanently short table:
        the cap then pins the frontier at the first chunk forever, the save loop
        sees ``aligned == saved`` on every later step, and everything past the
        first chunk is never offloaded at all. Measured on GLM-5.2 with a
        16,384-token budget: 20k-token prompts stored exactly 16,384 tokens and
        nothing else, so half of every long prefix was invisible to the external
        tier while every metric reported success.

        Only long prompts reach this: a chat-sized prompt is allocated in one
        go, so the two quantities never separate and every gsm8k-scale test
        passes. It took an ISL-90k aiperf run to surface.

        Preemptions are settled before any of that, and deferred frees after it
        -- both for ordering reasons spelled out at their helpers.
        """
        preempted = self._handle_preempted(scheduler_output)
        # Virtual, not physical: under decode context parallelism one scheduler
        # block id covers `kv_cache_block_size * decode_context_parallel_size`
        # tokens, and the block table this frontier is clamped against is the
        # same one ATOM indexes in virtual units (`chunked_scheduler`). Pricing
        # it physically would under-report coverage by the DCP factor and clamp
        # away the tail of every prompt. Identity when DCP=1.
        block_size = int(self._scheduler.virtual_block_size)
        for req_id, new_blocks, replaces in _scheduled_block_growth(scheduler_output):
            seq = self._seqs.get(req_id)
            if seq is None:
                continue
            # Attention group only, for the reason given in
            # `update_state_after_alloc`: the codec strides one block table, and
            # the recurrent group's ids are addressed by the boundary hand-off,
            # never positionally. Concatenating the groups here would put the
            # recurrent ids at attention token offsets.
            groups = _group_block_ids(new_blocks)
            grown = list(groups[self._attn_group_id]) if groups else []
            seq.set_block_table(grown if replaces else seq.block_table + grown)
        frontiers: dict[str, int] = {}
        for req_id, num_tokens in _scheduled_frontiers(scheduler_output):
            # Unclamped, unlike the SeqView's copy below: that clamp exists to
            # keep the dense codec from striding past the *attention* block
            # table it was handed, and the recurrent leg does not index that
            # table at all -- it resolves its blocks by hash. Clamping here too
            # would drop the tail boundaries of exactly the long prompts this
            # leg is for.
            frontiers[str(req_id)] = int(num_tokens)
            seq = self._seqs.get(req_id)
            if seq is not None:
                covered = len(seq.block_table) * block_size
                seq.set_num_cached_tokens(min(int(num_tokens), covered))
        inner = self._scheduler.build_connector_meta()
        self._check_promised_loads(inner)
        # Before `_collect_releases`, so a save abandoned on this step turns
        # into a release on this step rather than on the next one.
        self._reconcile_stale_saves()
        kda_stores = self._collect_kda_stores(scheduler_output, preempted, frontiers)
        kda_loads = (
            self._kda_planner.take_loads() if self._kda_planner is not None else []
        )
        return AtomOffloadMetadata(
            inner,
            preempted,
            self._collect_releases(),
            kda_stores,
            kda_loads,
        )

    def _collect_kda_stores(self, scheduler_output, preempted, frontiers) -> list:
        """This step's recurrent boundary stores, from both sources.

        The hand-off has to be taken in the step it exists: the KV cache
        manager hands the pending offloads over while the step is being built
        and is not shipped to the workers, so there is no later step to read it
        in. It is also, on its own, empty on any model whose mamba block size
        equals its hash block size -- Kimi-K3 among them -- which is why the
        content-addressed sweep runs beside it rather than as a fallback. See
        ``kda_state``'s module docstring for why the two sources do not overlap.
        """
        if self._kda_planner is None:
            return []
        skip = set(preempted)
        skip.update(
            str(r) for r in getattr(scheduler_output, "finished_req_ids", None) or ()
        )
        for req_id in skip:
            self._kda_planner.forget_request(req_id)
        stores: list = []
        offloads = step_boundary_offloads(scheduler_output)
        if offloads:
            stores.extend(
                self._kda_planner.collect_stores(offloads, self._requests, skip)
            )
        stores.extend(
            self._kda_planner.collect_cached_boundary_stores(
                frontiers, self._requests, skip
            )
        )
        self._kda_planner.log_stats()
        return stores

    def _handle_preempted(self, scheduler_output) -> list[str]:
        """Forget the block table of every request vLLM just preempted.

        Preemption returns a request's blocks to the pool and tells the
        connector nothing: no `request_finished`, no completion, no deferral.
        What is left behind is a SeqView whose block table names blocks that now
        belong to somebody else and a frontier claiming tokens that are no
        longer resident -- and the save loop reads exactly those two, so the
        next step would offer up another request's KV under this request's
        token ids.

        Runs before the frontier refresh below because a preempted request is
        not in this step's scheduled set: nothing later would overwrite the
        stale values.

        Resetting placement is enough on the save side. Chunks already stored
        stay stored -- LMCache keys them by token content, not by location -- so
        the recomputed prefix is not saved twice. A load that was queued but not
        yet dispatched is cancelled outright: there is no longer a block table
        to load into.

        The ids also travel to the worker half, which has to fence the
        transfers that are already running. See `handle_preemptions`.
        """
        req_ids = [
            str(r) for r in getattr(scheduler_output, "preempted_req_ids", None) or ()
        ]
        for req_id in req_ids:
            if self._kda_planner is not None:
                # The recurrent destination was this request's boundary block,
                # which is back in the pool. Nothing can be loaded into it, and
                # leaving the claim pending would keep the index from ever
                # deciding whether that boundary is still there.
                self._kda_planner.forget_pending(req_id)
                # And the sweep cursor: the request comes back with its
                # frontier rewound, and a cursor left at the old high-water
                # mark would skip every boundary it recomputes.
                self._kda_planner.forget_request(req_id)
            seq = self._seqs.get(req_id)
            if seq is None:
                continue
            self._scheduler.cancel_pending_load(seq)
            seq.reset_for_preemption()
        if req_ids:
            logger.debug("ATOM LMCache offload: preempted %s", req_ids)
        return req_ids

    def _reconcile_stale_saves(self) -> None:
        """Abandon a deferred save whose rank reports are never coming.

        Every gate on the release path waits for all ranks: a save is terminal
        once `_save_reports` reaches world size, an operation's lease is dropped
        once `_completion_reports` does. Neither wait had a bound. One lost
        report -- a worker that died mid-store, a rank whose `store()` parked
        inside LMCache and neither returned nor raised -- leaves
        `should_defer_free` true forever, so vLLM holds that request's blocks
        and this adapter holds its SeqView for the life of the server. The pool
        loses that capacity permanently, and `has_pending_push_work` keeps the
        engine stepping over a request that can never finish.

        ATOM's native engine already bounds exactly this, on the same clock
        (`model_engine.scheduler._reconcile_stalled_saves`); the plugin path
        simply had no equivalent. The window is LMCache's own pin timeout plus a
        margin, which is what makes abandoning safe rather than merely
        convenient: past it the copy the deferral was protecting is provably not
        running, because `_guard` reports on both the return and the raise path,
        so a missing report means the save is parked inside LMCache and LMCache
        has already force-unpinned its source. See
        `offload_save_abandon_timeout_s`; a non-positive
        `LMCACHE_EC_PIN_TIMEOUT_SEC` disables reclamation, and this is then a
        no-op, matching the native path.

        This is the request half of the native reconcile, and it now covers
        only the requests that could not be narrowed to exact blocks. The other
        half, `reclaim_stale_leases`, reclaims *block ids*; an earlier revision
        of this docstring called that unimplementable on a vLLM connector,
        which was wrong -- `bind_gpu_block_pool` hands us vLLM's own pool, so
        `request_finished` can take refcount shares on exactly the blocks a
        save still reads and `update_connector_output` can give them back. That
        path is driven there, on this same window, and does not come through
        here. `abandon_save` is keyed by request id, which *is* the unit vLLM
        frees on the fallback path, and it clears precisely the state
        `should_defer_free` reads (`_save_inflight`, `_save_operation_*`,
        `_save_tracker`) -- so the release falls out of the `_collect_releases`
        pass that runs immediately after this.

        The quorum entries are dropped rather than forced through as a failed
        completion, because a failed store keeps its ranges unsafe by design:
        forcing one would free nothing and would also clear a newer save
        generation. A rank report arriving after the drop opens a fresh entry
        whose clock starts then -- bounded, and harmless once the save it
        belongs to has already been abandoned.
        """
        timeout = offload_save_abandon_timeout_s()
        if timeout <= 0 or not self._deferred_frees:
            return
        now = time.monotonic()
        if now < self._next_save_reconcile_at:
            return
        self._next_save_reconcile_at = now + _SAVE_RECONCILE_INTERVAL_S

        # `setdefault`, so a deferral with no recorded start -- a request parked
        # before this bookkeeping existed, or a test that seeds the set directly
        # -- starts its clock now instead of being abandoned on sight.
        stalled = [
            req_id
            for req_id in sorted(self._deferred_frees)
            if now - self._deferred_free_at.setdefault(req_id, now) >= timeout
        ]
        if not stalled:
            return
        for req_id in stalled:
            self._scheduler.abandon_save(req_id)
            self._save_reports.pop(req_id, None)
            self._load_failure_reports.pop(req_id, None)
            for key in [
                key for key in self._completion_reports if _req_id_of(key[1]) == req_id
            ]:
                del self._completion_reports[key]
        logger.warning(
            "ATOM LMCache offload: abandoned %d deferred save(s) with no "
            "completion report after %.0fs (LMCache force-unpins a stalled save "
            "without reporting it); their blocks are released so the engine does "
            "not stall: %s",
            len(stalled),
            timeout,
            stalled,
        )

    def _collect_releases(self) -> list[str]:
        """Requests whose deferred free is now safe to hand back to vLLM.

        `request_finished` returned True for each of these, so vLLM is holding
        their blocks because an ATOM save was still reading them. Nothing in
        vLLM revisits that decision -- the blocks stay held until the connector
        names the id in `finished_sending` -- so the check has to be re-run
        every step.

        After the step's own save dispatch, not before: `build_connector_meta`
        can issue a fresh save for a deferred request, and releasing it in the
        same step would free the blocks that save is about to read.

        `source_blocks_released` is what pops ATOM's save tracker, which is in
        turn what stops the save loop from ever emitting a save against blocks
        vLLM is about to reassign. Not `request_finished` a second time: that
        one also takes the P/D send claim, and the two calls are the same split
        the native `Scheduler._maybe_release_deferred` makes.
        """
        if not self._deferred_frees:
            return []
        released = []
        for req_id in sorted(self._deferred_frees):
            seq = self._seqs.get(req_id)
            if seq is not None:
                if self._scheduler.should_defer_free(seq):
                    continue
                self._scheduler.source_blocks_released(seq)
            self._seqs.drop(req_id)
            released.append(req_id)
        self._deferred_frees.difference_update(released)
        for req_id in released:
            self._deferred_free_at.pop(req_id, None)
        self._releases_in_flight.update(released)
        return released

    def has_pending_push_work(self) -> bool:
        """Keep the engine stepping while a deferred free is still owed.

        A request whose free vLLM is holding is, as far as the engine loop is
        concerned, finished; with nothing else to run it stops calling `step()`.
        Then `build_connector_meta` is never called again, the save completion
        never reaches the scheduler half, the release is never produced, and the
        blocks are never returned -- a server that has gone idle silently loses
        KV capacity. This is the hook that says there is still work.

        A pinned boundary block is the same problem in the other tier: its
        release also rides on a worker report that only a step can deliver.
        """
        if self._kda_planner is not None and self._kda_planner.has_pending_work():
            return True
        return bool(self._deferred_frees or self._releases_in_flight)

    # Steps a promised load may go undispatched before it is called out. Loads
    # are emitted on the step after the promise, so anything past a handful of
    # steps is already wrong; the margin is only so a busy scheduler does not
    # produce noise.
    _PROMISE_GRACE_STEPS = 50

    def _check_promised_loads(self, inner) -> None:
        """Name any request parked on a load that was never dispatched.

        Nothing here can rescue it: vLLM releases a parked request only when the
        worker reports the id in `finished_recving`, and the scheduler half
        cannot inject that. The promise is gated on ATOM's own park decision, so
        this should stay empty -- but when it does not, the symptom is an engine
        spinning in `schedule()` with every GPU idle and no log line at all,
        which costs hours to trace back. One line here names the request.
        """
        if not self._promised_loads:
            return
        for meta in getattr(inner, "requests", ()) or ():
            self._promised_loads.pop(str(getattr(meta, "req_id", meta)), None)
        stuck = []
        for req_id in list(self._promised_loads):
            self._promised_loads[req_id] += 1
            if self._promised_loads[req_id] > self._PROMISE_GRACE_STEPS:
                stuck.append(req_id)
                del self._promised_loads[req_id]
        if stuck:
            logger.error(
                "ATOM LMCache offload: promised a load for %s but none was "
                "dispatched within %d steps; those requests are parked in "
                "WAITING_FOR_REMOTE_KVS and cannot be released",
                sorted(stuck),
                self._PROMISE_GRACE_STEPS,
            )

    def update_connector_output(self, connector_output) -> None:
        """Feed the worker's completions back into ATOM's scheduler state.

        vLLM splits a connector across two processes and only the worker half
        sees ATOM's completion objects; this is the scheduler half's only news
        of them. Without it nothing ever clears: `_save_inflight` and the load
        lifecycle grow for the life of the process, `has_pending_work()` never
        goes quiet, and -- the one that actually hurts -- the SeqView of every
        deferred request is retained, each holding that request's prompt token
        ids. At M3's context lengths that is the difference between a bounded
        server and one that grows by most of a megabyte per request.

        The ids arrive as plain strings (that is all vLLM's KVConnectorOutput
        carries), so the `*_by_request` resolvers recover the exact operation
        identity ATOM parked.
        """
        self._absorb_worker_meta(
            getattr(connector_output, "kv_connector_worker_meta", None)
        )
        # Return the shares whose save reported, then force-return any whose
        # save never did. `reclaim_stale_leases` is ATOM's own bounded fallback
        # and it is given the same window as the whole-request deferral above,
        # so a lease can never outlive the deferral it replaced.
        if self._gpu_block_pool is not None:
            drain = getattr(self._scheduler, "take_source_safe_releases", None)
            if drain is not None:
                self._release_lease_sets(drain())
            reclaim = getattr(self._scheduler, "reclaim_stale_leases", None)
            if reclaim is not None:
                stale = reclaim(offload_save_abandon_timeout_s())
                if stale:
                    logger.warning(
                        "ATOM LMCache offload: force-releasing %d stale block "
                        "lease set(s) -- their saves never reported",
                        len(stale),
                    )
                    self._release_lease_sets(stale)
        for req_id in connector_output.finished_recving or ():
            rid = str(req_id)
            self._promised_loads.pop(rid, None)
            state_failed = self._state_load_failure_reports.pop(rid, 0) > 0
            if self._kda_planner is not None:
                # Retract the index's claim on a boundary whose bytes are gone,
                # so the next lookup caps at a boundary that is really there
                # instead of failing the same load again.
                self._kda_planner.on_load_result(rid, not state_failed)
            if state_failed or self._load_failure_reports.pop(rid, 0) > 0:
                # One rank that could not fill its shard makes the whole load a
                # failure. Routing it through `load_finished` instead would pop
                # the floor recording that the [HBM, LMCache) range is NOT
                # persisted, and the recomputed chunks would never be saved.
                self._scheduler.load_failed_by_request(rid)
            else:
                self._scheduler.load_finished_by_request(rid)
        for req_id in connector_output.finished_sending or ():
            # Our own release, echoed back: vLLM has freed the blocks. All the
            # bookkeeping happened in `_collect_releases`, before the id was
            # ever handed to the worker.
            self._releases_in_flight.discard(str(req_id))

    def _absorb_worker_meta(self, meta) -> None:
        """Tally each rank's reports and act only when the last one is in.

        A save is complete when every rank has written its shard; acting on the
        first report would clear `_save_inflight` -- and with it
        `should_defer_free` -- while a slower rank is still reading the blocks.
        Load failures are the opposite: any single rank failing means the KV is
        incomplete, so those are consumed as a flag by the `finished_recving`
        loop above rather than counted to quorum. The two are in step anyway,
        since a rank reports a failure in the same call it reports the id, and
        vLLM's aggregator holds `finished_recving` until every rank has.

        Without this, nothing ever clears `_save_inflight`: ATOM keeps at most
        one save per request in flight, so a chunked long prompt would offload
        its first chunk and silently skip the rest.
        """
        if meta is None:
            return
        self._apply_completions(getattr(meta, "completions", None) or ())
        if self._kda_planner is not None:
            # Before the loops below: a store quorum releases a pinned block and
            # publishes the boundary, and both want to be true by the time this
            # step's `finished_recving` ids are resolved.
            self._kda_planner.absorb_reports(
                getattr(meta, "state_stored", None),
                getattr(meta, "state_store_failed", None),
            )
        for req_id, count in (getattr(meta, "state_load_failed", None) or {}).items():
            rid = str(req_id)
            self._state_load_failure_reports[rid] = (
                self._state_load_failure_reports.get(rid, 0) + int(count)
            )
        for req_id, count in (getattr(meta, "load_failed", None) or {}).items():
            rid = str(req_id)
            self._load_failure_reports[rid] = self._load_failure_reports.get(
                rid, 0
            ) + int(count)
        for req_id, count in (getattr(meta, "saved", None) or {}).items():
            rid = str(req_id)
            quorums, remainder = divmod(
                self._save_reports.get(rid, 0) + int(count), self._world_size
            )
            if remainder:
                self._save_reports[rid] = remainder
            else:
                self._save_reports.pop(rid, None)
            if quorums:
                self._scheduler.save_finished_by_request(rid)

    def _apply_completions(self, completions) -> None:
        """Hand ATOM's connector-owned completions on, once every rank agrees.

        These are the events `should_defer_free` waits for. The dense layout
        emits two channels per save -- the staged source ranges becoming safe to
        overwrite, and the store landing -- and the second is what pops the
        operation's block lease. Drop them and `_save_operation_owner` only
        grows: every finished request defers forever, each pinning its blocks
        and its SeqView, until the pool has no free block left and the engine
        sits at zero running requests with one waiting on capacity.

        Quorum is by count, as for `saved`: the worker's completion set is
        drained on each `get_finished`, so a rank reports one event once.
        Failure is dominant -- a store that failed on any rank did not persist
        that range, and `_store_finished` keeps the lease so the range is not
        advertised as source-safe.

        The return value is deliberately ignored. ATOM's native worker treats a
        ``True`` here as ALSO a terminal save; on this path the same save
        already travels the legacy `finished_saving` channel (the dense
        connector emits both), and completing it twice would let a delayed
        report clear a newer save generation.
        """
        for completion in completions:
            key = completion.key
            report = self._completion_reports.get(key)
            if report is None:
                report = self._completion_reports[key] = [0, True, time.monotonic()]
            report[0] += 1
            report[1] = report[1] and bool(completion.succeeded)
            if report[0] < self._world_size:
                continue
            del self._completion_reports[key]
            handled = self._scheduler.connector_completion(
                ConnectorCompletion(key[0], key[1], report[1])
            )
            if handled is False:
                logger.warning(
                    "ATOM LMCache offload: unhandled completion channel %s", key[0]
                )

    def _release_lease_sets(self, block_id_sets) -> None:
        """Drop the refcount shares a lease transferred to this connector.

        Tail-first within each set, matching vLLM's own convention for the same
        operation, so a shared prefix is the last thing eligible for eviction.

        One share per block per emission. The scheduler keys leases by
        ``id(seq)`` and every release path (`_mark_source_safe`,
        `_release_operation_lease`, `abandon_save`, `reclaim_stale_leases`)
        removes what it emits from that request's lease set first, so a block
        arrives here exactly once per `activate_block_leases` call that
        contained it. That pairs 1:1 with the `touch` in `request_finished`,
        including when two requests protect the same shared block -- which then
        stays alive until the last of them reports.
        """
        pool = self._gpu_block_pool
        if pool is None:
            return
        for block_ids in block_id_sets or ():
            if not block_ids:
                continue
            ordered = sorted(block_ids, reverse=True)
            pool.free_blocks([pool.blocks[block_id] for block_id in ordered])

    def request_finished_all_groups(
        self, request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict | None]:
        """The finish callback for every model, hybrid or not.

        Declaring ``SupportsHMA`` makes vLLM route *all* models through this
        method and stop calling ``request_finished`` at all, so this is the one
        body rather than a hybrid-only variant of one.

        ``block_ids`` is unused for the same reason it was unused before: the
        decision to defer is about whether a save is still reading this
        request's blocks, which ATOM tracks by request, and the blocks
        themselves are vLLM's to free once it is told it may.
        """
        return self._request_finished(request)

    def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
        """Unreachable while this connector declares ``SupportsHMA``.

        vLLM dispatches on that base and never calls this for a connector that
        has it. Kept because the abstract base still declares it, and delegating
        rather than raising means a future vLLM that calls it gets the right
        behaviour instead of a crash in production.
        """
        return self._request_finished(request)

    def _request_finished(self, request) -> tuple[bool, dict | None]:
        req_id = request.request_id
        self._promised_loads.pop(req_id, None)
        self._requests.pop(str(req_id), None)
        if self._kda_planner is not None:
            self._kda_planner.forget_pending(str(req_id))
            self._kda_planner.forget_request(str(req_id))
        seq = self._seqs.get(req_id)
        if seq is not None:
            self._scheduler.request_finished(seq)
            # Blocks may still be pinned by an in-flight save; ATOM says when.
            if self._scheduler.should_defer_free(seq):
                # Ask for the exact blocks first. `protected_block_ids` returns
                # None when it cannot narrow the protection -- the layout has no
                # early-release support, or a load is in flight -- and only then
                # is holding the whole table the right answer.
                protected = None
                if self._gpu_block_pool is not None:
                    getter = getattr(self._scheduler, "protected_block_ids", None)
                    if getter is not None:
                        protected = getter(seq)
                if protected is not None:
                    pool = self._gpu_block_pool
                    if protected:
                        pool.touch([pool.blocks[block_id] for block_id in protected])
                    self._scheduler.activate_block_leases(seq, frozenset(protected))
                    self._seqs.drop(req_id)
                    # False: vLLM frees the table now, and the refcounts taken
                    # above keep exactly the blocks the save still reads.
                    # `has_pending_work()` already counts `_save_lease_blocks`,
                    # so the engine keeps stepping until they are handed back.
                    return False, None
                # vLLM will not ask a second time -- it holds the blocks until
                # the connector names the id in `finished_sending`, and
                # `_collect_releases` is what eventually produces that.
                self._deferred_frees.add(req_id)
                self._deferred_free_at.setdefault(req_id, time.monotonic())
                return True, None
            self._seqs.drop(req_id)
        return False, None


def _req_id_of(completion_id) -> str:
    """Completion ids are a bare request id, or one tagged with a generation."""
    return str(getattr(completion_id, "req_id", completion_id))


def _group_block_ids(blocks) -> tuple[list[int], ...]:
    """vLLM's allocated-block structure as one list of ids per group.

    vLLM has spelled this several ways across versions (KVCacheBlocks with
    ``get_block_ids()``, a per-group tuple of lists, or already-flat ids).

    Grouping is preserved rather than flattened, which the earlier version of
    this helper did. Flattening is exactly wrong for a hybrid model: the two
    groups have different page geometries and different block-table lengths, so
    a concatenated list gives no position any meaning -- the attention codec
    would stride into the recurrent group's ids, and the boundary row would be
    counted from the wrong place.
    """
    getter = getattr(blocks, "get_block_ids", None)
    if getter is not None:
        blocks = getter()
    if (
        isinstance(blocks, (list, tuple))
        and blocks
        and isinstance(blocks[0], (list, tuple))
    ):
        return tuple([int(b) for b in group] for group in blocks)
    return ([int(b) for b in (blocks or [])],)


def _scheduled_block_growth(scheduler_output):
    """Yield ``(request_id, new_block_ids, replaces)`` for this step.

    ``update_state_after_alloc`` fires once, when a request leaves the waiting
    queue, and carries only the blocks allocated by then. Every block allocated
    after that -- each further chunk of a chunked prefill, and each block decode
    appends -- is announced only here. A request in ``resumed_req_ids`` is one
    vLLM re-admitted after preemption: its ids REPLACE the old table rather than
    extending it, because the blocks it held went back to the pool.
    """
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = getattr(cached, "req_ids", None) or []
    new_blocks = getattr(cached, "new_block_ids", None) or []
    resumed = getattr(cached, "resumed_req_ids", None) or ()
    for req_id, blocks in zip(req_ids, new_blocks):
        if blocks is not None:
            yield req_id, blocks, req_id in resumed


def _scheduled_frontiers(scheduler_output):
    """Yield ``(request_id, num_computed_tokens)`` for this step's requests."""
    for req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        yield req.req_id, getattr(req, "num_computed_tokens", 0)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = getattr(cached, "req_ids", None) or []
    computed = getattr(cached, "num_computed_tokens", None) or []
    yield from zip(req_ids, computed)
