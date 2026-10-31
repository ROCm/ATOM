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
from atom.utils import envs

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = logging.getLogger("atom")

# How often the stale-save reconcile actually runs. `build_connector_meta` is
# called every step, but the window it enforces is minutes long, so matching the
# native engine's reconcile cadence (`model_engine.scheduler`) keeps the per-step
# cost off the scheduling path.
_SAVE_RECONCILE_INTERVAL_S = 5.0


def _v41_checkpoint_group_ids(kv_cache_config) -> tuple[int, ...]:
    """The state-checkpoint groups in key-space order.

    Ascending group id, which is the order vLLM declared them in and the
    order of the checkpoint image's units: the block in group ``j`` is unit
    ``j``. The order *is* the key space: a transfer that names the units in
    one order cannot be read back in another, so both halves derive it the
    same way rather than passing it between them.
    """
    return tuple(
        sorted(int(gid) for gid in _v41_checkpoint_group_id_set(kv_cache_config))
    )


def _v41_checkpoint_group_id_set(kv_cache_config) -> set:
    """Group ids whose layers are ALL DeepSeek-V4.1 state checkpoints."""
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
        register_v41_state_checkpoint_adapter,
    )

    is_state_checkpoint_layer = (
        register_v41_state_checkpoint_adapter().is_state_checkpoint_layer
    )

    out = set()
    for gid, group in enumerate(
        getattr(kv_cache_config, "kv_cache_groups", None) or ()
    ):
        names = list(getattr(group, "layer_names", []) or [])
        if names and all(is_state_checkpoint_layer(n) for n in names):
            out.add(gid)
    return out


def _without_v41_checkpoint_groups(groups, kv_cache_config):
    """Drop DeepSeek-V4.1's checkpoint-checkpoint group from the recurrent leg.

    The checkpoint layers are declared as a `MambaSpec` so vLLM allocates and
    prefix-caches blocks for them, which makes `find_mamba_groups` pick them
    up -- but they are ATOM's own state checkpoints, written and read inside
    the worker, not a recurrent cache to hand to the tier.

    Left in, the recurrent leg produced `KdaStore`s for them and fed them to
    the V4.1 leg, which expects its own shape: measured as
    `AttributeError: 'KdaStore' object has no attribute 'req_id'` on the
    first request, every rank, with the engine down.
    """
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
        register_v41_state_checkpoint_adapter,
    )

    is_state_checkpoint_layer = (
        register_v41_state_checkpoint_adapter().is_state_checkpoint_layer
    )

    cfg_groups = list(getattr(kv_cache_config, "kv_cache_groups", None) or ())
    kept = []
    for group_id, spec in groups or ():
        names = []
        if 0 <= int(group_id) < len(cfg_groups):
            names = list(getattr(cfg_groups[int(group_id)], "layer_names", []) or [])
        if names and all(is_state_checkpoint_layer(n) for n in names):
            continue
        kept.append((group_id, spec))
    return kept


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
        abandoned_loads=(),
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
        # ``(req_id, block_ids)`` for requests this connector parked in
        # WAITING_FOR_REMOTE_KVS and then never issued a load for. Only the
        # worker half can release a parked request, so the scheduler half's
        # only way out is to ask it to report the load as failed. The block
        # ids are the ones the promise reserved and nothing filled; without
        # them vLLM takes the release at face value and serves them as KV.
        self.abandoned_loads = list(abandoned_loads)


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
        state_refusals=None,
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
        # Why the worker declined to snapshot, by reason. The cursor guard and
        # the staging ring live there, but the leg's stats line -- the one
        # place an operator reads -- is on the scheduler half, so the tallies
        # have to travel or the alarm cannot fire.
        self.state_refusals: dict[str, int] = dict(state_refusals or {})

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "AtomOffloadWorkerMetadata":
        for field in (
            "saved",
            "load_failed",
            "state_stored",
            "state_store_failed",
            "state_load_failed",
            "state_refusals",
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
            f"state_load_failed={self.state_load_failed}, "
            f"state_refusals={self.state_refusals})"
        )


def _v41_state_checkpoints_can_serve(vllm_config) -> bool:
    """Whether a V4.1 state checkpoint may back a LOCAL prefix hit.

    Takes the config: the scheduler calls `cap_local_hit` outside any
    `set_current_vllm_config` scope. Where the checkpoint module cannot be
    imported, local hits are refused.
    """
    try:
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            register_v41_state_checkpoint_adapter,
        )
        from atom.plugin.vllm.paged_state_checkpoint import state_checkpoints_on

        return bool(
            vllm_config is not None
            and state_checkpoints_on(
                register_v41_state_checkpoint_adapter(), vllm_config
            )
        )
    except ImportError:
        # Absent mechanism means no checkpoints; any other error is a real
        # fault and must not quietly turn every V4.1 HBM hit off.
        return False


#: Options that sized or selected an in-process state pool beside the MP
#: server. That pool is gone; see `_resolve_offload_backend`.
_RETIRED_STATE_OPTIONS = (
    "lmcache.mp.state_transport",
    "lmcache.mp.state_cpu_size_gb",
    "atom.offload.v41.state_interval",
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

    # Class-level so an instance built with `object.__new__` -- the idiom the
    # tests in `tests/plugin` use to exercise one method without a vLLM config
    # -- takes the non-V4.1 path rather than raising AttributeError from a
    # branch that has nothing to do with what is under test.
    _is_deepseek_v41 = False
    # The same, for the boundary planners: a stub that sets neither has none.
    _kda_planner = None
    _v41_planner = None
    #: Block size of the group this connector's codec owns, 0 when unknown.
    #: Distinct from the engine-wide `cache_config.block_size`, which is the
    #: smallest across groups once a second group exists.
    _attn_block_size = 0

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
        # Loads promised to vLLM at lookup: request id -> [steps undispatched,
        # first token, tokens promised, parked]. A promise no load honours is
        # withdrawn through `finished_recving` so vLLM releases the request;
        # see `_settle_promised_loads`.
        self._promised_loads: dict[str, list] = {}
        # Parked promises of requests that finished before a load dispatched.
        # vLLM holds their blocks until a `finished_recving` names them, so the
        # withdrawal still goes out with the next step's metadata.
        self._finished_withdrawals: list[tuple[str, list[int]]] = []
        # Requests admitted without waiting for their lookup, by HBM hit; the
        # lookup is sent once vLLM has allocated (`update_state_after_alloc`).
        self._detach_after_alloc: dict[str, int] = {}

        # Worker half: the release list handed over by the scheduler for this
        # step, plus the completions owed back to it. Both are drained by the
        # hooks the model runner calls once per step.
        self._pending_release_ids: list[str] = []
        self._worker_saved: dict[str, int] = {}
        self._worker_load_failed: dict[str, int] = {}
        # Independent of worker metadata: vLLM's receive-failure channel must
        # also carry MP failures and promises withdrawn before dispatch.
        self._pending_failed_recving: set[str] = set()
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
        self._mamba_groups = _without_v41_checkpoint_groups(
            find_mamba_groups(getattr(kv_cache_config, "kv_cache_groups", None) or ()),
            kv_cache_config,
        )
        # DeepSeek-V4.1 registers one proxy layer whose tensor is an opaque
        # byte arena, not a per-layer K/V pair; it is carved into PAGE planes
        # by ATOM itself. `register_kv_caches` takes those planes directly
        # rather than inferring them from the registration dict.
        #
        # Set BEFORE resolving the attention group, which reads it: the class
        # attribute defaults to False, so leaving the assignment after the
        # call made the V4.1 branch there dead without any error.
        from atom.plugin.vllm.deepseek_v41_bridge import is_deepseek_v41_vllm_config

        self._is_deepseek_v41 = is_deepseek_v41_vllm_config(vllm_config)
        # The groups the state checkpoints are written into, in key-space
        # order. Both halves read this one field rather than deriving the
        # order twice; see `_v41_checkpoint_group_ids`.
        self._v41_checkpoint_group_ids = (
            _v41_checkpoint_group_ids(kv_cache_config) if self._is_deepseek_v41 else ()
        )
        self._attn_group_id = self._resolve_attention_group(kv_cache_config)
        # Transports freeze the PAGE namespace during construction. Project the
        # resolved group's geometry into our private shim first, without changing
        # vLLM's engine-wide block size (which may belong to a smaller draft group).
        if self._attn_block_size > 0:
            self._config.kv_cache_block_size = self._attn_block_size
        # Worker half of the recurrent leg.
        self._kda_tier = None
        # Requests whose recurrent load is still outstanding, the dense halves
        # waiting on them, and the results waiting for a dense half. A load is
        # only reported to vLLM once both legs are in; see `_join_kda`.
        self._kda_expect: set[str] = set()
        self._kda_dense_done: set[str] = set()
        self._kda_results: dict[str, Any] = {}
        self._kda_error_blocks: set[int] = set()
        self._kda_error_reqs: set[str] = set()
        # Promises the scheduler half withdrew. Held here until the next
        # `get_finished`, which is the only hook that can release a park.
        self._abandoned_loads: set[str] = set()
        self._abandoned_error_blocks: set[int] = set()
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
        # DeepSeek-V4.1's per-request CSA2 state, which rides the PAGE object:
        # only the scheduler half has anything to decide. None on every other
        # model.
        self._v41_planner = None
        self._state_load_failure_reports: dict[str, int] = {}
        #: Worker polls with PAGE complete but STATE still pending. Diagnostic
        #: only: polling cannot cancel or fence a transfer, so it must never
        #: release a request whose state destination may still be written.
        self._kda_join_waiting: dict[str, int] = {}
        # vLLM Request objects for the requests this connector has seen. The
        # recurrent key is derived from `request.block_hashes`, and a boundary
        # hand-off names only a request id. These are the scheduler's own
        # objects, shared not copied, and dropped when the request finishes.
        self._requests: dict[str, Any] = {}

        self._offload_backend = self._resolve_offload_backend()
        if self._is_deepseek_v41:
            self._refuse_v41_without_ride()
        elif self._mamba_groups and self._offload_backend == "mp":
            # The MP path carries recurrent state only as V4.1's checkpoint
            # groups; a mamba group's boundary state has no leg there.
            raise ValueError(
                "ATOM offload connector: recurrent (mamba) state offloads "
                "in-process only; use atom.offload.backend='inproc'."
            )
        elif role == KVConnectorRole.SCHEDULER:
            self._warn_unless_recompute()

        if role == KVConnectorRole.WORKER:
            if self._offload_backend == "mp":
                from atom.kv_transfer.offload.mp.worker import LMCacheMPConnector

                from .mp_worker_adapter import MPOffloadWorkerAdapter
                from .page_ride import NULL_BLOCK_ID

                # The PAGE-only worker, not the shell in `offload/mp/
                # connector.py`: given V4.1's checkpoint spec the shell picks
                # the native-state worker, which restores an image into an
                # Active SLOT, while here the plugin restores it from the PAGE
                # units a load fills. vLLM never hands out its null block, so
                # the MP server's default `--null-block-id` of 0 marks a chunk
                # that holds no image.
                self._worker = MPOffloadWorkerAdapter(
                    LMCacheMPConnector(self._config, null_block_id=NULL_BLOCK_ID)
                )
            else:
                from atom.kv_transfer.offload.dense.connector import (
                    DenseOffloadConnector,
                )

                self._worker = DenseOffloadConnector(self._config)
                # vLLM 0.31 refuses a flat block set for a model with more than
                # one KV cache group, so failures also go out by request.
                self._worker.track_load_error_reqs()
        else:
            if self._offload_backend == "mp":
                # The shell in `offload/mp/connector.py` is bypassed on purpose:
                # it picks its implementation inside `bind_block_manager`, and
                # the block manager it inspects is ATOM's own. In plugin mode
                # vLLM owns the block pool and there is no such object, so the
                # scheduler is constructed directly: the PAGE-only one that
                # shell selects for a layout with no state checkpoints, or for
                # V4.1, whose checkpoints vLLM holds and every transfer
                # carries, `V41StateMPScheduler`. Both derive, as does
                # `DenseOffloadScheduler`, from `ChunkedOffloadSchedulerBase`,
                # which is where every method this connector calls on the
                # scheduler half is defined.
                from atom.kv_transfer.offload.mp.scheduler import (
                    LMCacheMPConnectorScheduler,
                )

                if self._state_rides_mp:
                    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
                        v41_state_checkpoint_sizing,
                    )
                    from atom.plugin.vllm.kv_transfer.v41_mp_scheduler import (
                        V41StateMPScheduler,
                    )

                    # Connected under the checkpoint spec's namespace, as the
                    # workers register (`_publish_v41_state_checkpoints`).
                    self._scheduler = V41StateMPScheduler(
                        self._config,
                        checkpoint_spec=v41_state_checkpoint_sizing(vllm_config).spec,
                    )
                else:
                    self._scheduler = LMCacheMPConnectorScheduler(self._config)
            else:
                from atom.kv_transfer.offload.dense.connector import (
                    DenseOffloadScheduler,
                )

                self._scheduler = DenseOffloadScheduler(self._config)
            self._init_kda_planner(vllm_config, kv_cache_config)
            self._init_v41_planner(vllm_config, kv_cache_config)

    def _resolve_offload_backend(self) -> str:
        """Which ATOM offload transport this connector drives.

        ``inproc`` keeps LMCache inside the vLLM worker process, the only mode
        this connector has had. ``mp`` hands the KV planes to a separate
        LMCache server process, which is what lets one CPU tier be shared by
        every rank instead of being cut into per-rank slices.

        Under ``mp`` V4.1's state checkpoint rides the PAGE object as a
        native-state image; see :attr:`_state_rides_mp`.
        """
        options = self._config.kv_transfer_config
        retired = sorted(k for k in _RETIRED_STATE_OPTIONS if k in options)
        if retired:
            # Refused rather than ignored: each of these asked for a state pool
            # of this connector's own, and quietly riding the PAGE object
            # instead would change where a deployment's state lives with
            # nothing in its config saying so.
            raise ValueError(
                f"ATOM offload connector: {', '.join(retired)} no longer "
                "exist. Under 'mp' recurrent state always rides the PAGE "
                "object; there is no separate state pool to choose or size."
            )
        raw = options.get("atom.offload.backend", "inproc")
        backend = str(raw).strip().lower()
        if backend not in ("inproc", "mp"):
            raise ValueError(
                "ATOM offload connector: atom.offload.backend must be "
                f"'inproc' or 'mp', got {raw!r}"
            )
        return backend

    @property
    def _state_rides_mp(self) -> bool:
        """True when DeepSeek-V4.1's checkpoint groups ride the PAGE object.

        A checkpoint is PAGE units of the same proxy pool, so every transfer
        names them as a native-state image (`v41_mp_scheduler`) and the worker
        registers them through the native layout.
        """
        return (
            self._offload_backend == "mp"
            and self._is_deepseek_v41
            and bool(self._v41_checkpoint_group_ids)
        )

    def _refuse_v41_without_ride(self) -> None:
        """DeepSeek-V4.1 offloads only through the LMCache MP server.

        Its state is carried by the checkpoint groups riding the PAGE object;
        there is no other leg. In-process LMCache would need one of this
        connector's own, and without checkpoint groups there is nothing for the
        state to ride in -- either way a prefix would come back with the PAGE
        bytes and no state, which the forward refuses.
        """
        if self._offload_backend != "mp":
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 offloads through the "
                "LMCache MP server only; set atom.offload.backend='mp'."
            )
        cache_config = getattr(
            getattr(self, "_vllm_config", None), "cache_config", None
        )
        if not self._v41_checkpoint_group_ids or not getattr(
            cache_config, "enable_prefix_caching", False
        ):
            # Both halves: the groups exist with the switch alone, but nothing
            # writes or restores them without prefix caching, so the first
            # tier hit would load a checkpoint no restore consumes.
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 offload needs its state "
                "checkpoints live (ATOM_V41_STATE_CHECKPOINTS=1 with "
                "--enable-prefix-caching and --mamba-cache-mode align); the "
                "checkpoint is what carries the state with the PAGE bytes."
            )
        kv_transfer_config = getattr(
            getattr(self, "_vllm_config", None), "kv_transfer_config", None
        )
        policy = getattr(kv_transfer_config, "kv_load_failure_policy", "fail")
        if policy != "recompute":
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 offload needs "
                "kv_load_failure_policy='recompute'. Its KV groups report every "
                "load that does not land through `failed_recving` -- a declined "
                "state ride or a lost tier prefix as well as a fault -- and "
                f"{policy!r} turns each into a failed request."
            )

    #: Async lookups `_lookup_in_flight` sent in the current scheduling pass;
    #: reset by `build_connector_meta`.
    _lookups_sent_this_step = 0

    def _warn_unless_recompute(self) -> None:
        """Name what vLLM's default load-failure policy does to a declined load.

        A load declined at dispatch or a withdrawn promise is reported through
        `failed_recving`. Under `fail` that request ends with an error rather
        than recomputing; V4.1 refuses the policy outright, other models are
        told once at startup.
        """
        kv_transfer_config = getattr(self._vllm_config, "kv_transfer_config", None)
        policy = getattr(kv_transfer_config, "kv_load_failure_policy", "fail")
        if policy != "recompute":
            logger.warning(
                "ATOM offload connector: kv_load_failure_policy=%r ends a request "
                "whose tier load is declined or lost with an error; set "
                "'recompute' to recompute it instead.",
                policy,
            )

    def _page_tp_replication_factor(self) -> int:
        """How many TP ranks hold byte-identical copies of this PAGE layout.

        The multiprocess tier can store one copy for the whole TP group instead
        of one per rank, but only where the bytes really are identical: handing
        rank 0's KV to rank 3 is a wrong answer that nothing reports. ATOM's MP
        deployment predicts the property from the model config (``auto`` reads
        ``kv_lora_rank``) and then refuses to register unless the backend
        declares it as well, so this side must answer from something other than
        that same config or the cross-check checks nothing.

        vLLM's own cache spec is that independent instrument. ``MLAAttentionSpec``
        means the group caches the shared latent, which every rank computes
        before any TP-sharded projection -- the same property ATOM's MLA backend
        declares natively (``aiter_mla.py``), index planes included. Anything
        else keeps the sharded answer: vLLM splits KV heads across TP, so the
        ranks' bytes differ.

        A sparse-MLA group does not arrive as one ``MLAAttentionSpec``. GLM-5.3
        puts 78 latent layers and 21 DSA indexer layers in one group with two
        different page sizes, which vLLM merges into a ``UniformTypeKVCacheSpecs``
        holding both. The declaration covers the whole published layout, so it
        is the merged members that have to be MLA -- every one of them, since a
        single sharded plane in the middle of the layout would be served to the
        wrong rank exactly as a sharded layout would.
        """
        # `getattr`, not an attribute read: this method is exercised against
        # stubs that carry only what the spec branch needs, and a V4.1 flag
        # that has to exist on every caller would make the narrow fix below
        # a wide requirement.
        if getattr(self, "_is_deepseek_v41", False):
            # The spec instrument is blind here by construction. V4.1 publishes
            # through one proxy layer whose `FullAttentionSpec` is a byte
            # container, not a description of attention, so reading it would
            # answer "sharded" for a pool that is not -- costing a factor of
            # TP in tier capacity for every V4.1 run.
            #
            # What stands in for it is the pool geometry: `build_v41_pool_geometry`
            # takes the text config and the block size and NOTHING about tensor
            # parallelism. A cache with any dimension split across ranks could
            # not be sized without knowing how many ranks there are, so a
            # geometry that never asks is a geometry that does not shard. The
            # native backend reaches the same conclusion from the same model
            # and says so in its own publisher
            # (`DeepseekV41MetadataBuilder.get_kv_transfer_tensors`:
            # "No cache dimension is split across TP"), and a test holds the
            # two together so they cannot drift apart silently.
            #
            # The limit, stated rather than left to be discovered: this is not
            # the independent instrument the rest of this method uses. It is
            # two code paths in one tree agreeing about one model. If V4.1 ever
            # shards a cache dimension, both have to change, and only the test
            # would notice if one did not.
            return max(1, int(getattr(self._config, "tensor_parallel_size", 1) or 1))
        groups = getattr(self._kv_cache_config, "kv_cache_groups", None) or ()
        if not groups:
            return 1
        spec = getattr(groups[self._attn_group_id], "kv_cache_spec", None)
        try:
            from vllm.v1.kv_cache_interface import MLAAttentionSpec
        except ImportError:
            # Not a reason to guess: an unreadable spec vocabulary means the
            # sharded answer, which costs tier capacity rather than correctness.
            return 1
        members = getattr(spec, "kv_cache_specs", None)
        specs = list(members.values()) if isinstance(members, dict) else [spec]
        if not specs or not all(
            isinstance(member, MLAAttentionSpec) for member in specs
        ):
            return 1
        return int(getattr(self._config, "tensor_parallel_size", 1) or 1)

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
        # V4.1 names its group positively rather than by elimination. The
        # codec registers the proxy's planes, so the proxy's group is the
        # one it owns -- and eliminating everything else needs a new clause
        # per group anyone else adds. DSpark was the second: its draft KV is
        # its own group and the engine refused to start with
        # `found [0, 1]`.
        if getattr(self, "_is_deepseek_v41", False):
            from atom.plugin.vllm.deepseek_v41_bridge import (
                ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
            )

            for gid, group in enumerate(groups):
                names = list(getattr(group, "layer_names", []) or [])
                if ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME in names:
                    self._attn_block_size = int(
                        getattr(
                            getattr(group, "kv_cache_spec", None),
                            "block_size",
                            0,
                        )
                        or 0
                    )
                    return gid
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 is configured but no "
                "kv cache group holds the proxy layer "
                f"{ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME!r}; the codec has "
                "nothing to register"
            )
        mamba_ids = {group_id for group_id, _ in self._mamba_groups}
        # V4.1's state-checkpoint group is neither: it is dropped from the
        # recurrent leg above, and counting it here as an attention group
        # made this refuse to start with `found [0, 1]`. It is ATOM's own
        # state and the connector never touches it in either direction.
        checkpoint_unit_ids = _v41_checkpoint_group_id_set(kv_cache_config)
        others = [
            gid
            for gid in range(len(groups))
            if gid not in mamba_ids and gid not in checkpoint_unit_ids
        ]
        if len(others) == 1:
            self._attn_block_size = int(
                getattr(
                    getattr(groups[others[0]], "kv_cache_spec", None),
                    "block_size",
                    0,
                )
                or 0
            )
        if len(others) != 1:
            raise ValueError(
                "ATOM offload connector: expected exactly one non-recurrent KV "
                f"cache group, found {others}; the byte codec registers one "
                "block geometry and cannot span several"
            )
        return others[0]

    def _init_v41_planner(self, vllm_config, kv_cache_config) -> None:
        """Stand up the scheduler half of V4.1's STATE leg."""
        if not self._is_deepseek_v41:
            return
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        from .v41_state import V41BoundaryPlanner

        _, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)
        # The grid a STATE boundary may sit on is V4.1's PAGE block, which
        # is the attention group's. `hash_block_size` is engine-wide and
        # becomes the smallest group's, so with a draft group the two part
        # company and a boundary chosen on the hash grid is not a PAGE edge.
        grid = int(getattr(self, "_attn_block_size", 0) or 0) or int(hash_block_size)
        # When checkpoints are registered the two mechanisms have to agree on one
        # grid: the planner decides where a boundary may sit, and the checkpoint
        # group decides which column vLLM hashes. A boundary the planner
        # accepts on a different grid is a store nothing can look up, and the
        # symptom is a silent zero hit rate rather than an error.
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            register_v41_state_checkpoint_adapter,
        )

        adapter = register_v41_state_checkpoint_adapter()
        checkpoint_block = int(adapter.block_size)
        if checkpoint_block and checkpoint_block != grid:
            raise ValueError(
                f"ATOM LMCache offload: V4.1 boundary grid {grid} does not match the "
                f"state checkpoint block {checkpoint_block}. A boundary claimed on one grid is "
                "hashed on the other, so nothing stored would ever be found."
            )
        self._refuse_v41_ride_without_dense_checkpoints(int(grid))
        self._v41_planner = V41BoundaryPlanner(
            hash_block_size=int(hash_block_size),
            boundary_grid=grid,
            chunk_size=int(self._scheduler.chunk_size),
            state_checkpoints_can_serve=_v41_state_checkpoints_can_serve(vllm_config),
            checkpoint_group_ids=self._v41_checkpoint_group_ids,
            prefill_tail=adapter.prefill_tail(vllm_config),
        )
        self._scheduler.install_hit_cap_hook(self._v41_planner.cap_hit)
        self._scheduler.bind_planner(self._v41_planner, self._v41_request_of)
        logger.info(
            "ATOM LMCache offload: V4.1 state leg on (hash_block=%d, chunk=%d, "
            "checkpoint_groups=%s)",
            int(hash_block_size),
            int(self._scheduler.chunk_size),
            ",".join(str(g) for g in self._v41_checkpoint_group_ids) or "-",
        )

    def _refuse_v41_ride_without_dense_checkpoints(self, grid: int) -> None:
        """A save needs a state checkpoint at the boundary it ends on.

        One save spans many chunks and carries one checkpoint image, the one
        at its end; its earlier chunks name the null block, and the MP server
        stores no image for them. The server serves a hit length only if every
        image group holds the length's last chunk (with
        `--separate-object-groups`, which keeps the image groups apart from
        PAGE), so a hit never ends inside a save: the hit grain is the set of
        saved forward ends, not the chunk. `cap_hit` relies on that.

        What has to hold on this side is that a save's end has a checkpoint
        at all. A prefill forward ends on a rung of the checkpoint interval
        (`Placement.cut`) and a save ends on a chunk boundary; with the two
        equal they coincide. With V4.1's default interval of 8192 against a
        chunk of 256, most saves would end where no forward did, find no
        checkpoint, and be dropped -- correct, but a tier that fills only at
        interval multiples and says nothing about it. Refused instead, naming
        the setting that fixes it.
        """
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            V41CheckpointPlacement,
        )

        placement = V41CheckpointPlacement.from_env()
        interval = int(placement.interval)
        chunk = int(self._scheduler.chunk_size)
        if interval == chunk:
            return
        raise ValueError(
            "ATOM offload connector: DeepSeek-V4.1 state cannot ride the PAGE "
            f"object with a checkpoint interval of {interval} tokens against "
            f"an LMCache chunk of {chunk}. A save carries the checkpoint at the "
            "boundary it ends on and a prefill forward ends on a rung of the "
            "interval, so with the two unequal most saves find no checkpoint "
            "and are dropped. Bring the two together from either side -- "
            f"ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS={chunk}, or an LMCache "
            f"chunk of {interval} on the server and the client, which is the "
            "cheaper one: the interval decides how often a checkpoint is "
            "written, while the chunk only decides how coarse a tier hit is, "
            "and a V4.1 hit is already quantised to a checkpoint."
        )

    def _v41_request_of(self, seq):
        """The vLLM request a scheduler ``SeqView`` stands for, or None.

        The planner needs it for its block hashes. It is read off the
        ``SeqView``, which the scheduler holds for as long as a save is still
        owed -- including after the request finished, when ``_requests`` has
        already let it go.
        """
        return getattr(seq, "_request", None) or self._requests.get(str(seq.id))

    @property
    def _planners(self) -> tuple:
        """This model's boundary planners, in a fixed order; at most one.

        Each keeps one recurrent-state leg and the KV prefix at one boundary.
        Every scheduler hook drives them through `BoundaryPlanner`'s API, in a
        loop over this, rather than by name -- so a rule a hook applies to one
        leg reaches every leg.
        """
        return tuple(
            planner
            for planner in (self._kda_planner, self._v41_planner)
            if planner is not None
        )

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
        if self._is_deepseek_v41:
            per_group = {}
            tensors, num_blocks, leading_dim = self._v41_page_tensors()
        else:
            attention_caches = per_group[self._attn_group_id]
            tensors = build_kv_cache_tensors(attention_caches, self._attention_layers())
            if not tensors:
                raise ValueError("ATOM offload connector: vLLM registered no KV caches")

            # Every segment's per-block stride is derived from num_blocks, so it
            # has to be the count of blocks vLLM's block tables name, which is
            # what `resolve_block_count` takes from the config and reconciles
            # against the tensor. The leading dimension is NOT that count on
            # ATOM's MLA backend: it asks for a kernel block size of 1, so vLLM
            # allocates one row per token and dim 0 comes back 1536x too large
            # on Kimi-K3.
            leading_dim = int(tensors[0].k_cache.shape[0])
            num_blocks = resolve_block_count(
                leading_dim,
                int(getattr(self._kv_cache_config, "num_blocks", 0)),
                int(self._config.kv_cache_block_size),
            )

        if self._offload_backend == "mp":
            # The multiprocess worker ignores the layer dict entirely: its
            # server process addresses bytes, so it needs the address form.
            from .mp_page_layout import (
                build_mp_transfer_tensors,
                summarize_layout,
            )

            replication = self._page_tp_replication_factor()
            transfer_tensors = build_mp_transfer_tensors(
                tensors,
                num_blocks=num_blocks,
                tp_replication_factor=replication,
            )
            if self._state_rides_mp:
                self._publish_v41_state_checkpoints(transfer_tensors, replication)
            logger.info(
                "ATOM LMCache offload: publishing PAGE layout (%s), "
                "tp_replication_factor=%d",
                summarize_layout(transfer_tensors),
                replication,
            )
            self._worker.register_kv_caches(
                {str(t.layer_num): t for t in tensors},
                transfer_tensors,
                num_blocks,
            )
        else:
            self._worker.register_kv_caches(
                {str(t.layer_num): t for t in tensors},
                num_blocks=num_blocks,
            )
        # After the dense registration, not before: the recurrent codec shares
        # the engine, its storage manager and its LMCache identity, and none of
        # those exist until the call above has run. Skipped entirely when the
        # state rides the PAGE object -- there is then no second engine, no
        # second pool and no leg of its own to drive.
        if not self._state_rides_mp:
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

    def _publish_v41_state_checkpoints(self, transfer_tensors, replication: int):
        """Declare the PAGE units' checkpoint images, as native V4.1 does.

        The same four fields `DeepseekV41MetadataBuilder.get_kv_transfer_tensors`
        publishes, so the MP worker registers this pool through the native
        layout: every published plane is part of the checkpoint's PAGE unit
        (the proxy group has no draft pool of its own), and the images are
        replicated exactly as the PAGE KV is, for the reason
        `_page_tp_replication_factor` gives. The copy callback is the one this
        worker's checkpoint stores and restores already run through; the MP
        worker restores no SLOT and only checks that it exists.
        """
        from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
            current_state_checkpoint_worker,
            v41_state_checkpoint_sizing,
        )

        worker = current_state_checkpoint_worker()
        copier = getattr(worker, "copier", None)
        if copier is None:
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 state checkpoints are on "
                "but this worker has no checkpoint copier after binding the "
                "proxy pool; there would be no checkpoint for a transfer to carry"
            )
        transfer_tensors.paged_state_checkpoint_spec = v41_state_checkpoint_sizing(
            self._vllm_config
        ).spec
        transfer_tensors.execute_paged_state_copies = copier.execute_paged_state_copies
        transfer_tensors.paged_state_region_count = len(transfer_tensors.pages)
        transfer_tensors.native_state_tp_replication_factor = int(replication)

    def _v41_page_tensors(self):
        """DeepSeek-V4.1's PAGE planes as the codec's `(tensors, blocks, dim)`.

        The standard path cannot produce these. vLLM sees one proxy layer whose
        tensor is `(2, blocks, 256, 1, head_size)` of uint8 -- the leading `2`
        is a byte-accounting axis, not K and V -- so `build_kv_cache_tensors`
        would take the whole arena as a single opaque run and
        `resolve_block_count` would read a block count of 2. (It raises rather
        than believing it, which is why this is a bypass and not a bug fix.)

        What is actually movable is the carving ATOM already did: the main page
        plane plus one index plane per compression owner, each dense in pages.
        Each becomes its own segment, in `unit_regions` order, because that
        order is the stored object's key space.

        `num_blocks` is the pool's page count, which `bind_deepseek_v41_proxy_cache`
        took from `cache_config.num_gpu_blocks` -- the blocks vLLM's tables
        name. It is deliberately not the proxy tensor's leading dimension:
        that allocation also holds the withheld tail the per-request STATE
        lives in, and a mover sized from it would stride into another
        request's rings.
        """
        from atom.config import KVCacheTensor
        from atom.plugin.vllm.deepseek_v41_bridge import (
            ensure_v41_proxy_bound,
            v41_page_planes,
        )

        cache, _ = ensure_v41_proxy_bound(self._vllm_config)
        planes = v41_page_planes(cache)
        num_blocks = int(cache.num_pages)
        scheduler_blocks = int(getattr(self._kv_cache_config, "num_blocks", 0))
        if num_blocks != scheduler_blocks:
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 pool holds "
                f"{num_blocks} PAGEs but vLLM addresses {scheduler_blocks} "
                "blocks. The two must be the same number -- a vLLM block id "
                "is a V4.1 page id."
            )
        unit_bytes = []
        tensors = []
        for layer_num, (role, plane) in enumerate(planes):
            if not plane.is_contiguous():
                raise ValueError(
                    f"ATOM offload connector: V4.1 plane {role} is not "
                    "contiguous; the byte codec slices each segment by stride"
                )
            if plane.numel() % num_blocks:
                raise ValueError(
                    f"ATOM offload connector: V4.1 plane {role} holds "
                    f"{plane.numel()} elements, not a whole number of "
                    f"{num_blocks} blocks"
                )
            unit_bytes.append(plane.numel() // num_blocks * plane.element_size())
            tensors.append(KVCacheTensor(layer_num=layer_num, k_cache=plane))
        # The self-check the whole registration rests on: what is published has
        # to be exactly the PAGE unit the kernels address. Cross-checked against
        # two independent statements of it -- the cache's own region stream and
        # the geometry's declared total -- because agreeing with only one would
        # leave a carving error that moved bytes between planes invisible.
        declared = [size for _, size in cache.unit_regions()]
        paged_bytes = int(cache.geometry.paged_bytes)
        if unit_bytes != declared or sum(unit_bytes) != paged_bytes:
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 PAGE unit mismatch: "
                f"published={unit_bytes}, cache regions={declared}, "
                f"geometry.paged_bytes={paged_bytes}"
            )
        # The arena has to balance, not just the PAGE side of it. Every check
        # above compares the published planes against another statement of the
        # PAGE unit, so all of them still pass if a plane is missing from both
        # -- or if the STATE region the other leg moves is not the rest of the
        # allocation. This is the one identity that spans both legs:
        #
        #     Sigma(published unit bytes) * pages + slots * state_bytes == arena
        #
        # The left side is what this connector will move; the right is what the
        # cache was built to hold, from `paged_extents` and `state_fields`. A
        # plane dropped from `v41_page_planes` shortens the left and nothing
        # else, so it reads as a shortfall of exactly that plane.
        #
        # What it does NOT catch, stated so the next reader does not trust it
        # too far: a field missing from `state_fields` moves both sides at once,
        # because the STATE leg sends the entry as flat bytes sized from the
        # same geometry. The guard for that is that there is no per-field
        # selection on this path at all -- `state_entry_views` hands over
        # `state_bytes[slot]` whole -- which the tests assert directly.
        arena = int(cache.backing.numel())
        alignment = int(cache.geometry.alignment)
        moved = sum(unit_bytes) * num_blocks + int(cache.num_slots) * int(
            cache.geometry.state_bytes
        )
        # Not equality: `paged_extents` rounds the PAGE region up to the
        # geometry's alignment, so the two regions are separated by up to
        # `alignment - 1` bytes that belong to neither leg. The slack is
        # bounded and has to be, or the identity stops noticing a plane --
        # the smallest index plane here is 2,112 B/block, so any missing one
        # shorts the sum by far more than an alignment gap.
        slack = arena - moved
        if not 0 <= slack < alignment:
            raise ValueError(
                "ATOM offload connector: DeepSeek-V4.1 arena does not balance: "
                f"{len(planes)} PAGE planes x {num_blocks} blocks "
                f"({sum(unit_bytes) * num_blocks} B) + {cache.num_slots} STATE "
                f"slots x {int(cache.geometry.state_bytes)} B = {moved} B "
                f"against a pool of {arena} B, a difference of {slack} B that "
                f"is not the inter-region padding (which is under {alignment}). "
                "A plane missing from `v41_page_planes` is the usual cause; it "
                "would otherwise offload nothing and restore stale bytes in "
                "its place."
            )
        logger.info(
            "ATOM LMCache offload: DeepSeek-V4.1 PAGE unit = %d B/block over "
            "%d planes %s, num_blocks=%d, state entry=%d B x %d slots",
            paged_bytes,
            len(planes),
            unit_bytes,
            num_blocks,
            int(cache.geometry.state_bytes),
            int(cache.num_slots),
        )
        return tensors, num_blocks, int(planes[0][1].shape[0])

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
                # The PAGE half landed and the STATE half has not. The
                # request stays in `_kda_dense_done` and waits. Counted,
                # because waiting forever and waiting one more step produce
                # the same output here, and one of them is a livelock: a
                # promised load that is never discharged leaves the request
                # in WAITING with `num_computed` at zero for the life of the
                # process.
                self._kda_join_waiting[str(req_id)] = (
                    self._kda_join_waiting.get(str(req_id), 0) + 1
                )
                n = self._kda_join_waiting[str(req_id)]
                if n in (1, 8, 64) or n % 512 == 0:
                    logger.warning(
                        "ATOM LMCache offload: %s has its dense half and no "
                        "recurrent state result after %d worker polls; "
                        "its promised load cannot be discharged until one "
                        "arrives",
                        req_id,
                        n,
                    )
                continue
            self._kda_join_waiting.pop(str(req_id), None)
            self._kda_dense_done.discard(req_id)
            self._kda_expect.discard(req_id)
            released.add(req_id)
            if not result.ok:
                self._kda_error_blocks.update(result.error_block_ids)
                self._kda_error_reqs.add(str(req_id))
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
        for req_id, block_ids in getattr(metadata, "abandoned_loads", ()) or ():
            # The scheduler half parked this request and then found it had no
            # load to issue. Only this half can release it.
            self._abandoned_loads.add(str(req_id))
            self._abandoned_error_blocks.update(int(b) for b in block_ids)

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
        if self._abandoned_loads:
            # Not a transfer failure -- no transfer was ever started -- but the
            # same two things have to happen: release the park and truncate to
            # what HBM really holds. `failed_loading` is that pair.
            failed |= self._abandoned_loads
            self._abandoned_loads = set()
        if failed:
            logger.warning(
                "ATOM LMCache offload: load failed for %s; recomputing", sorted(failed)
            )
            self._pending_failed_recving.update(failed)
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
        if self._abandoned_error_blocks:
            blocks = set(blocks) | self._abandoned_error_blocks
            self._abandoned_error_blocks = set()
        if self._multi_group_failures():
            # vLLM 0.31 raises on a non-empty set here when the model has more
            # than one KV cache group, because a flat block id cannot be mapped
            # back to a request. The failures are not dropped: they go out as
            # `failed_recving` from `get_transfer_results` instead. Draining
            # above rather than returning early is deliberate -- the sets must
            # not accumulate across steps and reappear once the group count
            # changes.
            return set()
        return blocks

    def _multi_group_failures(self) -> bool:
        """Whether this model must report load failures by request, not block.

        One group is the common case and keeps the finer channel: vLLM
        truncates `num_computed_tokens` to the first bad block and recomputes
        from there, where a request-level failure discards the whole external
        prefix. More than one group -- a hybrid model, or V4.1 once its state
        checkpoints become their own groups -- has no choice.
        """
        # `getattr` on self, not just on the config: the base class sets
        # `_kv_cache_config` in `__init__`, so a connector without it is one
        # built past `__init__` -- a test fixture, never a served engine. Such
        # a connector reads as single-group, which keeps the finer channel and
        # leaves today's behaviour alone; the multi-group refusal it would miss
        # cannot be reached without a real config.
        config = getattr(self, "_kv_cache_config", None)
        groups = getattr(config, "kv_cache_groups", None) or ()
        return len(groups) > 1

    def get_transfer_results(self, finished_req_ids):
        """Sends, receives and receive failures, as vLLM 0.31 asks for them.

        The base class derives this from `get_finished` with no failures, which
        is right only while every failure can travel as a block id. It cannot
        on a multi-group model, so the requests whose load failed are named
        here. Failed receives must ALSO remain in `finished_recving`: vLLM
        holds the failure until that receive reaches its worker quorum, then
        marks the prefix invalid before unblocking the request.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorTransferResults,
        )

        sending, recving = self.get_finished(finished_req_ids)
        # get_finished retained failures from the common terminal channel.
        # MP has no take_load_error_reqs hook, and a withdrawn promise has no
        # worker transfer at all. A PAGE failure may precede the STATE result:
        # vLLM holds this flag until finished_recving reaches its rank quorum.
        failed, self._pending_failed_recving = self._pending_failed_recving, set()
        drain = getattr(self._worker, "take_load_error_reqs", None)
        if callable(drain):
            failed.update(drain())
        if self._kda_error_reqs:
            failed |= self._kda_error_reqs
            self._kda_error_reqs = set()
        if not self._multi_group_failures():
            # Single group: the block channel already carried them, and naming
            # them twice would discard prefixes the block channel only trims.
            failed = set()
        return KVConnectorTransferResults(
            finished_sending=set(sending or ()),
            finished_recving=set(recving or ()),
            failed_recving=failed,
        )

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
        kept until every rank reports; the planner owns both ends of it.

        The pool is also kept on `self` for the exact-lease release path. vLLM
        calls this unconditionally on the scheduler half before any request
        runs, and only that half may touch the pool; the worker half never sees
        this call and its `_gpu_block_pool` stays None, which is what keeps the
        whole-request fallback correct on a vLLM too old to call this at all.
        """
        self._gpu_block_pool = gpu_block_pool
        for planner in self._planners:
            planner.bind_gpu_block_pool(gpu_block_pool)

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
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
        # A detach recorded for an admission that then failed allocation: this
        # question is asked afresh, and the old frontier must not reach
        # `update_state_after_alloc` and overwrite the save floor.
        self._detach_after_alloc.pop(str(request.request_id), None)
        if self._lookup_detached(seq, num_computed_tokens):
            return 0, False
        if self._lookup_in_flight(seq):
            return None, False
        planners = self._planners
        for planner in planners:
            # ATOM's scheduler hands the cap hook a SeqView, which carries no
            # block hashes; this is how the hook reaches the vLLM Request whose
            # lookup is running. Armed per lookup and cleared unconditionally,
            # so a hook firing outside one has nothing stale to read.
            planner.begin_lookup(request)
        try:
            need, _ = self._scheduler.get_num_new_matched_tokens(seq)
        finally:
            for planner in planners:
                planner.end_lookup()
        parked = need > 0 and self._scheduler.should_park_for_load_after_alloc(seq)
        if not parked:
            return 0, False
        # [steps_undispatched, first_token_of_promise, tokens_promised,
        #  parked]. The range is kept because a promise that is never honoured
        # has to be withdrawn, and withdrawing it means naming the blocks it
        # reserved. `parked` starts False: this is the lookup, and vLLM has not
        # yet decided it can allocate for the request. Only
        # `update_state_after_alloc` -- which vLLM calls once it has, and
        # which is followed by parking the request in WAITING_FOR_REMOTE_KVS --
        # makes it True. See `_settle_promised_loads` for why that matters.
        self._promised_loads[request.request_id] = [
            0,
            int(num_computed_tokens),
            int(need),
            False,
        ]
        return need, True

    def _lookup_detached(self, seq, num_computed_tokens: int) -> bool:
        """Admit `seq` without its lookup when no answer could make it load.

        vLLM's local hit leaves less than the minimum load uncached, so the
        tier cannot be asked for anything -- but the lookup is still sent, for
        its L1 recency and for the save floor it corrects afterwards
        (`reconcile_detached_lookups`, run from `build_connector_meta`).
        Skipping it outright would have every such request re-save its whole
        prompt. Waiting for it instead cost local-hit runs 14% of their
        throughput: each request was held a step for an answer it could not
        use.

        Safe to decide here, before vLLM allocates: unlike ATOM's own
        allocator, vLLM never keeps less of the local hit than it offered. The
        lookup itself waits for the allocation, as `detach_lookup` requires: a
        request vLLM cannot allocate for is asked again next step, and sending
        it here resent the whole prompt's lookup on every attempt.
        """
        can_detach = getattr(self._scheduler, "can_detach_lookup", None)
        if (
            can_detach is None
            or getattr(self._scheduler, "detach_lookup", None) is None
        ):
            return False
        hbm = int(num_computed_tokens)
        if not can_detach(seq, hbm):
            return False
        self._detach_after_alloc[str(seq.id)] = hbm
        return True

    def _lookup_in_flight(self, seq) -> bool:
        """Send `seq`'s tier lookup without waiting; True while it is out.

        A synchronous lookup is a round trip in which the server hashes the
        whole prompt, and vLLM runs this hook inside `schedule()`, so no
        forward starts until it returns. Measured on DeepSeek-V4.1 with
        7k-token prompts: 17.6 ms p50 and 43 ms p90 per new request, which put
        `schedule()` at 14 ms p90 against 0.5 ms with the tier idle.

        Returning None makes vLLM pass over the request and ask again next
        step, and the answer is then consumed from the client without a second
        round trip -- the prefetch ATOM's own scheduler already does, with the
        same bounds: `OFFLOAD_ASYNC_LOOKUP_DEPTH=0` keeps lookups synchronous,
        and past `OFFLOAD_LOOKUP_DEFER_S` the lookup is waited for as before.
        A client that cannot look up asynchronously never reports one pending.

        The depth is one budget per scheduling pass, as ATOM's own scheduler
        spends it on the head of its waiting queue. vLLM asks once per waiting
        request, so submitting (and pumping every lookup in flight) on each
        call restarted the budget per request; past it a request is passed over
        and sent next pass.
        """
        prefetch = getattr(self._scheduler, "prefetch_lookups", None)
        pending = getattr(self._scheduler, "lookup_pending", None)
        if prefetch is None or pending is None:
            return False
        if pending(seq):
            return True
        depth = envs.OFFLOAD_ASYNC_LOOKUP_DEPTH
        if depth > 0 and self._lookups_sent_this_step >= depth:
            return True
        prefetch([seq])
        sent = bool(pending(seq))
        if sent:
            self._lookups_sent_this_step += 1
        return sent

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
        hbm = self._detach_after_alloc.pop(str(request.request_id), None)
        if hbm is not None:
            self._scheduler.detach_lookup(seq, hbm)
        if int(num_external_tokens) > 0:
            promise = self._promised_loads.get(request.request_id)
            if promise is not None:
                promise[3] = True
        groups = _group_block_ids(blocks)
        # ATOM's byte codec addresses the attention group only; the recurrent
        # group is addressed by the boundary hand-off, never positionally.
        attention_blocks = groups[self._attn_group_id] if groups else []
        num_total_computed = int(seq.num_cached_tokens) + int(num_external_tokens)
        seq.set_block_table(list(attention_blocks))
        self._scheduler.update_state_after_alloc(seq)
        if int(num_external_tokens) == 0 and (
            self._promised_loads.pop(request.request_id, None) is not None
        ):
            # vLLM admitted the request without the tokens the lookup promised
            # (`skip_reading_prefix_cache`, a local sub-block tail, a hit no
            # longer than the replay window). It runs RUNNING and computes them
            # itself, so the queued load would write blocks this step's forward
            # fills and report `finished_recving` for a request never parked.
            self._scheduler.cancel_pending_load(seq)
        if int(num_external_tokens) > 0:
            for planner in self._planners:
                planner.admit_load(
                    request,
                    group_blocks=groups,
                    num_total_computed=num_total_computed,
                    num_external_tokens=int(num_external_tokens),
                    attention_group_id=self._attn_group_id,
                    attention_block_size=int(self._config.kv_cache_block_size),
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
        self._lookups_sent_this_step = 0
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
                # Priced with the ATTENTION group's block size, not the
                # engine-wide one. `cache_config.block_size` is a single
                # value and becomes the smallest group's: with DSpark the
                # draft's 32 replaced the proxy's 256, so six proxy blocks
                # were priced at 192 tokens, every save frontier floored to
                # zero chunks, and the tier was never written. Measured as
                # `computed=192` stuck for the life of every request while
                # the same prompts reached 1280 without drafts.
                covered = len(seq.block_table) * block_size
                seq.set_num_cached_tokens(min(int(num_tokens), covered))
        inner = self._scheduler.build_connector_meta()
        abandoned_loads = self._settle_promised_loads(inner)
        # Before `_collect_releases`, so a save abandoned on this step turns
        # into a release on this step rather than on the next one.
        self._reconcile_stale_saves()
        state_stores, state_loads = self._collect_state_transfers(
            scheduler_output, preempted, frontiers
        )

        # After this step's saves: an operation built here is in the live set
        # the deadline walks.
        self._scheduler.enforce_transfer_deadlines()
        return AtomOffloadMetadata(
            inner,
            preempted,
            self._collect_releases(),
            state_stores,
            state_loads,
            abandoned_loads,
        )

    def _collect_state_transfers(
        self, scheduler_output, preempted, frontiers
    ) -> tuple[list, list]:
        """This step's recurrent-state stores and loads, from every planner.

        A request preempted or finished in this step is forgotten first: its
        blocks are going away, and a preempted one comes back with its frontier
        rewound, so whatever a planner kept for it (the KDA sweep cursor, a
        V4.1 load destination) has to be gone before this step's stores are
        collected. The stores are collected here because vLLM's boundary
        hand-off exists only in the step it is built for.
        """
        planners = self._planners
        if not planners:
            return [], []
        skip = set(preempted)
        skip.update(
            str(r) for r in getattr(scheduler_output, "finished_req_ids", None) or ()
        )
        stores: list = []
        loads: list = []
        for planner in planners:
            for req_id in skip:
                planner.forget_request(req_id)
            stores.extend(
                planner.collect_step_stores(
                    scheduler_output, frontiers, self._requests, skip
                )
            )
            loads.extend(planner.take_loads())
            planner.log_stats()
        return stores, loads

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
            for planner in self._planners:
                # The recurrent destination was this request's own block,
                # which is back in the pool, and the request comes back with
                # its frontier rewound: drop both the pending load's claim and
                # what the planner kept for its old life.
                planner.abandon_pending_load(req_id)
                planner.forget_request(req_id)
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
        if not getattr(self._scheduler, "abandons_saves", True):
            # MP gives a save's blocks back only on its report. Dropping the
            # ranks' reports here would free nothing and leave the quorum
            # unreachable; `enforce_transfer_deadlines` fail-stops instead.
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
        release also rides on a worker report, or a reconciliation, that only a
        step can deliver.
        """
        if any(planner.has_pending_work() for planner in self._planners):
            return True
        # The MP scheduler pumps lookups and ends sessions from a step as well.
        pending = getattr(getattr(self, "_scheduler", None), "has_pending_work", None)
        if pending is not None and pending():
            return True
        return bool(self._deferred_frees or self._releases_in_flight)

    def has_pending_block_frees(self) -> bool:
        """Blocks a transfer still in flight is about to hand back (vLLM 0.31).

        vLLM asks this when an allocation just failed, immediately before it
        would preempt a running request to make room (`Scheduler.schedule`,
        0.31). True means "wait a step, these are coming back"; the default
        False means every such moment costs a preemption that one step would
        have avoided.

        Three sources, and the third is the one a reader would miss: a save
        holds its source blocks by refcount with the request already finished
        and its table returned, so nothing in vLLM's view of that request says
        those blocks are spoken for. `has_pending_push_work` does not count
        them -- it answers a different question (is the engine owed a step at
        all) and a lease does not owe one, because the next request's step
        drains it.
        """
        if self._releases_in_flight:
            return True
        # Only recent deferrals, like the leases below: on MP a finished
        # request whose save stalls stays deferred until the transfer deadline
        # fail-stops it, and counting it would trade every preemption in that
        # window for a step that frees nothing.
        oldest = time.monotonic() - self._PENDING_FREE_HORIZON_S
        if any(
            self._deferred_free_at.get(req_id, oldest) > oldest
            for req_id in self._deferred_frees
        ):
            return True
        if any(
            planner.has_pending_frees(self._PENDING_FREE_HORIZON_S)
            for planner in self._planners
        ):
            return True
        scheduler = self._scheduler
        # Only recent leases. Waiting stalls the running loop for the step,
        # and a save past this horizon is stuck rather than landing (MP has no
        # stale-lease reclaim; its transfer deadline fail-stops later). The
        # leased blocks are not the ones a preemption frees, so preempting
        # instead is safe.
        return bool(
            scheduler is not None
            and getattr(scheduler, "has_active_save_leases", None) is not None
            and scheduler.has_active_save_leases(max_age_s=self._PENDING_FREE_HORIZON_S)
        )

    # How long a save lease counts as blocks about to come back.
    _PENDING_FREE_HORIZON_S = 10.0

    # Steps a promised load may go undispatched before it is called out. Loads
    # are emitted on the step after the promise, so anything past a handful of
    # steps is already wrong; the margin is only so a busy scheduler does not
    # produce noise.
    _PROMISE_GRACE_STEPS = 50

    def _settle_promised_loads(self, inner) -> list[tuple[str, list[int]]]:
        """Withdraw any promise this step did not turn into a dispatched load.

        The scheduler half cannot inject `finished_recving`, so what it returns
        here is a request for the worker half to report the load as failed --
        which releases the park and, because the reserved blocks travel with
        it, truncates the request back to what is really in HBM.

        This exists because the park decision and the dispatch decision are
        taken at different points and do not ask the same question. Parking
        happens in `get_num_new_matched_tokens`, before `allocate_slots`;
        `build_connector_meta` then declines a load on conditions the park gate
        never evaluated -- a lookup pin the tier has since lost, or (hybrid
        models only) a recurrent boundary state that was evicted between the
        lookup and the dispatch. ATOM's own engine parks a sequence only when
        it registers the load operation, so there the decline is genuinely
        before the park and those paths are safe; on this path they are not.

        Settlement is on a load actually being dispatched, not on the request
        appearing in `inner.requests`: a save for the same request rides the
        same list, and treating that as the load landing is what let this go
        unseen. A dispatched load carries a `load_operation`/`load_spec`.
        """
        finished, self._finished_withdrawals = self._finished_withdrawals, []
        if not self._promised_loads:
            return finished
        # Declined at dispatch this step: no later step will dispatch them.
        declined = getattr(self._scheduler, "declined_loads", None) or set()
        for meta in getattr(inner, "requests", ()) or ():
            if (
                getattr(meta, "load_operation", None) is None
                and getattr(meta, "load_spec", None) is None
            ):
                continue
            self._promised_loads.pop(str(getattr(meta, "req_id", meta)), None)
        withdrawn: list[tuple[str, list[int]]] = []
        expired: list[str] = []
        for req_id in list(self._promised_loads):
            state = self._promised_loads[req_id]
            if not state[3]:
                # Promised at lookup, never parked: vLLM could not allocate
                # for the request this step (or did not reach it), so it is
                # still WAITING and will be looked up -- and promised -- again.
                # Withdrawing it would report `finished_recving` for a request
                # vLLM never parked, and `_update_from_kv_xfer_finished`
                # asserts on exactly that. Measured on the second gsm8k pass of
                # a DSpark run against a 950-block pool, where allocation after
                # a tier hit is what fails: one withdrawal "(unknown)" after 50
                # steps, four ranks reporting it, and the engine down.
                self._promised_loads.pop(req_id)
                continue
            if req_id not in declined:
                state[0] += 1
                if state[0] <= self._PROMISE_GRACE_STEPS:
                    continue
                expired.append(req_id)
            _, start, need, _parked = self._promised_loads.pop(req_id)
            withdrawn.append((req_id, self._promised_block_ids(req_id, start, need)))
        if expired:
            logger.error(
                "ATOM LMCache offload: promised a load for %s but none was "
                "dispatched within %d steps; withdrawing the promise so the "
                "request leaves WAITING_FOR_REMOTE_KVS and recomputes",
                sorted(expired),
                self._PROMISE_GRACE_STEPS,
            )
        return finished + withdrawn

    def _promised_block_ids(self, req_id: str, start: int, need: int) -> list[int]:
        """The blocks a withdrawn promise reserved and nothing wrote.

        An empty list is not a safe default here -- vLLM would then cache the
        whole external prefix as though it had arrived -- but it is the only
        honest answer when the SeqView is already gone, and the release still
        beats a permanent park.
        """
        seq = self._seqs.get(req_id)
        if seq is None:
            return []
        # Virtual, as in `build_connector_meta`: under DCP one block id covers
        # `kv_cache_block_size * decode_context_parallel_size` tokens.
        block_size = int(
            getattr(self._scheduler, "virtual_block_size", 0)
            or self._config.kv_cache_block_size
        )
        if block_size <= 0:
            return []
        table = list(seq.block_table)
        first = int(start) // block_size
        last = -(-(int(start) + int(need)) // block_size)
        return [b for b in table[first:last]]

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
        failed_recving = getattr(connector_output, "failed_recving", None) or ()
        for req_id in connector_output.finished_recving or ():
            rid = str(req_id)
            self._promised_loads.pop(rid, None)
            state_failed = self._state_load_failure_reports.pop(rid, 0) > 0
            page_failed = self._load_failure_reports.pop(rid, 0) > 0
            for planner in self._planners:
                # Retract a claim on a boundary whose bytes are gone, so the
                # next lookup caps at a boundary that is really there instead
                # of failing the same load again.
                planner.on_load_result(rid, not state_failed)
            if state_failed or page_failed or rid in failed_recving:
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
        for planner in self._planners:
            # Before the loops below: a store quorum releases a pinned block and
            # publishes the boundary, and both want to be true by the time this
            # step's `finished_recving` ids are resolved.
            planner.absorb_reports(
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
        promise = self._promised_loads.pop(req_id, None)
        if promise is not None and promise[3]:
            # Parked, never dispatched: nothing else would send the
            # `finished_recving` vLLM frees an aborted parked request's blocks on.
            _, start, need, _parked = promise
            self._finished_withdrawals.append(
                (req_id, self._promised_block_ids(req_id, start, need))
            )
        self._requests.pop(str(req_id), None)
        self._detach_after_alloc.pop(str(req_id), None)
        for planner in self._planners:
            planner.abandon_pending_load(str(req_id))
            planner.forget_request(str(req_id))
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
                    # above keep exactly the blocks the save still reads. They
                    # are reported through `has_pending_block_frees` so an
                    # allocation that just failed waits for them instead of
                    # preempting; `reclaim_stale_leases` bounds the wait if the
                    # save never reports.
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
