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
"""

import logging
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)

from atom.plugin.vllm.kv_transfer.kv_cache_layout import build_kv_cache_tensors
from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config
from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext

logger = logging.getLogger("atom")


class AtomOffloadMetadata(KVConnectorMetadata):
    """Carries ATOM's offload metadata through vLLM's connector plumbing.

    ATOM's ``LMCacheOffloadMetadata`` derives from ATOM's own ConnectorMetadata,
    not vLLM's, so it cannot be returned directly from ``build_connector_meta``.
    Wrapping keeps ATOM's descriptors intact instead of flattening them into a
    vLLM-shaped copy that would then have to be kept in sync.
    """

    def __init__(self, inner, preempted_req_ids=(), release_req_ids=()) -> None:
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
    """

    def __init__(self, saved=None, load_failed=None) -> None:
        self.saved: dict[str, int] = dict(saved or {})
        self.load_failed: dict[str, int] = dict(load_failed or {})

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "AtomOffloadWorkerMetadata":
        for field in ("saved", "load_failed"):
            mine = getattr(self, field)
            for req_id, count in (getattr(other, field, None) or {}).items():
                mine[req_id] = mine.get(req_id, 0) + int(count)
        return self

    def __repr__(self) -> str:
        return (
            f"AtomOffloadWorkerMetadata(saved={self.saved}, "
            f"load_failed={self.load_failed})"
        )


class AtomLMCacheOffloadConnector(KVConnectorBase_V1):
    """Drives ``atom.kv_transfer.offload`` from vLLM's connector API."""

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

        # Scheduler half: requests whose free vLLM is holding for us, split by
        # what they are still waiting for -- the save to land, or the worker to
        # echo the release back. See `_collect_releases`.
        self._deferred_frees: set[str] = set()
        self._releases_in_flight: set[str] = set()
        # Per-request rank tallies for the two facts that travel as worker
        # metadata rather than as one of vLLM's two id sets.
        self._save_reports: dict[str, int] = {}
        self._load_failure_reports: dict[str, int] = {}
        self._world_size = max(
            1, int(getattr(vllm_config.parallel_config, "world_size", 1) or 1)
        )

        if role == KVConnectorRole.WORKER:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

            self._worker = DenseOffloadConnector(self._config)
        else:
            from atom.kv_transfer.offload.dense.connector import DenseOffloadScheduler

            self._scheduler = DenseOffloadScheduler(self._config)

    # ---- worker side --------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Translate vLLM's flat registration and hand it to ATOM's codec.

        The layer modules go along with the tensors: M3 keeps its fp8 KV scales
        (one fp32 per token per head) on the layer, not in this dict, and a
        transfer that moves the mantissas without them silently dequantises a
        restored block against the previous occupant's scale.
        """
        tensors = build_kv_cache_tensors(kv_caches, self._attention_layers())
        if not tensors:
            raise ValueError("ATOM offload connector: vLLM registered no KV caches")

        # Every segment's per-block stride is derived from num_blocks, so it has
        # to be the physical block count -- not a token count. Block-major KV
        # carries it in dim 0; taking it from the first mapped k_cache keeps the
        # value consistent with the very tensors the codec will slice.
        num_blocks = int(tensors[0].k_cache.shape[0])

        self._worker.register_kv_caches(
            {str(t.layer_num): t for t in tensors},
            num_blocks=num_blocks,
        )
        logger.info(
            "ATOM LMCache offload: registered %d layers, num_blocks=%d",
            len(tensors),
            num_blocks,
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

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Inert: transfers are per-request, not per-layer.

        A load is published to the scheduler through ``get_finished`` only once
        every block of the request has landed, so there is no partially-loaded
        layer for a forward to wait on.
        """

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs) -> None:
        """Inert: saves are issued per request from ``build_connector_meta``."""

    def wait_for_save(self) -> None:
        """Inert: saves are fire-and-forget on ATOM's save executor.

        Blocking the forward on them would put offload on the critical path,
        which is the opposite of what the tier is for. Completion still reaches
        the scheduler via ``get_finished``.
        """

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
        """
        return self._worker.take_load_error_blocks()

    def build_connector_worker_meta(self) -> AtomOffloadWorkerMetadata | None:
        """Hand this rank's save and load-failure reports to the scheduler half.

        Called once per step, immediately after ``get_finished``, so it carries
        exactly the completions that call observed. ``None`` when there are
        none: the aggregator folds only non-``None`` metadata, and a step with
        no offload activity should not pickle an empty object per rank.
        """
        if not (self._worker_saved or self._worker_load_failed):
            return None
        meta = AtomOffloadWorkerMetadata(self._worker_saved, self._worker_load_failed)
        self._worker_saved = {}
        self._worker_load_failed = {}
        return meta

    def shutdown(self) -> None:
        for side in (self._worker, self._scheduler):
            close = getattr(side, "close", None) or getattr(side, "shutdown", None)
            if close is not None:
                close()

    # ---- scheduler side -----------------------------------------------

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
        """
        seq = self._seqs.get_or_create(request)
        seq.set_num_cached_tokens(num_computed_tokens)
        need, _ = self._scheduler.get_num_new_matched_tokens(seq)
        if need <= 0:
            return 0, False
        if not self._scheduler.should_park_for_load_after_alloc(seq):
            return 0, False
        self._promised_loads[request.request_id] = 0
        return need, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        seq = self._seqs.get_or_create(request)
        seq.set_block_table(_block_ids(blocks))
        self._scheduler.update_state_after_alloc(seq)

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
        once the allocation catches up.

        Only long prompts reach this: a chat-sized prompt is allocated in one
        go, so the two quantities never separate and every gsm8k-scale test
        passes. It took an ISL-90k aiperf run to surface.

        Preemptions are settled before any of that, and deferred frees after it
        -- both for ordering reasons spelled out at their helpers.
        """
        preempted = self._handle_preempted(scheduler_output)
        block_size = int(self._config.kv_cache_block_size)
        for req_id, num_tokens in _scheduled_frontiers(scheduler_output):
            seq = self._seqs.get(req_id)
            if seq is not None:
                covered = len(seq.block_table) * block_size
                seq.set_num_cached_tokens(min(int(num_tokens), covered))
        inner = self._scheduler.build_connector_meta()
        self._check_promised_loads(inner)
        return AtomOffloadMetadata(inner, preempted, self._collect_releases())

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
            seq = self._seqs.get(req_id)
            if seq is None:
                continue
            self._scheduler.cancel_pending_load(seq)
            seq.reset_for_preemption()
        if req_ids:
            logger.debug("ATOM LMCache offload: preempted %s", req_ids)
        return req_ids

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

        The final `request_finished` is what pops ATOM's save tracker, which is
        in turn what stops the save loop from ever emitting a save against
        blocks vLLM is about to reassign.
        """
        if not self._deferred_frees:
            return []
        released = []
        for req_id in sorted(self._deferred_frees):
            seq = self._seqs.get(req_id)
            if seq is not None:
                if self._scheduler.should_defer_free(seq):
                    continue
                self._scheduler.request_finished(seq)
            self._seqs.drop(req_id)
            released.append(req_id)
        self._deferred_frees.difference_update(released)
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
        """
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
        for req_id in connector_output.finished_recving or ():
            rid = str(req_id)
            self._promised_loads.pop(rid, None)
            if self._load_failure_reports.pop(rid, 0) > 0:
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

    def request_finished(self, request, block_ids) -> tuple[bool, dict | None]:
        req_id = request.request_id
        self._promised_loads.pop(req_id, None)
        seq = self._seqs.get(req_id)
        if seq is not None:
            self._scheduler.request_finished(seq)
            # Blocks may still be pinned by an in-flight save; ATOM says when.
            if self._scheduler.should_defer_free(seq):
                # vLLM will not ask a second time -- it holds the blocks until
                # the connector names the id in `finished_sending`, and
                # `_collect_releases` is what eventually produces that.
                self._deferred_frees.add(req_id)
                return True, None
            self._seqs.drop(req_id)
        return False, None


def _req_id_of(completion_id) -> str:
    """Completion ids are a bare request id, or one tagged with a generation."""
    return str(getattr(completion_id, "req_id", completion_id))


def _block_ids(blocks) -> list[int]:
    """Flatten vLLM's allocated-block structure to plain ids.

    vLLM has spelled this several ways across versions (KVCacheBlocks with
    ``get_block_ids()``, a per-group tuple of lists, or already-flat ids), and
    the offload codec only ever needs the ids.
    """
    getter = getattr(blocks, "get_block_ids", None)
    if getter is not None:
        blocks = getter()
    if (
        isinstance(blocks, (list, tuple))
        and blocks
        and isinstance(blocks[0], (list, tuple))
    ):
        return [int(b) for group in blocks for b in group]
    return [int(b) for b in (blocks or [])]


def _scheduled_frontiers(scheduler_output):
    """Yield ``(request_id, num_computed_tokens)`` for this step's requests."""
    for req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
        yield req.req_id, getattr(req, "num_computed_tokens", 0)
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    req_ids = getattr(cached, "req_ids", None) or []
    computed = getattr(cached, "num_computed_tokens", None) or []
    yield from zip(req_ids, computed)
