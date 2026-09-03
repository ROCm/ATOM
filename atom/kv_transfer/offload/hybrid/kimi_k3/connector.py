# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache offload for Kimi-K3: dense paged KV plus a recurrent-state tier.

K3's paged KV *is* ordinary dense MLA -- block-addressed, sliceable, moved by
chunk -- so this extends the dense layout rather than being written beside it,
and the KV leg's behaviour is unchanged. What it adds is one more leg for the
KDA recurrent state, because a prefix whose KV came back from LMCache is useless
without the state that goes with it.

**The state load leg rides the request and shares its task.** It travels as
`LMCacheReqMeta.state_load_spec`, exactly as DSV4's `slot_load_spec` already
does, and `_do_load_req` runs it immediately after the KV leg inside the same
task, emitting exactly ONE completion for both. That is what lets the engine's
`StateOffloadIndex` be the sole owner of a load's lifecycle and state the
invariant `dispatched == settled + outstanding`.

**The state store leg cannot ride a request**, because a checkpoint outlives the
request that produced it (see `page_unit_checkpoint`). It keeps one named
completion channel, which `MultiConnector.get_finished` already forwards
generically, so this layout needs no new shell plumbing under `multi`.
"""

from __future__ import annotations

import logging

from atom.kv_transfer.disaggregation.types import ConnectorCompletion
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import pp_aware_rank_and_world
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.hybrid.kimi_k3.staging import StagedTransfer
from atom.kv_transfer.offload.hybrid.kimi_k3.state_object import StateByteCodec
from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier
from atom.kv_transfer.offload.metadata import (
    LMCacheOffloadMetadata,
    LMCacheReqMeta,
    LoadSpec,
    StateLoadSpec,
)

logger = logging.getLogger("atom")

#: The one connector-owned channel this layout adds. Three event kinds ride it,
#: distinguished by the first element of the operation id:
#:   ("src", op_id)  the PAGE units are free -- the device has stopped reading
#:   ("put", op_id)  the CPU put resolved; `succeeded` says whether it landed
#:   ("miss", hash)  a state `get` itself missed, so the index must retract it
#: One channel rather than three, because the two store phases are two fields of
#: one record with one owner, and a miss belongs to the same owner.
STATE_CHANNEL = "k3_state"


class KimiK3OffloadConnector(DenseOffloadConnector):
    """Worker half: dense KV, plus the recurrent-state legs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_tier: StateOffloadTier | None = None
        # Store ops handed to a rank that has no tier. Per instance: a mutable
        # class attribute would be shared by every connector in the process.
        self._state_no_tier: list = []

    def register_kv_caches(self, *args, **kwargs):
        out = super().register_kv_caches(*args, **kwargs)
        transfer_tensors = kwargs.get("transfer_tensors")
        if transfer_tensors is None and args:
            transfer_tensors = args[-1]
        self._build_state_tier(transfer_tensors)
        return out

    def close(self) -> None:
        # Before the base tears its executors down: a store still in flight is
        # holding PAGE units pinned, and its report is what releases them.
        tier, self._state_tier = self._state_tier, None
        if tier is not None:
            tier.shutdown()
        close = getattr(super(), "close", None)
        if close is not None:
            close()

    # ---------------------------- construction ----------------------------- #
    def _build_state_tier(self, transfer_tensors) -> None:
        from aiter.dist.parallel_state import get_tp_group

        # PP is refused outright, not half-supported: `CacheEngineKey` has no PP
        # component, so two stages at the same TP rank would overwrite each
        # other's images. Loud at startup rather than a warning that leaves the
        # engine running with the feature silently off -- if PP is wanted, drop
        # the offload connector from the launch line.
        pp_size = int(getattr(self._config, "pipeline_parallel_size", 1) or 1)
        if pp_size > 1:
            raise ValueError(
                "kimi_k3 offload: the recurrent-state tier does not support "
                f"pipeline parallelism (pipeline_parallel_size={pp_size}). The "
                "LMCache key carries no PP component, so two stages at one TP "
                "rank would overwrite each other's state images. Run with "
                "pipeline_parallel_size=1, or drop --kv-transfer-config."
            )

        backend = getattr(transfer_tensors, "state_backend", None)
        if backend is None:
            logger.warning(
                "kimi_k3 offload: no attention backend published; state tier off."
            )
            return
        # The geometry the bytes are written under, folded into every key so a
        # changed build cannot read another build's images back as valid. Read
        # from the runtime rather than recomputed, so HBM and CPU share one
        # owner of the string.
        spec = getattr(getattr(backend, "model_runner", None), "state_runtime", None)
        spec = getattr(spec, "checkpoint_spec", None)
        layout_id = getattr(spec, "layout_id", None)
        if not layout_id:
            logger.warning(
                "kimi_k3 offload: no checkpoint layout id published; state tier "
                "off. Without it a build that changed the state geometry could "
                "read another's images back as valid."
            )
            return
        try:
            views = backend.state_entry_views(0)
            entry_bytes = sum(int(v.numel()) * v.element_size() for v in views)
        except (NotImplementedError, AttributeError):
            # No per-request state on this backend. IndexError is deliberately
            # not caught: a zero-entry pool with the tier on is a sizing bug.
            logger.warning(
                "kimi_k3 offload: %s owns no per-request state views; tier off.",
                type(backend).__name__,
            )
            return
        # The store reads `page_unit_views`, a DIFFERENT method from the load's
        # `state_entry_views`. A backend with one but not the other would build
        # the tier, pass every load, then raise on the first store -- which the
        # tier's blind `except` would mask as an endlessly failing store. Probe
        # it here so the mismatch fails fast and visibly.
        if not callable(getattr(backend, "page_unit_views", None)):
            logger.warning(
                "kimi_k3 offload: %s has state_entry_views but no callable "
                "page_unit_views; the store path needs it, so a tier would fail "
                "every store silently. State tier off.",
                type(backend).__name__,
            )
            return
        # The store reads PAGE units and the load writes an Active Slot, so the
        # blob must be the same length both ways. A model where they differ
        # would truncate the store or over-read the load.
        image_bytes = int(getattr(spec, "image_bytes", 0) or 0)
        if image_bytes and image_bytes != entry_bytes:
            logger.warning(
                "kimi_k3 offload: a checkpoint image is %d B but an Active Slot "
                "is %d B; the store reads units and the load writes a slot, so "
                "they must match. State tier off.",
                image_bytes,
                entry_bytes,
            )
            return

        tp = get_tp_group()
        rank, world = pp_aware_rank_and_world(self._config, tp)
        cfg = offcfg.build_lmcache_config(
            getattr(self._config, "kv_transfer_config", None)
        )
        meta = offcfg.build_lmcache_metadata(self._config, cfg, world, rank)
        gpu_connector = self._engine.gpu_connector
        # Sized to a whole entry rather than shared with the KV staging buffer,
        # which is sized in LMCache chunks and is routinely an order of
        # magnitude smaller.
        staged = StagedTransfer(
            gpu_connector.device,
            staging_buffer_bytes=entry_bytes,
            release_after_transfer=gpu_connector.release_gpu_staging_after_transfer,
        )
        codec = StateByteCodec(
            backend,
            staged,
            entry_bytes,
            model_name=meta.model_name,
            world_size=world,
            worker_id=rank,
            layout_id=layout_id,
        )
        # ONE pool, shared with paged KV. A request writes its KV chunks and its
        # one state object inside the same prefill window, so both enter
        # LMCache's LRU together and cool at the same rate -- which is what a
        # joint boundary needs, since a boundary whose KV is gone is worthless.
        # `LMCACHE_MAX_LOCAL_CPU_SIZE` is the one size to tune.
        codec.bind_storage_manager(self._engine.storage_manager)
        self._state_tier = StateOffloadTier(codec)
        logger.info(
            "kimi_k3 offload: state tier up, entry=%.2f MiB rank=%d, "
            "sharing the paged-KV CPU pool, layout=%s",
            entry_bytes / (1 << 20),
            rank,
            layout_id,
        )

    # ------------------------------- per step ------------------------------ #
    def start_load_kv(self, metadata) -> None:
        self._start_state_stores(metadata)
        super().start_load_kv(metadata)

    def _do_load_req(self, req: LMCacheReqMeta) -> None:
        """The fusion: both legs in one task, exactly one completion.

        The state leg runs only if the KV leg landed, because state at the
        boundary is the compressed history of exactly the prefix the KV leg was
        asked to complete -- restoring it over a prefix whose KV never arrived
        would have the forward resume on a history it does not hold.

        `_finish_load` is reached on every path including an exception, so one
        dispatch produces one report whatever happens.
        """
        try:
            ok = self._load_kv_bytes(req)
            if ok and req.state_load_spec is not None:
                ok = self._load_state_bytes(req)
        except Exception:
            logger.warning(
                "kimi_k3 offload: load failed for req=%s", req.req_id, exc_info=True
            )
            ok = False
        self._finish_load(req, ok)

    def _load_state_bytes(self, req: LMCacheReqMeta) -> bool:
        spec = req.state_load_spec
        tier = self._state_tier
        if tier is None:
            # Nothing can serve this leg, so the request must recompute. Failing
            # here rather than passing the KV leg through as a success is the
            # difference between a recompute and the engine counting a state
            # restore that never happened.
            logger.warning(
                "kimi_k3 offload: no state tier on this rank; failing the load "
                "for req=%s so it recomputes.",
                req.req_id,
            )
            return False
        return tier.load_state(spec.boundary_hash, spec.destination_slot)

    def _start_state_stores(self, metadata) -> None:
        stores = getattr(metadata, "state_stores", None) or ()
        if not stores:
            return
        tier = self._state_tier
        for spec in stores:
            if tier is None:
                # The engine pinned PAGE units for a store this rank cannot do.
                # Report both phases so the units come straight back and the
                # store is counted as failed, rather than waiting on a reclaimer.
                self._state_no_tier.append(spec.op_id)
                continue
            tier.submit_store(spec.op_id, spec.prefix_hash, spec.unit_ids)

    def get_finished(self):
        out = super().get_finished()
        completions = out.connector_completions
        no_tier, self._state_no_tier = self._state_no_tier, []
        for op_id in no_tier:
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("src", op_id), True))
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("put", op_id), False))
        tier = self._state_tier
        if tier is None:
            return out
        released, stored, failed = tier.take_store_reports()
        for op_id in released:
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("src", op_id), True))
        for op_id in stored:
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("put", op_id), True))
        for op_id in failed:
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("put", op_id), False))
        for h in tier.take_missed_hashes():
            # The ONLY event that may retract a hash from the engine's index. A
            # fused load verdict of "failed" may mean the KV leg, and retracting
            # on that would permanently deny state bytes that are still present.
            completions.add(ConnectorCompletion(STATE_CHANNEL, ("miss", int(h)), False))
        return out


class KimiK3OffloadScheduler(DenseOffloadScheduler):
    """Engine half: aims both legs at one boundary and owns the store queue."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._state_stores: list = []
        self._state_events: list[tuple] = []
        # Sequences with a state load armed this pass, so the metadata post-pass
        # can reach the seq the base builder does not hand it.
        self._state_load_seqs: dict = {}

    # --------------------------- what the engine asks ---------------------- #
    def state_tier_capability(self) -> dict:
        """What the engine needs to build its index, reported once at attach.

        Reported rather than re-derived: the engine parsing this connector's own
        config gives two derivations of one fact, and the chunk grid is the fact
        that must not be derived twice -- the engine floors a boundary against
        it while the worker's transfer is validated against it.
        """
        return {
            "can_store": bool(self._do_save),
            "can_load": bool(self._do_load),
            "chunk_tokens": int(self.chunk_size or 0),
        }

    def enqueue_state_stores(self, specs) -> bool:
        self._state_stores.extend(specs)
        return True

    def take_state_events(self) -> list[tuple]:
        """Terminal state events for the engine to apply, drained per pass."""
        events, self._state_events = self._state_events, []
        return events

    def connector_completion(self, completion) -> bool:
        if completion.channel != STATE_CHANNEL:
            base = getattr(super(), "connector_completion", None)
            return False if base is None else base(completion)
        kind, value = completion.operation_id
        self._state_events.append((kind, value, completion.succeeded))
        return True

    def update_state_after_alloc(self, seq) -> None:
        super().update_state_after_alloc(seq)
        if not self._do_load:
            return
        joint = getattr(seq, "offload_joint", None)
        if joint is None or int(joint.state_load_hash) == -1:
            return
        sid = str(seq.id)
        self._state_load_seqs[sid] = seq
        if self._load_specs.get(sid) is not None:
            return
        # A state-only load: the KV is already resident, so the lookup found
        # nothing extra and armed no spec. Arm a no-op one so the leg travels
        # the ORDINARY load path -- same metadata, same task, same completion.
        # Giving it a path of its own is what previously required a second
        # channel and a park to reconcile the two.
        hbm = int(seq.num_cached_tokens)
        self._load_specs[sid] = LoadSpec(
            hbm_cached_tokens=hbm, lmcache_cached_tokens=hbm, can_load=True
        )
        self._reqs_need_recv[sid] = seq

    # ------------------------------ the two legs --------------------------- #
    def _decide_load_after_alloc(self, seq, ls):
        """Clamp a hybrid's KV leg to the boundary the state leg is aimed at.

        A hybrid's per-request state is the compressed history of exactly
        `[0, B)`. Raising the KV-loaded length past `B` would have the forward
        skip `[B, lmc)` while the linear layers never see it: wrong output, no
        exception. So a hybrid loads only where `can_allocate` picked one
        boundary for both legs, and this clamps the KV leg down to it.
        """
        if not getattr(seq, "has_per_req_cache", False):
            return super()._decide_load_after_alloc(seq, ls)

        hbm = int(seq.num_cached_tokens)
        lmc = int(ls.lmcache_cached_tokens)
        # From the spec's own grid, not re-derived: see `state_tier_capability`.
        chunk = self.chunk_size or 256
        joint = int(seq.offload_joint.boundary_tokens or 0)
        if joint <= hbm:
            return False, "per_req_cache_state_boundary", hbm, lmc, lmc - hbm, chunk
        # Where the transfer STARTS -- not what the request may call cached.
        # `allocate` claimed every matched block, not only resumable ones, so the
        # KV below this is already resident; asking LMCache to resend it would
        # land a second copy in HBM.
        start = max(hbm, int(seq.offload_joint.claim_tokens or 0))
        if start % chunk != 0:
            return False, "joint_unaligned_hbm_prefill", start, lmc, lmc - start, chunk
        kv_target = int(seq.offload_joint.kv_tokens or 0) or joint
        if joint > lmc or kv_target > lmc:
            return False, "joint_boundary_above_lookup", start, lmc, lmc - start, chunk
        if kv_target <= start:
            # The whole boundary is already resident: a state-only load. Emitted
            # as a KV no-op rather than refused, so the state leg still travels
            # the ordinary load path and produces the ordinary completion.
            ls.hbm_cached_tokens = start
            ls.lmcache_cached_tokens = start
            return True, "joint_state_only", start, start, 0, chunk
        # Both ends together: the worker reads `[hbm_cached_tokens,
        # lmcache_cached_tokens)`, so leaving the start where the lookup put it
        # would fetch from token 0 every time.
        ls.hbm_cached_tokens = start
        ls.lmcache_cached_tokens = kv_target
        # Deliberately past the min-load floor: the boundary was chosen for both
        # legs, and refusing on size would leave the state leg claiming a prefix
        # whose KV never came.
        return True, "joint_state_and_kv", start, kv_target, kv_target - start, chunk

    def _claim_after_load(self, seq, hbm: int, lmc: int) -> int:
        """How far the request may call itself cached once the load lands.

        For a joint load that is the STATE boundary, which sits at or below the
        transfer's end: the KV leg is aimed at the chunk covering it, and
        claiming the rounded-up figure would have the forward skip tokens the
        recurrent state does not cover. This is the KDA red line.
        """
        joint = int(seq.offload_joint.boundary_tokens or 0)
        claim = max(hbm, min(joint, lmc)) if joint else max(hbm, lmc)
        assert not joint or claim <= joint, (
            "a hybrid may never claim past its state boundary: "
            f"claim={claim} boundary={joint}"
        )
        return claim

    def should_park_for_load_after_alloc(self, seq) -> bool:
        # A state-only load has no extra KV to fetch, so the base predicate --
        # which keys off the KV load spec -- cannot see it. It still has to park:
        # its recurrent state is in flight and the forward must not start
        # without it.
        if super().should_park_for_load_after_alloc(seq):
            return True
        joint = getattr(seq, "offload_joint", None)
        return joint is not None and int(joint.state_load_hash) != -1

    def build_connector_meta(self) -> LMCacheOffloadMetadata:
        meta = super().build_connector_meta()
        meta.state_stores = tuple(self._state_stores)
        self._state_stores = []
        # Attached in a post-pass rather than by overriding the base's builder:
        # the KV leg's metadata is dense's to construct, and forking that loop
        # is how a layout stops receiving dense's fixes.
        for req in meta.requests:
            sid = str(req.req_id)
            seq = self._state_load_seqs.pop(sid, None)
            if seq is None:
                continue
            joint = seq.offload_joint
            h = int(joint.state_load_hash)
            if h == -1:
                continue
            req.state_load_spec = StateLoadSpec(
                boundary_tokens=int(joint.boundary_tokens or 0),
                boundary_hash=h,
                destination_slot=int(seq.state_slot),
                chunk_tokens=int(self.chunk_size or 0),
            )
        return meta

    def has_pending_work(self) -> bool:
        # The base counts only the KV trackers, so a step whose sole outstanding
        # work is a state store would look idle -- and the idle path is what
        # drains the store reports that release the pinned PAGE units.
        return bool(super().has_pending_work() or self._state_stores)
