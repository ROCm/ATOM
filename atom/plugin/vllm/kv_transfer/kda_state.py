# SPDX-License-Identifier: MIT
"""KDA recurrent-state offload for the vLLM plugin path.

Kimi-K3 is hybrid: MLA full-attention layers plus KDA recurrent layers. vLLM
builds one KV cache group per family, and a restored MLA prefix is only correct
if the KDA state at the *same token boundary* is restored with it. Half a
restore is not a crash and not a log line -- it is wrong output.

The two groups are moved by two different mechanisms, and that asymmetry is
forced, not chosen:

* The MLA group is ordinary paged KV. ``DenseKVByteCodec`` gathers whole blocks
  by position out of the request's block table, exactly as it does for M3 and
  GLM-5.2. Nothing here changes it.
* The KDA group in ``--mamba-cache-mode align`` cannot be read positionally at
  all. vLLM's own store connector says why (``mooncake/store/coordinator.py``,
  ``store_mask``): an align-mode mamba block table is not append-only -- a
  superseded state block is freed and nulled, and speculative blocks relocate in
  place -- so indexing it by ``token // block_size`` can land on a null, freed,
  or live speculative block and persist those bytes under a valid prefix hash.
  The only safe source is vLLM's explicit hand-off --
  ``SchedulerOutput.partial_tail_offloads`` on 0.28, the same payload under
  ``kv_connector_block_state.boundary_state_offloads`` on 0.29 -- which names
  the exact block holding a committed boundary state.

So a KDA boundary is stored as one whole opaque page under the prefix hash at
that boundary -- one key, one block -- through
:class:`~atom.kv_transfer.offload.hybrid.kimi_k3.state_object.StateByteCodec`,
which already speaks that shape on ATOM's native path and shares the paged-KV
LMCache ``StorageManager`` so the two tiers compete for one pool rather than two.

Correctness of the *pair* is enforced on lookup, not on save: the reported
external hit is capped at the largest boundary whose KDA state this index still
claims (:meth:`KdaBoundaryPlanner.cap_hit`). A KDA state that was never stored,
or was evicted, therefore shortens the prefix instead of corrupting it, and the
save side needs no cross-group parking.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch

from atom.model_engine.state_offload import StateOffloadIndex

logger = logging.getLogger("atom")

#: vLLM's placeholder block. ``vllm.v1.attention.backends.utils.NULL_BLOCK_ID``
#: is the authority; it is mirrored rather than imported so this module stays
#: importable in a unit test that has no vLLM.
NULL_BLOCK_ID = 0

#: How many chunk-sized steps :meth:`KdaBoundaryPlanner.cap_hit` will walk back
#: looking for a boundary the index still claims. Stores are emitted for every
#: chunk-aligned boundary of a saved prefix, so the state should be present at
#: the same boundary the KV hit reports; more than a couple of misses means the
#: two tiers have diverged (independent eviction), and walking the whole prefix
#: to find that out costs the scheduler thread for no gain.
_MAX_CAP_DESCENT = 8


def unwrap_kv_cache_spec(spec: Any) -> Any:
    """The concrete spec behind a ``UniformTypeKVCacheSpecs`` wrapper.

    vLLM wraps a group whose layers share a type, and ``isinstance(spec,
    MambaSpec)`` is False for the wrapper -- so classifying on the wrapper
    silently treats a mamba group as attention, which is the one mistake this
    module exists to prevent.
    """
    inner = getattr(spec, "kv_cache_specs", None)
    if inner:
        specs = list(inner.values())
        if specs:
            return specs[0]
    return spec


def find_mamba_group(kv_cache_groups) -> tuple[int, Any] | None:
    """``(group_id, spec)`` of the single mamba group, or None.

    Classification is by ``MambaSpec``, never by tensor shape: a mamba page and
    an attention page are both ``[num_blocks, ...]`` uint8 to this connector, so
    shape cannot tell them apart and a wrong guess moves the wrong bytes.
    """
    try:
        from vllm.v1.kv_cache_interface import MambaSpec
    except ImportError:  # pragma: no cover - vLLM is absent in unit tests
        return None
    found: list[tuple[int, Any]] = []
    for group_id, group in enumerate(kv_cache_groups or ()):
        spec = unwrap_kv_cache_spec(group.kv_cache_spec)
        if isinstance(spec, MambaSpec):
            found.append((group_id, spec))
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "ATOM offload connector: more than one mamba KV cache group "
            f"({[g for g, _ in found]}); the joint hit gate reconciles the "
            "attention hit against exactly one recurrent state"
        )
    return found[0]


def step_boundary_offloads(scheduler_output):
    """This step's recurrent boundary hand-offs, or None.

    Shape either way is ``{req_id: [(group_id, block_id, boundary_tokens)]}``:
    the block vLLM copied the committed recurrent state into, named explicitly
    rather than found by indexing a block table that is not append-only.

    Two spellings carry it. vLLM 0.28 -- the version ATOM pins -- puts it flat
    on the scheduler output as ``partial_tail_offloads``
    (``SchedulerOutput``, fed by ``KVCacheManager.take_partial_tail_offloads``).
    0.29 moved the same payload under
    ``kv_connector_block_state.boundary_state_offloads``. Both are read so this
    connector keeps working across that upgrade instead of going quietly
    hit-less on one side of it.
    """
    offloads = getattr(scheduler_output, "partial_tail_offloads", None)
    if offloads:
        return offloads
    state = getattr(scheduler_output, "kv_connector_block_state", None)
    return getattr(state, "boundary_state_offloads", None) if state else None


def boundary_prefix_hash(block_hash: bytes) -> int:
    """Fold vLLM's ``BlockHash`` into the 64-bit int the state codec keys on.

    ``Request.block_hashes`` is already chained over the whole prefix and
    already folds in everything that makes two identical token runs different
    (LoRA, multimodal inputs, cache salt), so deriving the state key from it
    keeps the two tiers keyed by the same notion of "same prefix". It is bytes
    of unbounded width; ``StateByteCodec.key`` takes a 64-bit int.

    ``xxh64`` rather than ``hash(bytes)``: Python salts ``hash`` per process, so
    a restart would orphan every entry written before it -- silently, as a cache
    that simply never hits.
    """
    import xxhash

    return xxhash.xxh64(bytes(block_hash)).intdigest()


class KdaPageViews:
    """Address one mamba block as the ordered byte stream the codec moves.

    :class:`StateByteCodec` was written against ATOM's native slot model and
    asks its backend two questions -- ``page_unit_views`` for a store,
    ``state_entry_views`` for a load. On the plugin path both resolve to the
    same thing, the per-layer views of one vLLM block, because vLLM allocates
    the destination block for an external hit and the mamba group's own block
    table is what the resuming forward reads. Keeping both methods (rather than
    collapsing them) is what lets the native codec be reused verbatim.

    Layer order is ``group.layer_names``, vLLM's own canonical order, so a
    stream gathered on one rank is read back in the same order on the next run.
    """

    def __init__(self, tensors: list[torch.Tensor], *, layout_id: str) -> None:
        if not tensors:
            raise ValueError("KDA state offload: the mamba group registered no tensors")
        self._tensors = list(tensors)
        self.layout_id = layout_id
        self.num_blocks = int(self._tensors[0].shape[0])
        for tensor in self._tensors:
            if int(tensor.shape[0]) != self.num_blocks:
                raise ValueError(
                    "KDA state offload: mamba layers disagree on block count "
                    f"({[int(t.shape[0]) for t in self._tensors]}); a boundary "
                    "block id addresses every layer, so one stream would be "
                    "gathered at the wrong offset"
                )
        self.entry_bytes = sum(
            int(t[0].numel()) * t[0].element_size() for t in self._tensors
        )

    def _views(self, block_id: int) -> list[torch.Tensor]:
        block_id = int(block_id)
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(
                f"KDA state offload: block {block_id} is outside the mamba "
                f"group's {self.num_blocks} blocks"
            )
        return [t[block_id] for t in self._tensors]

    def page_unit_views(self, unit_ids) -> list[torch.Tensor]:
        """Store source. Exactly one block; a state image is never split."""
        ids = list(unit_ids)
        if len(ids) != 1:
            raise ValueError(
                f"KDA state offload: a boundary is one block, got {len(ids)}"
            )
        return self._views(ids[0])

    def state_entry_views(self, slot) -> list[torch.Tensor]:
        """Load destination: the block vLLM allocated for the external hit."""
        return self._views(slot)


def build_layout_id(spec: Any, tensors: list[torch.Tensor]) -> str:
    """Name the geometry the bytes were written under.

    Folded into the storage key by ``StateByteCodec.key``. One prefix hash maps
    to a different image under a different mamba block size, speculative-block
    count, dtype or TP shape, and KV and state entries share one LMCache pool
    with no field saying what an entry is -- so without this a config change
    reads back another layout's bytes as state, which is silent wrong output
    rather than a miss.
    """
    mamba_type = getattr(spec, "mamba_type", None)
    parts = [
        "vllm-kda",
        str(getattr(mamba_type, "name", mamba_type)),
        f"bs={int(getattr(spec, 'block_size', 0))}",
        f"page={int(getattr(spec, 'page_size_bytes', 0))}",
        f"spec_blocks={int(getattr(spec, 'num_speculative_blocks', 0))}",
        f"tp_replicated={int(bool(getattr(spec, 'tp_replicated', False)))}",
        f"layers={len(tensors)}",
        ";".join(f"{tuple(t.shape[1:])}:{t.dtype}" for t in tensors),
    ]
    return "|".join(parts)


@dataclass(frozen=True)
class KdaStore:
    """One boundary state on its way out: op id, key, and source block."""

    op_id: int
    prefix_hash: int
    block_id: int


@dataclass(frozen=True)
class KdaLoad:
    """One boundary state on its way back in.

    ``error_block_ids`` are the *attention* group's blocks that this request's
    dense load is filling. They ride along because a KDA miss has to invalidate
    them: vLLM otherwise takes the request's appearance in ``finished_recving``
    at face value and caches the whole external prefix, serving an MLA prefix
    whose recurrent state was never restored.

    ``block_id <= 0`` means the destination could not be resolved; the tier
    fails it without touching the device, which is the same outcome as a miss.
    """

    req_id: str
    prefix_hash: int
    block_id: int
    error_block_ids: tuple[int, ...] = ()


@dataclass
class _PendingStore:
    block_id: int
    prefix_hash: int
    reports: int = 0
    failures: int = 0


@dataclass
class _KdaLoadResult:
    ok: bool
    error_block_ids: tuple[int, ...] = field(default=())


class KdaStateTier:
    """Worker half: moves boundary-state bytes, decides nothing.

    Its own executors rather than the dense connector's, for the reason
    ``StateOffloadTier`` splits its lanes: a load is on the TTFT critical path
    and a store is not, and one queue makes that unenforceable. One thread each,
    because ``StagedTransfer`` keys its staging buffer per thread -- more
    threads is more standing HBM, not more bandwidth.

    Reports only. The index lives in the scheduler process, so both directions
    hand sets back rather than recording anything here; neither side can then
    hold a second opinion about what is stored.
    """

    def __init__(self, codec, *, thread_name_prefix: str = "atom-kda") -> None:
        self._codec = codec
        self._store_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{thread_name_prefix}-store"
        )
        self._load_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{thread_name_prefix}-load"
        )
        self._lock = threading.Lock()
        self._stored: dict[int, int] = {}
        self._store_failed: dict[int, int] = {}
        self._load_results: dict[str, _KdaLoadResult] = {}
        self._inflight: dict[str, list] = {}

    # -- submission ------------------------------------------------------
    def submit_store(self, store: KdaStore, ready_event) -> None:
        """Persist the boundary block named by *store*.

        ``ready_event`` is not optional and not a convenience. The block's
        contents are written by ``preprocess_mamba``'s copy-on-write copy on the
        forward's compute stream in this very step, while ``StagedTransfer``
        issues its gather on a private ``pack_stream`` that -- by its own
        docstring -- never waits on the producer. Without the event the gather
        is free to run first and store the previous occupant's state under this
        boundary's hash. vLLM's own store connector fences the same way
        (``mooncake/store/scheduler.py``: "the CoW copy is enqueued before the
        connector event records, so this step's event fences the exact block").
        """
        self._track(
            f"store:{store.op_id}",
            self._store_executor.submit(self._do_store, store, ready_event),
        )

    def submit_load(self, load: KdaLoad) -> None:
        self._track(
            f"load:{load.req_id}", self._load_executor.submit(self._do_load, load)
        )

    def _track(self, key: str, future) -> None:
        with self._lock:
            self._inflight.setdefault(key, []).append(future)

        def _done(fut) -> None:
            with self._lock:
                pending = self._inflight.get(key)
                if pending is None:
                    return
                if fut in pending:
                    pending.remove(fut)
                if not pending:
                    self._inflight.pop(key, None)

        future.add_done_callback(_done)

    # -- transfers -------------------------------------------------------
    def _do_store(self, store: KdaStore, ready_event) -> None:
        ok = False
        try:
            if ready_event is not None:
                # Host-side, on the store thread: the forward is long past its
                # launch by the time this runs, and blocking the compute stream
                # instead would put offload on the critical path.
                ready_event.synchronize()
            ok = bool(self._codec.put(int(store.prefix_hash), [int(store.block_id)]))
        except Exception:  # deliberately blind
            # `put` reaches into LMCache, whose failure modes are its own. A
            # store that cannot happen costs one boundary -- not this thread,
            # whose death would strand every request parked on a later load.
            logger.warning(
                "KDA state offload: store of hash %d (block %d) failed",
                store.prefix_hash,
                store.block_id,
                exc_info=True,
            )
        with self._lock:
            target = self._stored if ok else self._store_failed
            target[store.op_id] = target.get(store.op_id, 0) + 1

    def _do_load(self, load: KdaLoad) -> None:
        ok = False
        try:
            if load.block_id > NULL_BLOCK_ID:
                ok = bool(self._codec.get(int(load.prefix_hash), int(load.block_id)))
        except Exception:  # a failed load is a normal path
            # LMCache's LRU can drop bytes under a hash the index still
            # advertises. Retracting that claim is the scheduler's job; the
            # report below is what tells it to.
            logger.warning(
                "KDA state offload: load of hash %d failed",
                load.prefix_hash,
                exc_info=True,
            )
        with self._lock:
            self._load_results[load.req_id] = _KdaLoadResult(
                ok, () if ok else tuple(load.error_block_ids)
            )

    # -- drains ----------------------------------------------------------
    def take_store_reports(self) -> tuple[dict[int, int], dict[int, int]]:
        with self._lock:
            stored, failed = self._stored, self._store_failed
            self._stored, self._store_failed = {}, {}
        return stored, failed

    def take_load_results(self) -> dict[str, _KdaLoadResult]:
        with self._lock:
            results = self._load_results
            self._load_results = {}
        return results

    def wait_for_requests(self, req_ids) -> None:
        """Fence the transfers still reading a just-preempted request's blocks.

        Only loads are keyed by request; a store is keyed by its boundary and
        its source block is pinned in the block pool until every rank reports,
        so preemption cannot hand that block to anyone else while it is read.
        """
        futures = []
        with self._lock:
            for req_id in req_ids:
                futures.extend(self._inflight.get(f"load:{req_id}", ()))
        for future in futures:
            try:
                future.result()
            except Exception:  # already reported by the task itself
                logger.debug("KDA state offload: fenced a failed load", exc_info=True)

    def close(self) -> None:
        self._store_executor.shutdown(wait=True)
        self._load_executor.shutdown(wait=True)


class KdaBoundaryPlanner:
    """Scheduler half: what to store, what to load, and how far a hit is valid.

    Owns the index of boundaries believed present, the block-pool pins that keep
    a handed-off boundary block alive across its asynchronous store, and the cap
    that makes a missing recurrent state shorten a prefix rather than corrupt it.
    """

    def __init__(
        self,
        *,
        group_id: int,
        mamba_block_size: int,
        hash_block_size: int,
        chunk_size: int,
        world_size: int,
        can_store: bool = True,
        can_load: bool = True,
    ) -> None:
        self.group_id = int(group_id)
        self.mamba_block_size = int(mamba_block_size)
        self.hash_block_size = int(hash_block_size)
        self.chunk_size = int(chunk_size)
        self._world_size = max(1, int(world_size))
        self._index = StateOffloadIndex(can_store=can_store, can_load=can_load)
        self._pool = None
        self._next_op_id = 0
        self._pending_stores: dict[int, _PendingStore] = {}
        self._stores: list[KdaStore] = []
        self._loads: list[KdaLoad] = []
        # Set by the connector immediately before it delegates a lookup, and
        # read by `cap_hit` -- ATOM's scheduler hands the hook a SeqView, which
        # carries no block hashes of its own.
        self._lookup_ctx: tuple[str, Any] | None = None
        if self.chunk_size % self.mamba_block_size != 0:
            raise ValueError(
                f"KDA state offload: LMCache chunk size {self.chunk_size} is not "
                f"a multiple of the mamba block size {self.mamba_block_size}; a "
                "chunk would then end between two boundaries and no joint hit "
                "could ever be reported"
            )

    # -- block pool ------------------------------------------------------
    def bind_gpu_block_pool(self, pool) -> None:
        self._pool = pool

    # -- lookup ----------------------------------------------------------
    def begin_lookup(self, request) -> None:
        self._lookup_ctx = (str(request.request_id), request)

    def end_lookup(self) -> None:
        self._lookup_ctx = None

    def cap_hit(self, seq, hit: int) -> int:
        """Shorten *hit* to the last boundary this index still claims.

        The whole joint-correctness argument is this method. An MLA prefix
        restored past the last stored KDA boundary is served with a recurrent
        state that belongs to some other prefix -- no exception, no log line,
        just wrong tokens. Capping turns that into a shorter hit, which is only
        ever a performance loss.

        Only chunk-aligned boundaries are probed because only those are stored
        (see :meth:`collect_stores`) and because the dense tier's own hit
        descends in chunk steps, so a finer probe could not produce a usable
        pair anyway.
        """
        hit = int(hit)
        if hit <= 0 or self._lookup_ctx is None:
            return hit
        req_id, request = self._lookup_ctx
        if str(seq.id) != req_id:
            # The hook is armed per lookup; a mismatch means the scheduler
            # called it for a different sequence than the one the connector is
            # in. Refusing the hit is the safe direction.
            logger.warning(
                "KDA state offload: hit cap armed for %s but called for %s; "
                "declining the external hit",
                req_id,
                seq.id,
            )
            return 0
        block_hashes = getattr(request, "block_hashes", None) or ()
        boundary = (hit // self.chunk_size) * self.chunk_size
        for _ in range(_MAX_CAP_DESCENT):
            if boundary <= 0:
                return 0
            h = self.boundary_hash(block_hashes, boundary)
            if h is not None and self._index.could_serve(h):
                return min(hit, boundary)
            boundary -= self.chunk_size
        logger.debug(
            "KDA state offload: no stored boundary within %d chunks of hit %d "
            "for %s; declining the external hit",
            _MAX_CAP_DESCENT,
            hit,
            req_id,
        )
        return 0

    def boundary_hash(self, block_hashes, boundary_tokens: int) -> int | None:
        """The prefix-hash key for the state committed at *boundary_tokens*.

        None when vLLM has not hashed that far -- which happens routinely, since
        block hashes only cover the tokens the request has actually committed.
        """
        if boundary_tokens <= 0 or boundary_tokens % self.hash_block_size:
            return None
        index = boundary_tokens // self.hash_block_size - 1
        if index < 0 or index >= len(block_hashes):
            return None
        return boundary_prefix_hash(block_hashes[index])

    # -- load ------------------------------------------------------------
    def resolve_load(
        self,
        request,
        group_blocks: tuple[list[int], ...],
        num_total_computed: int,
        attention_group_id: int,
        num_external_tokens: int,
        attention_block_size: int,
    ) -> None:
        """Queue the KDA leg of a request whose external hit was just allocated.

        The destination is positional and that is safe here, unlike on the save
        side: vLLM has just allocated this request's mamba blocks for the hit,
        so row ``num_total_computed // mamba_block_size - 1`` is the block the
        resuming forward will read its initial state from.

        A destination that cannot be resolved is queued as a failing load rather
        than dropped. Dropping it would leave the dense leg to report success on
        its own, and vLLM would cache an MLA prefix whose state never arrived.
        """
        if num_external_tokens <= 0:
            return
        req_id = str(request.request_id)
        block_hashes = getattr(request, "block_hashes", None) or ()
        h = self.boundary_hash(block_hashes, num_total_computed)
        error_blocks = self._attention_error_blocks(
            group_blocks,
            attention_group_id,
            num_total_computed,
            num_external_tokens,
            attention_block_size,
        )
        block_id = self._boundary_block(group_blocks, num_total_computed)
        if h is None or block_id <= NULL_BLOCK_ID:
            logger.warning(
                "KDA state offload: %s hit %d tokens but its boundary state has "
                "no %s; failing the load so the prefix is recomputed",
                req_id,
                num_total_computed,
                "key" if h is None else "destination block",
            )
            self._loads.append(KdaLoad(req_id, int(h or 0), 0, error_blocks))
            return
        self._index.request_load(req_id, h)
        self._loads.append(KdaLoad(req_id, h, block_id, error_blocks))

    def _boundary_block(
        self, group_blocks: tuple[list[int], ...], num_total_computed: int
    ) -> int:
        if self.group_id >= len(group_blocks):
            return 0
        blocks = group_blocks[self.group_id]
        row = num_total_computed // self.mamba_block_size - 1
        if row < 0 or row >= len(blocks):
            return 0
        return int(blocks[row])

    def _attention_error_blocks(
        self,
        group_blocks: tuple[list[int], ...],
        attention_group_id: int,
        num_total_computed: int,
        num_external_tokens: int,
        attention_block_size: int,
    ) -> tuple[int, ...]:
        """The attention blocks a failed KDA leg has to invalidate.

        Exactly the range the dense load fills: from the HBM frontier to the end
        of the external hit. Naming fewer would leave vLLM serving part of an
        unusable prefix; naming more would throw away blocks the GPU prefix
        cache legitimately owns.
        """
        if attention_group_id >= len(group_blocks) or attention_block_size <= 0:
            return ()
        blocks = group_blocks[attention_group_id]
        local = max(0, num_total_computed - num_external_tokens)
        start = local // attention_block_size
        end = -(-num_total_computed // attention_block_size)
        return tuple(int(b) for b in blocks[start:end] if int(b) > NULL_BLOCK_ID)

    def take_loads(self) -> list[KdaLoad]:
        loads, self._loads = self._loads, []
        return loads

    def on_load_result(self, req_id: str, ok: bool) -> None:
        if ok:
            self._index.complete_load(req_id)
        else:
            self._index.fail_load(req_id)

    def forget_pending(self, req_id: str) -> None:
        self._index.abandon_load(req_id)

    # -- store -----------------------------------------------------------
    def collect_stores(
        self, offloads, requests_by_id, skip_req_ids=()
    ) -> list[KdaStore]:
        """Turn this step's boundary hand-offs into pinned store jobs.

        Consumed in the step it arrives, because that is the only step in
        which it exists: the KV cache manager hands the pending offloads over
        once, while the step is being built, and forgets them.

        The filters, in the order they reject:

        * not the mamba group -- other groups are saved positionally by the
          dense tier and would be stored twice, under two different schemes;
        * the null block -- vLLM's placeholder, never real state;
        * a boundary that is not a whole mamba block -- the sub-block
          copy-on-write tail, which no chunk end can ever line up with;
        * a boundary that is not chunk-aligned -- :meth:`cap_hit` descends in
          chunk steps and can never select it, so storing it is pure volume;
        * a request that finished or was preempted in this same step -- its
          blocks are going away, so the pin would be taken on a block that is
          already someone else's.
        """
        accepted: list[KdaStore] = []
        pool = self._pool
        for req_id, entries in (offloads or {}).items():
            req_id = str(req_id)
            if req_id in skip_req_ids:
                continue
            request = requests_by_id.get(req_id)
            if request is None:
                logger.debug(
                    "KDA state offload: dropping boundary hand-off for unknown "
                    "request %s",
                    req_id,
                )
                continue
            block_hashes = getattr(request, "block_hashes", None) or ()
            for group_id, block_id, boundary_tokens in entries:
                if int(group_id) != self.group_id:
                    continue
                if int(block_id) <= NULL_BLOCK_ID:
                    continue
                boundary_tokens = int(boundary_tokens)
                if boundary_tokens % self.mamba_block_size:
                    continue
                if boundary_tokens % self.chunk_size:
                    continue
                h = self.boundary_hash(block_hashes, boundary_tokens)
                if h is None:
                    continue
                self._next_op_id += 1
                store = KdaStore(self._next_op_id, h, int(block_id))
                self._pending_stores[store.op_id] = _PendingStore(
                    store.block_id, store.prefix_hash
                )
                if pool is not None:
                    pool.touch([pool.blocks[store.block_id]])
                accepted.append(store)
        return accepted

    def absorb_reports(self, stored, failed) -> None:
        """Unpin a boundary block once every rank has reported on it.

        Quorum over ``stored | failed`` rather than ``stored`` alone: a rank
        that could not write its shard never sends a second report, so waiting
        for one would pin the block for the life of the process. The hash is
        indexed only when no rank failed -- a partially stored state is a state
        that cannot be restored.
        """
        touched: set[int] = set()
        for source, attr in ((stored or {}, "reports"), (failed or {}, "failures")):
            for op_id, count in source.items():
                pending = self._pending_stores.get(int(op_id))
                if pending is None:
                    continue
                pending.reports += int(count)
                if attr == "failures":
                    pending.failures += int(count)
                touched.add(int(op_id))
        pool = self._pool
        for op_id in touched:
            pending = self._pending_stores[op_id]
            if pending.reports < self._world_size:
                continue
            del self._pending_stores[op_id]
            if pending.failures == 0:
                self._index.note_stored(pending.prefix_hash)
            if pool is not None:
                pool.free_blocks([pool.blocks[pending.block_id]])

    def has_pending_work(self) -> bool:
        """Keep the engine stepping while a boundary block is still pinned.

        Store completions only reach this process as worker metadata on a step.
        An engine that went idle with a pin outstanding would hold that block
        out of the pool forever -- KV capacity lost with nothing to point at.
        """
        return bool(self._pending_stores)

    def stats(self) -> dict[str, int]:
        out = dict(self._index.stats())
        out["pinned_stores"] = len(self._pending_stores)
        return out
