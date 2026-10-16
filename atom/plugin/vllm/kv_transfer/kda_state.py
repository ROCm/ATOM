# SPDX-License-Identifier: MIT
"""KDA recurrent-state offload for the vLLM plugin path.

Kimi-K3 is hybrid: MLA full-attention layers plus KDA recurrent layers. A
restored MLA prefix is only correct if the KDA state at the *same token
boundary* is restored with it. Half a restore is not a crash and not a log
line -- it is wrong output.

vLLM does not build one group per family. It builds equal-sized groups and
takes the size from the smallest family, so K3's 69 KDA layers against 29
full-attention layers come out as **three** mamba groups plus one attention
group (see :func:`find_mamba_groups`). The three are one recurrent state here:
they commit at the same token boundary and none of them is restorable alone.

The two families are moved by two different mechanisms, and that asymmetry is
forced, not chosen:

* The MLA group is ordinary paged KV. ``DenseKVByteCodec`` gathers whole blocks
  by position out of the request's block table, exactly as it does for M3 and
  GLM-5.2. Nothing here changes it.
* The KDA groups in ``--mamba-cache-mode align`` cannot be read positionally at
  all. vLLM's own store connector says why (``mooncake/store/coordinator.py``,
  ``store_mask``): an align-mode mamba block table is not append-only -- a
  superseded state block is freed and nulled, and speculative blocks relocate in
  place -- so indexing it by ``token // block_size`` can land on a null, freed,
  or live speculative block and persist those bytes under a valid prefix hash.

  Two sources name a boundary block without indexing that table, and both are
  used because neither covers the other:

  1. vLLM's explicit hand-off -- ``SchedulerOutput.partial_tail_offloads`` on
     0.28, the same payload under ``kv_connector_block_state`` on 0.29. It
     names the block holding a committed boundary state exactly. But it is, by
     its own definition, the *partial tail*: ``_cache_partial_tail_block``
     emits nothing when ``num_tokens % block_size == 0`` and nothing at all
     when ``block_size == hash_block_size``. On Kimi-K3 both mamba and hash
     block size are 1536 (vLLM raises the attention block size to satisfy
     "attention page >= mamba page", and ``hash_block_size`` is then their
     gcd), so this source is **empty** -- and every boundary it could ever
     name is a non-multiple of the chunk size, which :meth:`cap_hit` can never
     select. Measured: 3.96M external queries, exactly 0 hits, over 900s.
  2. The block pool's own content-addressed cache, ``get_cached_block(hash,
     group_ids)``. This is the map vLLM itself uses to serve a local mamba
     prefix hit, so a whole-block boundary is in it precisely when that
     boundary's state is committed and intact. Looking a boundary up by hash
     rather than by row is what makes it safe: a freed, nulled or relocated
     block is not in the map under that hash, so the failure mode the
     positional read has -- wrong bytes under a valid key -- cannot occur. It
     is also all-or-nothing across the mamba groups in one call, which is the
     atomicity this leg needs anyway.

  (1) feeds :meth:`collect_stores`, (2) feeds
  :meth:`collect_cached_boundary_stores`. The second is what actually produces
  chunk-aligned boundaries, hence what makes a joint hit possible at all.

So a KDA boundary is stored as one whole opaque image under the prefix hash at
that boundary -- one key, one block per mamba group -- through
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
import time
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


def find_mamba_groups(kv_cache_groups) -> list[tuple[int, Any]]:
    """``(group_id, spec)`` for every mamba KV cache group, in group order.

    There is normally more than one, and the count is not a property of the
    model family. vLLM splits a hybrid model's layers into equal-sized groups
    and takes the group size from the *smallest* family
    (``_get_kv_cache_groups_uniform_page_size``: ``group_size =
    min_num_layers``, ``num_groups = cdiv(len(layers), group_size)``), so K3's
    69 KDA layers against 29 full-attention layers become **three** mamba
    groups of 23 plus one attention group. It is read here, never assumed.

    All mamba groups are one logical recurrent state to this connector: the
    split is an allocator artifact, every group commits its boundary at the
    same token count, and a boundary is restorable only if all of them are
    there. So a "boundary block" below is a tuple of block ids, one per group.

    Classification is by ``MambaSpec``, never by tensor shape: a mamba page and
    an attention page are both ``[num_blocks, ...]`` uint8 to this connector, so
    shape cannot tell them apart and a wrong guess moves the wrong bytes.
    """
    try:
        from vllm.v1.kv_cache_interface import MambaSpec
    except ImportError:  # pragma: no cover - vLLM is absent in unit tests
        return []
    found: list[tuple[int, Any]] = []
    for group_id, group in enumerate(kv_cache_groups or ()):
        spec = unwrap_kv_cache_spec(group.kv_cache_spec)
        if isinstance(spec, MambaSpec):
            found.append((group_id, spec))
    block_sizes = {int(getattr(spec, "block_size", 0)) for _, spec in found}
    if len(block_sizes) > 1:
        raise ValueError(
            "ATOM offload connector: mamba KV cache groups disagree on block "
            f"size ({sorted(block_sizes)}); one boundary is one block in every "
            "group at the same token count, which those sizes cannot all be"
        )
    return found


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
    """Address one boundary's mamba blocks as the ordered byte stream to move.

    One boundary is one block **per mamba group**, not one block. vLLM spreads
    the recurrent layers over several groups, each with its own block table,
    and commits all of them at the same token count -- so the unit addressed
    here is a tuple of block ids in group order, and the byte stream is every
    group's layers concatenated in that same order.

    :class:`StateByteCodec` was written against ATOM's native slot model and
    asks its backend two questions -- ``page_unit_views`` for a store,
    ``state_entry_views`` for a load. On the plugin path both resolve to the
    same thing, the per-layer views of one boundary's blocks, because vLLM
    allocates the destination blocks for an external hit and the mamba groups'
    own block tables are what the resuming forward reads. Keeping both methods
    (rather than collapsing them) is what lets the native codec be reused
    verbatim.

    Layer order within a group is ``group.layer_names``, vLLM's own canonical
    order, so a stream gathered on one rank is read back in the same order on
    the next run.
    """

    def __init__(
        self, tensors_by_group: list[list[torch.Tensor]], *, layout_id: str
    ) -> None:
        groups = [list(tensors) for tensors in tensors_by_group]
        if not groups or not all(groups):
            raise ValueError("KDA state offload: a mamba group registered no tensors")
        self._groups = groups
        self.layout_id = layout_id
        self.num_groups = len(groups)
        self.num_blocks: list[int] = []
        for tensors in groups:
            counts = sorted({int(t.shape[0]) for t in tensors})
            if len(counts) > 1:
                raise ValueError(
                    "KDA state offload: mamba layers disagree on block count "
                    f"({counts}); a boundary block id addresses every layer in "
                    "its group, so one stream would be gathered at the wrong "
                    "offset"
                )
            self.num_blocks.append(counts[0])
        self.entry_bytes = sum(
            int(t[0].numel()) * t[0].element_size()
            for tensors in groups
            for t in tensors
        )

    def _views(self, block_ids) -> list[torch.Tensor]:
        ids = [int(b) for b in block_ids]
        if len(ids) != self.num_groups:
            raise ValueError(
                "KDA state offload: a boundary is one block per mamba group, "
                f"expected {self.num_groups} block ids, got {len(ids)}"
            )
        views: list[torch.Tensor] = []
        for tensors, block_id, num_blocks in zip(self._groups, ids, self.num_blocks):
            if not 0 <= block_id < num_blocks:
                raise IndexError(
                    f"KDA state offload: block {block_id} is outside the mamba "
                    f"group's {num_blocks} blocks"
                )
            views.extend(t[block_id] for t in tensors)
        return views

    def page_unit_views(self, unit_ids) -> list[torch.Tensor]:
        """Store source: this boundary's block in each mamba group."""
        return self._views(unit_ids)

    def state_entry_views(self, slot) -> list[torch.Tensor]:
        """Load destination: the blocks vLLM allocated for the external hit."""
        return self._views(slot)


def build_layout_id(specs: list[Any], tensors_by_group) -> str:
    """Name the geometry the bytes were written under.

    Folded into the storage key by ``StateByteCodec.key``. One prefix hash maps
    to a different image under a different mamba block size, speculative-block
    count, dtype, TP shape *or group split* -- the last one because the byte
    stream is the groups concatenated in order, so re-splitting the same layers
    reorders it. KV and state entries share one LMCache pool with no field
    saying what an entry is, so without this a config change reads back another
    layout's bytes as state, which is silent wrong output rather than a miss.
    """
    spec = specs[0]
    mamba_type = getattr(spec, "mamba_type", None)
    parts = [
        "vllm-kda",
        str(getattr(mamba_type, "name", mamba_type)),
        f"bs={int(getattr(spec, 'block_size', 0))}",
        f"page={int(getattr(spec, 'page_size_bytes', 0))}",
        f"spec_blocks={int(getattr(spec, 'num_speculative_blocks', 0))}",
        f"tp_replicated={int(bool(getattr(spec, 'tp_replicated', False)))}",
        "groups=" + ",".join(str(len(t)) for t in tensors_by_group),
        ";".join(
            f"{tuple(t.shape[1:])}:{t.dtype}"
            for tensors in tensors_by_group
            for t in tensors
        ),
    ]
    return "|".join(parts)


@dataclass(frozen=True)
class KdaStore:
    """One boundary state on its way out: op id, key, and source blocks.

    ``block_ids`` is one block per mamba group, in group order -- the whole
    recurrent state at this boundary, which is the only unit that can be
    restored.
    """

    op_id: int
    prefix_hash: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class KdaLoad:
    """One boundary state on its way back in.

    ``error_block_ids`` are the *attention* group's blocks that this request's
    dense load is filling. They ride along because a KDA miss has to invalidate
    them: vLLM otherwise takes the request's appearance in ``finished_recving``
    at face value and caches the whole external prefix, serving an MLA prefix
    whose recurrent state was never restored.

    An empty ``block_ids``, or any entry ``<= NULL_BLOCK_ID``, means the
    destination could not be resolved; the tier fails it without touching the
    device, which is the same outcome as a miss.
    """

    req_id: str
    prefix_hash: int
    block_ids: tuple[int, ...]
    error_block_ids: tuple[int, ...] = ()


@dataclass
class _PendingStore:
    block_ids: tuple[int, ...]
    prefix_hash: int
    reports: int = 0
    failures: int = 0
    # Which request offered this store. Cleared when that request is
    # preempted or finishes, so the recomputed life can offer the boundary
    # again; while it is set, a later sweep of the same request must not pin
    # the same blocks a second time.
    req_id: str | None = None


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
            ok = bool(self._codec.put(int(store.prefix_hash), store.block_ids))
        except Exception:  # deliberately blind
            # `put` reaches into LMCache, whose failure modes are its own. A
            # store that cannot happen costs one boundary -- not this thread,
            # whose death would strand every request parked on a later load.
            logger.warning(
                "KDA state offload: store of hash %d (blocks %s) failed",
                store.prefix_hash,
                store.block_ids,
                exc_info=True,
            )
        with self._lock:
            target = self._stored if ok else self._store_failed
            target[store.op_id] = target.get(store.op_id, 0) + 1

    def _do_load(self, load: KdaLoad) -> None:
        ok = False
        try:
            if load.block_ids and all(b > NULL_BLOCK_ID for b in load.block_ids):
                ok = bool(self._codec.get(int(load.prefix_hash), load.block_ids))
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
        group_ids,
        mamba_block_size: int,
        hash_block_size: int,
        chunk_size: int,
        world_size: int,
        can_store: bool = True,
        can_load: bool = True,
    ) -> None:
        self.group_ids = tuple(int(g) for g in group_ids)
        if not self.group_ids:
            raise ValueError("KDA state offload: no mamba KV cache group to plan for")
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
        # How far each live request's sweep has contiguously resolved.
        # A boundary the pool does not hold yet stays behind this cursor, so
        # a later step can store it once the state is actually registered.
        # Advancing past that hole is what made a too-early miss permanent.
        self._swept: dict[str, int] = {}
        # Why this leg has nothing to show, when it has nothing to show. Until
        # these existed the whole leg was `logger.debug`, and a run that
        # produced exactly zero joint hits produced exactly zero lines saying
        # so -- 900 seconds of a by-construction null that read as a quiet
        # success. Every rejection below increments one of these, and
        # :meth:`log_stats` prints them at INFO.
        self._counters: dict[str, int] = {
            "handoff_entries": 0,
            "handoff_stores": 0,
            "sweep_offered": 0,
            "sweep_stores": 0,
            "sweep_no_hash": 0,
            "sweep_uncached": 0,
            "sweep_known": 0,
            "cap_kept": 0,
            "cap_declined": 0,
        }
        self._last_stats_log = 0.0
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
                self._counters["cap_declined"] += 1
                return 0
            h = self.boundary_hash(block_hashes, boundary)
            if h is not None and self._index.could_serve(h):
                self._counters["cap_kept"] += 1
                return min(hit, boundary)
            boundary -= self.chunk_size
        self._counters["cap_declined"] += 1
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
        so row ``(num_total_computed - 1) // mamba_block_size`` of *each* mamba
        group is the block the resuming forward will read its initial state
        from. That row is only well-defined when the hit ends on a mamba block
        boundary; otherwise the load is failed and the prefix recomputed.
        All of them, because the state is only whole across the groups.

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
        aligned = num_total_computed % self.mamba_block_size == 0
        block_ids = (
            self._boundary_blocks(group_blocks, num_total_computed) if aligned else ()
        )
        if h is None or not block_ids:
            # An unaligned end is not a row we can fill. preprocess_mamba reads
            # ``(num_computed - 1) // block_size``, which is a different row
            # from ``num_computed // block_size - 1`` unless the hit ends on a
            # block boundary. Writing the wrong row leaves the forward on the
            # previous occupant's state and ``get`` still returns success.
            logger.warning(
                "KDA state offload: %s hit %d tokens but its boundary state has "
                "no %s; failing the load so the prefix is recomputed",
                req_id,
                num_total_computed,
                (
                    "key"
                    if h is None
                    else (
                        "block-aligned boundary" if not aligned else "destination block"
                    )
                ),
            )
            self._loads.append(KdaLoad(req_id, int(h or 0), (), error_blocks))
            return
        self._index.request_load(req_id, h)
        self._loads.append(KdaLoad(req_id, h, block_ids, error_blocks))

    def _boundary_blocks(
        self, group_blocks: tuple[list[int], ...], num_total_computed: int
    ) -> tuple[int, ...]:
        """This boundary's destination block in each mamba group, or ``()``.

        All or nothing: a state scattered into some of its groups leaves the
        rest holding the previous occupant's recurrence, which is exactly the
        half restore this module exists to prevent. One unresolved group
        therefore fails the whole load.
        """
        # Same row ``preprocess_mamba`` reads as the initial state:
        # ``(num_computed_tokens - 1) // block_size``. Equal to
        # ``num_computed // block_size - 1`` only when num_computed is a
        # whole number of mamba blocks; the caller rejects the other case.
        if num_total_computed % self.mamba_block_size != 0:
            return ()
        row = (num_total_computed - 1) // self.mamba_block_size
        if row < 0:
            return ()
        block_ids: list[int] = []
        for group_id in self.group_ids:
            if group_id >= len(group_blocks):
                return ()
            blocks = group_blocks[group_id]
            if row >= len(blocks):
                return ()
            block_id = int(blocks[row])
            if block_id <= NULL_BLOCK_ID:
                return ()
            block_ids.append(block_id)
        return tuple(block_ids)

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
        if attention_group_id >= len(group_blocks):
            return ()
        blocks = group_blocks[attention_group_id]
        if attention_block_size <= 0:
            return tuple(int(b) for b in blocks if int(b) > NULL_BLOCK_ID)
        local = max(0, num_total_computed - num_external_tokens)
        start = local // attention_block_size
        end = -(-num_total_computed // attention_block_size)
        precise = tuple(int(b) for b in blocks[start:end] if int(b) > NULL_BLOCK_ID)
        if precise:
            return precise
        # The slice missed every allocated attention block (the table is
        # shorter than this block size, or every id in range is the null
        # placeholder). An empty set here is silent wrong output: the failed
        # KDA load still releases the request, and vLLM caches the MLA prefix
        # when ``invalid_block_ids`` is empty. Naming every real attention
        # block recomputes more than the external hit, which is the safe
        # direction.
        return tuple(int(b) for b in blocks if int(b) > NULL_BLOCK_ID)

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
          already someone else's;
        * a boundary that did not report a block in *every* mamba group -- the
          state is only whole across the groups, and storing the groups that
          did report would put an unrestorable image under a hash that
          :meth:`cap_hit` would then accept.
        """
        accepted: list[KdaStore] = []
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
            by_boundary: dict[int, dict[int, int]] = {}
            for group_id, block_id, boundary_tokens in entries:
                self._counters["handoff_entries"] += 1
                group_id = int(group_id)
                if group_id not in self.group_ids:
                    continue
                if int(block_id) <= NULL_BLOCK_ID:
                    continue
                boundary_tokens = int(boundary_tokens)
                if boundary_tokens % self.mamba_block_size:
                    continue
                if boundary_tokens % self.chunk_size:
                    continue
                by_boundary.setdefault(boundary_tokens, {})[group_id] = int(block_id)
            for boundary_tokens in sorted(by_boundary):
                blocks = by_boundary[boundary_tokens]
                if len(blocks) != len(self.group_ids):
                    logger.debug(
                        "KDA state offload: boundary %d of %s reported %d of %d "
                        "mamba groups; dropping the partial state",
                        boundary_tokens,
                        req_id,
                        len(blocks),
                        len(self.group_ids),
                    )
                    continue
                h = self.boundary_hash(block_hashes, boundary_tokens)
                if h is None:
                    continue
                block_ids = tuple(blocks[group_id] for group_id in self.group_ids)
                accepted.append(self._issue_store(h, block_ids, req_id=req_id))
                self._counters["handoff_stores"] += 1
        return accepted

    def _issue_store(
        self, prefix_hash: int, block_ids: tuple[int, ...], req_id: str | None = None
    ) -> KdaStore:
        """Pin a boundary's blocks and queue the job that copies them out.

        The pin is taken here, at submission, and released in
        :meth:`absorb_reports` once every rank has reported. Nothing else keeps
        the block alive: vLLM is free to evict it the moment the request that
        produced it stops referencing it, and the D2H runs asynchronously on
        the worker, so an unpinned source is a race whose loser writes another
        prefix's state under this hash.
        """
        self._next_op_id += 1
        store = KdaStore(self._next_op_id, prefix_hash, block_ids)
        self._pending_stores[store.op_id] = _PendingStore(
            block_ids, prefix_hash, req_id=None if req_id is None else str(req_id)
        )
        pool = self._pool
        if pool is not None:
            pool.touch([pool.blocks[block_id] for block_id in block_ids])
        return store

    def collect_cached_boundary_stores(
        self, frontiers, requests_by_id, skip_req_ids=()
    ) -> list[KdaStore]:
        """Offer every chunk-aligned boundary vLLM has committed and cached.

        This is the source that actually produces joint hits; the hand-off
        (:meth:`collect_stores`) produces none on a model whose mamba block
        size equals its hash block size, because vLLM emits a *partial tail*
        only when the boundary is **not** a whole block -- and a boundary that
        is not a whole block is never chunk-aligned either, so :meth:`cap_hit`
        could not select it even if it arrived. The two acceptance domains are
        disjoint by construction. See the module docstring.

        The block ids come from ``BlockPool.get_cached_block``, keyed by the
        same ``BlockHash`` vLLM uses to serve a local mamba hit, never by
        indexing a block table. That is the whole safety argument: an
        align-mode mamba block that was superseded, freed, nulled or relocated
        is not registered under that hash any more, so a stale row cannot be
        mistaken for a live boundary. The call spans every mamba group at once
        and returns ``None`` unless all of them are present, which is exactly
        the all-or-nothing this state needs -- a boundary stored for some of
        its groups is an image that cannot be restored, sitting under a hash
        :meth:`cap_hit` would accept.

        *frontiers* is ``{req_id: num_computed_tokens}`` for the requests this
        step scheduled. A boundary is offered only once per request (the
        ``_swept`` cursor) and only after the frontier has passed it, which is
        also when vLLM has had a chance to register it.
        """
        accepted: list[KdaStore] = []
        if self._pool is None or not self._index.can_store:
            return accepted
        for req_id, frontier in (frontiers or {}).items():
            req_id = str(req_id)
            if req_id in skip_req_ids:
                continue
            request = requests_by_id.get(req_id)
            if request is None:
                continue
            block_hashes = getattr(request, "block_hashes", None) or ()
            frontier = int(frontier)
            cursor = self._swept.get(req_id, 0)
            boundary = ((cursor // self.chunk_size) + 1) * self.chunk_size
            # Advance only through a contiguous resolved prefix. A hole (no
            # hash yet, or the pool does not hold the state yet) stays at the
            # cursor so the next step retries it. Later boundaries in this
            # step are still offered; an in-flight offer is not repeated.
            advanced = cursor
            blocked = False
            while boundary <= frontier:
                self._counters["sweep_offered"] += 1
                store, retry = self._store_for_cached_boundary(
                    block_hashes, boundary, req_id
                )
                if store is not None:
                    accepted.append(store)
                    self._counters["sweep_stores"] += 1
                if retry:
                    blocked = True
                elif not blocked:
                    advanced = boundary
                boundary += self.chunk_size
            self._swept[req_id] = max(cursor, advanced)
        return accepted

    def _store_for_cached_boundary(
        self, block_hashes, boundary_tokens: int, req_id: str
    ) -> tuple[KdaStore | None, bool]:
        """The store job for one chunk-aligned boundary, and whether to retry.

        ``(None, True)`` is a hole: vLLM has not hashed that far, or the pool
        does not hold the state yet. The sweep must not move its cursor past
        it. ``(None, False)`` is resolved without a new store (already indexed,
        or already pinned for this request).
        """
        h = self.boundary_hash(block_hashes, boundary_tokens)
        if h is None:
            self._counters["sweep_no_hash"] += 1
            return None, True
        if h in self._index.hashes or self._offer_in_flight(h, req_id):
            # Already stored once, or pinned and waiting for rank quorum.
            # Re-storing would move the same bytes under the same key. A
            # failed load forgets the hash, and ``forget_request`` drops the
            # in-flight claim, so a later sweep can offer it again.
            self._counters["sweep_known"] += 1
            return None, False
        blocks = self._pool.get_cached_block(
            block_hashes[boundary_tokens // self.hash_block_size - 1],
            list(self.group_ids),
        )
        if not blocks:
            self._counters["sweep_uncached"] += 1
            return None, True
        block_ids = tuple(int(block.block_id) for block in blocks)
        if any(block_id <= NULL_BLOCK_ID for block_id in block_ids):
            self._counters["sweep_uncached"] += 1
            return None, True
        return self._issue_store(h, block_ids, req_id=req_id), False

    def _offer_in_flight(self, prefix_hash: int, req_id: str) -> bool:
        for pending in self._pending_stores.values():
            if pending.prefix_hash == prefix_hash and pending.req_id == req_id:
                return True
        return False

    def forget_request(self, req_id: str) -> None:
        """Drop a finished or preempted request's sweep cursor.

        A preempted request comes back with its blocks reallocated and its
        frontier rewound; keeping the cursor would skip every boundary it
        recomputes. Dropping it is also what keeps the dict bounded.
        """
        req_id = str(req_id)
        self._swept.pop(req_id, None)
        # The in-flight pin stays until quorum -- those blocks may still be
        # mid-copy. Dropping the request id is what lets the recomputed life
        # offer the boundary again instead of treating the old pin as "known".
        for pending in self._pending_stores.values():
            if pending.req_id == req_id:
                pending.req_id = None

    def absorb_reports(self, stored, failed) -> None:
        """Unpin a boundary's blocks once every rank has reported on them.

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
                pool.free_blocks(
                    [pool.blocks[block_id] for block_id in pending.block_ids]
                )

    def has_pending_work(self) -> bool:
        """Keep the engine stepping while a boundary's blocks are still pinned.

        Store completions only reach this process as worker metadata on a step.
        An engine that went idle with a pin outstanding would hold that block
        out of the pool forever -- KV capacity lost with nothing to point at.
        """
        return bool(self._pending_stores)

    def stats(self) -> dict[str, int]:
        out = dict(self._index.stats())
        out["pinned_stores"] = len(self._pending_stores)
        out.update(self._counters)
        return out

    def log_stats(self, *, interval_s: float = 60.0, force: bool = False) -> None:
        """Print this leg's counters at INFO, at most once per *interval_s*.

        It is printed unconditionally rather than only on trouble, because the
        failure this leg actually had was indistinguishable from health at
        every level a reader can see: the hit counter was a flat zero, the tier
        logged stores, and the one line that would have explained it was
        ``logger.debug``. A leg that reports nothing cannot be shown to be
        working either, so the numbers ship at INFO -- one line a minute.
        """
        now = time.monotonic()
        if not force and now - self._last_stats_log < interval_s:
            return
        self._last_stats_log = now
        stats = self.stats()
        logger.info(
            "ATOM LMCache offload: recurrent leg %s",
            " ".join(f"{k}={v}" for k, v in sorted(stats.items())),
        )


def summarize_layout_id(layout_id: str) -> str:
    """``build_layout_id``'s string with its per-layer tail run-length encoded.

    That tail is one ``shape:dtype`` per layer, and on Kimi-K3 those 69 entries
    are identical -- 4.5 KiB on one line, 36 KiB per boot across TP8, which
    leaves the `grep` the recipe tells a reader to run scrolling past its own
    answer. Encoding runs is lossless and order preserving, so the line still
    shows every difference there is between two boots.

    For the log only. The storage key folds in the full string from
    ``build_layout_id``; shortening what a human reads must not shorten what
    distinguishes one layout's bytes from another's.
    """
    head, sep, tail = layout_id.rpartition("|")
    if not sep:
        return layout_id
    runs: list[list] = []
    for item in tail.split(";"):
        if runs and runs[-1][0] == item:
            runs[-1][1] += 1
        else:
            runs.append([item, 1])
    return head + "|" + ";".join(f"{n}x{item}" if n > 1 else item for item, n in runs)
