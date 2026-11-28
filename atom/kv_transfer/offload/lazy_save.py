# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache complement ("lazy") saves for the in-process offload path.

The eager save policy stores every prompt chunk a prefill computes, so the CPU
tier holds a superset of the GPU prefix cache. The lazy policy stores a block
only when the GPU prefix cache is about to evict it, so the tier holds what
the GPU evicted (a block loaded back from it stays in both) and the two
together hold more prefixes. No request stores its own prompt:

* Keys. When a request finishes or is preempted, each of its published prompt
  blocks is keyed by pool block id with its GPU block hash, its LMCache chunk
  hash (the request's prehash chain, one chunk per block) and whether the tier
  holds it. A key holds while the block still carries that GPU hash.
* Demotion. The keyed, unsaved, canonical blocks among those the next
  allocations would evict (`BlockPool.next_evicted`) are pinned
  (`BlockPool.claim_unreused`) and stored as one `PrehashedSave` whose
  chunk i is `block_ids[i]`. A reported store marks them saved; reported or
  not (a failure, which also pauses demotions, or no report within the save
  abandon window) they are unpinned to the head of their queue
  (`BlockPool.free(coldest=True)`), where they were found.
* Lookup. The tier holds what HBM evicted rather than the prefix HBM still
  has, so a lookup asks only past the HBM hit, which the scheduler probes first
  (`needs_hbm_probe_before_lookup`).

The shell builds :class:`LazyOffloadScheduler` for the layouts whose scheduler
is the dense one (dense and MiniMax-M3), with a worker that stores a
`PrehashedSave`: :class:`LazyDenseOffloadConnector`, or M3's own.

Where it falls short:

* A block is saved only if a demotion reaches it before HBM evicts it. Stores
  that take longer than about two steps to report, a step that allocates more
  blocks than the eviction window holds less one demotion, prefix hits inside
  the window (up to one more demotion's worth), or a free list of two
  demotions' worth or less (pins would leave the next allocation only a
  preemption) let keyed blocks be evicted unsaved (`lazy_evicted_unsaved`
  counts those it rekeys).
* Pins hold their blocks until the store reports. They leave two demotions'
  worth free when taken, but requests that grow through that before a slow
  store reports are preempted.
* A free list shorter than the window (a nearly full pool) is walked whole on
  most decode steps: about 60 us a step at a 3,000-block free list in
  a CPU model, against about 18 us for an eager build.
* The saved bit does not follow the CPU tier's own evictions: a block whose
  chunk LMCache dropped stays marked saved, and is lost from both tiers when
  HBM evicts it.
* A demotion is as best-effort as an eager save: the worker reports success
  after storing a prefix of the batch when the tier refuses an allocation, and
  after storing nothing when its engine is unhealthy or frozen.
* Under LRU, a block a request hits and releases while its demotion is in
  flight goes back to the cold end with the demotion (SLRU keeps it
  protected).
"""

from __future__ import annotations

import logging
import time

from atom.kv_transfer.disaggregation.types import SaveOperationId, req_id_of
from atom.kv_transfer.offload._offload_common import tier_cache_salt
from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_STORE_CHANNEL
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import PrehashedSave, SaveSpec
from atom.kv_transfer.offload.prehashed import (
    PrehashedSaveSchedulerMixin,
    PrehashedSaveWorkerMixin,
)

logger = logging.getLogger("atom")

# The request id of every demotion's save operation; an ATOM request's is an int.
_DEMOTION_REQ_ID = "lmcache-lazy-demotion"
# Demotions in flight at once; each pins its blocks until it reports.
_MAX_DEMOTIONS_IN_FLIGHT = 2
# Seconds without a new demotion after one reports a failure.
_DEMOTION_BACKOFF_S = 1.0


class LazySaveSchedulerMixin:
    """Scheduler half of ``save_policy: lazy`` (see the module doc).

    Mixed in ahead of a ``PrehashedSaveSchedulerMixin`` scheduler; call
    :meth:`_init_lazy_save` at the end of ``__init__``. The worker is the
    prehashed one, unchanged.
    """

    needs_hbm_probe_before_lookup = True

    def _init_lazy_save(self) -> None:
        chunk = int(self.chunk_size)
        if not self._do_save:
            raise ValueError(
                f"save_policy 'lazy' needs a kv_role that saves, not {self.kv_role!r}"
            )
        # Under DCP a block-table entry spans the KV block size times the DCP
        # size, and the worker maps chunks onto that same grid.
        if chunk != self.virtual_block_size:
            raise ValueError(
                f"save_policy 'lazy' stores one LMCache chunk per KV block: the "
                f"LMCache chunk size ({chunk}) must equal the tokens a block "
                f"spans ({self.virtual_block_size}: the block size, times the "
                f"DCP size under DCP)"
            )
        # A demotion reports per stage, and only the engine can retire a stage
        # that never reports (`PPKVAggregator.forget`).
        if int(getattr(self._config, "pipeline_parallel_size", 1) or 1) > 1:
            raise ValueError("save_policy 'lazy' does not support pipeline parallelism")
        if not self._prehashed:
            raise ValueError(
                "save_policy 'lazy' needs prehashed saves, which are off (the "
                "'LMCache prehashed saves off' line above says why)"
            )
        if self._prehash_ship_tokens:
            raise ValueError(
                "save_policy 'lazy' cannot emit LMCache KV events: a demotion "
                "stores blocks of many prefixes as one save, whose events would "
                "name the wrong parents and tokens"
            )
        # Checked now rather than on the first prompt (`_prefix_chunk_hashes`):
        # a demotion has no token path to fall back to, so a mismatch found
        # while serving would leave the tier empty.
        tokens = list(range(2 * chunk))
        chain: list[int] = []
        self._extend_prehash_chain(chain, tokens, 2)
        if not self._verify_prehash_chain(tokens, chain, 2):
            raise ValueError(
                "save_policy 'lazy': the scheduler's chunk hashes do not match "
                "LMCache's"
            )
        # No request saves its prompt: the save tracker, the save loop and the
        # deferred free all follow `_do_save`.
        self._do_save = False
        # Pool block id -> (GPU block hash, LMCache chunk hash, saved): one
        # entry per block at most, valid while the block has that hash.
        self._block_keys: dict[int, tuple[int, int, bool]] = {}
        # sid -> (seq, first chunk, end chunk) a finished load filled from the
        # tier; their GPU hashes are published only when the request wakes.
        self._loaded_chunks: dict[str, tuple[object, int, int]] = {}
        # Demotions in flight: operation -> (pinned block ids, dispatch time).
        self._demotions: dict[SaveOperationId, tuple[list[int], float]] = {}
        # A demotion's size: the blocks one step's tokens fill, the most
        # unsaved content a step adds (a load's blocks are the tier's already).
        self._demotion_blocks = -(-int(self._config.max_num_batched_tokens) // chunk)
        # The free list as the last look left it, and how far the window may
        # slide from there unseen, or None for any change (`_look_due`); None
        # when a look is owed.
        self._looked_at: tuple[int, int, int, int | None] | None = None
        self._backoff_until = 0.0
        self.total_lazy_evicted_unsaved = 0

    def bind_block_manager(self, block_manager) -> None:
        if not block_manager.enable_prefix_caching:
            raise ValueError("save_policy 'lazy' keys blocks by prefix-cache hash")
        super().bind_block_manager(block_manager)

    def _tier_lookup_start(self, seq) -> int:
        return int(seq.offload_hbm_probe_tokens)

    # -- keys ----------------------------------------------------------------
    def _key_blocks(self, seq) -> None:
        """Key `seq`'s published prompt blocks, which it is about to free.

        A block keyed under the same hash keeps its saved bit: every request
        that finishes on a shared prefix keys it again. One keyed under another
        hash, unsaved, held content HBM evicted that the tier never got.
        """
        loaded = self._loaded_chunks.pop(str(seq.id), None)
        if tier_cache_salt(seq):
            # A demotion carries no salt, so a salted prompt's keys could not.
            self._perf_bump("lazy_skipped_salted")
            return
        chunk = int(self.chunk_size)
        # As far as eager saves reach: never into the speculative placeholders.
        n = min(int(seq.num_hashed_tokens), int(seq.num_prompt_tokens)) // chunk
        if n <= 0:
            return
        # The request frees these next, and a free is no slide: under SLRU a
        # probationary block lands ahead of every protected one, inside a
        # window that reaches the protected queue.
        self._looked_at = None
        first, end = loaded[1:] if loaded is not None and loaded[0] is seq else (0, 0)
        chain = self._prefix_chunk_hashes(seq, n)
        blocks = self._block_manager.kv.blocks
        keys = self._block_keys
        for i, block_id in enumerate(seq.block_table[:n]):
            h = blocks[block_id].hash
            old = keys.get(block_id)
            if old is None or old[0] != h:
                saved = False
                self.total_lazy_evicted_unsaved += old is not None and not old[2]
            else:
                saved = old[2]
            keys[block_id] = (h, chain[i], saved or first <= i < end)

    def request_finished(self, seq) -> None:
        self._key_blocks(seq)
        super().request_finished(seq)

    def release_stalled_save(self, seq) -> None:
        """Preemption's hook, named for the eager save it reclaims there: key the
        published blocks the preempted request is about to free."""
        self._key_blocks(seq)
        super().release_stalled_save(seq)

    def source_blocks_released(self, seq) -> None:
        # A free another connector deferred (a P/D send): as `_key_blocks`.
        self._looked_at = None
        super().source_blocks_released(seq)

    def load_finished(self, req_id) -> bool:
        active = self._active_load_operations.get(req_id_of(req_id))
        if not super().load_finished(req_id):
            return False
        if active is not None:
            seq, chunk = active[0], int(self.chunk_size)
            self._loaded_chunks[str(seq.id)] = (
                seq,
                seq.offload_load_start_tokens // chunk,
                seq.offload_loaded_tokens // chunk,
            )
        return True

    def get_statistics(self) -> dict[str, int]:
        return {
            **super().get_statistics(),
            "lazy_evicted_unsaved": self.total_lazy_evicted_unsaved,
        }

    # -- demotion ------------------------------------------------------------
    def build_connector_meta(self):
        meta = super().build_connector_meta()
        self._demote(meta)
        return meta

    def _look_due(self, pool) -> bool:
        """Whether the eviction window may hold a candidate no look could take.

        A look that spent its budget left candidates there: the next one is
        due whatever moved since. Otherwise, while the window ends inside the
        free list, a candidate enters it as it slides (blocks evicted off its
        cold end, vacant ones spent before it, blocks pinned off it; a prefix
        hit claiming a block inside it slides it too, uncounted), or as a
        request frees keyed blocks into it, which owes a look (`_key_blocks`).
        A window that reached past the end of the free list gains every block
        freed, and a look with no room for a pin waits for a free, so any
        change makes a look due after those.
        """
        if self._looked_at is None:
            return True
        free, reusable, evicted, slack = self._looked_at
        if slack is None:
            return (pool.num_free, pool.num_reusable_free, pool.blocks_evicted) != (
                free,
                reusable,
                evicted,
            )
        if slack <= 0:
            # Vacant blocks freed since move the window back, not the
            # candidates the budget left.
            return True
        spent = (free - reusable) - (pool.num_free - pool.num_reusable_free)
        return pool.blocks_evicted - evicted + spent >= slack

    def _demote(self, meta) -> None:
        """Store the keyed blocks the next allocations would evict.

        Runs on every metadata build, thousands a second on an idle engine, so
        it looks at the free list only when a look is due (`_look_due`).
        """
        pool = self._block_manager.kv
        if self._demotions:
            self._abandon_stalled_demotions()
        # Both slots taken leaves no budget. Not looking then also keeps the
        # last look's accounting, so a slot that comes back is used at once.
        if (
            len(self._demotions) >= _MAX_DEMOTIONS_IN_FLIGHT
            or not self._look_due(pool)
            or time.monotonic() < self._backoff_until
        ):
            return
        size = self._demotion_blocks
        # Two demotions' worth, or 2% of the pool when that is more: one
        # admission allocates its whole prompt at once.
        window = max(2 * size, pool.num_usable_blocks // 50)
        reach = window - (pool.num_free - pool.num_reusable_free)
        inside = pool.num_reusable_free >= reach
        # A pinned block is off the free list: leave two demotions' worth
        # there, or the next allocation preempts a request instead.
        budget = min(
            (_MAX_DEMOTIONS_IN_FLIGHT - len(self._demotions)) * size,
            pool.num_free - 2 * size,
        )
        keys, blocks, indexed = self._block_keys, pool.blocks, pool.index_get
        picked: list[tuple[int, int]] = []
        for block_id in pool.next_evicted(window) if budget > 0 else ():
            key = keys.get(block_id)
            if key is None or key[2]:
                continue
            h = blocks[block_id].hash
            # A stale key, or a duplicate of the canonical block no lookup finds.
            if key[0] != h or indexed(h) != block_id:
                continue
            picked.append((block_id, key[1]))
            if len(picked) == budget:
                break
        now, chunk = time.monotonic(), int(self.chunk_size)
        for start in range(0, len(picked), size):
            batch = picked[start : start + size]
            block_ids = [block_id for block_id, _ in batch]
            for block_id in block_ids:
                pool.claim_unreused(block_id)
            operation = SaveOperationId(_DEMOTION_REQ_ID, self._save_nonce)
            self._save_nonce += 1
            meta.add_request(
                self._request_meta_type(
                    req_id=_DEMOTION_REQ_ID,
                    token_ids=[],
                    block_ids=block_ids,
                    save_spec=SaveSpec(skip_leading_tokens=0),
                    save_operation=operation,
                    prehashed_save=PrehashedSave(
                        start=0, hashes=tuple(h for _, h in batch)
                    ),
                )
            )
            self._demotions[operation] = (block_ids, now)
            self._track_save_statistics(operation, len(block_ids) * chunk)
        # A window past the free list's end gains every block freed, and a look
        # with no room for a pin waits for a free: either wants any change, a
        # slot's unpin included. Otherwise the window may slide unseen by what
        # it holds beyond two demotions' worth, one demotion at most, or by
        # nothing (a look on any slide). The pins slid it already, so a spent
        # budget leaves no slack: the next build looks. A block that slides in
        # is first seen at least two demotions' worth, less the step's
        # allocation, before HBM evicts it, so only a step that allocates more
        # than the window less one demotion evicts blocks no look has seen. A
        # prefix hit claiming blocks inside the window slides it uncounted:
        # blocks that brings in can be evicted before the next look, at most
        # one more demotion's worth.
        slack = max(1, min(size, window - 2 * size)) - len(picked)
        self._looked_at = (
            pool.num_free,
            pool.num_reusable_free,
            pool.blocks_evicted,
            slack if inside and budget > 0 else None,
        )

    def _abandon_stalled_demotions(self) -> None:
        """Unpin a demotion that has not reported within the window an eager
        save gets (`save_abandon_timeout_s`), without marking its blocks."""
        timeout = self.save_abandon_timeout_s()
        if timeout <= 0:
            return
        now = time.monotonic()
        for operation, (block_ids, at) in list(self._demotions.items()):
            if now - at >= timeout:
                logger.warning(
                    "LMCache lazy demotion %s of %d blocks did not report in "
                    "%.0fs; unpinning it unsaved",
                    operation,
                    len(block_ids),
                    timeout,
                )
                self._end_demotion(operation, stored=False)

    def _end_demotion(self, operation, *, stored: bool) -> None:
        """Mark a demotion's blocks saved if it `stored`, and unpin them to the
        head of their queues, where it found them, either way."""
        block_ids, _ = self._demotions.pop(operation)
        if stored:
            keys = self._block_keys
            for block_id in block_ids:
                keys[block_id] = (*keys[block_id][:2], True)
            self._finish_save_statistics(operation)
        else:
            self._cancel_save_statistics(operation)
            # Back in the window unsaved.
            self._looked_at = None
        pool = self._block_manager.kv
        # Last first, so the first ends up coldest, as it was.
        for block_id in reversed(block_ids):
            pool.free(block_id, coldest=True)

    def connector_completion(self, completion):
        # Only the store report ends a demotion: a source-safe or quiescent
        # one comes before it, or beside a failure.
        if (
            completion.channel != DENSE_PAGE_STORE_CHANNEL
            or completion.operation_id not in self._demotions
        ):
            return super().connector_completion(completion)
        self._end_demotion(completion.operation_id, stored=completion.succeeded)
        if not completion.succeeded:
            self._perf_bump("lazy_demotion_failures")
            self._backoff_until = time.monotonic() + _DEMOTION_BACKOFF_S
        # Handled, and no request's save terminal.
        return None

    def has_pending_work(self) -> bool:
        # A look that is due counts too: an idle engine builds metadata only
        # while there is work, and the blocks the last step freed or evicted
        # may bring the next step's first evictions within the window.
        return (
            bool(self._demotions)
            or self._look_due(self._block_manager.kv)
            or super().has_pending_work()
        )


class LazyOffloadScheduler(
    LazySaveSchedulerMixin, PrehashedSaveSchedulerMixin, DenseOffloadScheduler
):
    """``save_policy: lazy`` for the dense and MiniMax-M3 layouts, which share
    the dense scheduler: it sees token counts, block tables and the tier's
    answers, never the KV bytes a layout's worker moves."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._init_prehashed_save()
        self._init_lazy_save()


class LazyDenseOffloadConnector(PrehashedSaveWorkerMixin, DenseOffloadConnector):
    """The dense worker under ``save_policy: lazy``: it stores a demotion's
    `PrehashedSave`, which the eager dense worker would take for a save of no
    tokens and report done. M3's worker stores one under either policy."""


__all__ = [
    "LazyDenseOffloadConnector",
    "LazyOffloadScheduler",
    "LazySaveSchedulerMixin",
]
