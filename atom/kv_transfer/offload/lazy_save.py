# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache complement ("lazy") saves for the in-process offload path.

The eager save policy stores every prompt chunk a prefill computes, so the CPU
tier holds a superset of the GPU prefix cache. The lazy policy stores a block
only when the GPU prefix cache is about to evict it, so the tier holds what
the GPU evicted (a block loaded back from it stays in both, unless the tiers
are exclusive) and the two together hold more prefixes. No request stores its
own prompt:

* Keys. When a request finishes or is preempted, each of its published prompt
  blocks is keyed by pool block id with its GPU block hash, its LMCache chunk
  hash (the request's prehash chain, one chunk per block) and whether the tier
  holds it. A key holds while the block still carries that GPU hash.
* Demotion. The keyed, unsaved, canonical blocks among those the next
  allocations would evict (`BlockPool.next_evicted`) are pinned
  (`BlockPool.claim_unreused`) and stored as one `PrehashedSave` whose
  chunk i is `block_ids[i]` (`must_store_all`: a store the tier holds only
  part of reports a failure). A reported store marks them saved; reported or
  not (a failure, which also pauses demotions, or no report within the save
  abandon window) they are unpinned to the head of their queue
  (`BlockPool.free(coldest=True)`), where they were found.
* Write-back. Keyed unsaved blocks are the pool's dirty blocks
  (`BlockPool.set_write_back`): an allocation evicts every clean block with
  content (vacant, keyless, or one the tier holds) before it evicts one, and
  the blocks it passed over are the first a demotion takes
  (`BlockPool.held_blocks`). A block is lost unsaved only when the pool holds
  nothing clean to evict instead (`lazy_evicted_unsaved`).
* Exclusive tiers (`ATOM_OFFLOAD_LAZY_EXCLUSIVE`, off by default). A block a
  load filled is keyed unsaved instead of saved, and its chunk hash rides the
  next metadata (`tier_cold_chunks`); the worker puts the tier's copy first in
  its eviction order before that step's stores. HBM demotes the block again
  before it evicts it: a store of a chunk the tier kept is skipped and makes
  it recent, one it dropped is stored again. A returning conversation's
  prefix then takes room in one tier, not both: otherwise the tier spends
  most of its protected half on copies of what HBM holds (M3.1 C192 trace).
  Demotions double (about 1.8x the blocks on that trace), and so do the
  demotion slots: every block a load brought back is dirty.
* Lookup. The tier holds what HBM evicted rather than the prefix HBM still
  has, so a lookup asks only past the HBM hit, which the scheduler probes first
  (`needs_hbm_probe_before_lookup`).
* Pipeline parallelism. This scheduler and its pool live once, on the head;
  every stage holds its own layers under the same block ids. The head sends
  each metadata it builds, demotions included, to every stage, whose workers
  store and look up under their own LMCache worker id (`pp_rank * tp +
  tp_rank`, `pp_aware_rank_and_world`), and its `PPKVAggregator` reports a
  store or a load only once every stage has, failed if any stage failed: the
  pins hold until the last stage stores, and a stage's failure ends the
  demotion unsaved. A lookup takes the shortest run any worker it asks holds,
  so with every worker asked (`lookup_server_worker_ids` unset or all of
  them) a chunk counts only if every stage holds it. Each demotion is a
  request of its own, so the head retires an abandoned one, and only it,
  from the aggregator (`bind_save_abandoned`).

The shell builds :class:`LazyOffloadScheduler` for the layouts whose scheduler
is the dense one (dense and MiniMax-M3), with a worker that stores a
`PrehashedSave`: :class:`LazyDenseOffloadConnector`, or M3's own.

Where it falls short:

* Dirty blocks outlive clean ones, hotter ones included: HBM keeps what the
  tier lacks rather than strictly the most recent. Dirty blocks are lost when
  the free list holds nothing else, which takes demotions slower than the
  prefills that make new blocks (four demotions of a step's blocks in flight)
  or a tier that refuses them for long.
* Pins hold their blocks until the store reports. They leave two demotions'
  worth free when taken, but requests that grow through that before a slow
  store reports are preempted.
* A free list shorter than the window (a nearly full pool) is walked whole on
  most decode steps: about 60 us a step at a 3,000-block free list in
  a CPU model, against about 18 us for an eager build.
* The saved bit does not follow the CPU tier's own evictions: a block whose
  chunk LMCache dropped stays marked saved, and is lost from both tiers when
  HBM evicts it.
* A failed demotion is retried whole, after a backoff; the worker skips the
  chunks the tier already holds.
* Under LRU, a block a request hits and releases while its demotion is in
  flight goes back to the cold end with the demotion (SLRU keeps it
  protected).
"""

from __future__ import annotations

import logging
import time
from itertools import chain

from atom.kv_transfer.disaggregation.types import SaveOperationId, req_id_of
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_STORE_CHANNEL
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, PrehashedSave, SaveSpec
from atom.kv_transfer.offload.prehashed import (
    PrehashedSaveSchedulerMixin,
    PrehashedSaveWorkerMixin,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# The prefix of a demotion's request id, which its generation completes: under
# PP the head retires an abandoned save by request id (`PPKVAggregator.forget`),
# which must not take the other demotions with it. An ATOM request's id is an int.
_DEMOTION_REQ_ID = "lmcache-lazy-demotion"
# Demotions in flight at once; each pins its blocks until it reports. Four, so
# demotions outpace a step's new blocks when a store takes three or four steps
# to report (two kept up only at two).
_MAX_DEMOTIONS_IN_FLIGHT = 4
# Seconds without a new demotion after one reports a failure.
_DEMOTION_BACKOFF_S = 1.0


class LazySaveSchedulerMixin:
    """Scheduler half of ``save_policy: lazy`` (see the module doc).

    Mixed in ahead of a ``PrehashedSaveSchedulerMixin`` scheduler; call
    :meth:`_init_lazy_save` at the end of ``__init__``. The worker is the
    prehashed one, unchanged.
    """

    needs_hbm_probe_before_lookup = True
    # How log lines name the tier, and the request-id prefix of its demotions.
    _lazy_label = "LMCache lazy"
    _lazy_demotion_req_id = _DEMOTION_REQ_ID

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
        self._check_lazy_backend(chunk)
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
        # unsaved content a step adds (a load's blocks are the tier's already,
        # unless the tiers are exclusive).
        self._demotion_blocks = -(-int(self._config.max_num_batched_tokens) // chunk)
        # The free list as the last look left it, and how far the window may
        # slide from there unseen, or None for any change (`_look_due`); None
        # when a look is owed.
        self._looked_at: tuple[int, int, int, int | None] | None = None
        self._backoff_until = 0.0
        # Demotions reported failed (the tier refused one of their chunks, a
        # fence or an unhealthy engine): their blocks stay dirty.
        self.total_lazy_demotions_failed = 0
        # The held count the last look picked nothing from, so an idle engine
        # does not look again until it moves; -1 when a look is owed.
        self._held_looked = -1
        self._last_unsaved_warning = (0.0, 0)
        # The PP head's observer of demotions that will never report
        # (`bind_save_abandoned`); None without PP.
        self._report_abandoned = None
        # ATOM_OFFLOAD_LAZY_EXCLUSIVE: a chunk HBM loaded from the tier goes
        # first in the tier's eviction order and its block is keyed unsaved, so
        # a chunk both tiers hold costs the tier no room. The chunk hashes to
        # send with the next build.
        self._exclusive = bool(envs.ATOM_OFFLOAD_LAZY_EXCLUSIVE)
        self._tier_cold: list[int] = []
        # Exclusive tiers demote every block a load brought back too (about
        # 1.8x the blocks on the M3.1 C192 trace): twice the slots. With four,
        # a 1/8-scale replay of that trace lost 3.4x the blocks it lost dirty
        # with eight.
        # Under PP a demotion reports only once every stage stored it: each
        # stage takes its metadata in dispatch order, one stage-step behind the
        # last, and its report rides that stage's next forward to the head. The
        # round trip grows with the stage count, so the slots do too, or the
        # demotion rate falls below the prefill's (M3.1 PP4 C192: 94% of HBM
        # dirty within three minutes, then ~45K unsaved blocks evicted a
        # minute). The free-list budget in `_demote` still bounds the pins.
        pp = max(1, int(getattr(self._config, "pipeline_parallel_size", 1) or 1))
        self._max_demotions = (
            _MAX_DEMOTIONS_IN_FLIGHT * (2 if self._exclusive else 1) * pp
        )

    def _check_lazy_backend(self, chunk: int) -> None:
        """Refuse a tier this policy cannot run on. The LMCache tier: it needs
        prehashed saves, no LMCache KV events, and the scheduler's chunk hashes
        must be LMCache's own. A tier with its own keys overrides this."""
        if not self._prehashed:
            raise ValueError(
                "save_policy 'lazy' needs prehashed saves "
                "(ATOM_OFFLOAD_PREHASHED_SAVE=1), which are off (with the flag "
                "set, the 'ATOM_OFFLOAD_PREHASHED_SAVE ignored' line above says "
                "why)"
            )
        kvc = getattr(self._config, "kv_transfer_config", None)
        if getattr(offcfg.build_lmcache_config(kvc), "enable_kv_events", False):
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

    def _demotion_request(
        self, operation: SaveOperationId, block_ids: list[int], hashes: list
    ) -> LMCacheReqMeta:
        """The save that stores a demotion: chunk i is `block_ids[i]`, keyed
        by `hashes[i]`. The LMCache tier takes it as a `PrehashedSave`; a tier
        with its own keys overrides this."""
        return LMCacheReqMeta(
            req_id=operation.req_id,
            token_ids=[],
            block_ids=block_ids,
            save_spec=SaveSpec(skip_leading_tokens=0),
            save_operation=operation,
            prehashed_save=PrehashedSave(
                start=0,
                hashes=tuple(hashes),
                must_store_all=True,
            ),
        )

    def bind_save_abandoned(self, report) -> None:
        """Take the PP head's observer of abandoned saves, called with the
        request id of each demotion that did not report: the head drops its
        stages' partial tally (`PPEngineCoreProc._forget_pp_save_quorum`)."""
        self._report_abandoned = report

    def bind_block_manager(self, block_manager) -> None:
        if not block_manager.enable_prefix_caching:
            raise ValueError("save_policy 'lazy' keys blocks by prefix-cache hash")
        if not hasattr(block_manager.kv, "set_write_back"):
            # Demotion takes the blocks ATOM's pool is about to evict
            # (`next_evicted`, `claim_unreused`). A pool a connector provides
            # chooses its own evictions, out of ATOM's sight.
            raise ValueError(
                "save_policy 'lazy' saves the blocks ATOM's block pool is about "
                f"to evict, and {type(block_manager.kv).__name__} chooses its "
                "own evictions (umbp_pool): use save_policy 'eager' beside it"
            )
        super().bind_block_manager(block_manager)
        block_manager.kv.set_write_back(self._is_dirty)

    def _is_dirty(self, block_id: int) -> bool:
        """Whether HBM evicting `block_id` loses content the tier lacks: a
        valid key not yet saved, on the block a lookup would find."""
        key = self._block_keys.get(block_id)
        if key is None or key[2]:
            return False
        pool = self._block_manager.kv
        h = pool.blocks[block_id].hash
        return key[0] == h and pool.index_get(h) == block_id

    def _tier_lookup_start(self, seq) -> int:
        return int(seq.offload_hbm_probe_tokens)

    def _tier_probe_start(self, num_hbm_tokens: int) -> int:
        return int(num_hbm_tokens)

    # -- keys ----------------------------------------------------------------
    def _key_blocks(self, seq) -> None:
        """Key `seq`'s published prompt blocks, which it is about to free.

        A block keyed under the same hash keeps its saved bit: every request
        that finishes on a shared prefix keys it again. One keyed under another
        hash, unsaved, held content HBM evicted that the tier never got.
        """
        loaded = self._loaded_chunks.pop(str(seq.id), None)
        if getattr(seq, "cache_seed", -1) != -1:
            # A multimodal prompt's blocks hold KV its media seeded, which
            # token-hashed tier keys cannot tell apart.
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
        pool = self._block_manager.kv
        blocks, indexed = pool.blocks, pool.index_get
        keys = self._block_keys
        exclusive = self._exclusive
        for i, block_id in enumerate(seq.block_table[:n]):
            h = blocks[block_id].hash
            old = keys.get(block_id)
            if old is None or old[0] != h:
                saved = False
            else:
                saved = old[2]
            if first <= i < end:
                if not exclusive:
                    saved = True
                elif h != -1 and indexed(h) == block_id:
                    # HBM holds the chunk now: the tier's copy is the first it
                    # evicts, and this block is demoted again before HBM
                    # evicts it (a store the worker skips if the copy stayed).
                    saved = False
                    self._tier_cold.append(chain[i])
            keys[block_id] = (h, chain[i], saved)

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
        seq = active[0] if active is not None else None
        start = getattr(seq, "offload_load_start_tokens", None)
        if start is not None:
            chunk = int(self.chunk_size)
            self._loaded_chunks[str(seq.id)] = (
                seq,
                start // chunk,
                seq.offload_loaded_tokens // chunk,
            )
        return True

    def get_statistics(self) -> dict[str, int]:
        pool = self._block_manager.kv if self._block_manager is not None else None
        return {
            **super().get_statistics(),
            "lazy_evicted_unsaved": pool.blocks_evicted_dirty if pool else 0,
            "lazy_held_blocks": pool.num_held if pool else 0,
            "lazy_demotions_failed": self.total_lazy_demotions_failed,
        }

    # -- demotion ------------------------------------------------------------
    def build_connector_meta(self):
        meta = super().build_connector_meta()
        if self._tier_cold:
            # The worker reorders these before this build's demotions, so a
            # demotion of the same chunk leaves it recent.
            meta.tier_cold_chunks = self._tier_cold
            self._perf_bump("lazy_tier_cooled", len(self._tier_cold))
            self._tier_cold = []
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
        # Blocks an allocation passed over as dirty are the next to demote.
        if pool.num_held and pool.num_held != self._held_looked:
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
        # Every slot taken leaves no budget. Not looking then also keeps the
        # last look's accounting, so a slot that comes back is used at once.
        if (
            len(self._demotions) >= self._max_demotions
            or not self._look_due(pool)
            or time.monotonic() < self._backoff_until
        ):
            return
        size = self._demotion_blocks
        # Two demotions' worth, or 2% of the pool when that is more: one
        # admission allocates its whole prompt at once.
        window = max(2 * size, pool.num_blocks // 50)
        reach = window - (pool.num_free - pool.num_reusable_free)
        inside = pool.num_reusable_free >= reach
        # A pinned block is off the free list: leave two demotions' worth
        # there, or the next allocation preempts a request instead.
        budget = min(
            (self._max_demotions - len(self._demotions)) * size,
            pool.num_free - 2 * size,
        )
        keys, blocks, indexed = self._block_keys, pool.blocks, pool.index_get
        picked: list[tuple[int, int]] = []
        candidates = chain(pool.held_blocks(), pool.next_evicted(window))
        for block_id in candidates if budget > 0 else ():
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
            nonce = self._save_nonce
            operation = SaveOperationId(f"{self._lazy_demotion_req_id}-{nonce}", nonce)
            self._save_nonce += 1
            meta.add_request(
                self._demotion_request(operation, block_ids, [h for _, h in batch])
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
        self._held_looked = -1 if picked else pool.num_held
        self._warn_evicted_unsaved(pool)
        self._looked_at = (
            pool.num_free,
            pool.num_reusable_free,
            pool.blocks_evicted,
            slack if inside and budget > 0 else None,
        )

    def _warn_evicted_unsaved(self, pool) -> None:
        """At most one line a minute while dirty blocks are being lost."""
        at, count = self._last_unsaved_warning
        now = time.monotonic()
        if pool.blocks_evicted_dirty == count or now - at < 60.0:
            return
        logger.warning(
            "%s: %d unsaved blocks evicted from HBM since the last "
            "report (no clean block left to evict); %d held, %d demotions "
            "failed so far",
            self._lazy_label,
            pool.blocks_evicted_dirty - count,
            pool.num_held,
            self.total_lazy_demotions_failed,
        )
        self._last_unsaved_warning = (now, pool.blocks_evicted_dirty)

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
                    "%s demotion %s of %d blocks did not report in "
                    "%.0fs; unpinning it unsaved",
                    self._lazy_label,
                    operation,
                    len(block_ids),
                    timeout,
                )
                self._end_demotion(operation, stored=False)
                if self._report_abandoned is not None:
                    self._report_abandoned(operation.req_id)

    def _end_demotion(self, operation, *, stored: bool) -> None:
        """Mark a demotion's blocks saved if it `stored`, and unpin them to the
        head of their queues, where it found them, either way."""
        block_ids, _ = self._demotions.pop(operation)
        if stored:
            keys = self._block_keys
            for block_id in block_ids:
                keys[block_id] = (*keys[block_id][:2], True)
            if self._tier_index is not None:
                self._tier_index.touch(keys[block_id][1] for block_id in block_ids)
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
            self.total_lazy_demotions_failed += 1
            failed = self.total_lazy_demotions_failed
            if failed & (failed - 1) == 0:
                # The 1st, 2nd, 4th, ...: a line, not a line per demotion.
                logger.warning(
                    "%s demotion failed (%d so far): the CPU tier "
                    "refused a chunk or the store failed; its blocks stay "
                    "unsaved and demotions pause %.0fs",
                    self._lazy_label,
                    failed,
                    _DEMOTION_BACKOFF_S,
                )
            self._backoff_until = time.monotonic() + _DEMOTION_BACKOFF_S
        # Handled, and no request's save terminal.
        return None

    def has_pending_work(self) -> bool:
        # A look that is due counts too: an idle engine builds metadata only
        # while there is work, and the blocks the last step freed or evicted
        # may bring the next step's first evictions within the window.
        return (
            bool(self._demotions)
            or bool(self._tier_cold)
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
