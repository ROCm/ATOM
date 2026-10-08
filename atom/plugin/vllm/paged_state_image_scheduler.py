"""Where checkpoint images go, on vLLM's scheduler (EngineCore side).

A slot only holds the state of the position its last forward ended at, so an image
can only be taken at a forward's end. Images go where native ATOM puts them
(``ImagePlacement``): every ``state_checkpoint_interval_tokens`` rung, the
prompt-end anchor and a demand rung, with prefill cut so a forward ends there and
nowhere else, and in generation native's spacing rule; nothing else. A request
holds at most two images' blocks. The worker stores exactly where the scheduler
says (``scheduler_output.atom_image_stores``).

Model-agnostic: the family's adapter (``paged_state_image``) supplies the block
size, the placement defaults and the layer names that identify the groups.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

from atom.plugin.vllm.paged_state_image import adapter_for_kv_cache_config

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class ImagePlacement:
    """Where an image is kept, and where prefill is cut so a forward ends there.

    Line for line native ``BlockManager.checkpoint_limit``, ``_record_checkpoint_end``,
    ``_record_checkpoint_demand``, ``checkpoint_cut`` and
    ``checkpointers_at`` for a PAGE checkpoint (``successor_room = 0``, not
    readable mid-step), at the family's block size; generation uses the
    unaimed spacing rule (``decode_keeps``).

    ``interval``: >0 a rung every N tokens, 0 no images at all, -1 no rungs but the
    anchor and the demand rung still place images (native semantics).
    """

    interval: int
    demand: bool
    block: int

    @classmethod
    def from_env(
        cls, defaults: tuple[int, bool], block: int, **kwargs
    ) -> ImagePlacement:
        interval, demand = defaults
        raw = os.environ.get("ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS")
        if raw:
            interval = int(raw)
        raw = os.environ.get("ATOM_STATE_CHECKPOINT_DEMAND")
        if raw:
            demand = raw == "1"
        interval = max(-1, interval)
        if interval > 0 and interval % block:
            snapped = interval // block * block
            logger.warning(
                "state_checkpoint_interval_tokens=%d is not a multiple of the "
                "%d-token block; snapping to %s.",
                interval,
                block,
                snapped or "off (0)",
            )
            interval = snapped
        return cls(interval=interval, demand=demand, block=block, **kwargs)

    def limit(self, num_prompt_tokens: int) -> int:
        if self.interval <= 0:
            return 0
        return num_prompt_tokens // self.interval * self.interval

    def anchor(self, num_prompt_tokens: int, num_tokens: int) -> int:
        if self.interval == 0:
            return 0
        end = num_prompt_tokens // self.block * self.block
        # Never past the last block a lookup can match (the last one is always
        # recomputed for logits).
        end = min(end, (-(-num_tokens // self.block) - 1) * self.block)
        return max(end, 0)

    def demand_pos(self, hit: int, compressed_hit: int, has_room: bool) -> int:
        if self.interval == 0 or not self.demand or compressed_hit <= hit:
            return 0
        return compressed_hit if has_room else 0

    def cut(
        self, start: int, end: int, num_prompt_tokens: int, demand: int, anchor: int
    ) -> int:
        """Earliest position in ``(start, end]`` a forward must end on, or 0."""
        rung = 0
        if limit := self.limit(num_prompt_tokens):
            rung = min(end, limit)
            rung -= rung % self.interval
        candidates = [p for p in (rung, demand, anchor) if p and start < p <= end]
        return min(candidates) if candidates else 0

    def keeps(self, pos: int, num_prompt_tokens: int, demand: int, anchor: int) -> bool:
        """Whether a prefill forward ending at ``pos`` leaves an image there."""
        if self.interval == 0 or pos <= 0 or pos > num_prompt_tokens:
            return False
        on_grid = self.interval > 0 and pos % self.interval == 0
        return on_grid or pos == demand or pos == anchor

    def decode_keeps(self, pos: int, last_checkpoint_pos: int) -> bool:
        """Native ``checkpointers_at(aimed=False)`` for a plain decode step
        (``next_forward_tokens=1``): a block boundary at least one interval
        past the request's last checkpoint."""
        return (
            self.interval > 0
            and pos % self.block == 0
            and pos - last_checkpoint_pos >= self.interval
        )

    def images_per_request(self) -> int:
        """Images one request holds at once: the current one and the one it
        replaces; none when checkpointing is off."""
        return 0 if self.interval == 0 else 2


def request_marks(request) -> tuple[int, int]:
    return (
        int(getattr(request, "_atom_img_demand", 0)),
        int(getattr(request, "_atom_img_anchor", 0)),
    )


class SchedulerImages:
    """Native placement bound to one vLLM ``Scheduler`` (EngineCore side).

    Three things have to agree for an image to be usable: where prefill is cut
    (``chunk_len``), which image block vLLM files under a prefix hash
    (``cache_image_blocks``), and where the worker copies the slot out
    (``scheduler_output.atom_image_stores``). All three read ``placement``.
    """

    #: Family name in errors (set from the adapter).
    name = "paged-state"

    def __init__(
        self, scheduler, placement: ImagePlacement, proxy_gid, image_gids, name=None
    ):
        if name is not None:
            self.name = name
        self.scheduler = scheduler
        self.placement = placement
        self.proxy_gid = proxy_gid
        self.image_gids = list(image_gids)
        self.k = len(self.image_gids)
        kvm = scheduler.kv_cache_manager
        self.coordinator = kvm.coordinator
        self.block_pool = kvm.block_pool
        self.cache_end = -1
        # req_id -> (boundary, [(group id, block)]) hashed during this schedule().
        self.step_keeps: dict[str, tuple[int, list]] = {}
        # req_id -> last position a native-placed image was kept at (native
        # ``Sequence.last_checkpoint_pos``: set on a keep, cleared on preemption).
        self.last_ckpt: dict[str, int] = {}
        # req_id -> decode step end on a block boundary, from this schedule().
        self.decode_intents: dict[str, int] = {}
        # req_id -> block index of a decode keep whose prefix hash is not known
        # yet (async scheduling: the token closing the block is still in flight).
        self.pending_hash: dict[str, int] = {}
        # Placement decisions, for tests: (kind, req_id, boundary).
        self.events: list | None = None
        # req_id -> first block index worth scanning (everything below is null).
        self._scan_from: dict[str, int] = {}
        # req_id -> block indices the release pass keeps this step (computed on
        # the first image group, applied to all k; they allocate in lockstep).
        self._release_keep: dict[str, tuple[int, set]] = {}
        self._release_step = 0

        self.check_blocks = (
            os.environ.get("ATOM_STATE_IMAGE_CHECK_BLOCKS")
            or os.environ.get("ATOM_V4_IMAGE_CHECK_BLOCKS")
        ) == "1"
        self.counters = {
            "chunks_cut_for_rung": 0,
            "chunks_cut_for_anchor": 0,
            "chunks_cut_for_demand": 0,
            "images_hashed": 0,
            "demands_recorded": 0,
            "demands_declined_no_room": 0,
            "stores_unhashed": 0,
            "decode_keeps": 0,
            "duplicate_images_skipped": 0,
            "max_image_blocks_per_request": 0,
        }

    # -- admission (each get_computed_blocks call = native can_allocate) ------
    def on_admission(self, request, blocks, hit: int, shared_prefix_boundary: int):
        p = self.placement
        request._atom_img_anchor = p.anchor(
            int(request.num_prompt_tokens), int(request.num_tokens)
        )
        # The proxy-only hit: vLLM reports it as the shared-prefix junction
        # exactly when the image groups cut the hit short.
        compressed_hit = shared_prefix_boundary or hit
        has_room = True
        if p.demand and p.interval != 0 and compressed_hit > hit:
            bs = self.placement.block
            proxy_hit = blocks.blocks[self.proxy_gid][: hit // bs] if hit else []
            live = -(-int(request.num_tokens) // bs) - sum(
                1 for b in proxy_hit if b.ref_cnt > 0
            )
            has_room = self.block_pool.get_num_free_blocks() >= live + self.k
        demand = p.demand_pos(hit, compressed_hit, has_room)
        if (
            p.demand
            and p.interval != 0
            and compressed_hit > hit
            and not has_room
            and not getattr(request, "_atom_img_demand_declined", False)
        ):
            self.counters["demands_declined_no_room"] += 1
            request._atom_img_demand_declined = True
        if demand and not getattr(request, "_atom_img_demand_counted", False):
            self.counters["demands_recorded"] += 1
            request._atom_img_demand_counted = True
        request._atom_img_demand = demand

    # -- chunk length (replaces _mamba_block_aligned_split) -------------------
    def chunk_len(self, request, num_new_tokens: int, start: int) -> int:
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            end = start + num_new_tokens
            last = self.last_ckpt.get(request.request_id, 0)
            if self.placement.decode_keeps(end, last):
                # Kept in take_store_plan once the step is scheduled: with async
                # scheduling cache_blocks lags a decode step by a token.
                self.decode_intents[request.request_id] = end
            return num_new_tokens
        end = start + num_new_tokens
        demand, anchor = request_marks(request)
        target = self.placement.cut(
            start, end, int(request.num_prompt_tokens), demand, anchor
        )
        request._atom_img_step = ("prefill", target or end)
        if not target:
            return num_new_tokens
        if target < end:
            rung = (
                target % self.placement.interval == 0
                if self.placement.interval > 0
                else False
            )
            if target == demand:
                self.counters["chunks_cut_for_demand"] += 1
            elif rung:
                self.counters["chunks_cut_for_rung"] += 1
            else:
                self.counters["chunks_cut_for_anchor"] += 1
        return target - start

    # -- which image block gets a prefix hash ---------------------------------
    def cache_image_blocks(self, mgr, request, num_tokens: int) -> None:
        rid = request.request_id
        bs = mgr.block_size
        num_cached = mgr.num_cached_block.get(rid, 0)
        num_full = num_tokens // bs
        if num_cached >= num_full:
            return
        end = self.cache_end
        decision = self._decide(request, end) if end == num_full * bs else None
        keep = decision == "hash"
        blocks = mgr.req_to_blocks[rid]
        mask = [False] * (num_full - num_cached)
        pend = self.pending_hash.get(rid)
        deferred = pend is not None and num_cached <= pend < num_full
        if deferred:
            mask[pend - num_cached] = True
        if decision is not None and blocks[num_full - 1].is_null:
            raise RuntimeError(
                f"{self.name} image for {rid} at {end}: the running image block "
                "is null; vLLM align did not allocate where the forward ends"
            )
        if keep:
            mask[-1] = True
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=num_cached,
            num_full_blocks=num_full,
            block_size=bs,
            kv_cache_group_id=mgr.kv_cache_group_id,
            block_mask=mask,
        )
        mgr.num_cached_block[rid] = num_full
        if deferred:
            blk = blocks[pend]
            if blk.block_hash is not None:
                mgr.cached_blocks_this_step.add(blk.block_hash)
            if mgr.kv_cache_group_id == self.image_gids[-1]:
                self.pending_hash.pop(rid, None)
        if decision is not None:
            blk = blocks[num_full - 1]
            if keep and blk.block_hash is not None:
                mgr.cached_blocks_this_step.add(blk.block_hash)
            entry = self.step_keeps.setdefault(rid, (end, []))
            entry[1].append((mgr.kv_cache_group_id, blk))

    def _decide(self, request, end: int):
        """Whether a prefill forward ending at ``end`` keeps a native-placed
        image ("hash") or nothing (None). Decided once per request and step
        (all k groups ask). Decode steps are decided in ``take_store_plan``.
        """
        memo = getattr(request, "_atom_img_decided", None)
        if memo is not None and memo[0] == self._release_step and memo[1] == end:
            return memo[2]
        rid = request.request_id
        p = self.placement
        kind, step_end = getattr(request, "_atom_img_step", (None, -1))
        decision = None
        if step_end == end and p.interval != 0 and kind == "prefill":
            demand, anchor = request_marks(request)
            if p.keeps(end, int(request.num_prompt_tokens), demand, anchor):
                self.last_ckpt[rid] = end
                if self._image_cached(request, end // p.block - 1):
                    self.counters["duplicate_images_skipped"] += 1
                else:
                    decision = "hash"
        if decision is not None and self.events is not None:
            self.events.append((decision, rid, end))
        request._atom_img_decided = (self._release_step, end, decision)
        return decision

    # -- per-request image blocks: at most two images -------------------------
    def release_image_blocks(self, mgr, request_id: str) -> None:
        """Replaces vLLM align's ``remove_skipped_blocks`` for the image groups.

        Keeps the last non-null block (the previous step's running block, whose
        store may still be in flight) and frees the rest. A hashed image stays in
        the prefix cache once freed. Called before every allocation, so after it
        a request holds at most two images, with or without async scheduling
        (vLLM's own pass lags by the in-flight step and can leave a third behind
        until the request ends).
        """
        blocks = mgr.req_to_blocks.get(request_id)
        if not blocks:
            return
        if mgr.kv_cache_group_id == self.image_gids[0]:
            start = self._scan_from.get(request_id, 0)
            held = [i for i in range(start, len(blocks)) if not blocks[i].is_null]
            keep = {held[-1]} if held else set()
            self._release_keep[request_id] = (self._release_step, keep, held)
            if keep:
                self._scan_from[request_id] = min(keep)
        entry = self._release_keep.get(request_id)
        if entry is None or entry[0] != self._release_step:
            return
        _step, keep, held = entry
        freed = []
        for i in held:
            if i not in keep and i < len(blocks) and not blocks[i].is_null:
                freed.append(blocks[i])
                blocks[i] = mgr._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    def on_free(self, request) -> None:
        """Before vLLM frees a request's blocks.

        A decode keep stored on the request's last step whose prefix hash was
        not known yet is hashed now; a preempted request (still RUNNING when
        ``_preempt_request`` frees it) drops it, and like native forgets its
        last checkpoint position.
        """
        rid = request.request_id
        pend = self.pending_hash.pop(rid, None)
        self.decode_intents.pop(rid, None)
        if request.is_finished() and pend is not None:
            self._hash_block(request, pend)
        self.last_ckpt.pop(rid, None)
        self._scan_from.pop(rid, None)
        self._release_keep.pop(rid, None)
        request._atom_img_decided = None

    def image_blocks_held(self, request_id: str) -> int:
        n = 0
        start = self._scan_from.get(request_id, 0)
        for gid in self.image_gids:
            blocks = self.coordinator.single_type_managers[gid].req_to_blocks.get(
                request_id, ()
            )
            n += sum(1 for b in blocks[start:] if not b.is_null)
        return n

    def check_image_blocks(self) -> None:
        """Instrumented runs: no request may hold more than two images' blocks."""
        limit = self.placement.images_per_request() * self.k
        for req in self.scheduler.running:
            n = self.image_blocks_held(req.request_id)
            self.counters["max_image_blocks_per_request"] = max(
                self.counters["max_image_blocks_per_request"], n
            )
            if n > limit:
                raise RuntimeError(
                    f"{self.name} request {req.request_id} holds {n} image blocks, "
                    f"more than {limit} (2 images x k={self.k})"
                )

    # -- the worker's store plan ----------------------------------------------
    def take_store_plan(self, scheduler_output) -> dict[str, int]:
        keeps, self.step_keeps = self.step_keeps, {}
        scheduled = scheduler_output.num_scheduled_tokens
        plan = {}
        for rid, (end, blocks) in keeps.items():
            req = self.scheduler.requests.get(rid)
            if (
                req is not None
                and rid in scheduled
                and int(req.num_computed_tokens) == end
                and len(blocks) == self.k
            ):
                plan[rid] = end
                self.counters["images_hashed"] += 1
                continue
            # Hashed for a forward that is not going to run: take the hash back.
            self.block_pool.evict_blocks({b.block_id for _g, b in blocks})
            self.counters["stores_unhashed"] += 1
            logger.warning(
                "ATOM %s: image for %s at %d hashed but not scheduled; "
                "unhashed its %d blocks",
                self.name,
                rid,
                end,
                len(blocks),
            )
        intents, self.decode_intents = self.decode_intents, {}
        for rid, end in intents.items():
            req = self.scheduler.requests.get(rid)
            if (
                req is None
                or rid not in scheduled
                or rid in plan
                or int(req.num_computed_tokens) != end
            ):
                continue
            plan[rid] = end
            self._keep_decode(req, end)
        return plan

    def _keep_decode(self, request, end: int) -> None:
        """A scheduled decode step on which native's unaimed rule keeps an image
        (native keeps it on the request's last step too: ``_checkpoint_room``
        is 0 there and a PAGE checkpoint's ``successor_room`` is 0). Hashed as
        soon as the block's prefix hash exists."""
        rid = request.request_id
        idx = end // self.placement.block - 1
        for gid in self.image_gids:
            blocks = self.coordinator.single_type_managers[gid].req_to_blocks.get(
                rid, []
            )
            if idx >= len(blocks) or blocks[idx].is_null:
                raise RuntimeError(
                    f"{self.name} decode image for {rid} at {end}: no running image "
                    "block there"
                )
        self.last_ckpt[rid] = end
        self._hash_or_defer(request, idx)
        self.counters["decode_keeps"] += 1
        self.counters["images_hashed"] += 1
        if self.events is not None:
            self.events.append(("hash", rid, end))

    def _image_cached(self, request, idx: int) -> bool:
        """Whether this boundary already has a whole image (native skips a store
        whose prefix hash is already present or pending: `take_checkpoint_ops`).

        vLLM evicts block by block, so a boundary can have lost some groups'
        blocks; it is stored again then, and a later hit may take some groups
        from one store and the rest from another. Every group's block is the
        state after the same hashed prefix, as in vLLM's own hybrid Mamba."""
        if len(request.block_hashes) <= idx:
            return False
        return (
            self.block_pool.get_cached_block(request.block_hashes[idx], self.image_gids)
            is not None
        )

    def _hash_or_defer(self, request, idx: int) -> None:
        rid = request.request_id
        mgr0 = self.coordinator.single_type_managers[self.image_gids[0]]
        if mgr0.num_cached_block.get(rid, 0) > idx and len(request.block_hashes) > idx:
            self._hash_block(request, idx)
        else:
            self.pending_hash[rid] = idx

    def _hash_block(self, request, idx: int) -> bool:
        rid = request.request_id
        hashed = False
        if self._image_cached(request, idx):
            self.counters["duplicate_images_skipped"] += 1
            return False
        for gid in self.image_gids:
            mgr = self.coordinator.single_type_managers[gid]
            blocks = mgr.req_to_blocks.get(rid, [])
            if idx >= len(blocks) or blocks[idx].is_null:
                raise RuntimeError(
                    f"{self.name} image of {rid} at {(idx + 1) * self.placement.block} "
                    "is no longer held; the release pass dropped it"
                )
            if blocks[idx].block_hash is not None:
                continue
            self.block_pool.cache_full_blocks(
                request=request,
                blocks=blocks,
                num_cached_blocks=idx,
                num_full_blocks=idx + 1,
                block_size=self.placement.block,
                kv_cache_group_id=gid,
                block_mask=[True],
            )
            hashed = True
        return hashed


def find_image_groups(kv_cache_config, adapter):
    proxy, images = None, []
    for gid, g in enumerate(kv_cache_config.kv_cache_groups):
        if adapter.proxy_layer_name in g.layer_names:
            proxy = gid
        elif any(adapter.is_image_layer(n) for n in g.layer_names):
            images.append(gid)
    return proxy, images


def bind_scheduler_images(scheduler) -> SchedulerImages | None:
    """Attach native placement to a prefix-caching scheduler of a registered
    model family; else None."""
    kvm = getattr(scheduler, "kv_cache_manager", None)
    cfg = getattr(scheduler, "kv_cache_config", None)
    if kvm is None or cfg is None or not getattr(kvm, "enable_caching", False):
        return None
    adapter = adapter_for_kv_cache_config(cfg)
    if adapter is None:
        return None
    proxy, images = find_image_groups(cfg, adapter)
    if proxy is None or not images:
        return None
    placement = ImagePlacement.from_env(
        adapter.checkpoint_defaults(), adapter.block_size
    )
    st = SchedulerImages(scheduler, placement, proxy, images, name=adapter.name)

    orig_gcb = kvm.get_computed_blocks

    def get_computed_blocks(request):
        blocks, hit, spb = orig_gcb(request)
        st.on_admission(request, blocks, int(hit), int(spb))
        return blocks, hit, spb

    kvm.get_computed_blocks = get_computed_blocks

    coordinator = st.coordinator
    orig_cache = coordinator.cache_blocks

    def coordinator_cache_blocks(request, num_computed_tokens):
        st.cache_end = int(num_computed_tokens)
        try:
            return orig_cache(request, num_computed_tokens)
        finally:
            st.cache_end = -1

    coordinator.cache_blocks = coordinator_cache_blocks
    for gid in images:
        mgr = coordinator.single_type_managers[gid]

        def image_cache_blocks(request, num_tokens, retention_interval=None, _m=mgr):
            st.cache_image_blocks(_m, request, num_tokens)

        mgr.cache_blocks = image_cache_blocks

        def image_remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens=None, _m=mgr
        ):
            st.release_image_blocks(_m, request_id)

        mgr.remove_skipped_blocks = image_remove_skipped_blocks

    orig_rsb = coordinator.remove_skipped_blocks

    def coordinator_remove_skipped_blocks(*args, **kwargs):
        st._release_step += 1
        return orig_rsb(*args, **kwargs)

    coordinator.remove_skipped_blocks = coordinator_remove_skipped_blocks

    for name in ("free", "pop_blocks_for_free"):
        orig_free = getattr(kvm, name)

        def wrapped_free(request, *a, _orig=orig_free, **kw):
            st.on_free(request)
            return _orig(request, *a, **kw)

        setattr(kvm, name, wrapped_free)
    logger.info(
        "ATOM %s image placement bound to the scheduler (%s): "
        "state_checkpoint_interval_tokens=%d state_checkpoint_demand=%s "
        "proxy_group=%d image_groups=%s image_block_check=%s",
        adapter.name,
        type(scheduler).__name__,
        st.placement.interval,
        st.placement.demand,
        proxy,
        images,
        st.check_blocks,
    )
    return st


def apply_scheduler_patches() -> bool:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler.__init__, "_atom_state_image_patched", False):
        return False
    orig_init = Scheduler.__init__
    orig_split = Scheduler._mamba_block_aligned_split
    orig_schedule = Scheduler.schedule

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._atom_state_images = bind_scheduler_images(self)

    @functools.wraps(orig_split)
    def _mamba_block_aligned_split(
        self,
        request,
        num_new_tokens,
        num_new_local_computed_tokens=0,
        num_external_computed_tokens=0,
    ):
        st = getattr(self, "_atom_state_images", None)
        if st is None:
            return orig_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        return st.chunk_len(request, num_new_tokens, start)

    @functools.wraps(orig_schedule)
    def schedule(self, *args, **kwargs):
        out = orig_schedule(self, *args, **kwargs)
        st = getattr(self, "_atom_state_images", None)
        if st is not None:
            out.atom_image_stores = st.take_store_plan(out)
            if st.check_blocks:
                st.check_image_blocks()
        return out

    __init__._atom_state_image_patched = True
    Scheduler.__init__ = __init__
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
    Scheduler.schedule = schedule
    return True
