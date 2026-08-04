# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SparseKV coordinator: paged CPU cold pool + GPU hot buffer for DSA decode.

GLM-5.2 (DSA) decode only reads the indexer's top-k tokens each step, but the
full KV cache otherwise occupies GPU HBM. SparseKV keeps the complete KV in a
paged shared CPU pinned cold pool and only a fixed-size hot buffer per request
on the GPU. Each decode step, for every layer: miss-detect the top-k against
the resident hot set, evict the least-recently-used slots, swap the missing
tokens in from the cold pool, and translate the top-k into hot-buffer slots so
MLA attention reads the hot buffer.

Index domain is LOGICAL: miss-detect runs on the indexer's per-request logical
top-k positions (``0..context_len``), not physical paged-KV slots. The cold pool
is a paged shared host pool addressed via ``req_to_host_pool``; ``translate``
maps logical top-k to hot-buffer slots.

The pure bookkeeping (miss-detect + LRU allocate + translate) is separated from
the GPU data movement so it is testable without a GPU. Only :meth:`swap_in_for_layer`
and the staging/backup helpers touch the swap kernel.
"""

import logging

import torch

from atom.utils import envs

logger = logging.getLogger("atom")

_EMPTY = -1  # sentinel for empty slot / non-resident token


class SparseKVCoordinator:
    """Manages the CPU cold pool and GPU hot buffer for one model runner.

    Args:
        num_layers: total number of KV-bearing layers.
        max_num_seqs: max concurrent decode requests (cold pool / hot buffer are
            sized for this many request slots).
        hot_buffer_size: per-request resident hot tokens (H). The hot buffer
            physically holds H+1 slots per request (one padding slot of headroom
            for the newest token); all H+1 are managed as one LRU pool.
        max_context_len: max logical tokens per request (cold pool depth).
        kv_dim: MLA compressed KV width per token (kv_lora_rank + qk_rope_head_dim).
        kv_dtype: cold pool / hot buffer element dtype.
        device: GPU device for the hot buffer.
    """

    def __init__(
        self,
        num_layers: int,
        max_num_seqs: int,
        hot_buffer_size: int,
        max_context_len: int,
        kv_dim: int,
        kv_dtype: torch.dtype,
        device: torch.device | str = "cuda",
        index_topk: int | None = None,
        shared_index_layers: list[bool] | None = None,
        host_to_device_ratio: int = 8,
        page_size: int = 16,
    ):
        self.num_layers = num_layers
        self.max_num_seqs = max_num_seqs
        self.hot_buffer_size = hot_buffer_size
        self.padded_hot_size = hot_buffer_size + 1  # H+1 physical slots
        self.max_context_len = max_context_len
        self.kv_dim = kv_dim
        self.kv_dtype = kv_dtype
        self.device = torch.device(device)
        self.item_size_bytes = kv_dim * torch.empty(0, dtype=kv_dtype).element_size()

        H1 = self.padded_hot_size
        R = max_num_seqs
        C = max_context_len

        self.host_page_size = page_size

        # Table stride: max logical positions per request, padded to a whole
        # number of pages so a request's page-crossing stays page-aligned.
        self.table_stride = ((C + page_size - 1) // page_size) * page_size
        # Shared host pool sized as a multiple of the GPU hot-buffer total
        # capacity; requests allocate pages on demand from a common free pool.
        host_tokens = host_to_device_ratio * R * H1
        self.num_host_pages = max(1, (host_tokens + page_size - 1) // page_size)
        num_host_slots = self.num_host_pages * page_size
        self.cold_pool = torch.zeros(
            (num_layers, num_host_slots, kv_dim), dtype=kv_dtype, pin_memory=True
        )
        # req_to_host_pool: logical token -> flat host slot (-1 = unallocated).
        self.req_to_host_pool = torch.full(
            (R, self.table_stride), _EMPTY, dtype=torch.int32, device=self.device
        )
        self._free_host_pages: list[int] = list(range(self.num_host_pages))
        self._req_pages: dict[int, list[int]] = {}
        self._req_host_alloc_len = [0] * R

        # GPU hot buffer: [num_layers, R * H1, kv_dim].
        self.hot_buffer = torch.zeros(
            (num_layers, R * H1, kv_dim), dtype=kv_dtype, device=self.device
        )

        # Device-mapped cold-pool pointer per layer (xnack- needs the translation;
        # a raw host VA faults). Filled lazily on first swap.
        self._cold_dev_ptr: list[int | None] = [None] * num_layers

        # Paged-host translation table: logical token -> host slot, consulted by
        # the swap kernels to indirect through the page table.
        self._empty_host_locs = torch.empty(0, dtype=torch.int32, device=self.device)
        self._host_cache_locs: torch.Tensor | None = self.req_to_host_pool
        self._host_stride = self.table_stride

        # Per (layer, request, slot) bookkeeping. Lives on ``device`` so the
        # fused swap kernel reads/writes it without any host sync; the pure-Python
        # reference path (CPU tests, debug bisection) uses the same tensors.
        # slot_token: logical token ID resident in each hot slot (-1 empty).
        self.slot_token = torch.full(
            (num_layers, R, H1), _EMPTY, dtype=torch.int32, device=self.device
        )
        # last_used: recency tick per slot (-1 empty); LRU victim = smallest.
        self.last_used = torch.full(
            (num_layers, R, H1), _EMPTY, dtype=torch.int64, device=self.device
        )
        # token_to_slot: reverse map logical token -> hot slot (-1 not resident).
        self.token_to_slot = torch.full(
            (num_layers, R, C), _EMPTY, dtype=torch.int32, device=self.device
        )
        # Per-request monotonic recency counter for the fused kernel (the GPU
        # analogue of ``self._tick``); advanced on-device per swap/backup.
        self.recency = torch.zeros(R, dtype=torch.int64, device=self.device)

        # Per-request context length (logical tokens currently staged).
        self.context_len = torch.zeros(max_num_seqs, dtype=torch.int64)
        self.slot_active = torch.zeros(max_num_seqs, dtype=torch.bool)

        # Stable request-id -> slot map (worker-side; keyed by ScheduledBatch
        # req_ids so the same request lands in the same cold-pool region every
        # step) and the free-slot pool.
        self._reqid_to_slot: dict[int, int] = {}
        self._free_slots: list[int] = list(range(max_num_seqs))
        # RDMA-direct: request ids whose hot buffer has been preloaded from the
        # (RDMA-filled) cold pool. The slot is acquired at recv by the connector,
        # but the hot load must wait for the first decode (KV is in the cold pool
        # by then); this tracks which acquired requests still need that first load.
        self._hot_loaded: set = set()
        # RDMA-direct: request ids acquired at recv (slot + host pages reserved,
        # prompt KV being RDMA'd in) that have not yet reached their first decode.
        # sync_active must NOT reclaim these — they are absent from the decode batch
        # only because their transfer is still in flight, not because they finished.
        self._awaiting_first_decode: set = set()

        # Monotonic recency tick, advanced each swap/backup.
        self._tick = 0

        logger.info(
            "SparseKVCoordinator: layers=%d max_seqs=%d hot=%d(+1) max_ctx=%d "
            "kv_dim=%d dtype=%s cold_pool=%.2fGB hot_buffer=%.2fGB",
            num_layers,
            max_num_seqs,
            hot_buffer_size,
            max_context_len,
            kv_dim,
            kv_dtype,
            self.cold_pool.numel() * self.cold_pool.element_size() / 1e9,
            self.hot_buffer.numel() * self.hot_buffer.element_size() / 1e9,
        )

        self._init_prefetch(index_topk, shared_index_layers)

    # ------------------------------------------------------------------
    # IndexShare group prefetch (Stage C)
    # ------------------------------------------------------------------
    def _init_prefetch(
        self, index_topk: int | None, shared_index_layers: list[bool] | None
    ) -> None:
        """Build the IndexShare prefetch groups, swap stream, events, plan buffers.

        A "shared" layer reuses its anchor (the most recent non-shared layer)
        top-k, so the anchor's swap plan is valid for every layer in the group.
        Within a group the per-request slot bookkeeping evolves identically across
        layers (same top-k, same current position, identical seed), so the anchor
        records its miss plan once and each shared layer replays the identical IO
        on a side stream, overlapping the intervening compute.
        """
        self._is_shared = list(shared_index_layers or [False] * self.num_layers)
        assert len(self._is_shared) == self.num_layers, (
            f"shared_index_layers length {len(self._is_shared)} "
            f"!= num_layers {self.num_layers}"
        )

        self._prefetch_groups: dict[int, list[int]] = {}
        self._anchor_of = list(range(self.num_layers))
        self._prefetch_slot = [0] * self.num_layers
        anchor = None
        for i, is_shared in enumerate(self._is_shared):
            if not is_shared:
                anchor = i
            else:
                assert anchor is not None, (
                    f"shared-index layer {i} has no preceding anchor layer; "
                    "the model's index-topk pattern is invalid"
                )
                g = self._prefetch_groups.setdefault(anchor, [])
                self._prefetch_slot[i] = len(g)
                self._anchor_of[i] = anchor
                g.append(i)

        self.enable_prefetch = (
            bool(self._prefetch_groups) and envs.ATOM_SPARSEKV_PREFETCH
        )
        if not self.enable_prefetch:
            return

        assert index_topk is not None, "index_topk required for SparseKV prefetch"
        self._prefetch_topk = index_topk
        max_group = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = torch.cuda.Stream()
        self._prefetch_events = [torch.cuda.Event() for _ in range(max_group)]
        R = self.max_num_seqs
        # Miss plan recorded by the current anchor, replayed by its shared layers.
        # Indexed by decode query token (0..n-1); one group is live at a time.
        self._plan_miss_tok = torch.zeros(
            (R, index_topk), dtype=torch.int32, device=self.device
        )
        self._plan_miss_slot = torch.zeros(
            (R, index_topk), dtype=torch.int32, device=self.device
        )
        self._plan_miss_count = torch.zeros((R,), dtype=torch.int32, device=self.device)
        logger.info(
            "SparseKV: IndexShare prefetch enabled; %d anchor group(s), "
            "%d shared layer(s) of %d total, max group size %d.",
            len(self._prefetch_groups),
            sum(self._is_shared),
            self.num_layers,
            max_group,
        )

    def is_prefetch_anchor(self, layer_id: int) -> bool:
        """True if this layer computes a top-k that shared layers will reuse."""
        return self.enable_prefetch and layer_id in self._prefetch_groups

    def is_shared_layer(self, layer_id: int) -> bool:
        """True if this layer reuses its anchor's top-k (skip its own indexer)."""
        return self.enable_prefetch and self._is_shared[layer_id]

    def anchor_of(self, layer_id: int) -> int:
        """The anchor layer whose top-k / slot table this layer shares."""
        return self._anchor_of[layer_id]

    def prefetch_group(self, anchor_layer: int, req_slots: torch.Tensor) -> None:
        """Issue every shared layer's swap-in on the side stream (from the anchor).

        The anchor already recorded its miss plan; each shared layer replays it
        into its own hot buffer with a pure copy kernel, overlapping the compute
        of the anchor and the layers between it and each shared layer.
        """
        group = self._prefetch_groups.get(anchor_layer)
        if not group:
            return
        from atom.sparsekv.swap_kernel import sparsekv_copy_planned

        n = int(req_slots.shape[0])
        topk = self._prefetch_topk
        cur = torch.cuda.current_stream()
        self.prefetch_stream.wait_stream(cur)
        with torch.cuda.stream(self.prefetch_stream):
            for skip in group:
                host_ptr = self._ensure_cold_dev_ptr(skip)
                sparsekv_copy_planned(
                    host_ptr,
                    self.hot_buffer[skip],
                    req_slots,
                    self._plan_miss_tok[:n],
                    self._plan_miss_slot[:n],
                    self._plan_miss_count[:n],
                    self._host_locs_arg(),
                    self._host_stride,
                    self.item_size_bytes,
                    self.padded_hot_size,
                    self.max_context_len,
                    topk,
                )
                self._prefetch_events[self._prefetch_slot[skip]].record(
                    self.prefetch_stream
                )

    def wait_prefetch(self, shared_layer: int) -> None:
        """Block the current stream on this shared layer's prefetched swap-in."""
        ev = self._prefetch_events[self._prefetch_slot[shared_layer]]
        ev.wait(torch.cuda.current_stream())

    # ------------------------------------------------------------------
    # request lifecycle
    # ------------------------------------------------------------------
    def register_request(self, req_slot: int, context_len: int) -> int:
        """Reserve a request slot and reset its hot-buffer / LRU state."""
        assert 0 <= req_slot < self.max_num_seqs, req_slot
        assert (
            context_len <= self.max_context_len
        ), f"context_len {context_len} > max_context_len {self.max_context_len}"
        self.slot_active[req_slot] = True
        self.context_len[req_slot] = context_len
        self.slot_token[:, req_slot, :] = _EMPTY
        self.last_used[:, req_slot, :] = _EMPTY
        self.token_to_slot[:, req_slot, :] = _EMPTY
        self.recency[req_slot] = 0
        return req_slot

    def unregister_request(self, req_slot: int) -> None:
        """Release a request slot and clear its state."""
        self.slot_active[req_slot] = False
        self.context_len[req_slot] = 0
        self.slot_token[:, req_slot, :] = _EMPTY
        self.last_used[:, req_slot, :] = _EMPTY
        self.token_to_slot[:, req_slot, :] = _EMPTY
        self.free_host_pages(req_slot)

    # ------------------------------------------------------------------
    # request-id (stable) management — worker-side, keyed by ScheduledBatch ids
    # ------------------------------------------------------------------
    def is_registered(self, req_id: int) -> bool:
        return req_id in self._reqid_to_slot

    def slot_for_req(self, req_id: int) -> int:
        """Return the cold-pool slot for a registered request id."""
        return self._reqid_to_slot[req_id]

    def acquire(self, req_id: int, context_len: int) -> int:
        """Assign a fresh slot to a new request id and return it."""
        if req_id in self._reqid_to_slot:
            return self._reqid_to_slot[req_id]
        if not self._free_slots:
            raise RuntimeError(
                "SparseKV: no free request slots "
                f"(max_num_seqs={self.max_num_seqs} exhausted)"
            )
        slot = self._free_slots.pop()
        self._reqid_to_slot[req_id] = slot
        self.register_request(slot, context_len)
        return slot

    def acquire_at_recv(self, req_id: int, num_tokens: int) -> int:
        """Reserve a slot + host pages for an inbound request (RDMA-direct).

        Called by the connector before it asks the producer to RDMA the prompt KV
        straight into this request's freshly allocated host pages. The hot buffer
        is preloaded later, on the request's first decode (see maybe_first_hot_load).
        """
        slot = self.acquire(req_id, num_tokens)
        self.alloc_host_pages(slot, 0, num_tokens)
        self._awaiting_first_decode.add(req_id)
        return slot

    def maybe_first_hot_load(self, req_id: int, present: int) -> None:
        """Preload the hot buffer for an RDMA-direct request on its first decode.

        Idempotent per request id: the prompt KV is already in the cold pool (RDMA),
        so this just seeds the resident hot set from it once.
        """
        if req_id in self._hot_loaded:
            return
        slot = self._reqid_to_slot.get(req_id)
        if slot is None:
            return
        self.load_initial_hot_set(slot, present)
        self._hot_loaded.add(req_id)
        # First decode reached: the request is now a normal decode-batch member,
        # so sync_active may reclaim it once it leaves the batch.
        self._awaiting_first_decode.discard(req_id)

    def release(self, req_id: int) -> None:
        """Free the slot held by a request id (idempotent)."""
        self._hot_loaded.discard(req_id)
        self._awaiting_first_decode.discard(req_id)
        slot = self._reqid_to_slot.pop(req_id, None)
        if slot is None:
            return
        self.unregister_request(slot)
        self._free_slots.append(slot)

    def sync_active(self, active_req_ids) -> None:
        """Release every registered request not in ``active_req_ids``.

        Decode requests are scheduled every step until they finish, so a
        registered id absent from the current batch has completed (or been
        preempted — it will re-stage cleanly on return).

        RDMA-direct exception: a request acquired at recv whose prompt KV is still
        being transferred is not yet in any decode batch. Skip those (tracked in
        ``_awaiting_first_decode``) so their reserved slot + host pages survive the
        recv window; they become reclaimable once their first decode runs.
        """
        active = set(active_req_ids)
        for req_id in list(self._reqid_to_slot.keys()):
            if req_id not in active and req_id not in self._awaiting_first_decode:
                self.release(req_id)

    # ------------------------------------------------------------------
    # cold-pool addressing
    # ------------------------------------------------------------------
    def _host_locs_arg(self) -> torch.Tensor:
        """Tensor to pass the swap kernels as ``host_cache_locs``.

        Returns the paged translation table when RDMA-direct paging is active, or
        a persistent empty tensor otherwise (kernel sees stride 0 -> nullptr ->
        dense ``req_slot*cold_depth + logical`` addressing, unchanged from Phase 0).
        """
        return (
            self._host_cache_locs
            if self._host_cache_locs is not None
            else self._empty_host_locs
        )

    # ------------------------------------------------------------------
    # paged host pool (RDMA-direct)
    # ------------------------------------------------------------------
    def alloc_host_pages(self, req_slot: int, start_pos: int, num_tokens: int) -> None:
        """Back logical ``[start_pos, start_pos+num_tokens)`` with host pool pages.

        Allocates whole pages on demand from the shared free pool and records each
        logical token's flat host slot in ``req_to_host_pool`` (slots within a page
        are contiguous, so an RDMA block write lands in one page). Growth is
        contiguous: ``start_pos`` must not exceed the request's already-backed
        length. Raises if the shared pool is exhausted (admission back-pressure).
        """
        if num_tokens <= 0:
            return
        page = self.host_page_size
        allocated_len = self._req_host_alloc_len[req_slot]
        end_pos = start_pos + num_tokens
        assert start_pos <= allocated_len, (
            f"non-contiguous host alloc: start_pos={start_pos} > "
            f"allocated_len={allocated_len} (req_slot={req_slot})"
        )
        page_end = ((end_pos + page - 1) // page) * page
        if page_end <= allocated_len:
            return
        num_new_pages = (page_end - allocated_len) // page
        if len(self._free_host_pages) < num_new_pages:
            raise RuntimeError(
                f"SparseKV host pool exhausted: need {num_new_pages} pages, "
                f"{len(self._free_host_pages)} free (num_host_pages="
                f"{self.num_host_pages}); raise ATOM_SPARSEKV_HOST_TO_DEVICE_RATIO "
                "or lower concurrency"
            )
        new_pages = [self._free_host_pages.pop() for _ in range(num_new_pages)]
        self._req_pages.setdefault(req_slot, []).extend(new_pages)
        # Flat host slots for the newly backed logical range, page-contiguous.
        slots = torch.empty(page_end - allocated_len, dtype=torch.int32)
        off = torch.arange(page, dtype=torch.int32)
        for i, p in enumerate(new_pages):
            slots[i * page : (i + 1) * page] = p * page + off
        self.req_to_host_pool[req_slot, allocated_len:page_end] = slots.to(self.device)
        self._req_host_alloc_len[req_slot] = page_end

    def free_host_pages(self, req_slot: int) -> None:
        """Return a request's host pages to the shared free pool and clear its row."""
        pages = self._req_pages.pop(req_slot, None)
        if pages:
            self._free_host_pages.extend(pages)
        self.req_to_host_pool[req_slot].fill_(_EMPTY)
        self._req_host_alloc_len[req_slot] = 0

    def grow_host_for_new_tokens(
        self, req_slots: list[int], positions: list[int]
    ) -> None:
        """Ensure each decode query token's logical position has a backing host slot.

        Called eagerly before the forward each step so the fused backup kernel can
        write cold pool at ``req_to_host_pool[r][pos]`` (the table update is a fixed-
        address scatter on an already-allocated tensor, so the captured forward that
        follows reads it correctly).
        """
        for r, pos in zip(req_slots, positions):
            if pos >= 0:
                self.alloc_host_pages(r, pos, 1)

    def _hot_base(self, req_slot: int) -> int:
        return req_slot * self.padded_hot_size

    def _ensure_cold_dev_ptr(self, layer_id: int) -> int:
        """Return (and cache) the device-mapped pointer for one layer's cold pool."""
        ptr = self._cold_dev_ptr[layer_id]
        if ptr is None:
            from atom.sparsekv.swap_kernel import host_get_device_pointer

            layer_view = self.cold_pool[layer_id]  # contiguous [R*C, kv_dim]
            ptr = host_get_device_pointer(layer_view)
            self._cold_dev_ptr[layer_id] = ptr
        return ptr

    # ------------------------------------------------------------------
    # staging (first decode)
    # ------------------------------------------------------------------
    def stage_kv_to_cold_pool(
        self,
        req_slot: int,
        gpu_kv_cache: torch.Tensor,
        token_phys_slots: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Copy a request's full KV from the GPU cache into the paged cold pool.

        ``gpu_kv_cache`` is ``[num_layers, num_blocks, block_size, kv_dim]``.
        ``token_phys_slots[i]`` is the physical slot (``block*block_size+offset``)
        of logical token ``i`` in the flattened per-layer cache. Allocates host
        pages for the logical range and scatters into the paged pool via
        ``req_to_host_pool``.
        """
        self.alloc_host_pages(req_slot, 0, num_tokens)
        phys = token_phys_slots[:num_tokens].to(gpu_kv_cache.device, dtype=torch.long)
        host_slots = self.req_to_host_pool[req_slot, :num_tokens].to(torch.long)
        for layer_id in range(self.num_layers):
            layer_cache = gpu_kv_cache[layer_id].reshape(-1, self.kv_dim)
            gathered = layer_cache.index_select(0, phys)  # [num_tokens, kv_dim]
            self.cold_pool[layer_id, host_slots] = gathered.to(self.cold_pool.dtype)

    def load_initial_hot_set(self, req_slot: int, num_tokens: int) -> None:
        """Preload the most recent min(H, num_tokens) tokens into the hot buffer."""
        h = min(self.hot_buffer_size, num_tokens)
        if h == 0:
            return
        start_tok = num_tokens - h  # most-recent window
        tokens = torch.arange(
            start_tok, num_tokens, dtype=torch.int32, device=self.device
        )
        slots = torch.arange(h, dtype=torch.int32, device=self.device)  # hot 0..h-1
        self._tick += 1
        # Seed the fused kernel's recency so its on-device ticks continue strictly
        # above the initial hot set (all initial tokens share this baseline tick).
        self.recency[req_slot] = self._tick
        hot_base = self._hot_base(req_slot)
        src = self.req_to_host_pool[req_slot, start_tok:num_tokens]
        dst = hot_base + slots
        for layer_id in range(self.num_layers):
            # bookkeeping
            self.slot_token[layer_id, req_slot, :h] = tokens
            self.last_used[layer_id, req_slot, :h] = self._tick
            self.token_to_slot[layer_id, req_slot, start_tok:num_tokens] = slots
            # data movement: cold[src] -> hot[hot_base+slot]
            self._run_swap(layer_id, src, dst)

    # ------------------------------------------------------------------
    # per-layer swap-in (decode hot path)
    # ------------------------------------------------------------------
    def plan_swap_for_request(
        self, layer_id: int, req_slot: int, topk_logical: torch.Tensor
    ):
        """Pure bookkeeping: miss-detect + LRU allocate + translate for one request.

        Args:
            layer_id: layer index.
            req_slot: request slot.
            topk_logical: 1-D int tensor of logical top-k positions for this
                request (values in ``0..context_len``; ``-1`` entries are padding
                and are ignored / translated to slot 0).

        Returns:
            (src_locs, dst_locs, translated) all CPU int32 1-D tensors:
              - src_locs/dst_locs: cold-pool / hot-buffer ABSOLUTE rows to swap in
                (paired, one per unique miss token).
              - translated: hot-buffer ABSOLUTE row per entry of ``topk_logical``
                (padding entries map to hot_base, harmless).
        """
        self._tick += 1
        tick = self._tick
        host_locs = self.req_to_host_pool[req_slot]
        hot_base = self._hot_base(req_slot)

        topk = topk_logical.to(torch.int64)
        # A position is valid only if in-range AND backed by an allocated cold-pool
        # page. Anything < 0, beyond the cold pool depth, or outside the request's
        # allocated range (``req_to_host_pool == -1``) is padding/garbage and is
        # ignored (mirrors the sparse gather kernel's ``pos < req_kv_len`` guard).
        # Without the backed check an in-range-but-unallocated position resolves to
        # host row -1 and the gather faults. Clamp so the lookup never faults.
        clamped = topk.clamp(min=0, max=self.max_context_len - 1)
        backed = host_locs[clamped] >= 0
        valid = (topk >= 0) & (topk < self.max_context_len) & backed

        tts = self.token_to_slot[layer_id, req_slot]  # [C]
        slots_for_topk = tts[clamped].to(torch.int64)  # [-1 if miss]

        # hits: refresh recency
        hit = valid & (slots_for_topk >= 0)
        hit_slots = slots_for_topk[hit]
        if hit_slots.numel() > 0:
            self.last_used[layer_id, req_slot, hit_slots] = tick

        # misses: unique logical tokens to bring in
        miss = valid & (slots_for_topk < 0)
        miss_tokens = torch.unique(topk[miss]) if miss.any() else topk.new_empty(0)

        src_locs = topk.new_empty(0, dtype=torch.int32)
        dst_locs = topk.new_empty(0, dtype=torch.int32)
        if miss_tokens.numel() > 0:
            m = miss_tokens.numel()
            lu = self.last_used[layer_id, req_slot]  # [H1]
            # victims = m slots with smallest last_used (empty slots = -1 first).
            chosen = torch.topk(lu, m, largest=False).indices  # [m], slot ids
            evicted_tokens = self.slot_token[layer_id, req_slot, chosen]  # [m]
            # clear reverse map for evicted (still-resident) tokens
            ev_valid = evicted_tokens >= 0
            if ev_valid.any():
                self.token_to_slot[
                    layer_id, req_slot, evicted_tokens[ev_valid].to(torch.int64)
                ] = _EMPTY
            # assign new tokens to chosen slots
            miss_tokens_i32 = miss_tokens.to(torch.int32)
            chosen_i32 = chosen.to(torch.int32)
            self.slot_token[layer_id, req_slot, chosen] = miss_tokens_i32
            self.last_used[layer_id, req_slot, chosen] = tick
            self.token_to_slot[layer_id, req_slot, miss_tokens] = chosen_i32
            src_locs = host_locs[miss_tokens].to(torch.int32)
            dst_locs = (hot_base + chosen_i32).to(torch.int32)

        # translate every top-k entry to its (now-resident) hot ABSOLUTE row
        final_slots = self.token_to_slot[layer_id, req_slot][clamped].to(torch.int64)
        final_slots = torch.where(valid, final_slots, torch.zeros_like(final_slots))
        translated = (hot_base + final_slots).to(torch.int32)
        return src_locs, dst_locs, translated

    def swap_in_for_layer(
        self,
        layer_id: int,
        batch_req_slots: list[int],
        topk_per_req: list[torch.Tensor],
        out_translated: torch.Tensor,
        out_indptr: torch.Tensor,
    ) -> None:
        """Decode hot path for one layer: miss-detect + swap + translate the batch.

        Fills ``out_translated`` (flat, hot-buffer absolute rows) in-place so the
        MLA kernel reads the hot buffer. ``out_indptr[i]:out_indptr[i+1]`` bounds
        request ``i``'s top-k run within ``out_translated`` (same layout as
        ``sparse_kv_indices_buffer``).
        """
        all_src = []
        all_dst = []
        for i, req_slot in enumerate(batch_req_slots):
            start = int(out_indptr[i].item())
            end = int(out_indptr[i + 1].item())
            if end <= start:
                # Padding / inactive query token (empty KV run): skip so its
                # garbage top-k never mutates a real request's LRU state.
                continue
            topk = topk_per_req[i]
            src, dst, translated = self.plan_swap_for_request(layer_id, req_slot, topk)
            n = min(end - start, translated.numel())
            out_translated[start : start + n] = translated[:n].to(out_translated.device)
            if src.numel() > 0:
                all_src.append(src)
                all_dst.append(dst)
        if all_src:
            src = torch.cat(all_src).to(self.device)
            dst = torch.cat(all_dst).to(self.device)
            self._run_swap(layer_id, src, dst)

    def _run_swap(
        self, layer_id: int, src_locs: torch.Tensor, dst_locs: torch.Tensor
    ) -> None:
        """Invoke the HIP gather kernel for one layer's swap-in."""
        if src_locs.numel() == 0:
            return
        from atom.sparsekv.swap_kernel import sparsekv_swap_in

        host_ptr = self._ensure_cold_dev_ptr(layer_id)
        sparsekv_swap_in(
            host_ptr,
            self.hot_buffer[layer_id],
            src_locs.to(torch.int32),
            dst_locs.to(torch.int32),
            self.item_size_bytes,
        )

    # ------------------------------------------------------------------
    # fused GPU hot path (production) — no host sync, CUDAGraph-capturable
    # ------------------------------------------------------------------
    def swap_in_for_layer_fused(
        self,
        layer_id: int,
        topk_logical: torch.Tensor,
        indptr: torch.Tensor,
        req_slots: torch.Tensor,
        out_translated: torch.Tensor,
        record_plan: bool = False,
    ) -> None:
        """Fused miss-detect + LRU evict + swap-in + translate for one layer.

        All arguments are GPU tensors; the kernel does its own per-request work so
        the launch shape is fixed. ``topk_logical`` is ``[n, K]`` (logical top-k per
        decode query token), ``indptr`` bounds each query token's run in
        ``out_translated`` (hot-buffer absolute rows written in place). Requires one
        query token per request (non-MTP); MTP uses the reference path.

        When ``record_plan`` is set (an IndexShare anchor), the kernel also records
        the per-query-token miss list it computed into the coordinator's plan
        buffers so the anchor's shared layers can replay the identical IO.
        """
        host_ptr = self._ensure_cold_dev_ptr(layer_id)
        if record_plan:
            from atom.sparsekv.swap_kernel import sparsekv_swap_and_translate_record

            n = int(req_slots.shape[0])
            sparsekv_swap_and_translate_record(
                host_ptr,
                self.hot_buffer[layer_id],
                topk_logical,
                indptr,
                req_slots,
                self.slot_token[layer_id],
                self.last_used[layer_id],
                self.token_to_slot[layer_id],
                self.recency,
                out_translated,
                self._plan_miss_tok[:n],
                self._plan_miss_slot[:n],
                self._plan_miss_count[:n],
                self._host_locs_arg(),
                self._host_stride,
                self.item_size_bytes,
                self.padded_hot_size,
                self.max_context_len,
                topk_logical.shape[1],
            )
            return

        from atom.sparsekv.swap_kernel import sparsekv_swap_and_translate

        sparsekv_swap_and_translate(
            host_ptr,
            self.hot_buffer[layer_id],
            topk_logical,
            indptr,
            req_slots,
            self.slot_token[layer_id],
            self.last_used[layer_id],
            self.token_to_slot[layer_id],
            self.recency,
            out_translated,
            self._host_locs_arg(),
            self._host_stride,
            self.item_size_bytes,
            self.padded_hot_size,
            self.max_context_len,
            topk_logical.shape[1],
        )

    def backup_into_assigned_fused(
        self,
        layer_id: int,
        anchor_layer: int,
        layer_kv: torch.Tensor,
        src_slots: torch.Tensor,
        req_slots: torch.Tensor,
        logical_pos: torch.Tensor,
    ) -> None:
        """Backup a shared layer's new token into the anchor-assigned hot slot.

        The anchor's swap already assigned ``logical_pos`` a hot slot in the shared
        (anchor's) slot table; this writes this layer's freshly generated KV into
        that same slot and the cold pool. Data only — no LRU/recency mutation, so
        the group's shared slot table stays authoritative on the anchor.
        """
        from atom.sparsekv.swap_kernel import sparsekv_backup_into_assigned

        host_ptr = self._ensure_cold_dev_ptr(layer_id)
        sparsekv_backup_into_assigned(
            host_ptr,
            self.hot_buffer[layer_id],
            layer_kv.reshape(-1, self.kv_dim),
            src_slots,
            req_slots,
            logical_pos,
            self.token_to_slot[anchor_layer],
            self._host_locs_arg(),
            self._host_stride,
            self.item_size_bytes,
            self.padded_hot_size,
            self.max_context_len,
        )

    def backup_new_tokens_fused(
        self,
        layer_id: int,
        layer_kv: torch.Tensor,
        src_slots: torch.Tensor,
        req_slots: torch.Tensor,
        logical_pos: torch.Tensor,
    ) -> None:
        """Persist every decode query token's freshly written KV for one layer.

        ``layer_kv`` is this layer's KV cache flattened to ``[num_slots, kv_dim]``;
        ``src_slots[i]`` is the physical row of query token ``i`` (its slot_mapping
        entry). Writes cold pool at ``logical_pos[i]`` and a fresh hot slot. All GPU.
        """
        from atom.sparsekv.swap_kernel import sparsekv_backup_new_token

        host_ptr = self._ensure_cold_dev_ptr(layer_id)
        sparsekv_backup_new_token(
            host_ptr,
            self.hot_buffer[layer_id],
            layer_kv.reshape(-1, self.kv_dim),
            src_slots,
            req_slots,
            logical_pos,
            self.slot_token[layer_id],
            self.last_used[layer_id],
            self.token_to_slot[layer_id],
            self.recency,
            self._host_locs_arg(),
            self._host_stride,
            self.item_size_bytes,
            self.padded_hot_size,
            self.max_context_len,
        )

    # ------------------------------------------------------------------
    # new-token backup (post decode step)
    # ------------------------------------------------------------------
    def backup_new_token(
        self,
        req_slot: int,
        layer_id: int,
        new_token_kv: torch.Tensor,
        logical_pos: int,
    ) -> None:
        """Persist a freshly generated token's KV to cold pool + hot buffer.

        Writes cold pool at ``logical_pos`` and allocates a hot slot for it with
        high recency (it will likely be selected next step).
        """
        assert logical_pos < self.max_context_len
        self.alloc_host_pages(req_slot, logical_pos, 1)
        cold_row = int(self.req_to_host_pool[req_slot, logical_pos].item())
        self.cold_pool[layer_id, cold_row].copy_(
            new_token_kv.reshape(self.kv_dim).to(self.cold_pool.dtype)
        )
        self._tick += 1
        tick = self._tick
        lu = self.last_used[layer_id, req_slot]
        slot = int(torch.topk(lu, 1, largest=False).indices.item())
        evicted = int(self.slot_token[layer_id, req_slot, slot].item())
        if evicted >= 0:
            self.token_to_slot[layer_id, req_slot, evicted] = _EMPTY
        self.slot_token[layer_id, req_slot, slot] = logical_pos
        self.last_used[layer_id, req_slot, slot] = tick
        self.token_to_slot[layer_id, req_slot, logical_pos] = slot
        hot_base = self._hot_base(req_slot)
        self.hot_buffer[layer_id, hot_base + slot].copy_(
            new_token_kv.reshape(self.kv_dim).to(self.hot_buffer.dtype)
        )
        if logical_pos + 1 > int(self.context_len[req_slot].item()):
            self.context_len[req_slot] = logical_pos + 1
