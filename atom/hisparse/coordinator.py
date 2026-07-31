# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""HiSparse coordinator: CPU cold pool + GPU hot buffer for DSA decode.

GLM-5.2 (DSA) decode only reads the indexer's top-k tokens each step, but the
full KV cache otherwise occupies GPU HBM. HiSparse keeps the complete KV in a
CPU pinned cold pool and only a fixed-size hot buffer per request on the GPU.
Each decode step, for every layer: miss-detect the top-k against the resident
hot set, evict the least-recently-used slots, swap the missing tokens in from
the cold pool, and translate the top-k into hot-buffer slots so MLA attention
reads the hot buffer.

Index domain is LOGICAL: miss-detect runs on the indexer's per-request logical
top-k positions (``0..context_len``), not physical paged-KV slots. The cold pool
is stored densely by logical position; ``translate`` maps logical top-k to hot
slots. This matches the design doc Appendix A experiments (per-token LRU on
logical top-k indices).

The pure bookkeeping (miss-detect + LRU allocate + translate) is separated from
the GPU data movement so it is testable without a GPU. Only :meth:`swap_in_for_layer`
and the staging/backup helpers touch the swap kernel.
"""

import logging

import torch

logger = logging.getLogger("atom")

_EMPTY = -1  # sentinel for empty slot / non-resident token


class HiSparseCoordinator:
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

        # CPU cold pool: [num_layers, R * C, kv_dim], pinned so the swap kernel
        # can gather from it directly (via a device-mapped pointer, see below).
        self.cold_pool = torch.zeros(
            (num_layers, R * C, kv_dim), dtype=kv_dtype, pin_memory=True
        )
        # GPU hot buffer: [num_layers, R * H1, kv_dim].
        self.hot_buffer = torch.zeros(
            (num_layers, R * H1, kv_dim), dtype=kv_dtype, device=self.device
        )

        # Device-mapped cold-pool pointer per layer (xnack- needs the translation;
        # a raw host VA faults). Filled lazily on first swap.
        self._cold_dev_ptr: list[int | None] = [None] * num_layers

        # Per (layer, request, slot) bookkeeping — CPU, small (4 bytes/slot).
        # slot_token: logical token ID resident in each hot slot (-1 empty).
        self.slot_token = torch.full((num_layers, R, H1), _EMPTY, dtype=torch.int32)
        # last_used: recency tick per slot (-1 empty); LRU victim = smallest.
        self.last_used = torch.full((num_layers, R, H1), _EMPTY, dtype=torch.int64)
        # token_to_slot: reverse map logical token -> hot slot (-1 not resident).
        self.token_to_slot = torch.full((num_layers, R, C), _EMPTY, dtype=torch.int32)

        # Per-request context length (logical tokens currently staged).
        self.context_len = torch.zeros(max_num_seqs, dtype=torch.int64)
        self.slot_active = torch.zeros(max_num_seqs, dtype=torch.bool)

        # Stable request-id -> slot map (worker-side; keyed by ScheduledBatch
        # req_ids so the same request lands in the same cold-pool region every
        # step) and the free-slot pool.
        self._reqid_to_slot: dict[int, int] = {}
        self._free_slots: list[int] = list(range(max_num_seqs))

        # Monotonic recency tick, advanced each swap/backup.
        self._tick = 0

        logger.info(
            "HiSparseCoordinator: layers=%d max_seqs=%d hot=%d(+1) max_ctx=%d "
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
        return req_slot

    def unregister_request(self, req_slot: int) -> None:
        """Release a request slot and clear its state."""
        self.slot_active[req_slot] = False
        self.context_len[req_slot] = 0
        self.slot_token[:, req_slot, :] = _EMPTY
        self.last_used[:, req_slot, :] = _EMPTY
        self.token_to_slot[:, req_slot, :] = _EMPTY

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
                "HiSparse: no free request slots "
                f"(max_num_seqs={self.max_num_seqs} exhausted)"
            )
        slot = self._free_slots.pop()
        self._reqid_to_slot[req_id] = slot
        self.register_request(slot, context_len)
        return slot

    def release(self, req_id: int) -> None:
        """Free the slot held by a request id (idempotent)."""
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
        """
        active = set(active_req_ids)
        for req_id in list(self._reqid_to_slot.keys()):
            if req_id not in active:
                self.release(req_id)

    # ------------------------------------------------------------------
    # cold-pool addressing
    # ------------------------------------------------------------------
    def _cold_base(self, req_slot: int) -> int:
        return req_slot * self.max_context_len

    def _hot_base(self, req_slot: int) -> int:
        return req_slot * self.padded_hot_size

    def _ensure_cold_dev_ptr(self, layer_id: int) -> int:
        """Return (and cache) the device-mapped pointer for one layer's cold pool."""
        ptr = self._cold_dev_ptr[layer_id]
        if ptr is None:
            from atom.hisparse.swap_kernel import host_get_device_pointer

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
        """Copy a request's full KV from the GPU cache into the CPU cold pool.

        ``gpu_kv_cache`` is ``[num_layers, num_blocks, block_size, kv_dim]``.
        ``token_phys_slots[i]`` is the physical slot (``block*block_size+offset``)
        of logical token ``i`` in the flattened per-layer cache. The cold pool
        stores tokens densely by logical position.
        """
        phys = token_phys_slots[:num_tokens].to(gpu_kv_cache.device, dtype=torch.long)
        base = self._cold_base(req_slot)
        for layer_id in range(self.num_layers):
            layer_cache = gpu_kv_cache[layer_id].reshape(-1, self.kv_dim)
            gathered = layer_cache.index_select(0, phys)  # [num_tokens, kv_dim]
            self.cold_pool[layer_id, base : base + num_tokens].copy_(
                gathered, non_blocking=False
            )

    def load_initial_hot_set(self, req_slot: int, num_tokens: int) -> None:
        """Preload the most recent min(H, num_tokens) tokens into the hot buffer."""
        h = min(self.hot_buffer_size, num_tokens)
        if h == 0:
            return
        start_tok = num_tokens - h  # most-recent window
        tokens = torch.arange(start_tok, num_tokens, dtype=torch.int32)
        slots = torch.arange(h, dtype=torch.int32)  # hot slots 0..h-1
        self._tick += 1
        cold_base = self._cold_base(req_slot)
        hot_base = self._hot_base(req_slot)
        for layer_id in range(self.num_layers):
            # bookkeeping
            self.slot_token[layer_id, req_slot, :h] = tokens
            self.last_used[layer_id, req_slot, :h] = self._tick
            self.token_to_slot[layer_id, req_slot, start_tok:num_tokens] = slots
            # data movement: cold[cold_base+tok] -> hot[hot_base+slot]
            src = (cold_base + tokens).to(self.device)
            dst = (hot_base + slots).to(self.device)
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
        cold_base = self._cold_base(req_slot)
        hot_base = self._hot_base(req_slot)

        topk = topk_logical.to(torch.int64)
        # A position is valid only if in-range; anything < 0 or beyond the cold
        # pool depth is padding/garbage and is ignored (mirrors the sparse gather
        # kernel's ``pos < req_kv_len`` guard). Clamp so the gather never faults.
        valid = (topk >= 0) & (topk < self.max_context_len)
        clamped = topk.clamp(min=0, max=self.max_context_len - 1)

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
            src_locs = (cold_base + miss_tokens_i32).to(torch.int32)
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
        from atom.hisparse.swap_kernel import hisparse_swap_in

        host_ptr = self._ensure_cold_dev_ptr(layer_id)
        hisparse_swap_in(
            host_ptr,
            self.hot_buffer[layer_id],
            src_locs.to(torch.int32),
            dst_locs.to(torch.int32),
            self.item_size_bytes,
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
        cold_base = self._cold_base(req_slot)
        self.cold_pool[layer_id, cold_base + logical_pos].copy_(
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
