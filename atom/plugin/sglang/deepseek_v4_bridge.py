from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger("atom.plugin.sglang.deepseek_v4_bridge")

ATOM_DEEPSEEK_V4_BLOCK_SIZE = 128


def _debug_enabled() -> bool:
    return os.environ.get("ATOM_SGLANG_V4_DEBUG") == "1"


def _aligned_index_dim(index_head_dim: int) -> int:
    # extra 4 bytes for scale, then 16-byte alignment.
    return ((int(index_head_dim) + 4 + 15) // 16) * 16


def _layer_counts(compress_ratios) -> tuple[list[int], int, int, int]:
    ratios = [int(r) for r in (compress_ratios or [])]
    dense = sum(1 for r in ratios if r == 0)
    csa = sum(1 for r in ratios if r == 4)
    hca = sum(1 for r in ratios if r == 128)
    return ratios, dense, csa, hca


try:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
except Exception:  # pragma: no cover - SGLang import-time fallback
    BaseSWAKVPool = object


class ATOMDeepSeekV4ProxyKVPool(BaseSWAKVPool):
    """SGLang-visible proxy KV pool whose bytes are owned by ATOM V4.

    SGLang still allocates full/SWA token indices through its regular SWA
    allocator, but the physical tensor here is a raw byte arena.  We carve that
    arena into the views expected by ATOM's native DeepSeek-V4 attention:
    per-layer SWA prefix, optional CSA/HCA main KV tail, and CSA indexer tail.
    """

    def __init__(
        self,
        max_num_reqs: int,
        swa_size: int,
        c4_size: int,
        c128_size: int,
        c4_state_pool_size: int,
        c128_state_pool_size: int,
        page_size: int,
        swa_page_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        compression_ratios: list[int],
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
    ) -> None:
        del c4_state_pool_size, c128_state_pool_size, dtype, state_dtype
        del enable_memory_saver, enable_hisparse

        self.max_num_reqs = int(max_num_reqs)
        self.swa_size = int(swa_size)
        self.c4_size = int(c4_size)
        self.c128_size = int(c128_size)
        # SGLang worker/scheduler code expects TokenToKVPool-like objects to
        # expose `size` as the externally visible token capacity.  The proxy
        # owns multiple internal ATOM views, but the SWA/full token capacity is
        # the right public capacity for scheduling/accounting.
        self.size = self.swa_size
        self.size_swa = self.swa_size
        self.page_size = int(page_size)
        self.swa_page_size = int(swa_page_size)
        self.device = device
        self.start_layer = 0 if start_layer is None else int(start_layer)
        self.end_layer = int(end_layer) if end_layer is not None else int(layer_num)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.indexer_head_dim = int(indexer_head_dim)
        self.index_dim = _aligned_index_dim(self.indexer_head_dim)
        self.compression_ratios = [int(r) for r in compression_ratios]
        self.stage_ratios = self.compression_ratios[self.start_layer : self.end_layer]
        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None

        # SGLang's SWA allocator only needs these attributes to exist so it can
        # create full/SWA index allocators and then call register_mapping().
        self.full_kv_pool = None
        self.swa_kv_pool = None

        self.num_slots = max(1, self.max_num_reqs)
        # SGLang's DSV4 allocator is initialized with page_size/swa_page_size=256
        # for paged-SWA bookkeeping, but ATOM V4-Pro attention uses a 128-token
        # SWA ring/window.  Keep the SGLang-facing size above intact and size all
        # ATOM cache views + metadata with the native V4 window.
        self.window_size = ATOM_DEEPSEEK_V4_BLOCK_SIZE
        # In the ATOM bridge layout one original-token block contributes one
        # HCA entry, so the HCA compressed-entry count is the physical block
        # count for the unified tails.
        self.num_blocks = max(1, self.c128_size)

        total_bytes = self._compute_raw_bytes()
        self.raw_arena = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        # SGLang's /get_internal_state path reports
        # token_to_kv_pool_allocator.get_kvcache().mem_usage.  The ATOM proxy
        # owns one raw arena instead of regular SGLang KV buffers, so report
        # the arena footprint in GiB to keep that API compatible.
        self.mem_usage = total_bytes / (1 << 30)
        self.views = self._slice_views()
        self.is_atom_v4_proxy_pool = True

        logger.info(
            "Initialized ATOM DeepSeek-V4 SGLang proxy KV pool: "
            "slots=%s blocks=%s layers=%s raw=%.2f MiB",
            self.num_slots,
            self.num_blocks,
            len(self.stage_ratios),
            total_bytes / (1 << 20),
        )

    def _compute_raw_bytes(self) -> int:
        total = 0
        swa_bytes = self.num_slots * self.window_size * self.head_dim * 2
        for ratio in self.stage_ratios:
            total += swa_bytes
            if ratio == 4:
                k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
                total += self.num_blocks * k * self.head_dim * 2
                total += self.num_blocks * k * self.index_dim
            elif ratio == 128:
                k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128
                total += self.num_blocks * k * self.head_dim * 2
        return max(1, total)

    def _take(self, offset: int, nbytes: int) -> torch.Tensor:
        end = offset + nbytes
        if end > self.raw_arena.numel():
            raise RuntimeError(
                f"ATOM V4 proxy arena too small: need {end}, have {self.raw_arena.numel()}"
            )
        return self.raw_arena[offset:end]

    def _slice_views(self) -> dict[str, list[torch.Tensor]]:
        try:
            from aiter import dtypes

            fp8_dtype = dtypes.fp8
        except Exception:
            fp8_dtype = torch.float8_e4m3fnuz

        offset = 0
        unified: list[torch.Tensor] = []
        swa: list[torch.Tensor] = []
        csa_main: list[torch.Tensor] = []
        csa_indexer: list[torch.Tensor] = []
        hca_main: list[torch.Tensor] = []

        for ratio in self.stage_ratios:
            layer_start = offset
            swa_bytes = self.num_slots * self.window_size * self.head_dim * 2
            swa_view = (
                self._take(offset, swa_bytes)
                .view(torch.bfloat16)
                .view(self.num_slots, self.window_size, self.head_dim)
            )
            offset += swa_bytes
            swa.append(swa_view)

            if ratio == 4:
                k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
                main_bytes = self.num_blocks * k * self.head_dim * 2
                main = (
                    self._take(offset, main_bytes)
                    .view(torch.bfloat16)
                    .as_strided(
                        size=(self.num_blocks, k, self.head_dim),
                        stride=(k * self.head_dim, self.head_dim, 1),
                    )
                )
                offset += main_bytes
                unified.append(
                    self.raw_arena[layer_start:offset]
                    .view(torch.bfloat16)
                    .view(
                        self.num_slots * self.window_size + self.num_blocks * k,
                        self.head_dim,
                    )
                )
                idx_bytes = self.num_blocks * k * self.index_dim
                idx = (
                    self._take(offset, idx_bytes)
                    .view(fp8_dtype)
                    .as_strided(
                        size=(self.num_blocks, k, self.index_dim),
                        stride=(k * self.index_dim, self.index_dim, 1),
                    )
                )
                offset += idx_bytes
                csa_main.append(main)
                csa_indexer.append(idx)
            elif ratio == 128:
                k = ATOM_DEEPSEEK_V4_BLOCK_SIZE // 128
                main_bytes = self.num_blocks * k * self.head_dim * 2
                main = (
                    self._take(offset, main_bytes)
                    .view(torch.bfloat16)
                    .as_strided(
                        size=(self.num_blocks, k, self.head_dim),
                        stride=(k * self.head_dim, self.head_dim, 1),
                    )
                )
                offset += main_bytes
                unified.append(
                    self.raw_arena[layer_start:offset]
                    .view(torch.bfloat16)
                    .view(
                        self.num_slots * self.window_size + self.num_blocks * k,
                        self.head_dim,
                    )
                )
                hca_main.append(main)
            else:
                unified.append(
                    swa_view.view(self.num_slots * self.window_size, self.head_dim)
                )

        return {
            "unified": unified,
            "swa": swa,
            "csa_main": csa_main,
            "csa_indexer": csa_indexer,
            "hca_main": hca_main,
        }

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        if self.full_to_swa_index_mapping is None:
            raise RuntimeError("ATOM V4 proxy pool has no full->SWA mapping")
        return self.full_to_swa_index_mapping[kv_indices]

    @staticmethod
    def _block_pairs(tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> torch.Tensor:
        """Convert SGLang token relocation into unique V4 block relocation pairs.

        SGLang calls move_kv_cache with logical token locations.  ATOM V4 stores
        persistent CSA/HCA history by 128-token blocks, so prefix-cache
        relocation must first collapse token locs to block ids and drop no-op
        copies within the same block.
        """
        if tgt_loc.numel() != src_loc.numel():
            raise ValueError(
                "ATOM V4 KV relocation expects matching target/source sizes: "
                f"{tgt_loc.numel()} vs {src_loc.numel()}"
            )
        if tgt_loc.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)

        tgt = tgt_loc.reshape(-1).to(dtype=torch.int64)
        src = src_loc.reshape(-1).to(dtype=torch.int64)
        valid = (tgt >= 0) & (src >= 0)
        if not bool(valid.any().item()):
            return torch.empty((0, 2), dtype=torch.long)

        tgt_blocks = torch.div(
            tgt[valid], ATOM_DEEPSEEK_V4_BLOCK_SIZE, rounding_mode="floor"
        )
        src_blocks = torch.div(
            src[valid], ATOM_DEEPSEEK_V4_BLOCK_SIZE, rounding_mode="floor"
        )
        keep = tgt_blocks != src_blocks
        if not bool(keep.any().item()):
            return torch.empty((0, 2), dtype=torch.long)

        pairs = torch.stack([tgt_blocks[keep], src_blocks[keep]], dim=1)
        return torch.unique(pairs.cpu(), dim=0)

    @staticmethod
    def _copy_block_views(views: list[torch.Tensor], block_pairs: torch.Tensor) -> None:
        """Copy compressed KV blocks between proxy views during radix relocation."""
        if not views or block_pairs.numel() == 0:
            return

        tgt_blocks = block_pairs[:, 0]
        src_blocks = block_pairs[:, 1]
        for view in views:
            num_blocks = int(view.shape[0])
            valid = (
                (src_blocks >= 0)
                & (src_blocks < num_blocks)
                & (tgt_blocks >= 0)
                & (tgt_blocks < num_blocks)
            )
            if not bool(valid.any().item()):
                continue
            src_idx = src_blocks[valid].to(device=view.device)
            tgt_idx = tgt_blocks[valid].to(device=view.device)
            view.index_copy_(0, tgt_idx, view.index_select(0, src_idx).clone())

    def set_swa_loc(self, loc: torch.Tensor) -> None:
        # SGLang 0.5.12 requires BaseSWAKVPool subclasses to expose this hook.
        # DSV4 pools do not use the generic precomputed SWA location path, and
        # ATOM writes the proxy arena through its own bridge metadata.
        pass

    def get_state_buf_infos(self):
        return ([], [], [])

    def get_contiguous_buf_infos(self):
        return ([self.raw_arena.data_ptr()], [self.raw_arena.nbytes], [1])

    def get_kv_buffer(self, layer_id: int):
        raise NotImplementedError("ATOM V4 proxy pool is not a regular SGLang KV pool")

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError("ATOM V4 proxy pool is written by ATOM kernels")

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError("ATOM V4 proxy pool is not a regular SGLang KV pool")

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError("ATOM V4 proxy pool is not a regular SGLang KV pool")

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        """Implement the KV relocation hook required when SGLang radix cache is on.

        Prefix cache lets SGLang move logical KV locations after cached blocks are
        inserted or reused.  The proxy pool mirrors that move into ATOM's
        block-addressed CSA/HCA views and remaps the first-block -> state-slot
        table so the SWA ring and compressor state continue to belong to the same
        logical request after relocation.
        """
        block_pairs = self._block_pairs(tgt_loc, src_loc)
        if block_pairs.numel() == 0:
            return

        # SGLang relocates by full-token locations.  ATOM V4 stores persistent
        # compressed history by 128-token blocks, while SWA history lives in a
        # per-request state slot keyed by the request's first block.
        self._copy_block_views(self.views["csa_main"], block_pairs)
        self._copy_block_views(self.views["csa_indexer"], block_pairs)
        self._copy_block_views(self.views["hca_main"], block_pairs)

        allocator = getattr(self, "_atom_v4_slot_allocator", None)
        if allocator is not None:
            allocator.remap_blocks(block_pairs[:, 1], block_pairs[:, 0])

        if _debug_enabled():
            logger.info(
                "ATOM V4 proxy relocated %d KV blocks for SGLang radix cache",
                block_pairs.shape[0],
            )


def install_deepseek_v4_proxy_pool_patch() -> None:
    """Patch SGLang's DSV4 pool constructor before ModelRunner._init_pools().

    This makes SGLang allocate the ATOM proxy pool instead of the stock DSV4 KV
    pool, while leaving SGLang's scheduler/radix-cache code unchanged.  The proxy
    still satisfies SGLang's TokenToKVPool contract but exposes ATOM V4's SWA,
    CSA, HCA, and indexer views to the model.
    """

    import sglang.srt.model_executor.model_runner_kv_cache_mixin as mixin
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as dsv4_pool

    if getattr(mixin, "DeepSeekV4TokenToKVPool", None) is ATOMDeepSeekV4ProxyKVPool:
        return
    mixin.DeepSeekV4TokenToKVPool = ATOMDeepSeekV4ProxyKVPool
    dsv4_pool.ATOMDeepSeekV4ProxyKVPool = ATOMDeepSeekV4ProxyKVPool
    logger.info("Installed ATOM DeepSeek-V4 proxy KV pool patch for SGLang")


def _bind_compressor_state(
    compressor,
    kv_cache: torch.Tensor,
    num_slots: int,
    *,
    is_indexer: bool = False,
    head_dim: Optional[int] = None,
) -> None:
    compressor.kv_state = torch.zeros(
        (num_slots, *compressor.kv_state.shape[1:]),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    compressor.score_state = torch.full(
        (num_slots, *compressor.score_state.shape[1:]),
        float("-inf"),
        dtype=torch.float32,
        device=kv_cache.device,
    )
    compressor.kv_cache = kv_cache
    if is_indexer:
        nb, k1, aligned_dim = kv_cache.shape
        if head_dim is None:
            raise ValueError("indexer compressor binding requires explicit head_dim")
        block_fp32_stride = (k1 * aligned_dim) // 4
        scale_fp32_offset = (k1 * head_dim) // 4
        compressor.cache_scale = (
            kv_cache.view(torch.float32)
            .view(-1)
            .as_strided(
                size=(nb, k1),
                stride=(block_fp32_stride, 1),
                storage_offset=scale_fp32_offset,
            )
        )
    else:
        compressor.cache_scale = None


def bind_deepseek_v4_proxy_cache_views(model, proxy_pool: Any) -> bool:
    """Bind the SGLang-visible proxy arena to ATOM V4 attention modules.

    Prefix cache stores and reuses SGLang logical KV indices, but the actual V4
    kernels read ATOM-owned views.  Binding once per arena keeps both sides
    looking at the same storage: SGLang manages logical locs, ATOM kernels read
    and write the carved SWA/CSA/HCA tensors.
    """
    if not getattr(proxy_pool, "is_atom_v4_proxy_pool", False):
        return False
    ptr = proxy_pool.raw_arena.untyped_storage().data_ptr()
    if getattr(model, "_atom_sglang_v4_proxy_cache_ptr", None) == ptr:
        return True

    csa_i = 0
    hca_i = 0
    for local_layer_id, block in enumerate(model.model.layers):
        attn = block.attn
        ratio = int(attn.compress_ratio)
        attn.unified_kv = proxy_pool.views["unified"][local_layer_id]
        attn.swa_kv = proxy_pool.views["swa"][local_layer_id]
        if ratio == 4:
            _bind_compressor_state(
                attn.compressor,
                proxy_pool.views["csa_main"][csa_i],
                proxy_pool.num_slots,
            )
            attn.indexer.kv_cache = proxy_pool.views["csa_indexer"][csa_i]
            attn.indexer._max_model_len_idx = max(
                1, proxy_pool.num_blocks * ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4
            )
            _bind_compressor_state(
                attn.indexer.compressor,
                proxy_pool.views["csa_indexer"][csa_i],
                proxy_pool.num_slots,
                is_indexer=True,
                head_dim=proxy_pool.indexer_head_dim,
            )
            csa_i += 1
        elif ratio == 128:
            _bind_compressor_state(
                attn.compressor,
                proxy_pool.views["hca_main"][hca_i],
                proxy_pool.num_slots,
            )
            hca_i += 1

    model._atom_sglang_v4_proxy_cache_ptr = ptr
    model._atom_v4_meta_params = SimpleNamespace(
        num_slots=proxy_pool.num_slots,
        window_size=proxy_pool.window_size,
        cs=proxy_pool.window_size,
        index_topk=int(getattr(model.args, "index_topk", 1024)),
    )
    logger.info("Bound ATOM DeepSeek-V4 proxy cache views to model")
    return True


class _V4StateSlotAllocator:
    """Track which ATOM per-request state slot belongs to each first KV block.

    SGLang radix cache identifies a cached request by logical KV blocks, while
    ATOM V4 keeps SWA ring and compressor state in a separate per-request slot.
    This allocator bridges the two: fresh prefills get/reset a slot, prefix hits
    reuse the slot mapped from their first block, and KV relocation updates that
    mapping.
    """

    def __init__(self, num_slots: int):
        self.num_slots = max(1, int(num_slots))
        self._block_to_slot: dict[int, int] = {}
        self._slot_to_block: list[int] = [-1] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0

    def assign(self, first_block_ids, fresh_mask) -> tuple[np.ndarray, set[int]]:
        """Return state slots for the batch and identify slots that need reset."""
        self._step += 1
        fb = (
            first_block_ids.tolist()
            if hasattr(first_block_ids, "tolist")
            else list(first_block_ids)
        )
        fresh = (
            fresh_mask.tolist() if hasattr(fresh_mask, "tolist") else list(fresh_mask)
        )
        active = set(int(x) for x in fb)
        slots = []
        reset: set[int] = set()
        for block_id, is_fresh in zip(fb, fresh):
            block_id = int(block_id)
            slot = self._block_to_slot.get(block_id)
            if slot is None:
                slot = self._acquire(active)
                self._block_to_slot[block_id] = slot
                self._slot_to_block[slot] = block_id
                reset.add(slot)
            elif bool(is_fresh):
                reset.add(slot)
            self._last_seen[slot] = self._step
            slots.append(slot)
        return np.asarray(slots, dtype=np.int32), reset

    def remap_blocks(self, src_block_ids, tgt_block_ids) -> None:
        """Move first-block -> state-slot ownership after SGLang KV relocation.

        Without this remap, a radix-cache relocation could copy CSA/HCA blocks to
        the new logical block id while decode/prefix prefill still looked up the
        SWA ring via the old first block.
        """
        src = (
            src_block_ids.tolist()
            if hasattr(src_block_ids, "tolist")
            else list(src_block_ids)
        )
        tgt = (
            tgt_block_ids.tolist()
            if hasattr(tgt_block_ids, "tolist")
            else list(tgt_block_ids)
        )
        updates: dict[int, int] = {}
        for src_block, tgt_block in zip(src, tgt):
            src_block = int(src_block)
            tgt_block = int(tgt_block)
            if src_block == tgt_block:
                continue
            slot = self._block_to_slot.get(src_block)
            if slot is not None:
                updates[tgt_block] = slot
        if not updates:
            return

        moved_slots = set(updates.values())
        for block, slot in list(self._block_to_slot.items()):
            if slot in moved_slots or block in updates:
                self._block_to_slot.pop(block, None)
                if slot not in moved_slots:
                    self._slot_to_block[slot] = -1
                    if slot not in self._free:
                        self._free.append(slot)

        for block, slot in updates.items():
            self._block_to_slot[block] = slot
            self._slot_to_block[slot] = block
            if slot in self._free:
                self._free.remove(slot)

    def _acquire(self, active: set[int]) -> int:
        if self._free:
            return self._free.pop()
        victim = 0
        victim_seen = None
        for slot, block_id in enumerate(self._slot_to_block):
            if block_id in active:
                continue
            if victim_seen is None or self._last_seen[slot] < victim_seen:
                victim = slot
                victim_seen = self._last_seen[slot]
        old = self._slot_to_block[victim]
        if old >= 0:
            self._block_to_slot.pop(old, None)
        self._slot_to_block[victim] = -1
        return victim


def _counts_to_indptr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int32)
    out[1:] = np.cumsum(counts, dtype=np.int32)
    return out


def _make_compress_plans(extend_lens_cpu, context_lens_cpu, device):
    from atom.model_ops.v4_kernels import make_compress_plans
    from atom.utils import CpuGpuBuffer

    total = max(1, int(np.asarray(extend_lens_cpu, dtype=np.int32).sum()))
    plan_buffers = {
        ratio: {
            "compress": CpuGpuBuffer(total, 4, dtype=torch.int32, device=device),
            "write": CpuGpuBuffer(
                total * max(1, ratio), 4, dtype=torch.int32, device=device
            ),
        }
        for ratio in (4, 128)
    }
    plans = make_compress_plans(
        np.ascontiguousarray(extend_lens_cpu, dtype=np.int32),
        np.ascontiguousarray(context_lens_cpu, dtype=np.int32),
        [(4, True), (128, False)],
        plan_buffers=plan_buffers,
        decode_capacity_per_ratio=None,
    )
    for plan in plans.values():
        plan.write_plan_gpu = plan.write_plan_gpu[: plan.num_write]
    return plans


class _V4SGLangDecodeGraphBuffers:
    """Persistent fixed-address decode metadata buffers for SGLang cuda graph.

    SGLang captures decode graphs once per padded batch size.  ATOM V4 attention
    kernels then replay using the tensor addresses captured during the warmup
    forward, so replay must refresh buffer contents in place instead of swapping
    metadata tensors.  This mirrors the vLLM bridge's decode persistent path.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        max_decode_tokens: int,
        window: int,
        index_topk: int,
        max_committed_hca: int,
        max_blocks: int,
        device: torch.device,
    ) -> None:
        from atom.utils import CpuGpuBuffer

        self.device = device
        self.num_slots = max(1, int(num_slots))
        self.max_decode_tokens = max(1, int(max_decode_tokens))
        self.window = int(window)
        self.index_topk = int(index_topk)
        self.max_committed_hca = max(1, int(max_committed_hca))
        self.max_blocks = max(1, int(max_blocks))

        def i32(*shape):
            return CpuGpuBuffer(*shape, dtype=torch.int32, device=device)

        t = self.max_decode_tokens
        s = self.num_slots
        win = self.window
        topk = self.index_topk
        hca = self.max_committed_hca

        self.cu_q = i32(t + 1)
        self.state_slot = i32(s)
        self.n_csa = i32(s)
        self.n_hca = i32(s)
        self.batch_id = CpuGpuBuffer(t, dtype=torch.int32, device=device)
        self.block_tables = i32(s, self.max_blocks)
        self.indptr_swa = i32(t + 1)
        self.indptr_csa = i32(t + 1)
        self.indptr_hca = i32(t + 1)
        self.idx_swa = i32(t * max(1, win))
        self.idx_csa = i32(t * max(1, win + topk))
        self.idx_hca = i32(t * max(1, win + hca))

        self.plan_buffers = {
            4: {
                "compress": i32(max(1, s), 4),
                "write": i32(max(1, s * 8), 4),
            },
            128: {
                "compress": i32(max(1, s), 4),
                "write": i32(max(1, s * 128), 4),
            },
        }
        self.decode_compress_cap = {4: max(1, s), 128: max(1, s)}

    def stage(self, buf, arr_np, n: Optional[int] = None):
        n = int(arr_np.shape[0]) if n is None else int(n)
        assert (
            n <= buf.np.shape[0]
        ), f"V4 graph buffer too small: need {n}, have {buf.np.shape[0]}"
        if n:
            buf.np[:n] = arr_np[:n]
        return buf.copy_to_gpu(n)


def _make_decode_graph_compress_plans(extend_lens_cpu, context_lens_cpu, bufs):
    from atom.model_ops.v4_kernels.compress_plan import make_compress_plans

    return make_compress_plans(
        np.ascontiguousarray(extend_lens_cpu, dtype=np.int32),
        np.ascontiguousarray(context_lens_cpu, dtype=np.int32),
        [(4, True), (128, False)],
        plan_buffers=bufs.plan_buffers,
        decode_capacity_per_ratio=bufs.decode_compress_cap,
    )


def _get_extend_lens_cpu(
    forward_batch, positions: Optional[torch.Tensor] = None
) -> Optional[np.ndarray]:
    """Read per-request suffix lengths from SGLang ForwardBatch.

    Prefix-cache hits have `seq_lens = cached prefix + suffix`, but ATOM's
    prefill metadata needs only the suffix token counts to build cu_seqlens_q and
    batch_id_per_token.  Different SGLang paths expose that length under slightly
    different fields, so this helper normalizes them.
    """
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is not None:
        return np.asarray(extend_lens, dtype=np.int32)

    extend_lens_t = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens_t is not None:
        return extend_lens_t.detach().cpu().numpy().astype(np.int32)

    extend_start_loc = getattr(forward_batch, "extend_start_loc", None)
    if extend_start_loc is None or positions is None:
        return None

    return np.diff(
        torch.nn.functional.pad(extend_start_loc, (0, 1), value=positions.numel())
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )


def _infer_atom_attn_state(forward_batch) -> Any:
    """Map SGLang forward mode to the ATOM V4 attention state.

    The important prefix-cache case is a prefill batch with non-zero
    `extend_prefix_lens`: SGLang is only forwarding the suffix, so ATOM must use
    PREFILL_PREFIX and read prefix_swa/prefix_csa/prefix_hca instead of treating
    the batch as a fresh PREFILL_NATIVE from position 0.
    """
    from atom.utils.forward_context import AttnState

    mode = forward_batch.forward_mode
    if mode.is_decode_or_idle():
        return AttnState.DECODE

    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix_lens is None:
        prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
    if prefix_lens is None:
        return AttnState.PREFILL_NATIVE

    batch_size = int(forward_batch.batch_size)
    if torch.is_tensor(prefix_lens):
        has_prefix = bool(prefix_lens[:batch_size].gt(0).any().item())
    else:
        has_prefix = any(x > 0 for x in prefix_lens[:batch_size])
    if has_prefix:
        return AttnState.PREFILL_PREFIX
    return AttnState.PREFILL_NATIVE


def _get_seq_lens_cpu(forward_batch) -> np.ndarray:
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()
    return seq_lens_cpu.numpy().astype(np.int32)


def _build_block_tables(
    req_to_token_pool, req_pool_indices, max_seq_len: int, block_size: int
) -> torch.Tensor:
    req_to_token = req_to_token_pool.req_to_token
    max_blocks = max(1, (int(max_seq_len) + block_size - 1) // block_size)
    return (
        req_to_token[req_pool_indices, : max_blocks * block_size : block_size]
        // block_size
    ).to(torch.int32)


def build_atom_v4_decode_graph_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    proxy_pool: ATOMDeepSeekV4ProxyKVPool,
    req_to_token_pool,
    model: Any = None,
):
    """Build fixed-address ATOM V4 decode metadata for SGLang graph replay.

    Decode graph capture reuses tensor addresses, so this path stages new
    SGLang req/block/slot information into persistent buffers instead of
    replacing metadata tensors.  Keeping the state-slot mapping here is required
    for cached-prefix requests after they leave prefill and enter decode.
    """
    from atom.model_ops.v4_kernels import write_v4_paged_decode_indices
    from atom.plugin.vllm.deepseek_v4_ops import write_v4_decode_hca_compress_tail
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    bs = int(forward_batch.batch_size)
    seq_np = _get_seq_lens_cpu(forward_batch)[:bs]
    if seq_np.size == 0:
        seq_np = np.ones(0, dtype=np.int32)

    actual_mode = getattr(
        forward_batch, "actual_forward_mode", forward_batch.forward_mode
    )
    is_idle = bool(getattr(actual_mode, "is_idle", lambda: False)())
    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    scheduled_bs = (
        0
        if is_idle
        else (
            min(bs, int(out_cache_loc.numel()))
            if torch.is_tensor(out_cache_loc)
            else bs
        )
    )
    total = scheduled_bs
    t_pad = bs

    max_blocks = max(1, proxy_pool.num_blocks)
    bufs = getattr(proxy_pool, "_atom_v4_decode_graph_buffers", None)
    if bufs is None or bufs.num_slots < bs or bufs.max_blocks < max_blocks:
        bufs = proxy_pool._atom_v4_decode_graph_buffers = _V4SGLangDecodeGraphBuffers(
            num_slots=proxy_pool.num_slots,
            max_decode_tokens=max(proxy_pool.num_slots, bs),
            window=proxy_pool.window_size,
            index_topk=1024,
            max_committed_hca=max_blocks,
            max_blocks=max_blocks,
            device=device,
        )

    lens = np.ones(bs, dtype=np.int32)
    q_np = np.arange(bs + 1, dtype=np.int32)
    cu_q = bufs.stage(bufs.cu_q, q_np, bs + 1)

    block_tables_live = _build_block_tables(
        req_to_token_pool,
        forward_batch.req_pool_indices[:bs],
        max_blocks * ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )
    bufs.block_tables.gpu[:bs, : block_tables_live.shape[1]].copy_(block_tables_live)
    # Keep a full-row slice from the persistent 2D buffer.  Some V4 kernels
    # require block_tables.is_contiguous(); slicing the column dimension can
    # produce a strided view even when the logical width matches.
    block_tables = bufs.block_tables.gpu[:bs]

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,
        max_seqlen_q=1,
        max_seqlen_k=int(seq_np.max()) if len(seq_np) else 1,
        slot_mapping=out_cache_loc,
        context_lens=forward_batch.seq_lens[:bs],
        block_tables=block_tables,
        state=AttnState.DECODE,
    )
    md.swa_num_slots = proxy_pool.num_slots
    md.swa_window = proxy_pool.window_size
    md.swa_cs = proxy_pool.window_size
    md.index_topk = 1024
    md.swa_pages = proxy_pool.num_slots * proxy_pool.window_size

    if total:
        pos_np = (seq_np[:total] - 1).astype(np.int32)
        batch_np = np.arange(total, dtype=np.int32)
    else:
        pos_np = np.zeros(0, dtype=np.int32)
        batch_np = np.zeros(0, dtype=np.int32)
    batch_pad = np.full(t_pad, -1, dtype=np.int32)
    if total:
        batch_pad[:total] = batch_np

    allocator = getattr(proxy_pool, "_atom_v4_slot_allocator", None)
    if allocator is None:
        allocator = proxy_pool._atom_v4_slot_allocator = _V4StateSlotAllocator(
            proxy_pool.num_slots
        )

    slot_arr = np.zeros(bs, dtype=np.int32)
    reset_slots: set[int] = set()
    if total:
        first_blocks = block_tables[:total, 0].detach().cpu().numpy().astype(np.int32)
        fresh_mask = pos_np == 0
        slot_real, reset_slots = allocator.assign(first_blocks, fresh_mask)
        slot_arr[:total] = slot_real

    if reset_slots and model is not None:
        reset_deepseek_v4_state_slots(model, reset_slots)

    # Graph replay updates/reset state outside the captured region.  Do not let
    # the wrapper repeat the reset inside capture, because allocating the index
    # tensor there is not graph-capturable on HIP.
    md.reset_slots = set()
    md.state_slot_mapping_cpu = slot_arr
    md.state_slot_mapping = bufs.stage(bufs.state_slot, slot_arr, bs)
    md.batch_id_per_token_cpu = batch_np
    md.batch_id_per_token = bufs.stage(bufs.batch_id, batch_pad, t_pad)
    n_csa = (seq_np // 4).astype(np.int32)
    n_hca = (seq_np // 128).astype(np.int32)
    if os.environ.get("ATOM_SGLANG_V4_DISABLE_COMPRESS_READ") == "1":
        n_csa = np.zeros_like(n_csa)
        n_hca = np.zeros_like(n_hca)
    md.n_committed_csa_per_seq_cpu = n_csa
    md.n_committed_hca_per_seq_cpu = n_hca
    md.n_committed_csa_per_seq = bufs.stage(bufs.n_csa, n_csa, bs)
    md.n_committed_hca_per_seq = bufs.stage(bufs.n_hca, n_hca, bs)
    md.compress_plans = _make_decode_graph_compress_plans(lens, seq_np, bufs)

    win = int(md.swa_window)
    index_topk = int(md.index_topk)
    if total:
        actual_swa = np.minimum(pos_np + 1, win).astype(np.int32)
        csa_valid = np.minimum(
            np.minimum((pos_np + 1) // 4, n_csa[:total]), index_topk
        ).astype(np.int32)
        hca_valid = n_hca[:total].astype(np.int32)
    else:
        actual_swa = csa_valid = hca_valid = np.zeros(0, dtype=np.int32)

    def indptr(counts):
        out = np.zeros(t_pad + 1, dtype=np.int32)
        if total:
            out[1 : total + 1] = np.cumsum(counts, dtype=np.int32)
        if t_pad > total:
            out[total + 1 :] = out[total]
        return out

    swa_indptr_np = indptr(actual_swa)
    csa_indptr_np = indptr(actual_swa + csa_valid)
    hca_indptr_np = indptr(actual_swa + hca_valid)
    swa_indptr = bufs.stage(bufs.indptr_swa, swa_indptr_np, t_pad + 1)
    csa_indptr = bufs.stage(bufs.indptr_csa, csa_indptr_np, t_pad + 1)
    hca_indptr = bufs.stage(bufs.indptr_hca, hca_indptr_np, t_pad + 1)

    positions_gpu = positions[:t_pad]
    write_v4_paged_decode_indices(
        state_slot_per_seq=md.state_slot_mapping,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr,
        csa_indptr=csa_indptr,
        hca_indptr=hca_indptr,
        swa_indices=bufs.idx_swa.gpu,
        csa_indices=bufs.idx_csa.gpu,
        hca_indices=bufs.idx_hca.gpu,
        T=t_pad,
        win=win,
        cs=int(md.swa_cs),
    )
    write_v4_decode_hca_compress_tail(
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        hca_indptr=hca_indptr,
        n_committed_hca_per_seq=md.n_committed_hca_per_seq,
        block_tables=md.block_tables,
        hca_indices=bufs.idx_hca.gpu,
        T=t_pad,
        win=win,
        swa_pages=int(md.swa_pages),
    )
    md.kv_indices_swa = bufs.idx_swa.gpu
    md.kv_indices_csa = bufs.idx_csa.gpu
    md.kv_indices_hca = bufs.idx_hca.gpu
    md.kv_indptr_swa = swa_indptr
    md.kv_indptr_csa = csa_indptr
    md.kv_indptr_hca = hca_indptr
    md.indexer_meta = {
        "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
    }
    return md


def build_atom_v4_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    proxy_pool: ATOMDeepSeekV4ProxyKVPool,
    req_to_token_pool,
):
    """Translate SGLang ForwardBatch into ATOM V4 attention metadata.

    This is the main bridge that makes prefix cache usable without changing
    SGLang.  SGLang supplies logical req_to_token/block tables plus suffix-only
    input tokens; this function reconstructs ATOM's state slots, committed
    CSA/HCA counts, prefix/extend index arrays, and the correct PREFILL_PREFIX
    state for radix-cache hits.
    """
    from atom.utils.forward_context import AttentionMetaData

    state = _infer_atom_attn_state(forward_batch)
    device = positions.device
    num_reqs = int(forward_batch.batch_size)
    seq_np = _get_seq_lens_cpu(forward_batch)[:num_reqs]
    is_decode = forward_batch.forward_mode.is_decode_or_idle()

    if is_decode:
        lens = np.ones(num_reqs, dtype=np.int32)
        q_np = np.arange(num_reqs + 1, dtype=np.int32)
        batch_np = np.arange(num_reqs, dtype=np.int32)
        pos_np = positions[:num_reqs].detach().cpu().numpy().astype(np.int32)
    else:
        extend_lens = _get_extend_lens_cpu(forward_batch, positions)
        if extend_lens is None:
            raise RuntimeError("SGLang DeepSeek-V4 prefill metadata lacks extend lens")
        lens = extend_lens[:num_reqs].astype(np.int32)
        q_np = np.zeros(num_reqs + 1, dtype=np.int32)
        q_np[1:] = np.cumsum(lens, dtype=np.int32)
        batch_np = np.repeat(np.arange(num_reqs, dtype=np.int32), lens)
        pos_np = positions[: int(lens.sum())].detach().cpu().numpy().astype(np.int32)

    total = int(lens.sum())
    max_seq_len = int(seq_np.max()) if len(seq_np) else 1
    cu_q = torch.from_numpy(q_np).to(device=device, dtype=torch.int32)
    block_tables = _build_block_tables(
        req_to_token_pool,
        forward_batch.req_pool_indices[:num_reqs],
        max_seq_len,
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,
        max_seqlen_q=int(lens.max()) if len(lens) else 0,
        max_seqlen_k=max_seq_len,
        slot_mapping=getattr(forward_batch, "out_cache_loc", None),
        context_lens=forward_batch.seq_lens[:num_reqs],
        block_tables=block_tables,
        state=state,
    )
    md.swa_num_slots = proxy_pool.num_slots
    md.swa_window = proxy_pool.window_size
    md.swa_cs = proxy_pool.window_size
    md.index_topk = 1024
    md.swa_pages = proxy_pool.num_slots * proxy_pool.window_size

    allocator = getattr(proxy_pool, "_atom_v4_slot_allocator", None)
    if allocator is None:
        allocator = proxy_pool._atom_v4_slot_allocator = _V4StateSlotAllocator(
            proxy_pool.num_slots
        )
    first_block_ids = block_tables[:num_reqs, 0].detach().cpu().numpy()
    fresh_mask = (
        pos_np[q_np[:-1]] == 0
        if total and len(q_np) > 1
        else np.zeros(num_reqs, dtype=bool)
    )
    slot_arr, reset_slots = allocator.assign(first_block_ids, fresh_mask)
    md.reset_slots = reset_slots
    md.state_slot_mapping_cpu = slot_arr
    md.state_slot_mapping = torch.from_numpy(slot_arr).to(
        device=device, dtype=torch.int32
    )
    md.batch_id_per_token_cpu = batch_np
    md.batch_id_per_token = torch.from_numpy(batch_np).to(device=device)
    md.n_committed_csa_per_seq_cpu = (seq_np // 4).astype(np.int32)
    md.n_committed_hca_per_seq_cpu = (seq_np // 128).astype(np.int32)
    if os.environ.get("ATOM_SGLANG_V4_DISABLE_COMPRESS_READ") == "1":
        md.n_committed_csa_per_seq_cpu = np.zeros_like(md.n_committed_csa_per_seq_cpu)
        md.n_committed_hca_per_seq_cpu = np.zeros_like(md.n_committed_hca_per_seq_cpu)
    md.n_committed_csa_per_seq = torch.from_numpy(md.n_committed_csa_per_seq_cpu).to(
        device=device
    )
    md.n_committed_hca_per_seq = torch.from_numpy(md.n_committed_hca_per_seq_cpu).to(
        device=device
    )
    md.compress_plans = _make_compress_plans(lens, seq_np, device)

    if is_decode:
        _populate_decode_indices(md, block_tables, pos_np, device)
    else:
        _populate_prefill_indices(md, block_tables, batch_np, pos_np, q_np, device)
    _populate_indexer(md, batch_np, positions[:total], device)
    if _debug_enabled():
        logger.info(
            "ATOM SGLang V4 metadata: mode=%s batch=%s total=%s positions=%s "
            "lens=%s seq=%s state_slots=%s padded_static_len=%s",
            getattr(forward_batch.forward_mode, "name", forward_batch.forward_mode),
            num_reqs,
            total,
            int(positions.numel()),
            lens.tolist(),
            seq_np.tolist(),
            slot_arr.tolist(),
            getattr(forward_batch, "padded_static_len", None),
        )
    return md


def _populate_decode_indices(md, block_tables, pos_np, device) -> None:
    from atom.model_ops.v4_kernels import write_v4_paged_decode_indices

    win = int(md.swa_window)
    cs = int(md.swa_cs)
    batch_np = md.batch_id_per_token_cpu
    if len(batch_np) == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        md.kv_indices_swa = md.kv_indices_csa = md.kv_indices_hca = empty
        md.kv_indptr_swa = md.kv_indptr_csa = md.kv_indptr_hca = zero
        return
    swa_counts = np.minimum(pos_np + 1, win).astype(np.int32)
    csa_counts = np.minimum(
        np.minimum((pos_np + 1) // 4, int(md.index_topk)),
        md.n_committed_csa_per_seq_cpu[batch_np],
    ).astype(np.int32)
    # Per-token causal cap, mirroring CSA above and the prefill kernel
    # (n_hca = min((pos+1)//128, committed)); without it the indptr over-reserves
    # vs the kernel's actual writes -> uninitialized HCA tail garbage.
    hca_counts = np.minimum(
        (pos_np + 1) // 128, md.n_committed_hca_per_seq_cpu[batch_np]
    ).astype(np.int32)
    swa_indptr_np = _counts_to_indptr(swa_counts)
    csa_indptr_np = _counts_to_indptr(swa_counts + csa_counts)
    hca_indptr_np = _counts_to_indptr(swa_counts + hca_counts)

    positions_gpu = torch.from_numpy(pos_np).to(device=device, dtype=torch.int64)
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device=device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device=device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device=device)
    swa_indices = torch.empty(
        max(1, int(swa_indptr_np[-1])), dtype=torch.int32, device=device
    )
    csa_indices = torch.empty(
        max(1, int(csa_indptr_np[-1])), dtype=torch.int32, device=device
    )
    hca_indices = torch.empty(
        max(1, int(hca_indptr_np[-1])), dtype=torch.int32, device=device
    )
    write_v4_paged_decode_indices(
        state_slot_per_seq=md.state_slot_mapping,
        batch_id_per_token=md.batch_id_per_token,
        positions=positions_gpu,
        swa_indptr=swa_indptr,
        csa_indptr=csa_indptr,
        hca_indptr=hca_indptr,
        swa_indices=swa_indices,
        csa_indices=csa_indices,
        hca_indices=hca_indices,
        T=len(batch_np),
        win=win,
        cs=cs,
    )
    # Fill HCA compressed section on CPU for the first-cut eager bridge.
    # `write_v4_paged_decode_indices` writes the SWA prefix at the TAIL of each
    # per-token slice, so HCA compressed entries must occupy the HEAD starting
    # at hca_indptr[t].  This mirrors native ATOM's _attach_v4_paged_decode_meta.
    hca_cpu = hca_indices.detach().cpu().numpy()
    for t, bid in enumerate(batch_np):
        n_hca = int(hca_counts[t])
        base = int(hca_indptr_np[t])
        if n_hca:
            hca_cpu[base : base + n_hca] = int(md.swa_pages) + block_tables[
                int(bid), :n_hca
            ].detach().cpu().numpy().astype(np.int32)
    hca_indices.copy_(torch.from_numpy(hca_cpu).to(device=device))
    md.kv_indices_swa = swa_indices[: int(swa_indptr_np[-1])]
    md.kv_indices_csa = csa_indices[: int(csa_indptr_np[-1])]
    md.kv_indices_hca = hca_indices[: int(hca_indptr_np[-1])]
    md.kv_indptr_swa = swa_indptr
    md.kv_indptr_csa = csa_indptr
    md.kv_indptr_hca = hca_indptr


def _populate_prefill_indices(md, block_tables, batch_np, pos_np, q_np, device) -> None:
    """Create ATOM V4 prefix/suffix index arrays for SGLang prefill.

    For a prefix-cache hit, SGLang forwards only suffix tokens while block_tables
    still describe the full logical sequence.  The generated indices split each
    token's attention into the freshly computed suffix (`kv_indices_extend`) and
    the reusable prefix windows/compressed blocks (`kv_indices_prefix_*`).
    """
    from atom.model_ops.v4_kernels import write_v4_paged_prefill_indices

    T = len(batch_np)
    if T == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        md.kv_indices_extend = md.kv_indices_prefix_swa = empty
        md.kv_indices_prefix_csa = md.kv_indices_prefix_hca = empty
        md.kv_indptr_extend = md.kv_indptr_prefix_swa = zero
        md.kv_indptr_prefix_csa = md.kv_indptr_prefix_hca = zero
        md.skip_prefix_len_csa = empty
        return
    win = int(md.swa_window)
    cs = int(md.swa_cs)
    chunk_start_per_seq = pos_np[q_np[:-1]]
    chunk_start_pt = chunk_start_per_seq[batch_np]
    token_pos_in_chunk = pos_np - chunk_start_pt
    swa_low = np.maximum(pos_np - win + 1, 0)
    extend_count = np.minimum(token_pos_in_chunk + 1, win).astype(np.int32)
    prefix_swa_count = np.maximum(chunk_start_pt - swa_low, 0).astype(np.int32)
    csa_valid_k = np.minimum(
        np.minimum((pos_np + 1) // 4, md.n_committed_csa_per_seq_cpu[batch_np]),
        int(md.index_topk),
    ).astype(np.int32)
    # Per-token causal cap, mirroring CSA above and the prefill kernel
    # (n_hca = min((pos+1)//128, committed)); without it the indptr over-reserves
    # vs the kernel's actual writes -> uninitialized HCA tail garbage.
    hca_count = np.minimum(
        (pos_np + 1) // 128, md.n_committed_hca_per_seq_cpu[batch_np]
    ).astype(np.int32)
    ext_indptr_np = _counts_to_indptr(extend_count)
    swa_indptr_np = _counts_to_indptr(prefix_swa_count)
    csa_indptr_np = _counts_to_indptr(prefix_swa_count + csa_valid_k)
    hca_indptr_np = _counts_to_indptr(prefix_swa_count + hca_count)

    def t(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(
            device=device, dtype=torch.int32
        )

    ext_indices = torch.empty(
        max(1, int(ext_indptr_np[-1])), dtype=torch.int32, device=device
    )
    swa_indices = torch.empty(
        max(1, int(swa_indptr_np[-1])), dtype=torch.int32, device=device
    )
    csa_indices = torch.empty(
        max(1, int(csa_indptr_np[-1])), dtype=torch.int32, device=device
    )
    hca_indices = torch.empty(
        max(1, int(hca_indptr_np[-1])), dtype=torch.int32, device=device
    )
    write_v4_paged_prefill_indices(
        positions=t(pos_np),
        bid_per_token=md.batch_id_per_token.to(torch.int64),
        chunk_start_per_seq=t(chunk_start_per_seq),
        cu_seqlens_q_per_seq=t(q_np[:-1]),
        state_slot_per_seq=md.state_slot_mapping,
        n_committed_hca_per_seq=md.n_committed_hca_per_seq,
        block_tables=block_tables,
        extend_indptr=t(ext_indptr_np),
        prefix_swa_indptr=t(swa_indptr_np),
        prefix_csa_indptr=t(csa_indptr_np),
        prefix_hca_indptr=t(hca_indptr_np),
        extend_indices=ext_indices,
        prefix_swa_indices=swa_indices,
        prefix_csa_indices=csa_indices,
        prefix_hca_indices=hca_indices,
        T=T,
        win=win,
        cs=cs,
        swa_pages=int(md.swa_pages),
    )
    md.kv_indices_extend = ext_indices[: int(ext_indptr_np[-1])]
    md.kv_indices_prefix_swa = swa_indices[: int(swa_indptr_np[-1])]
    md.kv_indices_prefix_csa = csa_indices[: int(csa_indptr_np[-1])]
    md.kv_indices_prefix_hca = hca_indices[: int(hca_indptr_np[-1])]
    md.kv_indptr_extend = t(ext_indptr_np)
    md.kv_indptr_prefix_swa = t(swa_indptr_np)
    md.kv_indptr_prefix_csa = t(csa_indptr_np)
    md.kv_indptr_prefix_hca = t(hca_indptr_np)
    md.skip_prefix_len_csa = t(prefix_swa_count)
    md.chunk_start_per_seq_cpu = chunk_start_per_seq.astype(np.int32)


def _populate_indexer(md, batch_np, positions, device) -> None:
    n_csa = md.n_committed_csa_per_seq_cpu
    cu = np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(n_csa, dtype=np.int32)])
    cu[-1] = max(int(cu[-1]), 1)
    cu_gpu = torch.from_numpy(cu).to(device=device, dtype=torch.int32)
    bid = md.batch_id_per_token
    if bid.numel() == 0:
        md.indexer_meta = {
            "total_committed": int(cu[-1]),
            "cu_committed_gpu": cu_gpu,
            "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
            "batch_id_per_token_gpu": bid,
            "seq_base_per_token_gpu": None,
            "cu_starts_gpu": None,
            "cu_ends_gpu": None,
        }
        return
    base = cu_gpu[bid].to(torch.int32)
    end = base + torch.minimum(
        (positions.to(torch.int32) + 1) // 4,
        md.n_committed_csa_per_seq[bid],
    ).to(torch.int32)
    md.indexer_meta = {
        "total_committed": int(cu[-1]),
        "cu_committed_gpu": cu_gpu,
        "n_committed_per_seq_gpu": md.n_committed_csa_per_seq,
        "batch_id_per_token_gpu": bid,
        "seq_base_per_token_gpu": base,
        "cu_starts_gpu": base,
        "cu_ends_gpu": end,
    }


def maybe_get_proxy_pool_from_sglang_backend():
    """Find the active ATOM proxy pool from SGLang runtime objects.

    Attention code may run either with the backend already installed in
    SGLang's forward context or through the plugin wrapper's current
    ForwardBatch.  Returning the proxy pool plus req_to_token_pool gives the V4
    metadata builder access to the same logical KV mapping used by radix cache.
    """
    backend = None
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
    except Exception:
        backend = None

    proxy_pool = getattr(backend, "token_to_kv_pool", None)
    req_to_token_pool = getattr(backend, "req_to_token_pool", None)
    if getattr(proxy_pool, "is_atom_v4_proxy_pool", False):
        return proxy_pool, req_to_token_pool

    try:
        from atom.plugin.sglang.runtime import get_current_forward_batch

        forward_batch = get_current_forward_batch()
    except Exception:
        forward_batch = None

    proxy_pool = getattr(forward_batch, "token_to_kv_pool", None)
    req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
    return proxy_pool, req_to_token_pool


def reset_deepseek_v4_state_slots(model, slots) -> None:
    """Clear SWA and compressor state for newly assigned fresh-prefill slots."""
    if not slots:
        return
    idx = None
    for block in getattr(model.model, "layers", []):
        attn = getattr(block, "attn", None)
        swa = getattr(attn, "swa_kv", None)
        if isinstance(swa, torch.Tensor):
            if idx is None:
                idx = torch.as_tensor(
                    sorted(slots), dtype=torch.long, device=swa.device
                )
            swa[idx] = 0
        for compressor in (
            getattr(attn, "compressor", None),
            getattr(getattr(attn, "indexer", None), "compressor", None),
        ):
            if compressor is None or idx is None:
                continue
            if isinstance(getattr(compressor, "kv_state", None), torch.Tensor):
                compressor.kv_state[idx] = 0
            if isinstance(getattr(compressor, "score_state", None), torch.Tensor):
                compressor.score_state[idx] = float("-inf")
