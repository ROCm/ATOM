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
        self.window_size = self.swa_page_size
        # In the ATOM bridge layout one original-token block contributes one
        # HCA entry, so the HCA compressed-entry count is the physical block
        # count for the unified tails.
        self.num_blocks = max(1, self.c128_size)

        total_bytes = self._compute_raw_bytes()
        self.raw_arena = torch.empty(total_bytes, dtype=torch.uint8, device=device)
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
                    .view(self.num_slots * self.window_size + self.num_blocks * k, self.head_dim)
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
                    .view(self.num_slots * self.window_size + self.num_blocks * k, self.head_dim)
                )
                hca_main.append(main)
            else:
                unified.append(swa_view.view(self.num_slots * self.window_size, self.head_dim))

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
        raise RuntimeError(
            "ATOM V4 proxy pool does not support SGLang KV relocation yet. "
            "Run with --disable-radix-cache for the first bridge version."
        )


def install_deepseek_v4_proxy_pool_patch() -> None:
    """Patch SGLang's DSV4 pool constructor before ModelRunner._init_pools()."""

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
            .as_strided(size=(nb, k1), stride=(block_fp32_stride, 1), storage_offset=scale_fp32_offset)
        )
    else:
        compressor.cache_scale = None


def bind_deepseek_v4_proxy_cache_views(model, proxy_pool: Any) -> bool:
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
            attn.indexer._max_model_len_idx = max(1, proxy_pool.num_blocks * ATOM_DEEPSEEK_V4_BLOCK_SIZE // 4)
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
    def __init__(self, num_slots: int):
        self.num_slots = max(1, int(num_slots))
        self._block_to_slot: dict[int, int] = {}
        self._slot_to_block: list[int] = [-1] * self.num_slots
        self._free: list[int] = list(range(self.num_slots - 1, -1, -1))
        self._last_seen: list[int] = [-1] * self.num_slots
        self._step = 0

    def assign(self, first_block_ids, fresh_mask) -> tuple[np.ndarray, set[int]]:
        self._step += 1
        fb = first_block_ids.tolist() if hasattr(first_block_ids, "tolist") else list(first_block_ids)
        fresh = fresh_mask.tolist() if hasattr(fresh_mask, "tolist") else list(fresh_mask)
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
            "write": CpuGpuBuffer(total * max(1, ratio), 4, dtype=torch.int32, device=device),
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


def _infer_atom_attn_state(forward_batch) -> Any:
    from atom.utils.forward_context import AttnState

    mode = forward_batch.forward_mode
    if mode.is_decode_or_idle():
        return AttnState.DECODE
    # Prefix-cache is intentionally not supported in the first proxy version.
    return AttnState.PREFILL_NATIVE


def _get_seq_lens_cpu(forward_batch) -> np.ndarray:
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()
    return seq_lens_cpu.numpy().astype(np.int32)


def _build_block_tables(req_to_token_pool, req_pool_indices, max_seq_len: int, block_size: int) -> torch.Tensor:
    req_to_token = req_to_token_pool.req_to_token
    max_blocks = max(1, (int(max_seq_len) + block_size - 1) // block_size)
    return (req_to_token[req_pool_indices, : max_blocks * block_size : block_size] // block_size).to(torch.int32)


def build_atom_v4_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    proxy_pool: ATOMDeepSeekV4ProxyKVPool,
    req_to_token_pool,
):
    from atom.utils.forward_context import AttentionMetaData

    state = _infer_atom_attn_state(forward_batch)
    device = positions.device
    num_reqs = int(forward_batch.batch_size)
    seq_np = _get_seq_lens_cpu(forward_batch)[:num_reqs]
    is_decode = forward_batch.forward_mode.is_decode_or_idle()

    if is_decode:
        lens = np.ones(num_reqs, dtype=np.int32)
        q_np = np.arange(num_reqs + 1, dtype=np.int32)
        batch_np = np.arange(num_reqs, dtype=np.int64)
        pos_np = positions[:num_reqs].detach().cpu().numpy().astype(np.int32)
    else:
        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            extend_lens_t = getattr(forward_batch, "extend_seq_lens", None)
            if extend_lens_t is not None:
                extend_lens = extend_lens_t.detach().cpu().numpy().astype(np.int32)
            else:
                extend_lens = np.diff(
                    torch.nn.functional.pad(forward_batch.extend_start_loc, (0, 1), value=positions.numel())
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
        else:
            extend_lens = np.asarray(extend_lens, dtype=np.int32)
        lens = extend_lens[:num_reqs].astype(np.int32)
        q_np = np.zeros(num_reqs + 1, dtype=np.int32)
        q_np[1:] = np.cumsum(lens, dtype=np.int32)
        batch_np = np.repeat(np.arange(num_reqs, dtype=np.int64), lens)
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
        allocator = proxy_pool._atom_v4_slot_allocator = _V4StateSlotAllocator(proxy_pool.num_slots)
    first_block_ids = block_tables[:num_reqs, 0].detach().cpu().numpy()
    fresh_mask = pos_np[q_np[:-1]] == 0 if total and len(q_np) > 1 else np.zeros(num_reqs, dtype=bool)
    slot_arr, reset_slots = allocator.assign(first_block_ids, fresh_mask)
    md.reset_slots = reset_slots
    md.state_slot_mapping_cpu = slot_arr
    md.state_slot_mapping = torch.from_numpy(slot_arr).to(device=device, dtype=torch.int32)
    md.batch_id_per_token_cpu = batch_np
    md.batch_id_per_token = torch.from_numpy(batch_np).to(device=device)
    md.n_committed_csa_per_seq_cpu = (seq_np // 4).astype(np.int32)
    md.n_committed_hca_per_seq_cpu = (seq_np // 128).astype(np.int32)
    if os.environ.get("ATOM_SGLANG_V4_DISABLE_COMPRESS_READ") == "1":
        md.n_committed_csa_per_seq_cpu = np.zeros_like(md.n_committed_csa_per_seq_cpu)
        md.n_committed_hca_per_seq_cpu = np.zeros_like(md.n_committed_hca_per_seq_cpu)
    md.n_committed_csa_per_seq = torch.from_numpy(md.n_committed_csa_per_seq_cpu).to(device=device)
    md.n_committed_hca_per_seq = torch.from_numpy(md.n_committed_hca_per_seq_cpu).to(device=device)
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
    hca_counts = md.n_committed_hca_per_seq_cpu[batch_np].astype(np.int32)
    swa_indptr_np = _counts_to_indptr(swa_counts)
    csa_indptr_np = _counts_to_indptr(swa_counts + csa_counts)
    hca_indptr_np = _counts_to_indptr(swa_counts + hca_counts)

    positions_gpu = torch.from_numpy(pos_np).to(device=device, dtype=torch.int64)
    swa_indptr = torch.from_numpy(swa_indptr_np).to(device=device)
    csa_indptr = torch.from_numpy(csa_indptr_np).to(device=device)
    hca_indptr = torch.from_numpy(hca_indptr_np).to(device=device)
    swa_indices = torch.empty(max(1, int(swa_indptr_np[-1])), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(1, int(csa_indptr_np[-1])), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(1, int(hca_indptr_np[-1])), dtype=torch.int32, device=device)
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
    # Append HCA compressed tail on CPU for the first-cut eager bridge.
    hca_cpu = hca_indices.detach().cpu().numpy()
    for t, bid in enumerate(batch_np):
        n_swa = int(swa_counts[t])
        n_hca = int(hca_counts[t])
        base = int(hca_indptr_np[t]) + n_swa
        if n_hca:
            hca_cpu[base : base + n_hca] = (
                int(md.swa_pages)
                + block_tables[int(bid), :n_hca].detach().cpu().numpy().astype(np.int32)
            )
    hca_indices.copy_(torch.from_numpy(hca_cpu).to(device=device))
    md.kv_indices_swa = swa_indices[: int(swa_indptr_np[-1])]
    md.kv_indices_csa = csa_indices[: int(csa_indptr_np[-1])]
    md.kv_indices_hca = hca_indices[: int(hca_indptr_np[-1])]
    md.kv_indptr_swa = swa_indptr
    md.kv_indptr_csa = csa_indptr
    md.kv_indptr_hca = hca_indptr


def _populate_prefill_indices(md, block_tables, batch_np, pos_np, q_np, device) -> None:
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
    hca_count = md.n_committed_hca_per_seq_cpu[batch_np].astype(np.int32)
    ext_indptr_np = _counts_to_indptr(extend_count)
    swa_indptr_np = _counts_to_indptr(prefix_swa_count)
    csa_indptr_np = _counts_to_indptr(prefix_swa_count + csa_valid_k)
    hca_indptr_np = _counts_to_indptr(prefix_swa_count + hca_count)

    def t(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=torch.int32)

    ext_indices = torch.empty(max(1, int(ext_indptr_np[-1])), dtype=torch.int32, device=device)
    swa_indices = torch.empty(max(1, int(swa_indptr_np[-1])), dtype=torch.int32, device=device)
    csa_indices = torch.empty(max(1, int(csa_indptr_np[-1])), dtype=torch.int32, device=device)
    hca_indices = torch.empty(max(1, int(hca_indptr_np[-1])), dtype=torch.int32, device=device)
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
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
    except Exception:
        return None, None
    return getattr(backend, "token_to_kv_pool", None), getattr(backend, "req_to_token_pool", None)


def reset_deepseek_v4_state_slots(model, slots) -> None:
    if not slots:
        return
    idx = None
    for block in getattr(model.model, "layers", []):
        attn = getattr(block, "attn", None)
        swa = getattr(attn, "swa_kv", None)
        if isinstance(swa, torch.Tensor):
            if idx is None:
                idx = torch.as_tensor(sorted(slots), dtype=torch.long, device=swa.device)
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
