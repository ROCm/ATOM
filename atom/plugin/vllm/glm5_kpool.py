import logging
from dataclasses import dataclass

import numpy as np
import torch
from aiter import dtypes
from aiter.ops.cache import cp_gather_indexer_k_quant_cache, indexer_k_quant_and_cache
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from atom.config import get_current_atom_config
from atom.model_ops.attentions.aiter_mla import aligned_index_cache_dim
from atom.model_ops.glm5_next import kpool
from atom.model_ops.glm5_next.geometry import speculative_kpool_history_size
from atom.model_ops.glm5_next.indexer import _kpool_write_completed_pools
from atom.model_ops.glm5_next.speculative import (
    build_speculative_pool_candidates,
    update_speculative_kpool_history,
)
from atom.model_ops.sparse_indexer_chunk import sparse_indexer_row_chunk
from atom.plugin.vllm.attention.backend import AiterMlaBackendForVllm
from atom.plugin.vllm.attention.layer_sparse_mla import (
    triton_convert_req_index_to_global_index,
)
from atom.plugin.vllm.req_id_passthrough_patch import get_current_req_ids
from atom.utils import envs, mark_spliting_op

logger = logging.getLogger("atom")

KPOOL_KERNEL_BLOCK_SIZE = 64

GLM5_NEXT_MLA_ROPE_PAD = 64


def index_proxy_layer_name(text_config) -> str:
    num_layers = int(text_config.num_hidden_layers)
    num_mtp = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
    return f"model.layers.{num_layers + num_mtp}.glm5_kpool_index"


def num_speculative_tokens(vllm_config) -> int:
    spec = getattr(vllm_config, "speculative_config", None)
    return int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec else 0


class _SlotAllocator:
    def __init__(self, num_slots: int) -> None:
        self.num_slots = num_slots
        self._slot_of: dict[str, int] = {}
        self._owner: list[str | None] = [None] * num_slots
        self._last_used = [-1] * num_slots
        self._step = 0

    def assign(self, req_ids) -> np.ndarray:
        self._step += 1
        batch = set(req_ids)
        slots = np.empty(len(req_ids), dtype=np.int32)
        for i, req_id in enumerate(req_ids):
            slot = self._slot_of.get(req_id)
            if slot is None:
                slot = min(
                    (s for s in range(self.num_slots) if self._owner[s] not in batch),
                    key=self._last_used.__getitem__,
                )
                self._slot_of.pop(self._owner[slot], None)
                self._slot_of[req_id] = slot
                self._owner[slot] = req_id
            self._last_used[slot] = self._step
            slots[i] = slot
        return slots


class Glm5KpoolTailStore:
    def __init__(self, vllm_config, text_config, num_layers: int) -> None:
        pool = int(text_config.index_kpool)
        num_spec = num_speculative_tokens(vllm_config)
        self.ring = speculative_kpool_history_size(pool, num_spec or None)
        max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.allocator = _SlotAllocator(2 * max_reqs)
        self.tensor = torch.zeros(
            (num_layers, 2 * max_reqs, 2, self.ring, int(text_config.index_head_dim)),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self.slots = torch.full((max_reqs + 1,), -1, dtype=torch.int32, device="cuda")

    def assign(self, num_reqs: int) -> torch.Tensor:
        host = torch.empty(num_reqs, dtype=torch.int32, pin_memory=True)
        req_ids = get_current_req_ids()
        if req_ids is None:
            host.copy_(torch.arange(num_reqs, dtype=torch.int32))
            host.remainder_(self.allocator.num_slots)
        else:
            live = min(len(req_ids), num_reqs)
            slots = self.allocator.assign(req_ids[:live])
            host[:live].copy_(torch.from_numpy(slots))
            host[live:].fill_(-1)
        self.slots[:num_reqs].copy_(host, non_blocking=True)
        return self.slots[:num_reqs]


@dataclass
class Glm5KpoolPrefillMetadata:
    query_start_loc: torch.Tensor
    req_idx: torch.Tensor
    chunk_start: torch.Tensor | None
    max_seq_len: int
    total_pools: int


@dataclass
class Glm5KpoolIndexMetadata:
    num_reqs: int
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    tail_store: Glm5KpoolTailStore
    prefill: Glm5KpoolPrefillMetadata | None


def _build_prefill(cm, num_decodes: int, index_kpool: int) -> Glm5KpoolPrefillMetadata:
    num_reqs = cm.num_reqs
    query_start_cpu = cm.query_start_loc_cpu[num_decodes : num_reqs + 1]
    query_start_cpu = (query_start_cpu - query_start_cpu[0]).to(torch.int32)
    query_lens_cpu = query_start_cpu[1:] - query_start_cpu[:-1]
    seq_lens_cpu = cm.seq_lens[num_decodes:num_reqs].cpu()
    context_cpu = seq_lens_cpu - query_lens_cpu
    req_idx_cpu = torch.repeat_interleave(
        torch.arange(num_reqs - num_decodes, dtype=torch.int64), query_lens_cpu
    )
    device = cm.seq_lens.device
    chunk_start = None
    if bool((context_cpu > 0).any()):
        chunk_start = context_cpu.to(torch.int64).to(device, non_blocking=True)
    return Glm5KpoolPrefillMetadata(
        query_start_loc=query_start_cpu.to(device, non_blocking=True),
        req_idx=req_idx_cpu.to(device, non_blocking=True),
        chunk_start=chunk_start,
        max_seq_len=int(seq_lens_cpu.max()),
        total_pools=int((seq_lens_cpu // index_kpool).sum()),
    )


def build_kpool_metadata(
    cm, *, decode_threshold: int, index_kpool: int, tail_store: Glm5KpoolTailStore
) -> Glm5KpoolIndexMetadata:
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(cm, decode_threshold=decode_threshold)
    )
    return Glm5KpoolIndexMetadata(
        num_reqs=cm.num_reqs,
        num_actual_tokens=cm.num_actual_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        query_start_loc=cm.query_start_loc,
        seq_lens=cm.seq_lens,
        block_table=cm.block_table_tensor,
        slots=tail_store.assign(cm.num_reqs),
        tail_store=tail_store,
        prefill=(
            _build_prefill(cm, num_decodes, index_kpool) if num_prefills else None
        ),
    )


def find_tail_store(vllm_config, text_config) -> Glm5KpoolTailStore:
    sfc = vllm_config.compilation_config.static_forward_context
    return sfc[index_proxy_layer_name(text_config)].tail_store


@dataclass
class Glm5KpoolProxyMetadata:
    num_actual_tokens: int


class Glm5KpoolIndexMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return Glm5KpoolProxyMetadata(common_attn_metadata.num_actual_tokens)


class Glm5KpoolIndexBackend(AiterMlaBackendForVllm):
    @staticmethod
    def get_name() -> str:
        return "ATOM_GLM5_KPOOL_INDEX"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [KPOOL_KERNEL_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return KPOOL_KERNEL_BLOCK_SIZE

    @staticmethod
    def get_builder_cls() -> type:
        return Glm5KpoolIndexMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class Glm5KpoolIndexProxy(nn.Module, AttentionLayerBase):
    def __init__(self, layer_name: str, mla_layer, indexers: list, tail_store):
        super().__init__()
        self.layer_name = layer_name
        self.kv_cache = torch.tensor([])
        self.__dict__["_mla_layer"] = mla_layer
        self.__dict__["_indexers"] = list(indexers)
        self.__dict__["tail_store"] = tail_store

    def add_indexers(self, indexers: list) -> None:
        self._indexers.extend(indexers)

    def get_attn_backend(self):
        return Glm5KpoolIndexBackend

    def get_kv_cache_spec(self, vllm_config):
        # MLA's spec puts these rows in the MLA group, so they share its block ids.
        return self._mla_layer.get_kv_cache_spec(vllm_config)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache
        num_pages, page_tokens = kv_cache.shape[:2]
        if page_tokens != KPOOL_KERNEL_BLOCK_SIZE:
            raise RuntimeError(
                f"GLM-5.3 k-pool index pages hold {page_tokens} tokens, expected "
                f"{KPOOL_KERNEL_BLOCK_SIZE}; the MLA group was split differently"
            )
        indexer = self._indexers[0]
        rows = page_tokens // indexer.index_kpool
        row_bytes = aligned_index_cache_dim(indexer.config)
        page_bytes = kv_cache[0].numel() * kv_cache.element_size()
        subs_per_page = page_bytes // (rows * row_bytes)
        if subs_per_page < len(self._indexers):
            raise RuntimeError(
                f"a {page_bytes}-byte MLA page fits {subs_per_page} k-pool index "
                f"sub-blocks, but {len(self._indexers)} indexer layers need one each"
            )
        if len(self._indexers) > self.tail_store.tensor.shape[0]:
            raise RuntimeError("more k-pool indexer layers than tail store layers")
        index_cache = kv_cache.view(torch.uint8).view(
            num_pages * subs_per_page, rows, row_bytes
        )
        for slot, idx in enumerate(self._indexers):
            idx.bind_kpool_index_cache(index_cache, slot, subs_per_page)
        logger.info(
            "GLM-5.3 k-pool index: %d indexer layers in %d-byte pages "
            "(%d sub-blocks of %d x %d B each), tail ring %d",
            len(self._indexers),
            page_bytes,
            subs_per_page,
            rows,
            row_bytes,
            self.tail_store.ring,
        )


def register_kpool_index_proxy(
    vllm_config, text_config, mla_layer, indexers, num_draft_layers: int = 0
):
    sfc = vllm_config.compilation_config.static_forward_context
    name = index_proxy_layer_name(text_config)
    if name in sfc:
        raise ValueError(f"Duplicate layer name: {name}")
    store = Glm5KpoolTailStore(
        vllm_config, text_config, len(indexers) + num_draft_layers
    )
    proxy = Glm5KpoolIndexProxy(name, mla_layer, indexers, store)
    sfc[name] = proxy
    return proxy


def _fill_causal_indices(out: torch.Tensor, positions: torch.Tensor) -> None:
    cols = torch.arange(out.shape[1], device=out.device, dtype=torch.int32)
    pos = positions.to(torch.int32)[:, None]
    out.copy_(torch.where(cols[None, :] <= pos, cols[None, :], -1))


def _run_decode(
    indexer,
    md: Glm5KpoolIndexMetadata,
    k,
    gate,
    q_fp8,
    weights,
    positions,
    tail,
    index_cache,
    pool_bt,
    topk_indices,
) -> None:
    bs = md.num_decode_tokens
    pool = indexer.index_kpool
    rows = index_cache.shape[1]
    device = k.device
    pos = positions[:bs]
    slots = md.slots[:bs]
    pooled = kpool.kpool_decode_stash_and_pool(
        tail,
        k[:bs],
        gate[:bs],
        pos,
        slots,
        indexer.index_kpool_compress_ape,
        pool,
    )
    pos64 = pos.to(torch.int64)
    closes = (pos64 % pool == pool - 1) & (slots >= 0)
    pool_ids = torch.where(closes, pos64 // pool, torch.full_like(pos64, -1))
    write_slots = kpool.pool_slot_mapping(
        pool_bt[:bs],
        pool_ids,
        torch.arange(bs, device=device, dtype=torch.int64),
        rows,
    )
    indexer_k_quant_and_cache(
        pooled,
        index_cache,
        write_slots,
        indexer.head_dim,
        indexer.scale_fmt,
        preshuffle=True,
    )

    seq_lens = md.seq_lens[:bs]
    pool_ctx = (seq_lens // pool).to(torch.int32)
    pool_max_len = -(-indexer.max_model_len // pool)
    n_head = q_fp8.shape[1]
    logits = torch.empty((bs, pool_max_len), dtype=torch.float32, device=device)
    deepgemm_fp8_paged_mqa_logits(
        q_fp8[:bs].view(bs, 1, n_head, indexer.head_dim),
        index_cache.unsqueeze(-2),
        weights[:bs],
        logits,
        pool_ctx,
        pool_bt[:bs],
        pool_max_len,
        KVBlockSize=rows,
        Preshuffle=True,
    )
    select_k = indexer.topk_tokens // pool
    pool_topk = torch.empty((bs, select_k), dtype=torch.int32, device=device)
    top_k_per_row_decode(
        logits,
        1,
        pool_ctx,
        pool_topk,
        bs,
        logits.stride(0),
        logits.stride(1),
        k=select_k,
        stable=indexer.stable_topk,
    )
    kpool.expand_pools_and_append_tail(
        pool_topk, seq_lens.to(torch.int32), pool, out=topk_indices[:bs]
    )


def _run_verify(
    indexer,
    md: Glm5KpoolIndexMetadata,
    k,
    gate,
    q_fp8,
    weights,
    positions,
    tail,
    index_cache,
    pool_bt,
    topk_indices,
) -> None:
    n = md.num_decode_tokens
    num_reqs = md.num_decodes
    pool = indexer.index_kpool
    rows = index_cache.shape[1]
    device = k.device
    keys, gates, pos = k[:n], gate[:n], positions[:n].to(torch.int64)
    query_start = md.query_start_loc[: num_reqs + 1]
    slots = md.slots[:num_reqs]
    pooled, req_idx = build_speculative_pool_candidates(
        tail,
        keys,
        gates,
        pos,
        query_start,
        slots,
        indexer.index_kpool_compress_ape,
        pool,
    )
    closes = (pos % pool == pool - 1) & (slots[req_idx] >= 0)
    pool_ids = torch.where(closes, pos // pool, torch.full_like(pos, -1))
    write_slots = kpool.pool_slot_mapping(pool_bt[:num_reqs], pool_ids, req_idx, rows)
    indexer_k_quant_and_cache(
        pooled,
        index_cache,
        write_slots,
        indexer.head_dim,
        indexer.scale_fmt,
        preshuffle=True,
    )
    update_speculative_kpool_history(tail, keys, gates, pos, query_start, slots, slots)

    seq_lens = (pos + 1).to(torch.int32)
    pool_lens = (seq_lens // pool).contiguous()
    token_bt = pool_bt[:num_reqs][req_idx].contiguous()
    max_pools = -(-indexer.max_model_len // pool)
    select_k = indexer.topk_tokens // pool
    n_head = q_fp8.shape[1]
    chunk = min(n, 128)
    logits = torch.empty((chunk, max_pools), dtype=torch.float32, device=device)
    pool_topk = torch.empty((chunk, select_k), dtype=torch.int32, device=device)
    for begin in range(0, n, chunk):
        end = min(n, begin + chunk)
        count = end - begin
        deepgemm_fp8_paged_mqa_logits(
            q_fp8[begin:end].view(count, 1, n_head, indexer.head_dim),
            index_cache.unsqueeze(-2),
            weights[begin:end],
            logits[:count],
            pool_lens[begin:end],
            token_bt[begin:end],
            max_pools,
            KVBlockSize=rows,
            Preshuffle=True,
        )
        top_k_per_row_decode(
            logits[:count],
            1,
            pool_lens[begin:end],
            pool_topk[:count],
            count,
            logits.stride(0),
            logits.stride(1),
            k=select_k,
            stable=indexer.stable_topk,
        )
        kpool.expand_pools_and_append_tail(
            pool_topk[:count], seq_lens[begin:end], pool, out=topk_indices[begin:end]
        )


def _run_prefill(
    indexer,
    md: Glm5KpoolIndexMetadata,
    k,
    gate,
    q_fp8,
    weights,
    positions,
    tail,
    index_cache,
    pool_bt,
    topk_indices,
) -> None:
    pf = md.prefill
    pool = indexer.index_kpool
    rows = index_cache.shape[1]
    device = k.device
    tok = slice(md.num_decode_tokens, md.num_decode_tokens + md.num_prefill_tokens)
    req = slice(md.num_decodes, md.num_reqs)
    k, gate, positions = k[tok], gate[tok], positions[tok]
    q_fp8, weights = q_fp8[tok], weights[tok]
    slots, pool_bt, out = md.slots[req], pool_bt[req], topk_indices[tok]

    _kpool_write_completed_pools(
        index_cache,
        k,
        gate,
        positions,
        pool_bt,
        pf.req_idx,
        indexer.index_kpool_compress_ape,
        pool,
        indexer.head_dim,
        indexer.scale_fmt,
        rows,
        chunk_start=pf.chunk_start,
        tail_cache=tail,
        state_slot_idx=slots,
    )
    kpool.kpool_seed_tail(tail, k, gate, positions, pf.query_start_loc, slots, pool)

    if pf.max_seq_len <= indexer.topk_tokens or pf.total_pools <= 0:
        _fill_causal_indices(out, positions)
        return

    seq_lens = md.seq_lens[req]
    pool_cu = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32, device=device)
    pool_cu[1:] = torch.cumsum(seq_lens // pool, 0).to(torch.int32)
    k_fp8 = torch.empty(
        (pf.total_pools, indexer.head_dim), device=device, dtype=dtypes.fp8
    )
    k_scale = torch.empty((pf.total_pools, 1), device=device, dtype=torch.float32)
    cp_gather_indexer_k_quant_cache(
        index_cache, k_fp8, k_scale.view(dtypes.fp8), pool_bt, pool_cu, preshuffle=True
    )
    kv_scales = k_scale.squeeze(-1).contiguous()

    pool_ks = pool_cu.to(torch.int64)[pf.req_idx]
    pool_ke = (pool_ks + (positions.to(torch.int64) + 1) // pool).to(torch.int32)
    pool_ks = pool_ks.to(torch.int32)
    n_tokens = k.shape[0]
    select_k = indexer.topk_tokens // pool
    pool_topk = torch.empty((n_tokens, select_k), dtype=torch.int32, device=device)
    chunk_rows = sparse_indexer_row_chunk(
        n_tokens, pf.total_pools, envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB
    )
    for start in range(0, n_tokens, chunk_rows):
        end = min(start + chunk_rows, n_tokens)
        row_starts, row_ends = pool_ks[start:end], pool_ke[start:end]
        logits = fp8_mqa_logits(
            Q=q_fp8[start:end],
            KV=k_fp8,
            kv_scales=kv_scales,
            weights=weights[start:end],
            cu_starts=row_starts,
            cu_ends=row_ends,
        )
        top_k_per_row_prefill(
            logits=logits,
            rowStarts=row_starts,
            rowEnds=row_ends,
            indices=pool_topk[start:end],
            values=None,
            numRows=end - start,
            stride0=logits.stride(0),
            stride1=logits.stride(1),
            k=select_k,
            stable=indexer.stable_topk,
        )
    kpool.expand_pools_and_append_tail(
        pool_topk,
        positions.to(torch.int32) + 1,
        pool,
        out=out,
        pool_base=pool_ks,
    )


@eager_break_during_capture
def _kpool_indexer_run(
    k: torch.Tensor,
    gate: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    sparse_kv_indices_buffer: torch.Tensor,
) -> None:
    fwd = get_forward_context()
    attn_metadata = fwd.attn_metadata
    if not isinstance(attn_metadata, dict):
        return
    atom_config = fwd.additional_kwargs.get("atom_config") or get_current_atom_config()
    indexer = atom_config.compilation_config.static_forward_context[layer_name]
    sparse_meta = attn_metadata.get(indexer.mla_layer_name)
    md = getattr(sparse_meta, "kpool", None)
    if md is None or md.num_actual_tokens == 0:
        return

    num_tokens = md.num_actual_tokens
    k, gate, positions = k[:num_tokens], gate[:num_tokens], positions[:num_tokens]
    q_fp8, weights = q_fp8[:num_tokens], weights[:num_tokens]
    pool_bt = (
        md.block_table[: md.num_reqs] * indexer.kpool_subs_per_page
        + indexer.kpool_index_slot
    )
    topk_indices = torch.full(
        (num_tokens, indexer.topk_out_width),
        -1,
        dtype=torch.int32,
        device=k.device,
    )
    tail_store = md.tail_store
    args = (
        indexer,
        md,
        k,
        gate,
        q_fp8,
        weights,
        positions,
        tail_store.tensor[indexer.kpool_index_slot],
        indexer.kpool_index_cache,
        pool_bt,
        topk_indices,
    )
    if md.num_decodes > 0:
        if tail_store.ring > indexer.index_kpool:
            _run_verify(*args)
        else:
            _run_decode(*args)
    if md.prefill is not None:
        _run_prefill(*args)

    triton_convert_req_index_to_global_index(
        sparse_meta.batch_id_per_q_token[:num_tokens],
        sparse_meta.block_table,
        topk_indices,
        sparse_meta.paged_kv_indptr,
        sparse_meta.paged_kv_indices,
        BLOCK_SIZE=sparse_meta.block_size,
        NUM_TOPK_TOKENS=sparse_meta.topk_tokens,
    )


def _glm5_kpool_indexer_fake(
    k: torch.Tensor,
    gate: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    sparse_kv_indices_buffer: torch.Tensor,
) -> None:
    return None


@mark_spliting_op(
    is_custom=True,
    gen_fake=_glm5_kpool_indexer_fake,
    mutates_args=["sparse_kv_indices_buffer"],
)
def glm5_kpool_indexer(
    k: torch.Tensor,
    gate: torch.Tensor,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    sparse_kv_indices_buffer: torch.Tensor,
) -> None:
    _kpool_indexer_run(
        k, gate, q_fp8, weights, positions, layer_name, sparse_kv_indices_buffer
    )
