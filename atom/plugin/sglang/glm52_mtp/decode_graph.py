"""Target decode CUDA-graph metadata buffers."""

from __future__ import annotations

import torch
from aiter import get_mla_metadata_v1
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

from atom.plugin.sglang.glm52_mtp.common import (
    DECODE_GRAPH_BUFFERS_ATTR,
    SHARED_SPARSE_INDICES_ATTR,
    get_index_topk,
    local_num_attention_heads,
    make_mla_work_buffers,
    metadata_dtype,
    validate_page_size,
    attention_page_size,
)


class GLM52DecodeGraphBuffers:
    def __init__(
        self,
        *,
        max_bs: int,
        max_context_len: int,
        indexer_page_size: int,
        attention_page_size: int,
        index_topk: int,
        num_heads: int,
        dtype_q,
        dtype_kv,
        device: torch.device,
    ) -> None:
        self.max_bs = int(max_bs)
        self.max_context_len = int(max_context_len)
        self.indexer_page_size = int(indexer_page_size)
        self.attention_page_size = int(attention_page_size)
        self.index_topk = int(index_topk)
        self.device = device

        max_blocks = max(
            1,
            (self.max_context_len + self.indexer_page_size - 1)
            // self.indexer_page_size,
        )
        self.cu_q = torch.arange(self.max_bs + 1, dtype=torch.int32, device=device)
        self.kv_indptr = torch.zeros(self.max_bs + 1, dtype=torch.int32, device=device)
        self.sparse_kv_indptr = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=device
        )
        self.kv_indices = torch.empty(
            self.max_bs * self.max_context_len, dtype=torch.int32, device=device
        )
        self.kv_last_page_lens = torch.ones(
            self.max_bs, dtype=torch.int32, device=device
        )
        self.block_tables = torch.empty(
            self.max_bs, max_blocks, dtype=torch.int32, device=device
        )
        self.context_lens = torch.zeros(self.max_bs, dtype=torch.int32, device=device)
        self.slot_mapping = torch.zeros(self.max_bs, dtype=torch.int64, device=device)
        self.shared_sparse = torch.empty(
            self.max_bs * self.index_topk, dtype=torch.int32, device=device
        )

        work = make_mla_work_buffers(
            cu_seqlens_q=self.cu_q,
            kv_indptr=self.sparse_kv_indptr,
            kv_last_page_lens=self.kv_last_page_lens,
            num_heads=num_heads,
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            page_size=self.attention_page_size,
        )
        self.work_meta_data = work["work_meta_data"]
        self.work_indptr = work["work_indptr"]
        self.work_info_set = work["work_info_set"]
        self.reduce_indptr = work["reduce_indptr"]
        self.reduce_final_map = work["reduce_final_map"]
        self.reduce_partial_map = work["reduce_partial_map"]

    def stage_block_tables(self, req_to_token_pool, req_pool_indices, bs: int) -> None:
        req_to_token = req_to_token_pool.req_to_token
        live = req_to_token[
            req_pool_indices[:bs],
            : self.max_context_len : self.indexer_page_size,
        ]
        self.block_tables[:bs, : live.shape[1]].copy_(
            (live // self.indexer_page_size).to(torch.int32)
        )


def get_or_create_decode_graph_buffers(
    token_to_kv_pool,
    *,
    max_bs: int,
    max_context_len: int,
    indexer_page_size: int,
    attention_page_size_val: int,
    atom_config,
    device: torch.device,
) -> GLM52DecodeGraphBuffers:
    topk = get_index_topk(atom_config)
    dtype_q = metadata_dtype(atom_config)
    bufs = getattr(token_to_kv_pool, DECODE_GRAPH_BUFFERS_ATTR, None)
    if (
        bufs is None
        or bufs.max_bs < int(max_bs)
        or bufs.max_context_len < int(max_context_len)
        or bufs.indexer_page_size != int(indexer_page_size)
        or bufs.attention_page_size != int(attention_page_size_val)
        or bufs.index_topk != int(topk)
        or bufs.device != device
    ):
        bufs = GLM52DecodeGraphBuffers(
            max_bs=max_bs,
            max_context_len=max_context_len,
            indexer_page_size=indexer_page_size,
            attention_page_size=attention_page_size_val,
            index_topk=topk,
            num_heads=local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            device=device,
        )
        setattr(token_to_kv_pool, DECODE_GRAPH_BUFFERS_ATTR, bufs)
        setattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
    return bufs


def build_atom_glm52_decode_graph_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
    max_bs: int | None = None,
    max_context_len: int | None = None,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    del positions
    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    seq_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
    if max_context_len is None:
        req_to_token = req_to_token_pool.req_to_token
        max_context_len = int(req_to_token.shape[1])
    if max_bs is None:
        max_bs = max(bs, int(getattr(req_to_token_pool, "size", bs)))

    indexer_page_size = validate_page_size(token_to_kv_pool, atom_config)
    attn_page_size = attention_page_size(token_to_kv_pool)
    topk = get_index_topk(atom_config)
    dtype_q = metadata_dtype(atom_config)

    bufs = get_or_create_decode_graph_buffers(
        token_to_kv_pool,
        max_bs=max_bs,
        max_context_len=max_context_len,
        indexer_page_size=indexer_page_size,
        attention_page_size_val=attn_page_size,
        atom_config=atom_config,
        device=device,
    )

    bufs.kv_indptr.zero_()
    bufs.kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
    bufs.sparse_kv_indptr.zero_()
    bufs.sparse_kv_indptr[1 : bs + 1] = torch.cumsum(
        torch.clamp(seq_lens, max=topk), dim=0
    )
    bufs.context_lens[:bs].copy_(seq_lens)
    bufs.kv_last_page_lens[:bs].fill_(1)

    out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
    if torch.is_tensor(out_cache_loc):
        copy_n = min(bs, int(out_cache_loc.numel()))
        if copy_n:
            bufs.slot_mapping[:copy_n].copy_(out_cache_loc[:copy_n])
        if bs > copy_n:
            scratch_slot = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
            bufs.slot_mapping[copy_n:bs].fill_(scratch_slot)
    else:
        scratch_slot = max(0, int(getattr(token_to_kv_pool, "size", 1)) - 1)
        bufs.slot_mapping[:bs].fill_(scratch_slot)

    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token_pool.req_to_token,
        forward_batch.req_pool_indices[:bs],
        seq_lens,
        bufs.kv_indptr[: bs + 1],
        None,
        bufs.kv_indices,
        req_to_token_pool.req_to_token.stride(0),
    )
    bufs.stage_block_tables(req_to_token_pool, forward_batch.req_pool_indices, bs)

    get_mla_metadata_v1(
        bufs.cu_q[: bs + 1],
        bufs.sparse_kv_indptr[: bs + 1],
        bufs.kv_last_page_lens[:bs],
        max(local_num_attention_heads(atom_config), 16),
        1,
        True,
        bufs.work_meta_data,
        bufs.work_info_set,
        bufs.work_indptr,
        bufs.reduce_indptr,
        bufs.reduce_final_map,
        bufs.reduce_partial_map,
        page_size=attn_page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        kv_granularity=max(attn_page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
    )

    setattr(token_to_kv_pool, SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
    md = AttentionMetaData(
        cu_seqlens_q=bufs.cu_q[: bs + 1],
        cu_seqlens_k=bufs.kv_indptr[: bs + 1],
        max_seqlen_q=1,
        max_seqlen_k=int(seq_lens.max().item()) if bs else 1,
        slot_mapping=bufs.slot_mapping[:bs],
        context_lens=bufs.context_lens[:bs],
        block_tables=bufs.block_tables[:bs],
        state=AttnState.DECODE,
        kv_indptr=bufs.kv_indptr[: bs + 1],
        kv_indices=bufs.kv_indices,
        kv_last_page_lens=bufs.kv_last_page_lens[:bs],
        sparse_kv_indptr=bufs.sparse_kv_indptr[: bs + 1],
        work_meta_data=bufs.work_meta_data,
        work_indptr=bufs.work_indptr,
        work_info_set=bufs.work_info_set,
        reduce_indptr=bufs.reduce_indptr,
        reduce_final_map=bufs.reduce_final_map,
        reduce_partial_map=bufs.reduce_partial_map,
    )
    md.sparse_kv_last_page_lens = bufs.kv_last_page_lens[:bs]
    md.dtype_q = dtype_q
    return md
