"""Bridge SGLang ForwardBatch metadata to ATOM GLM-5.2 sparse MLA."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

from atom.plugin.sglang.runtime.model_arch import is_glm52_dsa_config

_DECODE_GRAPH_BUFFERS_ATTR = "_atom_glm52_decode_graph_buffers"
_EMPTY_VALUE_CACHE_ATTR = "_atom_glm52_empty_value_cache"
_INDEXER_PAGE_SIZE_ATTR = "_atom_glm52_indexer_page_size"
_ATTENTION_PAGE_SIZE_ATTR = "_atom_glm52_attention_page_size"
_SHARED_SPARSE_INDICES_ATTR = "_atom_glm52_shared_sparse_kv_indices"
_CHUNK_K_WORKSPACE_ATTR = "_atom_glm52_chunk_k_workspace"
_CHUNK_V_WORKSPACE_ATTR = "_atom_glm52_chunk_v_workspace"


def is_glm52_dsa_arch(config) -> bool:
    return is_glm52_dsa_config(config)


def maybe_get_glm52_dsa_pools_from_sglang_backend(forward_batch=None):
    if forward_batch is not None:
        token_to_kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        if token_to_kv_pool is not None and req_to_token_pool is not None:
            return token_to_kv_pool, req_to_token_pool
    return None, None


def _get_seq_lens_cpu(forward_batch, bs: int) -> np.ndarray:
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()
    if torch.is_tensor(seq_lens_cpu):
        seq_lens_cpu = seq_lens_cpu.detach().cpu().numpy()
    return np.asarray(seq_lens_cpu[:bs], dtype=np.int32)


def _get_extend_lens_cpu(forward_batch, positions: torch.Tensor, bs: int) -> np.ndarray:
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_lens is None:
        extend_lens = getattr(forward_batch, "extend_seq_lens", None)
    if extend_lens is not None:
        if torch.is_tensor(extend_lens):
            extend_lens = extend_lens.detach().cpu().numpy()
        return np.asarray(extend_lens[:bs], dtype=np.int32)

    tokens_per_req = getattr(
        getattr(forward_batch, "spec_info", None), "num_tokens_per_req", None
    )
    if tokens_per_req is None:
        tokens_per_req = max(1, int(positions.numel()) // max(1, bs))
    return np.full(bs, int(tokens_per_req), dtype=np.int32)


def _build_token_table(
    forward_batch,
    req_to_token_pool,
    *,
    seq_lens: np.ndarray,
    extend_lens: np.ndarray | None,
    page_size: int,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    prefix_lens = None
    if extend_lens is not None and not forward_batch.forward_mode.is_decode_or_idle():
        # CUDA graph capture can present allocation-shaped dummy inputs where
        # seq_lens is smaller than the fixed draft width. The token table still
        # needs enough columns for the verifier/draft slots.
        prefix_lens = np.maximum(seq_lens - extend_lens, 0).astype(np.int32)
        table_lens = np.maximum(seq_lens, prefix_lens + extend_lens)
    else:
        table_lens = seq_lens
    max_seq_len = int(table_lens.max(initial=1))
    req_pool_indices = forward_batch.req_pool_indices[:bs]
    token_table = req_to_token_pool.req_to_token[req_pool_indices, :max_seq_len].clone()

    if extend_lens is not None and not forward_batch.forward_mode.is_decode_or_idle():
        offset = 0
        for req_idx, (prefix_len, query_len) in enumerate(
            zip(prefix_lens, extend_lens)
        ):
            prefix_len = int(prefix_len)
            query_len = int(query_len)
            if query_len > 0:
                token_table[req_idx, prefix_len : prefix_len + query_len] = (
                    forward_batch.out_cache_loc[offset : offset + query_len]
                )
            offset += query_len

    if page_size == 1:
        return token_table.to(dtype=torch.int32).contiguous()
    return (token_table[:, ::page_size] // page_size).to(dtype=torch.int32).contiguous()


def _flatten_kv_indices(token_table: torch.Tensor, lengths: np.ndarray) -> torch.Tensor:
    pieces = []
    for row, length in enumerate(lengths):
        if int(length) > 0:
            pieces.append(token_table[row, : int(length)])
    if not pieces:
        return torch.empty(0, dtype=torch.int32, device=token_table.device)
    return torch.cat(pieces).to(dtype=torch.int32).contiguous()


def _counts_to_indptr(counts: np.ndarray, device: torch.device) -> torch.Tensor:
    indptr = np.zeros(len(counts) + 1, dtype=np.int32)
    if len(counts):
        indptr[1:] = np.cumsum(counts, dtype=np.int32)
    return torch.from_numpy(indptr).to(device=device)


def _get_index_topk(atom_config) -> int:
    topk = getattr(atom_config.hf_config, "index_topk", None)
    if topk is None:
        raise RuntimeError("GLM-5.2 DSA bridge requires hf_config.index_topk")
    return int(topk)


def _local_num_attention_heads(atom_config) -> int:
    hf_config = atom_config.hf_config
    num_heads = int(getattr(hf_config, "num_attention_heads"))
    tp_size = int(getattr(atom_config, "tensor_parallel_size", 1))
    return max(1, num_heads // max(1, tp_size))


def _metadata_dtype(atom_config):
    kv_dtype = getattr(atom_config, "kv_cache_dtype", "bf16")
    if str(kv_dtype).startswith("fp8"):
        return dtypes.fp8
    return getattr(dtypes, "d_dtypes", {}).get(kv_dtype, torch.bfloat16)


def _make_mla_work_buffers(
    *,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_heads: int,
    dtype_q,
    dtype_kv,
    page_size: int,
) -> dict[str, torch.Tensor]:
    num_seqs = max(1, int(cu_seqlens_q.numel()) - 1)
    max_q_len = 1
    if cu_seqlens_q.numel() > 1:
        q_counts = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_q_len = max(1, int(q_counts.max().item()))
    padded_heads = max(num_heads, 16)
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_mla_metadata_info_v1(
        num_seqs,
        max_q_len,
        padded_heads,
        dtype_q,
        dtype_kv,
        is_sparse=True,
        fast_mode=True,
    )
    device = cu_seqlens_q.device
    work = {
        "work_meta_data": torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        ),
        "work_indptr": torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        ),
        "work_info_set": torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        ),
        "reduce_indptr": torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        ),
        "reduce_final_map": torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        ),
        "reduce_partial_map": torch.empty(
            reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
        ),
    }
    get_mla_metadata_v1(
        cu_seqlens_q,
        kv_indptr,
        kv_last_page_lens,
        padded_heads,
        1,
        True,
        work["work_meta_data"],
        work["work_info_set"],
        work["work_indptr"],
        work["reduce_indptr"],
        work["reduce_final_map"],
        work["reduce_partial_map"],
        page_size=page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=True,
    )
    return work


def _ensure_shared_sparse_buffer(
    token_to_kv_pool,
    *,
    num_tokens: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    required = max(1, int(num_tokens) * int(topk))
    buffer = getattr(token_to_kv_pool, _SHARED_SPARSE_INDICES_ATTR, None)
    if (
        buffer is None
        or buffer.device != device
        or buffer.dtype != torch.int32
        or buffer.numel() < required
    ):
        buffer = torch.empty(required, dtype=torch.int32, device=device)
        setattr(token_to_kv_pool, _SHARED_SPARSE_INDICES_ATTR, buffer)
    return buffer[:required]


def _ensure_chunk_workspace(
    token_to_kv_pool,
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype,
    device: torch.device,
):
    required = max(1, int(num_tokens))
    k_buf = getattr(token_to_kv_pool, _CHUNK_K_WORKSPACE_ATTR, None)
    v_buf = getattr(token_to_kv_pool, _CHUNK_V_WORKSPACE_ATTR, None)
    if (
        k_buf is None
        or k_buf.device != device
        or k_buf.dtype != dtype
        or k_buf.shape[0] < required
        or k_buf.shape[1] != int(num_heads)
        or k_buf.shape[2] != int(qk_head_dim)
    ):
        k_buf = torch.empty(
            (required, int(num_heads), int(qk_head_dim)),
            dtype=dtype,
            device=device,
        )
        setattr(token_to_kv_pool, _CHUNK_K_WORKSPACE_ATTR, k_buf)
    if (
        v_buf is None
        or v_buf.device != device
        or v_buf.dtype != dtype
        or v_buf.shape[0] < required
        or v_buf.shape[1] != int(num_heads)
        or v_buf.shape[2] != int(v_head_dim)
    ):
        v_buf = torch.empty(
            (required, int(num_heads), int(v_head_dim)),
            dtype=dtype,
            device=device,
        )
        setattr(token_to_kv_pool, _CHUNK_V_WORKSPACE_ATTR, v_buf)
    return k_buf[:required], v_buf[:required]


class _GLM52DecodeGraphBuffers:
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

        work = _make_mla_work_buffers(
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


def _get_or_create_decode_graph_buffers(
    token_to_kv_pool,
    *,
    max_bs: int,
    max_context_len: int,
    indexer_page_size: int,
    attention_page_size: int,
    atom_config,
    device: torch.device,
) -> _GLM52DecodeGraphBuffers:
    topk = _get_index_topk(atom_config)
    dtype_q = _metadata_dtype(atom_config)
    bufs = getattr(token_to_kv_pool, _DECODE_GRAPH_BUFFERS_ATTR, None)
    if (
        bufs is None
        or bufs.max_bs < int(max_bs)
        or bufs.max_context_len < int(max_context_len)
        or bufs.indexer_page_size != int(indexer_page_size)
        or bufs.attention_page_size != int(attention_page_size)
        or bufs.index_topk != int(topk)
        or bufs.device != device
    ):
        bufs = _GLM52DecodeGraphBuffers(
            max_bs=max_bs,
            max_context_len=max_context_len,
            indexer_page_size=indexer_page_size,
            attention_page_size=attention_page_size,
            index_topk=topk,
            num_heads=_local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            device=device,
        )
        setattr(token_to_kv_pool, _DECODE_GRAPH_BUFFERS_ATTR, bufs)
        setattr(token_to_kv_pool, _SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
    return bufs


def _validate_page_size(token_to_kv_pool, atom_config) -> int:
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    from atom.utils import envs

    atom_config.kv_cache_block_size = page_size
    setattr(token_to_kv_pool, _INDEXER_PAGE_SIZE_ATTR, page_size)
    setattr(token_to_kv_pool, _ATTENTION_PAGE_SIZE_ATTR, int(envs.ATOM_MLA_PAGE_SIZE))
    return page_size


def _attention_page_size(token_to_kv_pool) -> int:
    return int(getattr(token_to_kv_pool, _ATTENTION_PAGE_SIZE_ATTR, 1))


def _build_decode_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    del positions
    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    seq_lens = _get_seq_lens_cpu(forward_batch, bs)
    topk = _get_index_topk(atom_config)
    page_size = _validate_page_size(token_to_kv_pool, atom_config)

    cu_q = torch.arange(bs + 1, dtype=torch.int32, device=device)
    block_tables = _build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=None,
        page_size=page_size,
    )
    token_table = _build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=None,
        page_size=1,
    )
    kv_indptr = _counts_to_indptr(seq_lens, device)
    kv_indices = _flatten_kv_indices(token_table, seq_lens)
    # Sparse decode consumes a compact list of selected physical token ids, so
    # each selected entry behaves as a single-token page even when the backing
    # cache is stored in page64/segmented layout.
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)
    sparse_counts = np.minimum(seq_lens, topk).astype(np.int32)
    sparse_kv_indptr = _counts_to_indptr(sparse_counts, device)

    _ensure_shared_sparse_buffer(
        token_to_kv_pool,
        num_tokens=bs,
        topk=topk,
        device=device,
    )
    dtype_q = _metadata_dtype(atom_config)
    work = _make_mla_work_buffers(
        cu_seqlens_q=cu_q,
        kv_indptr=sparse_kv_indptr,
        kv_last_page_lens=kv_last_page_lens,
        num_heads=_local_num_attention_heads(atom_config),
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        page_size=_attention_page_size(token_to_kv_pool),
    )

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr,
        max_seqlen_q=1,
        max_seqlen_k=int(seq_lens.max(initial=1)),
        slot_mapping=forward_batch.out_cache_loc[:bs],
        context_lens=forward_batch.seq_lens[:bs].to(dtype=torch.int32),
        block_tables=block_tables,
        state=AttnState.DECODE,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        sparse_kv_indptr=sparse_kv_indptr,
        **work,
    )
    md.dtype_q = dtype_q
    return md


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

    indexer_page_size = _validate_page_size(token_to_kv_pool, atom_config)
    attention_page_size = _attention_page_size(token_to_kv_pool)
    topk = _get_index_topk(atom_config)
    dtype_q = _metadata_dtype(atom_config)

    bufs = _get_or_create_decode_graph_buffers(
        token_to_kv_pool,
        max_bs=max_bs,
        max_context_len=max_context_len,
        indexer_page_size=indexer_page_size,
        attention_page_size=attention_page_size,
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
        max(_local_num_attention_heads(atom_config), 16),
        1,
        True,
        bufs.work_meta_data,
        bufs.work_info_set,
        bufs.work_indptr,
        bufs.reduce_indptr,
        bufs.reduce_final_map,
        bufs.reduce_partial_map,
        page_size=attention_page_size,
        dtype_q=dtype_q,
        dtype_kv=dtype_q,
        kv_granularity=max(attention_page_size, 16),
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
    )

    setattr(token_to_kv_pool, _SHARED_SPARSE_INDICES_ATTR, bufs.shared_sparse)
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
    md.dtype_q = dtype_q
    return md


def _build_prefill_metadata(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    from atom.utils.forward_context import AttentionMetaData, AttnState

    device = positions.device
    bs = int(forward_batch.batch_size)
    seq_lens = _get_seq_lens_cpu(forward_batch, bs)
    is_target_verify = forward_batch.forward_mode.is_target_verify()
    if is_target_verify:
        draft_token_num = int(
            getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0)
            or 0
        )
        if draft_token_num <= 0:
            raise RuntimeError("GLM-5.2 DSA target_verify requires draft_token_num")
        extend_lens = np.full(bs, draft_token_num, dtype=np.int32)
        position_rows = positions.detach().cpu().numpy().astype(np.int32)
        if position_rows.size < bs * draft_token_num:
            raise RuntimeError(
                "GLM-5.2 DSA target_verify positions are shorter than "
                f"bs*draft_token_num: positions={position_rows.size}, "
                f"bs={bs}, draft_token_num={draft_token_num}"
            )
        # In graph replay, forward_batch.seq_lens may be a padded/capture tensor.
        # The verifier positions are the reliable SGLang token-layout truth:
        # [prefix, prefix+1, ...] per request.
        seq_lens = position_rows[: bs * draft_token_num : draft_token_num]
        # TARGET_VERIFY appends verifier draft slots after the committed prefix.
        # The attention metadata must expose those slots as part of the KV range.
        seq_lens = seq_lens + extend_lens
        force_total_kv = int(os.environ.get("ATOM_GLM52_TV_FORCE_TOTAL_KV", "0") or 0)
        if force_total_kv > 0:
            seq_lens = np.maximum(seq_lens, force_total_kv).astype(np.int32)
    else:
        extend_lens = _get_extend_lens_cpu(forward_batch, positions, bs)
        if getattr(
            forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False
        )(include_v2=True):
            num_position_rows = int(positions.numel())
            if int(extend_lens.sum()) != num_position_rows and bs > 0:
                if num_position_rows % bs != 0:
                    raise RuntimeError(
                        "GLM-5.2 DSA draft_extend positions cannot be evenly "
                        f"distributed: positions={num_position_rows}, bs={bs}, "
                        f"extend_lens_sum={int(extend_lens.sum())}"
                    )
                extend_lens = np.full(
                    bs, num_position_rows // bs, dtype=np.int32
                )
    topk = _get_index_topk(atom_config)
    page_size = _validate_page_size(token_to_kv_pool, atom_config)
    cached_lens = np.maximum(seq_lens - extend_lens, 0).astype(np.int32)
    seq_lens = np.maximum(seq_lens, cached_lens + extend_lens).astype(np.int32)

    q_np = np.zeros(bs + 1, dtype=np.int32)
    q_np[1:] = np.cumsum(extend_lens, dtype=np.int32)
    cu_q = torch.from_numpy(q_np).to(device=device)
    kv_indptr = _counts_to_indptr(seq_lens, device)
    block_tables = _build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=page_size,
    )
    token_table = _build_token_table(
        forward_batch,
        req_to_token_pool,
        seq_lens=seq_lens,
        extend_lens=extend_lens,
        page_size=1,
    )
    kv_indices = _flatten_kv_indices(token_table, seq_lens)
    has_cached = bool(np.any(cached_lens > 0))
    state = AttnState.PREFILL_PREFIX if has_cached else AttnState.PREFILL_NATIVE
    total_tokens = int(extend_lens.sum())
    kv_last_page_lens = torch.ones(bs, dtype=torch.int32, device=device)

    md = AttentionMetaData(
        cu_seqlens_q=cu_q,
        cu_seqlens_k=kv_indptr if has_cached else cu_q,
        max_seqlen_q=int(extend_lens.max(initial=1)),
        max_seqlen_k=int(seq_lens.max(initial=1)),
        slot_mapping=forward_batch.out_cache_loc[:total_tokens],
        context_lens=(
            torch.from_numpy((seq_lens - extend_lens).astype(np.int32)).to(
                device=device
            )
            if is_target_verify
            else forward_batch.seq_lens[:bs].to(dtype=torch.int32)
        ),
        block_tables=block_tables,
        state=state,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_lens=kv_last_page_lens,
        has_cached=has_cached,
        total_kv=int(seq_lens.sum()),
        num_cached_tokens=torch.from_numpy(cached_lens).to(device=device),
        seq_starts=torch.from_numpy(cached_lens).to(device=device),
    )
    dtype_q = _metadata_dtype(atom_config)
    md.dtype_q = dtype_q

    if (
        is_target_verify
        and has_cached
        and os.environ.get("ATOM_GLM52_TV_FORCE_CHUNKED", "0").lower()
        in ("1", "true", "yes", "on")
    ):
        prefix_lens = cached_lens.astype(np.int32)
        prefix_indptr = _counts_to_indptr(prefix_lens, device)
        prefix_indices = _flatten_kv_indices(token_table, prefix_lens)
        num_heads = _local_num_attention_heads(atom_config)
        qk_head_dim = int(getattr(atom_config.hf_config, "qk_nope_head_dim")) + int(
            getattr(atom_config.hf_config, "qk_rope_head_dim")
        )
        v_head_dim = int(getattr(atom_config.hf_config, "v_head_dim"))
        k_workspace, v_workspace = _ensure_chunk_workspace(
            token_to_kv_pool,
            num_tokens=int(prefix_lens.sum()),
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dtype=getattr(atom_config, "torch_dtype", torch.bfloat16),
            device=device,
        )
        md.mla_chunk_meta = SimpleNamespace(
            kv_indptr=[prefix_indptr],
            kv_indices=[prefix_indices],
            cu_seqlens_k=[prefix_indptr],
            total_tokens=[int(prefix_lens.sum())],
            max_seqlen_k=[int(prefix_lens.max(initial=0))],
            num_chunks=1,
            k_workspace=k_workspace,
            v_workspace=v_workspace,
            shuffle_kv_block_indptr=None,
            shuffle_kv_block_indices=None,
        )

    if md.max_seqlen_k > topk:
        counts = extend_lens.astype(np.int32)
        local_offsets = np.concatenate(
            [np.arange(int(count), dtype=np.int32) for count in counts]
        )
        if has_cached:
            seq_starts = kv_indptr[:-1].detach().cpu().numpy().astype(np.int32)
            repeated_seq_starts = np.repeat(seq_starts, counts)
            repeated_cached_lens = np.repeat(cached_lens, counts)
            cu_ks = repeated_seq_starts
            cu_ke = repeated_seq_starts + repeated_cached_lens + local_offsets + 1
            sparse_counts = repeated_cached_lens + local_offsets + 1
        else:
            cu_ks = np.repeat(q_np[:bs], counts)
            cu_ke = np.arange(total_tokens, dtype=np.int32) + 1
            sparse_counts = local_offsets + 1

        sparse_cu = torch.arange(total_tokens + 1, dtype=torch.int32, device=device)
        sparse_kv_indptr = _counts_to_indptr(
            np.minimum(sparse_counts, topk).astype(np.int32), device
        )
        sparse_last_page_lens = torch.ones(
            total_tokens, dtype=torch.int32, device=device
        )
        md.cu_seqlen_ks = torch.from_numpy(cu_ks.astype(np.int32)).to(device=device)
        md.cu_seqlen_ke = torch.from_numpy(cu_ke.astype(np.int32)).to(device=device)
        md.sparse_cu_seqlens_q = sparse_cu
        md.sparse_kv_indptr = sparse_kv_indptr
        md.kv_last_page_lens = sparse_last_page_lens
        md.token_to_seq_idxs = torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32, device=device),
            torch.from_numpy(counts.astype(np.int64)).to(device=device),
        )
        _ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=total_tokens,
            topk=topk,
            device=device,
        )
        sparse_work = _make_mla_work_buffers(
            cu_seqlens_q=sparse_cu,
            kv_indptr=sparse_kv_indptr,
            kv_last_page_lens=sparse_last_page_lens,
            num_heads=_local_num_attention_heads(atom_config),
            dtype_q=dtype_q,
            dtype_kv=dtype_q,
            page_size=_attention_page_size(token_to_kv_pool),
        )
        for key, value in sparse_work.items():
            setattr(md, f"sparse_prefill_{key}", value)
    else:
        _ensure_shared_sparse_buffer(
            token_to_kv_pool,
            num_tokens=max(1, total_tokens),
            topk=topk,
            device=device,
        )

    return md


def build_atom_glm52_attention_metadata_from_sglang(
    forward_batch,
    positions: torch.Tensor,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    if forward_batch.forward_mode.is_decode_or_idle():
        return _build_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    return _build_prefill_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
    )


def bind_glm52_dsa_cache_views(model, token_to_kv_pool) -> bool:
    if token_to_kv_pool is None or not hasattr(token_to_kv_pool, "get_key_buffer"):
        return False
    if not hasattr(token_to_kv_pool, "get_index_k_with_scale_buffer"):
        return False

    from atom.config import KVCacheTensor
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention
    from atom.utils.forward_context import get_forward_context, set_kv_cache_data

    shared_sparse = getattr(token_to_kv_pool, _SHARED_SPARSE_INDICES_ATTR, None)
    if shared_sparse is None:
        return False

    page_size = int(
        getattr(
            token_to_kv_pool,
            _INDEXER_PAGE_SIZE_ATTR,
            getattr(token_to_kv_pool, "page_size", 1),
        )
    )
    empty_value_cache = getattr(token_to_kv_pool, _EMPTY_VALUE_CACHE_ATTR, None)
    if empty_value_cache is None or empty_value_cache.device != shared_sparse.device:
        empty_value_cache = torch.empty(0, device=shared_sparse.device)
        setattr(token_to_kv_pool, _EMPTY_VALUE_CACHE_ATTR, empty_value_cache)
    kv_cache_data = {}
    for module in model.modules():
        if not isinstance(module, DeepseekV2MLAAttention):
            continue

        layer_id = int(module.layer_num)
        kv_cache_data[f"layer_{layer_id}"] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=token_to_kv_pool.get_key_buffer(layer_id),
            v_cache=empty_value_cache,
            k_scale=getattr(module.mla_attn, "_k_scale", None),
            v_scale=getattr(module.mla_attn, "_k_scale", None),
        )

        indexer = getattr(module, "indexer", None)
        if indexer is not None:
            index_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
            index_entry_dim = int(getattr(indexer, "head_dim")) + 4
            indexer.k_cache.kv_cache[0] = index_cache.view(
                -1, page_size, index_entry_dim
            )
            indexer.sparse_kv_indices_buffer = shared_sparse

        if hasattr(module.mla_attn, "sparse_kv_indices_buffer"):
            module.mla_attn.sparse_kv_indices_buffer = shared_sparse

    if not kv_cache_data:
        return False

    set_kv_cache_data(kv_cache_data)
    get_forward_context().kv_cache_data = kv_cache_data
    return True
