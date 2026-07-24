# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang plugin sparse MLA indexer support for DeepSeek-V3.2."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re
from typing import Optional

import numpy as np
import torch
from aiter import (
    cp_gather_indexer_k_quant_cache,
    dtypes,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
    indexer_k_quant_and_cache,
    indexer_qk_rope_quant_and_cache,
    top_k_per_row_decode,
    top_k_per_row_prefill,
)
from aiter.mla import mla_decode_fwd
from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
import triton
import triton.language as tl

from atom.utils.custom_register import direct_register_custom_op


logger = logging.getLogger("atom")


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _is_graph_warmup_or_capture() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return _is_stream_capturing()


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,
    block_table_ptr,
    token_indices_ptr,
    cu_seqlens_ptr,
    out_ptr,
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    req = tl.load(req_id_ptr + token_id)
    seq_start = tl.load(cu_seqlens_ptr + token_id)
    seq_end = tl.load(cu_seqlens_ptr + token_id + 1)
    if tile_id * BLOCK_N + seq_start >= seq_end:
        return

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)
    is_invalid_tok = tok < 0

    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    out_val = tl.where(
        is_invalid_tok | (~valid_block), 0, base * BLOCK_SIZE + inblock_off
    )
    out_ptr_ij = out_ptr + seq_start + indice_id
    out_ptr_ij_mask = (seq_start + indice_id) < seq_end
    tl.store(out_ptr_ij, out_val, mask=out_ptr_ij_mask)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0

    num_tokens = req_id.shape[0]
    _, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()

    grid = (num_tokens, tiles_per_row)
    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        cu_seqlens,
        paged_kv_indices,
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
    )


@triton.jit
def _build_graph_block_table_kernel(
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_table_ptr,
    req_to_token_stride0,
    block_table_stride0,
    max_num_blocks: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_offset = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = block_offset < max_num_blocks
    req_id = tl.load(req_pool_indices_ptr + batch_id)
    token_slot = tl.load(
        req_to_token_ptr
        + req_id * req_to_token_stride0
        + block_offset * PAGE_SIZE,
        mask=mask,
        other=0,
    )
    tl.store(
        block_table_ptr + batch_id * block_table_stride0 + block_offset,
        token_slot // PAGE_SIZE,
        mask=mask,
    )


@triton.jit
def _patch_graph_block_table_kernel(
    positions_ptr,
    out_cache_loc_ptr,
    block_table_ptr,
    block_table_stride0,
    max_num_blocks: tl.constexpr,
    TOKENS_PER_REQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    req_id = token_id // TOKENS_PER_REQ
    position = tl.load(positions_ptr + token_id)
    block_id = position // PAGE_SIZE
    slot = tl.load(out_cache_loc_ptr + token_id)
    mask = (block_id >= 0) & (block_id < max_num_blocks)
    # Match the eager token-table slice: only a page's first token changes
    # the page base retained by token_table[:, ::PAGE_SIZE].
    mask &= (position % PAGE_SIZE) == 0
    tl.store(
        block_table_ptr + req_id * block_table_stride0 + block_id,
        slot // PAGE_SIZE,
        mask=mask,
    )


@triton.jit
def _count_valid_topk_kernel(
    topk_indices_ptr,
    seq_len_ptr,
    topk_stride0,
    TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    values = tl.load(
        topk_indices_ptr + token_id * topk_stride0 + offsets,
        mask=offsets < TOPK_TOKENS,
        other=-1,
    )
    count = tl.sum(values >= 0, axis=0)
    tl.store(seq_len_ptr + token_id, count)


@triton.jit
def _build_graph_query_ranges_kernel(
    positions_ptr,
    starts_ptr,
    ends_ptr,
    TOKENS_PER_REQ: tl.constexpr,
    CONTEXT_CAPACITY: tl.constexpr,
):
    token_id = tl.program_id(0)
    req_id = token_id // TOKENS_PER_REQ
    req_base = req_id * CONTEXT_CAPACITY
    position = tl.load(positions_ptr + token_id)
    tl.store(starts_ptr + token_id, req_base)
    tl.store(ends_ptr + token_id, req_base + position + 1)


def _parse_layer_id_from_indexer_prefix(prefix: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", prefix)
    if match is None:
        raise RuntimeError(
            f"Cannot infer DeepSeek-V3.2 indexer layer id from prefix: {prefix!r}"
        )
    return int(match.group(1))


def _is_draft_extend_v2(forward_batch) -> bool:
    return bool(
        getattr(forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False)(
            include_v2=True
        )
    )


def _mtp_spec_tokens_per_req(forward_batch) -> int:
    spec_info = getattr(forward_batch, "spec_info", None)
    tokens_per_req = int(
        getattr(spec_info, "draft_token_num", 0)
        or getattr(spec_info, "num_tokens_per_req", 0)
        or 0
    )
    if tokens_per_req > 0:
        return tokens_per_req
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    positions = getattr(forward_batch, "positions", None)
    if bs > 0 and torch.is_tensor(positions) and int(positions.numel()) >= bs:
        return max(1, int(positions.numel()) // bs)
    return 0


def _is_mtp_spec_extend_like(forward_batch) -> bool:
    """Eager MTP phases with bs * K query rows (target_verify / draft_extend decode)."""
    if forward_batch.forward_mode.is_target_verify():
        return True
    if not _is_draft_extend_v2(forward_batch):
        return False
    override = os.environ.get("ATOM_GLM52_DRAFT_EXTEND_PATH", "").lower()
    return override not in ("prefill", "prefill_prefix")


def _build_mtp_spec_query_ranges(
    forward_batch,
    *,
    tokens_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = forward_batch.seq_lens.device
    bs = int(forward_batch.batch_size)
    if tokens_per_req <= 0:
        raise RuntimeError("MTP sparse MLA requires tokens_per_req > 0")
    positions = getattr(forward_batch, "positions", None)
    if positions is None:
        positions = getattr(getattr(forward_batch, "spec_info", None), "positions", None)
    if torch.is_tensor(positions) and int(positions.numel()) >= bs * tokens_per_req:
        prefix_lens = positions[: bs * tokens_per_req : tokens_per_req].to(
            dtype=torch.int32
        )
    else:
        prefix_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
    seq_lens = prefix_lens
    kv_lens = seq_lens + tokens_per_req
    base_offsets = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int32, device=device), kv_lens[:-1]]),
        dim=0,
    )
    starts = torch.repeat_interleave(base_offsets, tokens_per_req)
    per_req_end_base = torch.repeat_interleave(base_offsets + seq_lens, tokens_per_req)
    draft_offsets = torch.arange(
        1, tokens_per_req + 1, dtype=torch.int32, device=device
    ).repeat(bs)
    return starts.to(dtype=torch.int32), (per_req_end_base + draft_offsets).to(
        dtype=torch.int32
    )


def _mtp_eager_context_lens(forward_batch) -> torch.Tensor | None:
    if not _is_mtp_spec_extend_like(forward_batch):
        return None
    bs = int(forward_batch.batch_size)
    tokens_per_req = _mtp_spec_tokens_per_req(forward_batch)
    if tokens_per_req <= 0:
        return None
    positions = getattr(forward_batch, "positions", None)
    if positions is None:
        positions = getattr(getattr(forward_batch, "spec_info", None), "positions", None)
    if torch.is_tensor(positions) and int(positions.numel()) >= bs * tokens_per_req:
        prefix_lens = positions[: bs * tokens_per_req : tokens_per_req].to(
            dtype=torch.int32
        )
    else:
        prefix_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
    return prefix_lens + int(tokens_per_req)


def _build_mtp_eager_gather_indptr(forward_batch) -> torch.Tensor:
    context_lens = _mtp_eager_context_lens(forward_batch)
    if context_lens is None:
        return torch.nn.functional.pad(
            torch.cumsum(forward_batch.seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
    return torch.nn.functional.pad(
        torch.cumsum(context_lens, dim=0, dtype=torch.int32), (1, 0)
    )


def _mtp_eager_total_kv(forward_batch) -> int:
    context_lens = _mtp_eager_context_lens(forward_batch)
    if context_lens is None:
        return int(forward_batch.seq_lens_sum)
    return max(1, int(context_lens.sum().item()))


def _build_sglang_query_ranges(forward_batch) -> tuple[torch.Tensor, torch.Tensor]:
    device = forward_batch.seq_lens.device
    if forward_batch.forward_mode.is_decode_or_idle():
        bs = int(forward_batch.batch_size)
        starts = torch.zeros(bs, dtype=torch.int32, device=device)
        ends = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
        return starts, ends

    if _is_mtp_spec_extend_like(forward_batch):
        return _build_mtp_spec_query_ranges(
            forward_batch,
            tokens_per_req=_mtp_spec_tokens_per_req(forward_batch),
        )

    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = forward_batch.seq_lens
    query_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if query_lens_cpu is None:
        query_lens_cpu = query_lens.detach().cpu()
    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = forward_batch.seq_lens.detach().cpu()

    starts = []
    ends = []
    kv_offset = 0
    for q_len_raw, seq_len_raw in zip(query_lens_cpu, seq_lens_cpu):
        q_len = int(q_len_raw)
        seq_len = int(seq_len_raw)
        prefix_len = seq_len - q_len
        starts.extend([kv_offset] * q_len)
        ends.extend(kv_offset + prefix_len + i + 1 for i in range(q_len))
        kv_offset += seq_len

    return (
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(ends, dtype=torch.int32, device=device),
    )


def _build_sglang_block_table(forward_batch, page_size: int) -> torch.Tensor:
    req_pool_indices = forward_batch.req_pool_indices
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    token_table = req_to_token[req_pool_indices, :]
    if not forward_batch.forward_mode.is_decode_or_idle():
        token_table = token_table.clone()
        bs = int(forward_batch.batch_size)
        if _is_mtp_spec_extend_like(forward_batch):
            tokens_per_req = _mtp_spec_tokens_per_req(forward_batch)
            if tokens_per_req <= 0:
                raise RuntimeError("MTP sparse MLA requires tokens_per_req > 0")
            positions = getattr(forward_batch, "positions", None)
            if positions is None:
                positions = getattr(
                    getattr(forward_batch, "spec_info", None), "positions", None
                )
            if (
                torch.is_tensor(positions)
                and int(positions.numel()) >= bs * tokens_per_req
            ):
                prefix_lens_cpu = (
                    positions[: bs * tokens_per_req : tokens_per_req]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
            else:
                seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
                if seq_lens_cpu is None:
                    seq_lens_cpu = forward_batch.seq_lens[:bs].detach().cpu()
                prefix_lens_cpu = np.asarray(seq_lens_cpu[:bs], dtype=np.int32)
            query_lens_cpu = [tokens_per_req] * bs
        else:
            query_lens = getattr(forward_batch, "extend_seq_lens", None)
            if query_lens is None:
                query_lens = forward_batch.seq_lens
            prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
            if prefix_lens is None:
                prefix_lens = forward_batch.seq_lens - query_lens
            query_lens_cpu = query_lens[:bs].detach().cpu()
            prefix_lens_cpu = prefix_lens[:bs].detach().cpu()
        offset = 0
        for req_idx, (prefix_len_raw, query_len_raw) in enumerate(
            zip(prefix_lens_cpu, query_lens_cpu)
        ):
            prefix_len = int(prefix_len_raw)
            query_len = int(query_len_raw)
            remaining = int(forward_batch.out_cache_loc.shape[0]) - offset
            query_len = min(query_len, max(remaining, 0))
            if query_len > 0:
                token_table[req_idx, prefix_len : prefix_len + query_len] = (
                    forward_batch.out_cache_loc[offset : offset + query_len]
                )
            offset += query_len
    if page_size == 1:
        return token_table.to(dtype=torch.int32).contiguous()
    return (token_table[:, ::page_size] // page_size).to(dtype=torch.int32).contiguous()


def _build_sparse_req_id_per_token_for_sglang(
    forward_batch,
    device: torch.device,
) -> torch.Tensor:
    bs = int(forward_batch.batch_size)
    req_ids = torch.arange(bs, dtype=torch.int32, device=device)
    if forward_batch.forward_mode.is_decode_or_idle():
        return req_ids
    if _is_mtp_spec_extend_like(forward_batch):
        tokens_per_req = _mtp_spec_tokens_per_req(forward_batch)
        if tokens_per_req <= 0:
            raise RuntimeError("MTP sparse MLA requires tokens_per_req > 0")
        return torch.repeat_interleave(req_ids, tokens_per_req)
    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = forward_batch.seq_lens
    return torch.repeat_interleave(req_ids, query_lens[:bs].to(torch.int32))


def _supports_sparse_mla_fast_metadata(
    nhead: int,
    *,
    max_seqlen_qo: int,
    uni_seqlen_qo: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> bool:
    """Whether AITER get_mla_metadata_v1 supports this sparse MLA shape."""
    if nhead in (16, 64, 128):
        return True
    if uni_seqlen_qo == 1 and nhead % 16 == 0 and 2 <= nhead // 16 < 8:
        return True
    if nhead == 8 and max_seqlen_qo == 4:
        return (q_dtype == dtypes.fp8 and kv_dtype == dtypes.fp8) or (
            q_dtype == dtypes.bf16 and kv_dtype == dtypes.bf16
        )
    return False


@dataclass
class SparseMLAKernelMetadata:
    allocator_page_size: int
    topk_tokens: int
    req_id_per_token: torch.Tensor
    block_table: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    use_fast_metadata: bool
    work_metadata: Optional[torch.Tensor] = None
    work_indptr: Optional[torch.Tensor] = None
    work_info_set: Optional[torch.Tensor] = None
    reduce_indptr: Optional[torch.Tensor] = None
    reduce_final_map: Optional[torch.Tensor] = None
    reduce_partial_map: Optional[torch.Tensor] = None


@dataclass
class SparseMLAGraphBuffers:
    batch_size: int
    tokens_per_req: int
    num_tokens: int
    topk_tokens: int
    allocator_page_size: int
    q: torch.Tensor
    output: torch.Tensor
    seq_len: torch.Tensor
    req_id_per_token: torch.Tensor
    block_table: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    work_metadata: Optional[torch.Tensor]
    work_indptr: Optional[torch.Tensor]
    work_info_set: Optional[torch.Tensor]
    reduce_indptr: Optional[torch.Tensor]
    reduce_final_map: Optional[torch.Tensor]
    reduce_partial_map: Optional[torch.Tensor]


@dataclass
class SparseMLAIndexerGraphBuffers:
    batch_size: int
    tokens_per_req: int
    num_tokens: int
    context_capacity: int
    block_table: torch.Tensor
    gather_indptr: torch.Tensor
    cu_starts: torch.Tensor
    cu_ends: torch.Tensor
    k_fp8: torch.Tensor
    k_scale: torch.Tensor


def init_sparse_mla_graph_state(backend, max_bs: int, max_num_tokens: int) -> None:
    backend._atom_sparse_mla_graph_buffers = {}
    backend._atom_sparse_mla_indexer_graph_buffers = {}
    backend._atom_sparse_mla_graph_max_bs = int(max_bs)
    backend._atom_sparse_mla_graph_max_num_tokens = int(max_num_tokens)


def _get_or_create_sparse_mla_indexer_graph_buffers(
    *,
    forward_batch,
    page_size: int,
    head_dim: int,
) -> SparseMLAIndexerGraphBuffers:
    backend = getattr(forward_batch, "attn_backend", None)
    store = getattr(backend, "_atom_sparse_mla_indexer_graph_buffers", None)
    if store is None:
        raise RuntimeError(
            "Sparse MLA indexer graph buffers were not initialized on the backend"
        )
    batch_size = int(forward_batch.batch_size)
    tokens_per_req = _mtp_spec_tokens_per_req(forward_batch)
    num_tokens = int(forward_batch.input_ids.shape[0])
    seq_len_fill = int(
        os.environ.get("ATOM_GLM52_DRAFT_EXTEND_CG_SEQ_LEN_FILL", "4096")
    )
    context_capacity = seq_len_fill + tokens_per_req
    if batch_size <= 0 or tokens_per_req <= 0 or batch_size * tokens_per_req != num_tokens:
        raise RuntimeError(
            "Sparse MLA indexer graph requires fixed bs * tokens_per_req rows: "
            f"batch_size={batch_size}, tokens_per_req={tokens_per_req}, "
            f"num_tokens={num_tokens}"
        )
    key = (
        batch_size,
        tokens_per_req,
        context_capacity,
        page_size,
        head_dim,
    )
    buffers = store.get(key)
    if buffers is not None:
        return buffers
    if _is_stream_capturing():
        raise RuntimeError(
            "Sparse MLA indexer graph bucket was not allocated during warmup: "
            f"bs={batch_size}, tokens_per_req={tokens_per_req}"
        )
    device = forward_batch.input_ids.device
    max_num_blocks = (context_capacity + page_size - 1) // page_size
    buffers = SparseMLAIndexerGraphBuffers(
        batch_size=batch_size,
        tokens_per_req=tokens_per_req,
        num_tokens=num_tokens,
        context_capacity=context_capacity,
        block_table=torch.empty(
            (batch_size, max_num_blocks), dtype=torch.int32, device=device
        ),
        gather_indptr=torch.arange(
            0,
            (batch_size + 1) * context_capacity,
            context_capacity,
            dtype=torch.int32,
            device=device,
        ),
        cu_starts=torch.empty(num_tokens, dtype=torch.int32, device=device),
        cu_ends=torch.empty(num_tokens, dtype=torch.int32, device=device),
        k_fp8=torch.empty(
            (batch_size * context_capacity, head_dim),
            dtype=dtypes.fp8,
            device=device,
        ),
        k_scale=torch.empty(
            (batch_size * context_capacity, 1),
            dtype=torch.float32,
            device=device,
        ),
    )
    store[key] = buffers
    logger.info(
        "[ATOM_SPARSE_MLA_INDEXER_GRAPH] allocate mode=DRAFT_EXTEND_V2 "
        "bs=%d tokens_per_req=%d tokens=%d context_capacity=%d "
        "block_table_ptr=%d k_fp8_ptr=%d",
        batch_size,
        tokens_per_req,
        num_tokens,
        context_capacity,
        buffers.block_table.data_ptr(),
        buffers.k_fp8.data_ptr(),
    )
    return buffers


def _prepare_sparse_mla_indexer_graph_buffers(
    *,
    buffers: SparseMLAIndexerGraphBuffers,
    forward_batch,
) -> None:
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    max_num_blocks = int(buffers.block_table.shape[1])
    page_size = int(forward_batch.token_to_kv_pool.page_size)
    block_n = 128
    _build_graph_block_table_kernel[
        (buffers.batch_size, triton.cdiv(max_num_blocks, block_n))
    ](
        forward_batch.req_pool_indices,
        req_to_token,
        buffers.block_table,
        req_to_token.stride(0),
        buffers.block_table.stride(0),
        max_num_blocks,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
    )
    _patch_graph_block_table_kernel[(buffers.num_tokens,)](
        forward_batch.positions,
        forward_batch.out_cache_loc,
        buffers.block_table,
        buffers.block_table.stride(0),
        max_num_blocks,
        TOKENS_PER_REQ=buffers.tokens_per_req,
        PAGE_SIZE=page_size,
    )
    _build_graph_query_ranges_kernel[(buffers.num_tokens,)](
        forward_batch.positions,
        buffers.cu_starts,
        buffers.cu_ends,
        TOKENS_PER_REQ=buffers.tokens_per_req,
        CONTEXT_CAPACITY=buffers.context_capacity,
    )


def _allocate_sparse_mla_graph_buffers(
    *,
    backend,
    batch_size: int,
    tokens_per_req: int,
    num_tokens: int,
    topk_tokens: int,
    allocator_page_size: int,
    q: torch.Tensor,
    q_dtype: torch.dtype,
    k_buffer: torch.Tensor,
    num_heads: int,
    output_dtype: torch.dtype,
    output_dim: int,
) -> SparseMLAGraphBuffers:
    max_num_tokens = int(
        getattr(backend, "_atom_sparse_mla_graph_max_num_tokens", 0)
    )
    if num_tokens > max_num_tokens:
        raise RuntimeError(
            "Sparse MLA graph token count exceeds backend capacity: "
            f"tokens={num_tokens}, capacity={max_num_tokens}"
        )
    max_bs = int(getattr(backend, "_atom_sparse_mla_graph_max_bs", 0))
    if batch_size > max_bs:
        raise RuntimeError(
            "Sparse MLA graph batch size exceeds backend capacity: "
            f"batch_size={batch_size}, capacity={max_bs}"
        )
    if batch_size <= 0 or tokens_per_req <= 0 or batch_size * tokens_per_req != num_tokens:
        raise RuntimeError(
            "Sparse MLA graph requires a fixed bs * tokens_per_req layout: "
            f"batch_size={batch_size}, tokens_per_req={tokens_per_req}, "
            f"num_tokens={num_tokens}"
        )
    req_to_token = backend.req_to_token
    max_num_blocks = (int(req_to_token.shape[1]) + allocator_page_size - 1) // (
        allocator_page_size
    )
    buffer_specs = get_mla_metadata_info_v1(
        num_tokens,
        1,
        num_heads,
        q_dtype,
        k_buffer.dtype,
        is_sparse=True,
        fast_mode=True,
    )
    work_buffers = [
        torch.empty(size, dtype=dtype, device=q.device)
        for size, dtype in buffer_specs
    ]
    return SparseMLAGraphBuffers(
        batch_size=batch_size,
        tokens_per_req=tokens_per_req,
        num_tokens=num_tokens,
        topk_tokens=topk_tokens,
        allocator_page_size=allocator_page_size,
        q=torch.empty(q.shape, dtype=q_dtype, device=q.device),
        output=torch.empty(
            (num_tokens, num_heads, output_dim),
            dtype=output_dtype,
            device=q.device,
        ),
        seq_len=torch.empty(num_tokens, dtype=torch.int32, device=q.device),
        req_id_per_token=torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.int32, device=q.device),
            tokens_per_req,
        ),
        block_table=torch.empty(
            (batch_size, max_num_blocks), dtype=torch.int32, device=q.device
        ),
        qo_indptr=torch.arange(
            num_tokens + 1, dtype=torch.int32, device=q.device
        ),
        kv_indptr=torch.empty(
            num_tokens + 1, dtype=torch.int32, device=q.device
        ),
        kv_indices=torch.empty(
            num_tokens * topk_tokens, dtype=torch.int32, device=q.device
        ),
        last_page_len=torch.ones(
            num_tokens, dtype=torch.int32, device=q.device
        ),
        work_metadata=work_buffers[0],
        work_indptr=work_buffers[1],
        work_info_set=work_buffers[2],
        reduce_indptr=work_buffers[3],
        reduce_final_map=work_buffers[4],
        reduce_partial_map=work_buffers[5],
    )


def _get_or_create_sparse_mla_graph_buffers(
    *,
    forward_batch,
    batch_size: int,
    tokens_per_req: int,
    num_tokens: int,
    topk_tokens: int,
    allocator_page_size: int,
    q: torch.Tensor,
    q_dtype: torch.dtype,
    k_buffer: torch.Tensor,
    num_heads: int,
    output_dtype: torch.dtype,
    output_dim: int,
) -> Optional[SparseMLAGraphBuffers]:
    if not _is_graph_warmup_or_capture():
        return None
    backend = getattr(forward_batch, "attn_backend", None)
    store = getattr(backend, "_atom_sparse_mla_graph_buffers", None)
    if store is None:
        raise RuntimeError(
            "Sparse MLA graph buffers were not initialized on the per-step backend"
        )
    key = (
        batch_size,
        tokens_per_req,
        num_tokens,
        topk_tokens,
        allocator_page_size,
        q_dtype,
        k_buffer.dtype,
        num_heads,
        output_dtype,
        output_dim,
    )
    buffers = store.get(key)
    if buffers is None:
        if _is_stream_capturing():
            raise RuntimeError(
                "Sparse MLA graph bucket was not allocated during warmup: "
                f"tokens={num_tokens}, topk={topk_tokens}"
            )
        buffers = _allocate_sparse_mla_graph_buffers(
            backend=backend,
            batch_size=batch_size,
            tokens_per_req=tokens_per_req,
            num_tokens=num_tokens,
            topk_tokens=topk_tokens,
            allocator_page_size=allocator_page_size,
            q=q,
            q_dtype=q_dtype,
            k_buffer=k_buffer,
            num_heads=num_heads,
            output_dtype=output_dtype,
            output_dim=output_dim,
        )
        store[key] = buffers
        forward_mode = getattr(forward_batch, "forward_mode", None)
        logger.info(
            "[ATOM_SPARSE_MLA_GRAPH] allocate mode=%s bs=%d tokens_per_req=%d "
            "tokens=%d topk=%d page_size=%d q_ptr=%d output_ptr=%d "
            "kv_indices_ptr=%d backend=%s",
            getattr(forward_mode, "name", str(forward_mode)),
            batch_size,
            tokens_per_req,
            num_tokens,
            topk_tokens,
            allocator_page_size,
            buffers.q.data_ptr(),
            buffers.output.data_ptr(),
            buffers.kv_indices.data_ptr(),
            type(backend).__name__,
        )
    return buffers


def _prepare_sparse_mla_graph_metadata(
    *,
    buffers: SparseMLAGraphBuffers,
    forward_batch,
    topk_indices: torch.Tensor,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    num_heads: int,
) -> SparseMLAKernelMetadata:
    num_tokens = buffers.num_tokens
    batch_size = int(forward_batch.batch_size)
    if batch_size != buffers.batch_size:
        raise RuntimeError(
            "Sparse MLA graph batch size changed after bucket allocation: "
            f"batch_size={batch_size}, fixed_batch_size={buffers.batch_size}"
        )
    if batch_size * buffers.tokens_per_req != num_tokens:
        raise RuntimeError(
            "Sparse MLA graph token layout changed after bucket allocation: "
            f"batch_size={batch_size}, tokens_per_req={buffers.tokens_per_req}, "
            f"num_tokens={num_tokens}"
        )
    buffers.q.copy_(q)
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    max_num_blocks = int(buffers.block_table.shape[1])
    block_n = 128
    _build_graph_block_table_kernel[
        (batch_size, triton.cdiv(max_num_blocks, block_n))
    ](
        forward_batch.req_pool_indices,
        req_to_token,
        buffers.block_table,
        req_to_token.stride(0),
        buffers.block_table.stride(0),
        max_num_blocks,
        PAGE_SIZE=buffers.allocator_page_size,
        BLOCK_N=block_n,
    )
    if _is_mtp_spec_extend_like(forward_batch):
        _patch_graph_block_table_kernel[(num_tokens,)](
            forward_batch.positions,
            forward_batch.out_cache_loc,
            buffers.block_table,
            buffers.block_table.stride(0),
            max_num_blocks,
            TOKENS_PER_REQ=buffers.tokens_per_req,
            PAGE_SIZE=buffers.allocator_page_size,
        )
    count_block = triton.next_power_of_2(buffers.topk_tokens)
    _count_valid_topk_kernel[(num_tokens,)](
        topk_indices,
        buffers.seq_len,
        topk_indices.stride(0),
        TOPK_TOKENS=buffers.topk_tokens,
        BLOCK_N=count_block,
    )
    buffers.kv_indptr[0].zero_()
    torch.cumsum(buffers.seq_len, dim=0, out=buffers.kv_indptr[1:])
    triton_convert_req_index_to_global_index(
        buffers.req_id_per_token,
        buffers.block_table,
        topk_indices,
        buffers.kv_indptr,
        buffers.kv_indices,
        BLOCK_SIZE=buffers.allocator_page_size,
        NUM_TOPK_TOKENS=buffers.topk_tokens,
    )
    get_mla_metadata_v1(
        buffers.qo_indptr,
        buffers.kv_indptr,
        buffers.last_page_len,
        num_heads,
        1,
        True,
        buffers.work_metadata,
        buffers.work_info_set,
        buffers.work_indptr,
        buffers.reduce_indptr,
        buffers.reduce_final_map,
        buffers.reduce_partial_map,
        kv_granularity=16,
        page_size=1,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        dtype_q_nope=buffers.q.dtype,
        dtype_q_rope=buffers.q.dtype,
        dtype_kv_nope=k_buffer.dtype,
        dtype_kv_rope=k_buffer.dtype,
    )
    return SparseMLAKernelMetadata(
        allocator_page_size=buffers.allocator_page_size,
        topk_tokens=buffers.topk_tokens,
        req_id_per_token=buffers.req_id_per_token,
        block_table=buffers.block_table,
        qo_indptr=buffers.qo_indptr,
        kv_indptr=buffers.kv_indptr,
        kv_indices=buffers.kv_indices,
        last_page_len=buffers.last_page_len,
        use_fast_metadata=True,
        work_metadata=buffers.work_metadata,
        work_indptr=buffers.work_indptr,
        work_info_set=buffers.work_info_set,
        reduce_indptr=buffers.reduce_indptr,
        reduce_final_map=buffers.reduce_final_map,
        reduce_partial_map=buffers.reduce_partial_map,
    )


def _prepare_sparse_mla_kernel_metadata(
    *,
    forward_batch,
    topk_indices: torch.Tensor,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    num_heads: int,
) -> SparseMLAKernelMetadata:
    """Build eager sparse MLA metadata only from the current SGLang batch/pools."""
    num_tokens = int(q.shape[0])
    topk_tokens = int(topk_indices.shape[1])
    allocator_page_size = int(
        getattr(forward_batch.token_to_kv_pool, "page_size", 1)
    )
    req_id_per_token = _build_sparse_req_id_per_token_for_sglang(
        forward_batch, q.device
    )
    block_table = _build_sglang_block_table(
        forward_batch, allocator_page_size
    ).to(dtype=torch.int32)

    seq_len = (topk_indices != -1).sum(dim=-1).to(dtype=torch.int32)
    kv_indptr = torch.empty((num_tokens + 1,), dtype=torch.int32, device=q.device)
    kv_indptr[0].zero_()
    torch.cumsum(seq_len, dim=0, out=kv_indptr[1:])
    kv_indices = torch.empty(
        (num_tokens * topk_tokens,), dtype=torch.int32, device=q.device
    )
    triton_convert_req_index_to_global_index(
        req_id_per_token,
        block_table,
        topk_indices.to(dtype=torch.int32),
        kv_indptr,
        kv_indices,
        BLOCK_SIZE=allocator_page_size,
        NUM_TOPK_TOKENS=topk_tokens,
    )

    qo_indptr = torch.arange(num_tokens + 1, dtype=torch.int32, device=q.device)
    last_page_len = torch.ones(num_tokens, dtype=torch.int32, device=q.device)
    use_fast_metadata = (
        q.dtype == dtypes.fp8 or k_buffer.dtype == dtypes.fp8
    ) and _supports_sparse_mla_fast_metadata(
        num_heads,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        q_dtype=q.dtype,
        kv_dtype=k_buffer.dtype,
    )

    metadata = SparseMLAKernelMetadata(
        allocator_page_size=allocator_page_size,
        topk_tokens=topk_tokens,
        req_id_per_token=req_id_per_token,
        block_table=block_table,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        last_page_len=last_page_len,
        use_fast_metadata=use_fast_metadata,
    )
    if not use_fast_metadata:
        return metadata

    buffer_specs = get_mla_metadata_info_v1(
        num_tokens,
        1,
        num_heads,
        q.dtype,
        k_buffer.dtype,
        is_sparse=True,
        fast_mode=True,
    )
    buffers = [
        torch.empty(size, dtype=dtype, device=q.device)
        for size, dtype in buffer_specs
    ]
    (
        metadata.work_metadata,
        metadata.work_indptr,
        metadata.work_info_set,
        metadata.reduce_indptr,
        metadata.reduce_final_map,
        metadata.reduce_partial_map,
    ) = buffers
    get_mla_metadata_v1(
        metadata.qo_indptr,
        metadata.kv_indptr,
        metadata.last_page_len,
        num_heads,
        1,
        True,
        metadata.work_metadata,
        metadata.work_info_set,
        metadata.work_indptr,
        metadata.reduce_indptr,
        metadata.reduce_final_map,
        metadata.reduce_partial_map,
        kv_granularity=16,
        page_size=1,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        dtype_q_nope=q.dtype,
        dtype_q_rope=q.dtype,
        dtype_kv_nope=k_buffer.dtype,
        dtype_kv_rope=k_buffer.dtype,
    )
    return metadata


def forward_sparse_mla_for_sglang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    topk_indices: torch.Tensor,
    save_kv_cache: bool = True,
    input_dtype: Optional[torch.dtype] = None,
    q_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ATOM sparse MLA path for SGLang DeepSeek-V3.2."""
    if save_kv_cache and k is not None:
        assert v is not None
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v
        )

    q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    num_tokens = q.shape[0]
    topk_indices = topk_indices[:num_tokens]
    output_dtype = input_dtype or torch.bfloat16
    k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    align_fp8_q = os.environ.get(
        "ATOM_SGLANG_SPARSE_MLA_ALIGN_FP8_Q", "0"
    ).lower() in ("1", "true", "yes", "on")
    graph_mode = _is_graph_warmup_or_capture()
    fixed_graph_phase = bool(
        forward_batch.forward_mode.is_decode_or_idle()
        or _is_mtp_spec_extend_like(forward_batch)
    )
    use_fixed_graph_buffers = graph_mode and fixed_graph_phase
    q_kernel_dtype = (
        dtypes.fp8 if align_fp8_q and k_buffer.dtype == dtypes.fp8 else q.dtype
    )
    if not use_fixed_graph_buffers and q.dtype != q_kernel_dtype:
        q = q.to(dtypes.fp8)
    q_descale = None
    if q_kernel_dtype == dtypes.fp8:
        q_descale = q_scale
        if q_descale is None:
            q_descale = getattr(layer, "q_scale", None)
        if q_descale is None:
            q_descale = getattr(layer, "k_scale", None)
    allocator_page_size = int(
        getattr(forward_batch.token_to_kv_pool, "page_size", 1)
    )
    graph_buffers = None
    if use_fixed_graph_buffers:
        backend = getattr(forward_batch, "attn_backend", None)
        if backend is not None and _is_mtp_spec_extend_like(forward_batch):
            backend._atom_glm52_draft_extend_layer_id = int(layer.layer_id)
        batch_size = int(forward_batch.batch_size)
        tokens_per_req = 1
        if _is_mtp_spec_extend_like(forward_batch):
            tokens_per_req = _mtp_spec_tokens_per_req(forward_batch)
        if (
            batch_size <= 0
            or tokens_per_req <= 0
            or batch_size * tokens_per_req != num_tokens
        ):
            raise RuntimeError(
                "Sparse MLA graph requires fixed token-major request rows: "
                f"batch_size={batch_size}, tokens_per_req={tokens_per_req}, "
                f"num_tokens={num_tokens}, mode={forward_batch.forward_mode}"
            )
        graph_buffers = _get_or_create_sparse_mla_graph_buffers(
            forward_batch=forward_batch,
            batch_size=batch_size,
            tokens_per_req=tokens_per_req,
            num_tokens=num_tokens,
            topk_tokens=int(topk_indices.shape[1]),
            allocator_page_size=allocator_page_size,
            q=q,
            q_dtype=q_kernel_dtype,
            k_buffer=k_buffer,
            num_heads=layer.tp_q_head_num,
            output_dtype=output_dtype,
            output_dim=layer.v_head_dim,
        )
    if graph_buffers is None:
        o = q.new_empty(
            (num_tokens, layer.tp_q_head_num, layer.v_head_dim),
            dtype=output_dtype,
        )
        metadata = _prepare_sparse_mla_kernel_metadata(
            forward_batch=forward_batch,
            topk_indices=topk_indices,
            q=q,
            k_buffer=k_buffer,
            num_heads=layer.tp_q_head_num,
        )
    else:
        o = graph_buffers.output
        metadata = _prepare_sparse_mla_graph_metadata(
            buffers=graph_buffers,
            forward_batch=forward_batch,
            topk_indices=topk_indices,
            q=q,
            k_buffer=k_buffer,
            num_heads=layer.tp_q_head_num,
        )
        q = graph_buffers.q

    mla_decode_fwd(
        q,
        k_buffer.view(-1, 1, 1, layer.qk_head_dim),
        o,
        metadata.qo_indptr,
        metadata.kv_indptr,
        metadata.kv_indices,
        metadata.last_page_len,
        1,
        sm_scale=layer.scaling,
        logit_cap=layer.logit_cap,
        q_scale=q_descale,
        kv_scale=layer.k_scale,
        page_size=1,
        work_meta_data=metadata.work_metadata,
        work_indptr=metadata.work_indptr,
        work_info_set=metadata.work_info_set,
        reduce_indptr=metadata.reduce_indptr,
        reduce_final_map=metadata.reduce_final_map,
        reduce_partial_map=metadata.reduce_partial_map,
    )
    return o.view(num_tokens, layer.tp_q_head_num * layer.v_head_dim)


def sparse_attn_indexer_sglang_plugin_mode(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    from atom.plugin.sglang.models.base_model_wrapper import get_current_forward_batch

    del kv_cache, total_seq_lens
    forward_batch = get_current_forward_batch()
    if forward_batch is None or forward_batch.forward_mode.is_idle():
        return torch.zeros_like(weights, dtype=torch.float32)

    token_to_kv_pool = forward_batch.token_to_kv_pool
    if not hasattr(token_to_kv_pool, "get_index_k_with_scale_buffer"):
        raise RuntimeError(
            "[SGL+ATOM] DeepSeek-V3.2 sparse MLA requires SGLang NSA KV pool "
            "with index_k_with_scale_buffer support."
        )

    layer_id = _parse_layer_id_from_indexer_prefix(k_cache_prefix)
    index_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    page_size = int(getattr(token_to_kv_pool, "page_size", 1))
    kv_cache = index_cache.view(-1, page_size, head_dim + 4)
    preshuffle_cache = page_size != 1
    slot_mapping = forward_batch.out_cache_loc

    if use_qk_rope_cache_fusion:
        q_bf16 = q_input
        q_fp8 = torch.empty_like(q_bf16, dtype=dtypes.fp8)
        weights_out = torch.empty(
            weights.shape, device=weights.device, dtype=torch.float32
        )
        indexer_qk_rope_quant_and_cache(
            q_bf16,
            q_fp8,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            k_norm_weight,
            k_norm_bias,
            positions,
            cos_cache,
            sin_cache,
            k_norm_eps,
            quant_block_size,
            scale_fmt,
            weights_scale,
            preshuffle=preshuffle_cache,
            is_neox=is_neox_style,
        )
        weights = weights_out
    else:
        q_fp8 = q_input
        indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
            preshuffle=preshuffle_cache,
        )

    num_tokens = hidden_states.shape[0]
    topk_indices_buffer[:num_tokens] = -1
    indexer_graph_buffers = None
    if _is_graph_warmup_or_capture() and _is_mtp_spec_extend_like(forward_batch):
        indexer_graph_buffers = _get_or_create_sparse_mla_indexer_graph_buffers(
            forward_batch=forward_batch,
            page_size=page_size,
            head_dim=head_dim,
        )
        _prepare_sparse_mla_indexer_graph_buffers(
            buffers=indexer_graph_buffers,
            forward_batch=forward_batch,
        )
        block_table = indexer_graph_buffers.block_table
    else:
        block_table = _build_sglang_block_table(forward_batch, page_size)

    if forward_batch.forward_mode.is_decode_or_idle():
        bs = int(forward_batch.batch_size)
        if q_fp8.shape[0] < bs or weights.shape[0] < bs:
            raise RuntimeError(
                "[SGL+ATOM] sparse indexer decode expected at least "
                f"{bs} token rows, got q={q_fp8.shape[0]}, weights={weights.shape[0]}. "
                "This usually means TP-scattered indexer inputs were not gathered."
            )
        from atom.plugin.sglang.glm52_mtp.common import resolve_indexer_seq_lens

        seq_lens_i32 = resolve_indexer_seq_lens(forward_batch, bs)
        padded_q_fp8 = q_fp8[:bs].reshape(bs, 1, *q_fp8.shape[1:])
        logits = torch.empty([bs, max_model_len], dtype=torch.float32, device=k.device)
        deepgemm_fp8_paged_mqa_logits(
            padded_q_fp8,
            kv_cache.unsqueeze(-2),
            weights[:bs],
            logits,
            seq_lens_i32,
            block_table,
            max_model_len,
            ChunkK=256,
            Preshuffle=preshuffle_cache,
            KVBlockSize=page_size,
            WavePerEU=2,
        )
        top_k_per_row_decode(
            logits,
            1,
            seq_lens_i32,
            topk_indices_buffer[:bs, :topk_tokens],
            bs,
            logits.stride(0),
            logits.stride(1),
        )
        return weights

    if indexer_graph_buffers is None:
        cu_starts, cu_ends = _build_sglang_query_ranges(forward_batch)
        total_kv = _mtp_eager_total_kv(forward_batch)
        k_fp8 = torch.empty([total_kv, head_dim], device=k.device, dtype=dtypes.fp8)
        k_scale = torch.empty([total_kv, 1], device=k.device, dtype=torch.float32)
        gather_indptr = _build_mtp_eager_gather_indptr(forward_batch)
    else:
        cu_starts = indexer_graph_buffers.cu_starts
        cu_ends = indexer_graph_buffers.cu_ends
        k_fp8 = indexer_graph_buffers.k_fp8
        k_scale = indexer_graph_buffers.k_scale
        gather_indptr = indexer_graph_buffers.gather_indptr
    cp_gather_indexer_k_quant_cache(
        kv_cache,
        k_fp8,
        k_scale.view(dtypes.fp8),
        block_table,
        gather_indptr,
        preshuffle=preshuffle_cache,
    )
    logits = fp8_mqa_logits(
        Q=q_fp8[:num_tokens],
        KV=k_fp8,
        kv_scales=k_scale,
        weights=weights[:num_tokens],
        cu_starts=cu_starts,
        cu_ends=cu_ends,
    )
    assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
    topk_indices = topk_indices_buffer[:num_tokens, :topk_tokens]
    top_k_per_row_prefill(
        logits=logits,
        rowStarts=cu_starts,
        rowEnds=cu_ends,
        indices=topk_indices,
        values=None,
        numRows=logits.shape[0],
        stride0=logits.stride(0),
        stride1=logits.stride(1),
    )
    topk_indices.copy_(
        torch.where(topk_indices >= 0, topk_indices - cu_starts[:, None], topk_indices)
    )
    return weights


def sparse_attn_indexer_sglang_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_input: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: Optional[str],
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    k_norm_eps: float,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    weights_scale: float,
    is_neox_style: bool,
    use_qk_rope_cache_fusion: bool,
) -> torch.Tensor:
    del (
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q_input,
        k,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        total_seq_lens,
        topk_indices_buffer,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        positions,
        cos_cache,
        sin_cache,
        weights_scale,
        is_neox_style,
        use_qk_rope_cache_fusion,
    )
    return torch.empty(weights.shape, device=weights.device, dtype=torch.float32)


direct_register_custom_op(
    op_name="sparse_attn_indexer_sglang_plugin_mode",
    op_func=sparse_attn_indexer_sglang_plugin_mode,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_sglang_fake,
)
