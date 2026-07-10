from __future__ import annotations

from dataclasses import dataclass

import torch


import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128


def _is_stream_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


@dataclass
class MiniMaxM3SGLangMetadata:
    """Per-forward SGLang metadata for MiniMax-M3 sparse attention."""

    is_decode: bool
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_seq_len: int
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_query_len: int = 1


def validate_minimax_m3_page_size(page_size: int) -> None:
    """MiniMax-M3 sparse blocks must line up 1:1 with SGLang KV pages."""

    if int(page_size) != SPARSE_BLOCK_SIZE:
        raise ValueError(
            "MiniMax-M3 sparse attention requires SGLang page size 128 "
            f"(got {page_size}). Launch SGLang with --page-size 128."
        )


def _get_batch_size(forward_batch) -> int:
    return int(getattr(forward_batch, "batch_size"))


def _slice_i32(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    return tensor[:batch_size].to(dtype=torch.int32)


def _get_query_lens(forward_batch, batch_size: int) -> torch.Tensor:
    query_lens = getattr(forward_batch, "extend_seq_lens", None)
    if query_lens is None:
        query_lens = getattr(forward_batch, "seq_lens")
    return _slice_i32(query_lens, batch_size)


def _get_prefix_lens(
    forward_batch,
    batch_size: int,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
) -> torch.Tensor:
    prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)
    if prefix_lens is None:
        return (seq_lens - query_lens).to(dtype=torch.int32)
    return _slice_i32(prefix_lens, batch_size)


def _get_page_size(forward_batch) -> int:
    return int(getattr(forward_batch.token_to_kv_pool, "page_size", 1))


def _get_layer_id(layer) -> int:
    if hasattr(layer, "layer_id"):
        return int(layer.layer_id)
    return int(layer.layer_num)


def _is_fp8_kv_cache_tensor(kv_cache: torch.Tensor) -> bool:
    fp8_dtypes = (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
    )
    return kv_cache.dtype in {dtype for dtype in fp8_dtypes if dtype is not None}


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit
def _sgl_m3_sparse_fwd_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return

    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim

    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k = tl.load(
                k_cache_ptr
                + page * stride_k_blk
                + off_n[None, :] * stride_k_pos
                + pid_kh * stride_k_h
                + off_d[:, None] * stride_k_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if FP8_KV_CACHE:
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                v_cache_ptr
                + page * stride_v_blk
                + off_n[:, None] * stride_v_pos
                + pid_kh * stride_v_h
                + off_d[None, :] * stride_v_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if FP8_KV_CACHE:
                v = v.to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
    }
)
@triton.jit
def _sgl_m3_sparse_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_pos,
    stride_k_h,
    stride_k_d,
    stride_v_blk,
    stride_v_pos,
    stride_v_h,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk
    seq_len = tl.load(seq_lens + pid_b)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    topk_idx = tl.load(idx_base + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    real_topk = tl.sum((topk_idx >= 0).to(tl.int32), axis=0)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + pid_b * stride_bt_b
    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < seq_len
        k = tl.load(
            k_cache_ptr
            + page * stride_k_blk
            + off_n[None, :] * stride_k_pos
            + pid_kh * stride_k_h
            + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if FP8_KV_CACHE:
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            v_cache_ptr
            + page * stride_v_blk
            + off_n[:, None] * stride_v_pos
            + pid_kh * stride_v_h
            + off_d[None, :] * stride_v_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if FP8_KV_CACHE:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    l_ptrs = (
        lse_ptr
        + pid_c * stride_l_c
        + pid_b * stride_l_b
        + (pid_h + tl.arange(0, BLOCK_SIZE_H)) * stride_l_h
    )
    tl.store(
        l_ptrs,
        lse_i,
        mask=tl.arange(0, BLOCK_SIZE_H) < gqa_group_size,
    )


@triton.jit
def _sgl_m3_sparse_decode_merge_kernel(
    o_partial,
    lse_partial,
    output,
    batch_size,
    num_heads,
    head_dim,
    num_chunks: tl.constexpr,
    stride_oc,
    stride_ob,
    stride_oh,
    stride_od,
    stride_lc,
    stride_lb,
    stride_lh,
    stride_on,
    stride_oh_out,
    stride_od_out,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    m = tl.full((), float("-inf"), dtype=tl.float32)
    for c in range(num_chunks):
        lse_value = tl.load(
            lse_partial + c * stride_lc + pid_b * stride_lb + pid_h * stride_lh
        )
        m = tl.maximum(m, lse_value)
    acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    denom = tl.full((), 0.0, dtype=tl.float32)
    for c in range(num_chunks):
        lse_value = tl.load(
            lse_partial + c * stride_lc + pid_b * stride_lb + pid_h * stride_lh
        )
        w = tl.exp2(lse_value - m)
        vals = tl.load(
            o_partial
            + c * stride_oc
            + pid_b * stride_ob
            + pid_h * stride_oh
            + off_d * stride_od,
            mask=d_mask,
            other=0.0,
        )
        acc += vals.to(tl.float32) * w
        denom += w
    acc = acc / denom
    tl.store(
        output + pid_b * stride_on + pid_h * stride_oh_out + off_d * stride_od_out,
        acc.to(output.dtype.element_ty),
        mask=d_mask,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_split_kv(
    q: torch.Tensor,
    key_cache: torch.Tensor,  # [num_blocks, page_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, page_size, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
) -> None:
    total_q, num_heads, head_dim = q.shape
    del total_q
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    grid = (max_query_len, num_kv_heads, batch)
    _sgl_m3_sparse_fwd_kernel[grid](
        q,
        key_cache,
        value_cache,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        FP8_KV_CACHE=_is_fp8_kv_cache_tensor(key_cache),
        num_stages=1,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode_split_kv(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
) -> None:
    batch, num_heads, head_dim = q.shape
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    target = max(1, min(max_topk, 256 // max(1, batch * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, batch, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, batch, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (batch * num_topk_chunks, num_kv_heads)
    _sgl_m3_sparse_decode_kernel[grid](
        q,
        key_cache,
        value_cache,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        batch,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        FP8_KV_CACHE=_is_fp8_kv_cache_tensor(key_cache),
        num_stages=1,
    )
    merge_grid = (batch, num_heads)
    _sgl_m3_sparse_decode_merge_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        batch,
        num_heads,
        head_dim,
        num_topk_chunks,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
    )


def build_minimax_m3_block_table(forward_batch, page_size: int) -> torch.Tensor:
    """Build physical block ids from SGLang's request-token table."""

    validate_minimax_m3_page_size(page_size)
    batch_size = _get_batch_size(forward_batch)
    req_pool_indices = forward_batch.req_pool_indices[:batch_size]
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    token_table = req_to_token[req_pool_indices, :].clone()

    if not forward_batch.forward_mode.is_decode_or_idle():
        query_lens = _get_query_lens(forward_batch, batch_size)
        seq_lens = _slice_i32(forward_batch.seq_lens, batch_size)
        prefix_lens = _get_prefix_lens(forward_batch, batch_size, seq_lens, query_lens)
        out_cache_loc = forward_batch.out_cache_loc
        offset = 0
        for req_idx in range(batch_size):
            prefix_len = int(prefix_lens[req_idx].item())
            query_len = int(query_lens[req_idx].item())
            if query_len > 0:
                token_table[req_idx, prefix_len : prefix_len + query_len] = (
                    out_cache_loc[offset : offset + query_len]
                )
            offset += query_len

    seq_lens = _slice_i32(forward_batch.seq_lens, batch_size)
    if _is_stream_capturing() and forward_batch.forward_mode.is_decode_or_idle():
        max_blocks = int(token_table.shape[1]) // page_size
    else:
        max_seq_len = int(seq_lens.max().item()) if batch_size else 0
        max_blocks = (max_seq_len + page_size - 1) // page_size
    block_table = token_table[:, : max_blocks * page_size : page_size] // page_size
    return block_table.to(dtype=torch.int32).contiguous()


def build_minimax_m3_forward_metadata(
    forward_batch,
    block_table: torch.Tensor,
    page_size: int,
) -> MiniMaxM3SGLangMetadata:
    """Translate SGLang ForwardBatch fields into MiniMax-M3 sparse metadata."""

    validate_minimax_m3_page_size(page_size)
    batch_size = _get_batch_size(forward_batch)
    seq_lens = _slice_i32(forward_batch.seq_lens, batch_size)
    if _is_stream_capturing() and forward_batch.forward_mode.is_decode_or_idle():
        max_seq_len = int(block_table.shape[1]) * page_size
    else:
        max_seq_len = int(seq_lens.max().item()) if batch_size else 0

    if forward_batch.forward_mode.is_decode_or_idle():
        return MiniMaxM3SGLangMetadata(
            is_decode=True,
            seq_lens=seq_lens,
            block_table=block_table,
            max_seq_len=max_seq_len,
        )

    query_lens = _get_query_lens(forward_batch, batch_size)
    context_lens = _get_prefix_lens(forward_batch, batch_size, seq_lens, query_lens)
    cu_seqlens_q = torch.empty(
        batch_size + 1, dtype=torch.int32, device=seq_lens.device
    )
    cu_seqlens_k = torch.empty(
        batch_size + 1, dtype=torch.int32, device=seq_lens.device
    )
    cu_seqlens_q[0] = 0
    cu_seqlens_k[0] = 0
    torch.cumsum(query_lens, dim=0, out=cu_seqlens_q[1:])
    torch.cumsum(seq_lens, dim=0, out=cu_seqlens_k[1:])

    return MiniMaxM3SGLangMetadata(
        is_decode=False,
        seq_lens=seq_lens,
        block_table=block_table,
        max_seq_len=max_seq_len,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        context_lens=context_lens,
        max_query_len=int(query_lens.max().item()) if batch_size else 0,
    )


def _ensure_side_caches(
    layer,
    forward_batch,
    index_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_size = _get_page_size(forward_batch)
    validate_minimax_m3_page_size(page_size)

    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
        _get_layer_id(layer)
    )
    num_slots = int(k_buffer.shape[0])
    num_blocks = num_slots // page_size
    if num_blocks <= 0:
        raise RuntimeError("MiniMax-M3 sparse attention received an empty KV pool.")

    key_cache = k_buffer[: num_blocks * page_size].view(
        num_blocks, page_size, layer.num_kv_heads, layer.head_dim
    )
    value_cache = v_buffer[: num_blocks * page_size].view(
        num_blocks, page_size, layer.num_kv_heads, layer.head_dim
    )
    index_shape = (num_blocks, page_size, layer.idx_head_dim)

    index_cache = getattr(layer, "_sglang_m3_index_cache", None)
    if (
        index_cache is None
        or tuple(index_cache.shape) != index_shape
        or index_cache.device != index_key.device
        or index_cache.dtype != index_key.dtype
    ):
        index_cache = torch.empty(
            index_shape, dtype=index_key.dtype, device=index_key.device
        )
        layer._sglang_m3_index_cache = index_cache

    return key_cache, value_cache, index_cache


def _insert_sparse_cache(
    layer,
    forward_batch,
    key: torch.Tensor,
    value: torch.Tensor,
    index_key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_cache, value_cache, index_cache = _ensure_side_caches(
        layer, forward_batch, index_key
    )
    page_size = _get_page_size(forward_batch)
    slot_mapping = forward_batch.out_cache_loc[: key.shape[0]].to(dtype=torch.long)
    valid = slot_mapping >= 0
    is_capture = _is_stream_capturing()
    if is_capture:
        # CUDA graph capture uses fixed-size decode batches; boolean compaction
        # launches a dynamic-shape HIP op that is not graph-capture safe.
        slots = slot_mapping
        key = key.view(-1, layer.num_kv_heads, layer.head_dim)
        value = value.view(-1, layer.num_kv_heads, layer.head_dim)
        index_key = index_key.view(-1, layer.idx_head_dim)
    else:
        slots = slot_mapping[valid]
        key = key.view(-1, layer.num_kv_heads, layer.head_dim)[valid]
        value = value.view(-1, layer.num_kv_heads, layer.head_dim)[valid]
        index_key = index_key.view(-1, layer.idx_head_dim)[valid]
    block_ids = torch.div(slots, page_size, rounding_mode="floor")
    block_offsets = slots % page_size
    key_cache[block_ids, block_offsets] = key.to(key_cache.dtype)
    value_cache[block_ids, block_offsets] = value.to(value_cache.dtype)
    index_cache[block_ids, block_offsets] = index_key.to(index_cache.dtype)
    return key_cache, value_cache, index_cache


def minimax_m3_sparse_attention_for_sglang(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    index_query: torch.Tensor,
    index_key: torch.Tensor,
    layer,
    forward_batch=None,
    save_kv_cache: bool = True,
) -> torch.Tensor:
    """Run MiniMax-M3 lightning-indexer sparse attention in SGLang plugin mode."""

    if forward_batch is None:
        from atom.plugin.sglang.runtime import get_current_forward_batch

        forward_batch = get_current_forward_batch()
    if forward_batch is None:
        raise RuntimeError(
            "MiniMax-M3 sparse attention requires a SGLang ForwardBatch."
        )

    page_size = _get_page_size(forward_batch)
    validate_minimax_m3_page_size(page_size)
    if save_kv_cache:
        key_cache, value_cache, index_cache = _insert_sparse_cache(
            layer, forward_batch, key, value, index_key
        )
    else:
        key_cache, value_cache, index_cache = _ensure_side_caches(
            layer, forward_batch, index_key
        )

    block_table = build_minimax_m3_block_table(forward_batch, page_size)
    metadata = build_minimax_m3_forward_metadata(forward_batch, block_table, page_size)

    q = query.view(-1, layer.num_heads, layer.head_dim)
    index_q = index_query.view(-1, layer.num_idx_heads, layer.idx_head_dim)
    output = torch.empty_like(q)

    from atom.model_ops.minimax_m3.index_topk import (
        minimax_m3_index_topk,
        minimax_m3_index_topk_decode,
    )

    if metadata.is_decode:
        batch_size = metadata.seq_lens.shape[0]
        topk_idx = minimax_m3_index_topk_decode(
            index_q[:batch_size],
            index_cache,
            metadata.block_table,
            metadata.seq_lens,
            metadata.max_seq_len,
            layer.topk_blocks,
            layer.init_blocks,
            layer.local_blocks,
            layer.num_kv_heads,
            layer.scaling,
        )
        minimax_m3_sparse_attn_decode_split_kv(
            q[:batch_size],
            key_cache,
            value_cache,
            topk_idx,
            metadata.block_table,
            metadata.seq_lens,
            layer.num_kv_heads,
            layer.scaling,
            output[:batch_size],
        )
        if batch_size < output.shape[0]:
            output[batch_size:].zero_()
    else:
        assert metadata.cu_seqlens_q is not None
        assert metadata.context_lens is not None
        num_tokens = int(metadata.cu_seqlens_q[-1].item())
        topk_idx = minimax_m3_index_topk(
            index_q[:num_tokens],
            index_cache,
            metadata.block_table,
            metadata.cu_seqlens_q,
            metadata.seq_lens,
            metadata.context_lens,
            metadata.max_query_len,
            metadata.max_seq_len,
            layer.topk_blocks,
            layer.init_blocks,
            layer.local_blocks,
            layer.num_kv_heads,
            layer.scaling,
        )
        minimax_m3_sparse_attn_split_kv(
            q[:num_tokens],
            key_cache,
            value_cache,
            topk_idx,
            metadata.block_table,
            metadata.cu_seqlens_q,
            metadata.seq_lens,
            metadata.context_lens,
            metadata.max_query_len,
            layer.num_kv_heads,
            layer.scaling,
            output[:num_tokens],
        )
        if num_tokens < output.shape[0]:
            output[num_tokens:].zero_()

    return output.reshape_as(query)
