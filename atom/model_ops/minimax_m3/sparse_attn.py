# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for MiniMax M3 block-sparse GQA attention.

The main heads attend only to the blocks selected by the lightning indexer (see
``index_topk``). Ported from the sglang reference (minimax_sparse_ops), adapted
to vLLM's paged KV cache: the KV page size is forced to equal the sparse block
size (128), so one selected block maps to exactly one page.

Main K/V cache layout (vLLM):
  ``(num_blocks, 2, 128, num_kv_heads, head_dim)``  K=[:,0] V=[:,1]

Only the paths MiniMax M3 uses are implemented: no attention sink, base-2
(exp2/log2) softmax. The decode kernels use split-K (flash-decoding) over the
selected blocks with a separate merge step, since one query token per request
leaves the prefill kernels (which parallelize over the query dim) idle.
"""

from dataclasses import dataclass

import numpy as np
import torch

try:
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:
    import triton
    import triton.language as tl

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128


@dataclass
class MiniMaxM3SparsePrefillMetadata:
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    # Per-query-token request id and absolute position, layer-INVARIANT (depend
    # only on cu_seqlens_q + prefix_lens). Built ONCE in prepare_prefill and
    # reused across all sparse layers by the ASM prefill block-table builder, so
    # the per-layer build needs no host sync. None on the non-ASM path.
    query_req_id: torch.Tensor | None = None  # [total_q] int32
    query_abs_pos: torch.Tensor | None = None  # [total_q] int32
    # Per-token CSR for pa_fwd_asm prefill (each token a length-1 segment):
    # arange(total_q + 1). Layer-invariant, built once in prepare_prefill.
    per_token_qo_indptr: torch.Tensor | None = None  # [total_q+1] int32


@dataclass
class MiniMaxM3SparseDecodeMetadata:
    seq_lens: torch.Tensor
    block_table: torch.Tensor


@dataclass
class MiniMaxM3SparseSmallQDecodeMetadata:
    query_seq_lens: torch.Tensor
    query_block_table: torch.Tensor
    prefix_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    decode_seq_lens: torch.Tensor | None = None


@dataclass
class MiniMaxM3SparseMetadata:
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor
    num_prefills: int
    prefill: MiniMaxM3SparsePrefillMetadata | None = None
    decode: MiniMaxM3SparseDecodeMetadata | None = None
    small_q_decode: MiniMaxM3SparseSmallQDecodeMetadata | None = None


def _build_prefill_query_metadata(
    *,
    cu_seqlens_q_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build layer-invariant per-query metadata for the ASM prefill path."""
    cu_np = np.asarray(cu_seqlens_q_cpu, dtype=np.int64)
    seq_lens_np = np.asarray(seq_lens_cpu, dtype=np.int64)
    batch = seq_lens_np.shape[0]
    q_lens_np = cu_np[1 : batch + 1] - cu_np[:batch]
    total_q = int(cu_np[batch])
    prefix_np = (seq_lens_np - q_lens_np).astype(np.int64)
    req_id_np = np.repeat(np.arange(batch, dtype=np.int32), q_lens_np)
    abs_pos_np = (
        prefix_np[req_id_np] + (np.arange(total_q, dtype=np.int64) - cu_np[req_id_np])
    ).astype(np.int32)
    return (
        torch.from_numpy(req_id_np).to(device, non_blocking=True),
        torch.from_numpy(abs_pos_np).to(device, non_blocking=True),
        torch.arange(total_q + 1, dtype=torch.int32, device=device),
    )


def make_minimax_m3_sparse_prefill_metadata(
    *,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_prefills: int,
    seq_lens_cpu: np.ndarray | None = None,
    cu_seqlens_q_cpu: np.ndarray | None = None,
    include_query_metadata: bool = False,
) -> MiniMaxM3SparseMetadata:
    query_lens = cu_seqlens_q[1 : num_prefills + 1] - cu_seqlens_q[:num_prefills]
    prefix_lens = seq_lens - query_lens
    query_req_id = query_abs_pos = per_token_qo_indptr = None
    if include_query_metadata:
        if cu_seqlens_q_cpu is None or seq_lens_cpu is None:
            raise ValueError(
                "cu_seqlens_q_cpu and seq_lens_cpu are required when "
                "include_query_metadata=True."
            )
        query_req_id, query_abs_pos, per_token_qo_indptr = (
            _build_prefill_query_metadata(
                cu_seqlens_q_cpu=cu_seqlens_q_cpu,
                seq_lens_cpu=seq_lens_cpu,
                device=block_table.device,
            )
        )
    prefill = MiniMaxM3SparsePrefillMetadata(
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        context_lens=prefix_lens,
        block_table=block_table,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        query_req_id=query_req_id,
        query_abs_pos=query_abs_pos,
        per_token_qo_indptr=per_token_qo_indptr,
    )
    return MiniMaxM3SparseMetadata(
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        slot_mapping=slot_mapping,
        num_prefills=num_prefills,
        prefill=prefill,
        decode=None,
    )


def make_minimax_m3_sparse_decode_metadata(
    *,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    max_seq_len: int,
) -> MiniMaxM3SparseMetadata:
    decode = MiniMaxM3SparseDecodeMetadata(seq_lens=seq_lens, block_table=block_table)
    return MiniMaxM3SparseMetadata(
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        slot_mapping=slot_mapping,
        num_prefills=0,
        prefill=None,
        decode=decode,
    )


def make_minimax_m3_sparse_small_q_decode_metadata(
    *,
    seq_lens: torch.Tensor,
    query_seq_lens: torch.Tensor,
    query_block_table: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    decode_seq_lens: torch.Tensor | None = None,
) -> MiniMaxM3SparseMetadata:
    small_q_decode = MiniMaxM3SparseSmallQDecodeMetadata(
        query_seq_lens=query_seq_lens,
        query_block_table=query_block_table,
        prefix_lens=prefix_lens,
        block_table=block_table,
        max_query_len=max_query_len,
        decode_seq_lens=decode_seq_lens,
    )
    return MiniMaxM3SparseMetadata(
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        slot_mapping=slot_mapping,
        num_prefills=0,
        prefill=None,
        decode=None,
        small_q_decode=small_q_decode,
    )


# ---------------------------------------------------------------------------
# Fused qknorm + RoPE + KV insert (SHUFFLE main cache writer).
#
# Fused Gemma-RMSNorm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert.
# This lets AITER ASM paged-attention (``pa_fwd_asm``) read the M3 main KV
# cache during decode.
# ---------------------------------------------------------------------------
@triton.jit
def _gemma_norm_rope_head(
    row_ptr,  # pointer to this head's input row (head_dim contiguous)
    w_ptr,  # norm weight [head_dim]
    cos_ptr,  # [half] cos for this token
    sin_ptr,  # [half] sin for this token
    HEAD_DIM: tl.constexpr,
    ROT_HALF: tl.constexpr,  # rotary_dim // 2
    eps,
):
    """Gemma (1+w) RMSNorm in fp32 + partial NeoX RoPE; returns fp32 [HEAD_DIM].

    Processes the head as low/high halves so the rope pairing (d, d+half) is a
    plain elementwise op between the two half-vectors (no register permutation).
    """
    d = tl.arange(0, HEAD_DIM)
    vals = tl.load(row_ptr + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    var = tl.sum(vals * vals, axis=0) / HEAD_DIM
    normed = vals * tl.rsqrt(var + eps) * (1.0 + w)  # [HEAD_DIM] fp32

    # rotate-half partner: for d in [0,half) partner = normed[d+half];
    #                      for d in [half,rot) partner = normed[d-half].
    dh = tl.arange(0, HEAD_DIM)
    is_low = dh < ROT_HALF
    in_rot = dh < (2 * ROT_HALF)
    partner_idx = tl.where(is_low, dh + ROT_HALF, dh - ROT_HALF)
    # gather partner from `normed` via masked load of the head again (same source,
    # post-norm): recompute is cheap and avoids register permute. Load partner raw
    # then norm it with its own weight.
    pvals = tl.load(row_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    pw = tl.load(w_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    # partner shares the SAME rms variance (same head), so normed partner:
    p_normed = pvals * tl.rsqrt(var + eps) * (1.0 + pw)

    # cos/sin per d: index j = d for low, d-half for high (both in [0,half)).
    j = tl.where(is_low, dh, dh - ROT_HALF)
    cos = tl.load(cos_ptr + j, mask=in_rot, other=0.0)
    sin = tl.load(sin_ptr + j, mask=in_rot, other=0.0)
    # low:  normed*cos - partner*sin ; high: normed*cos + partner*sin
    sign = tl.where(is_low, -1.0, 1.0)
    roped = normed * cos + sign * p_normed * sin
    return tl.where(in_rot, roped, normed)


@triton.jit
def _fused_qknorm_rope_kv_insert_shuffle_kernel(
    qkv_ptr,  # [num_tokens, row_elems]
    q_norm_w_ptr,  # [head_dim]
    k_norm_w_ptr,  # [head_dim]
    iq_norm_w_ptr,  # [idx_head_dim]
    ik_norm_w_ptr,  # [idx_head_dim]
    cos_sin_ptr,  # [max_pos, rotary_dim]  (first half cos, second half sin)
    positions_ptr,  # [num_tokens] int64
    slot_mapping_ptr,  # [num_tokens] int64 (logical slot = block*128 + offset)
    q_out_ptr,  # [num_tokens, num_heads*head_dim]
    iq_out_ptr,  # [num_tokens, num_index_heads*idx_head_dim]
    kc_ptr,  # SHUFFLE K [nb, nkv, head_dim//x, 16, x]  (contiguous)
    vc_ptr,  # SHUFFLE V [nb, nkv, 16//x, head_dim, x]  (contiguous)
    index_cache_ptr,  # [*, idx_head_dim]  flat page-128 (contiguous)
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_index_heads: tl.constexpr,
    head_dim: tl.constexpr,
    idx_head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps,
    row_elems: tl.constexpr,
    x: tl.constexpr,  # 16 // itemsize
    ASM_PAGE: tl.constexpr,  # 16
):
    """Fused Gemma-RMSNorm + partial-NeoX-RoPE + SHUFFLE KV insert, one token/program.

    Sub-ops (match the PyTorch reference exactly):
      (1) q[num_heads]        : norm(q_norm) + rope            -> q_out
      (2) index_q[niq]        : norm(iq_norm) + rope           -> iq_out
      (3) k[num_kv_heads]     : norm(k_norm) + rope            -> SHUFFLE K cache
      (4) v[num_kv_heads]     : raw                            -> SHUFFLE V cache
      (5) index_k[1]          : norm(ik_norm) + rope           -> index_cache (page-128 flat)
    """
    tok = tl.program_id(0)
    half = rotary_dim // 2
    pos = tl.load(positions_ptr + tok)
    cos_row = cos_sin_ptr + pos * rotary_dim  # [:half] cos
    sin_row = cos_sin_ptr + pos * rotary_dim + half  # [half:] sin

    # qkv row layout: [q (nq*hd) | k (nkv*hd) | v (nkv*hd) | iq (niq*idx) | ik (idx)]
    q_base = 0
    k_base = num_heads * head_dim
    v_base = k_base + num_kv_heads * head_dim
    iq_base = v_base + num_kv_heads * head_dim
    ik_base = iq_base + num_index_heads * idx_head_dim
    row = qkv_ptr + tok * row_elems
    d = tl.arange(0, head_dim)

    # ----- (1) q heads -----
    for h in tl.static_range(num_heads):
        out = _gemma_norm_rope_head(
            row + q_base + h * head_dim,
            q_norm_w_ptr,
            cos_row,
            sin_row,
            head_dim,
            half,
            eps,
        )
        tl.store(
            q_out_ptr + tok * (num_heads * head_dim) + h * head_dim + d,
            out.to(q_out_ptr.dtype.element_ty),
        )

    # ----- (2) index_q heads -----
    for h in tl.static_range(num_index_heads):
        out = _gemma_norm_rope_head(
            row + iq_base + h * idx_head_dim,
            iq_norm_w_ptr,
            cos_row,
            sin_row,
            idx_head_dim,
            half,
            eps,
        )
        di = tl.arange(0, idx_head_dim)
        tl.store(
            iq_out_ptr + tok * (num_index_heads * idx_head_dim) + h * idx_head_dim + di,
            out.to(iq_out_ptr.dtype.element_ty),
        )

    slot = tl.load(slot_mapping_ptr + tok)
    page = slot // ASM_PAGE
    s = slot % ASM_PAGE
    valid_slot = slot >= 0

    # ----- (3) k heads -> SHUFFLE K, (4) v heads -> SHUFFLE V -----
    # K [nb, nkv, hd//x, 16, x]: off(d) = ((page*nkv+h)*(hd//x)+d//x)*16*x + s*x + d%x
    # V [nb, nkv, 16//x, hd, x]: off(d) = ((page*nkv+h)*(16//x)+s//x)*hd*x + d*x + s%x
    for h in tl.static_range(num_kv_heads):
        kout = _gemma_norm_rope_head(
            row + k_base + h * head_dim,
            k_norm_w_ptr,
            cos_row,
            sin_row,
            head_dim,
            half,
            eps,
        )
        k_off = (
            ((page * num_kv_heads + h) * (head_dim // x) + d // x) * (ASM_PAGE * x)
            + s * x
            + (d % x)
        )
        tl.store(kc_ptr + k_off, kout.to(kc_ptr.dtype.element_ty), mask=valid_slot)

        vvals = tl.load(row + v_base + h * head_dim + d)  # raw, no norm/rope
        v_off = (
            ((page * num_kv_heads + h) * (ASM_PAGE // x) + s // x) * (head_dim * x)
            + d * x
            + (s % x)
        )
        tl.store(vc_ptr + v_off, vvals.to(vc_ptr.dtype.element_ty), mask=valid_slot)

    # ----- (5) index_k -> index_cache page-128 flat scatter -----
    ikout = _gemma_norm_rope_head(
        row + ik_base, ik_norm_w_ptr, cos_row, sin_row, idx_head_dim, half, eps
    )
    di = tl.arange(0, idx_head_dim)
    tl.store(
        index_cache_ptr + slot * idx_head_dim + di,
        ikout.to(index_cache_ptr.dtype.element_ty),
        mask=valid_slot,
    )


@torch.no_grad()
def minimax_m3_fused_qknorm_rope_kv_insert_shuffle(
    qkv: torch.Tensor,  # [num_tokens, q_size + 2*kv_size + iq_size + ik_size]
    q_norm_weight: torch.Tensor,  # [head_dim]
    k_norm_weight: torch.Tensor,  # [head_dim]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    positions: torch.Tensor,  # [num_tokens] int
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor,  # [idx_head_dim]
    index_k_norm_weight: torch.Tensor,  # [idx_head_dim]
    num_index_heads: int,
    slot_mapping: torch.Tensor,  # [num_tokens] int64 logical slots
    kv_cache_k: torch.Tensor,  # SHUFFLE K cache [phys, num_kv_heads, head_dim//x, 16, x]
    kv_cache_v: torch.Tensor,  # SHUFFLE V cache [phys, num_kv_heads, 16//x, head_dim, x]
    index_cache: torch.Tensor,  # index K cache, viewable as [-1, idx_head_dim]
    q_out: torch.Tensor,  # [num_tokens, q_size] normed+roped q
    index_q_out: torch.Tensor,  # [num_tokens, iq_size] normed+roped index_q
    idx_head_dim: int,
) -> None:
    """Fused Gemma-RMSNorm + partial-NeoX-RoPE + page-16 SHUFFLE KV insert (Triton).

    One fused kernel doing q/index_q norm+rope (-> q_out/index_q_out), k norm+rope
    + raw v -> SHUFFLE K/V cache, and index_k norm+rope -> page-128 index cache.
    Math matches the AITER fused op oracle; K/V writes match
    ``reshape_and_cache(asm_layout=True)``.
    """
    num_tokens = qkv.shape[0]
    head_dim = q_norm_weight.shape[-1]
    x = 16 // kv_cache_k.element_size()
    assert head_dim == 128, "M3 fused shuffle insert requires head_dim == 128"
    assert kv_cache_k.is_contiguous() and kv_cache_v.is_contiguous()
    assert index_cache.is_contiguous()

    _fused_qknorm_rope_kv_insert_shuffle_kernel[(num_tokens,)](
        qkv,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        q_out,
        index_q_out,
        kv_cache_k,
        kv_cache_v,
        index_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        row_elems=qkv.shape[1],
        x=x,
        ASM_PAGE=16,
    )


def _is_fp8_kv_cache_tensor(kv_cache: torch.Tensor) -> bool:
    fp8_dtypes = (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
    )
    return kv_cache.dtype in {dtype for dtype in fp8_dtypes if dtype is not None}


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
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
def _gqa_sparse_fwd_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
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
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
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
                kv_cache_ptr
                + page * stride_kv_blk
                + 0 * stride_kv_kv
                + off_n[None, :] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if FP8_KV_CACHE:
                # Triton/ROCm does not support fp8 as RHS for tl.dot here.
                k = k.to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            # causal: q_abs_pos - k_off >= block_start (c)
            qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            v = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + 1 * stride_kv_kv
                + off_n[:, None] * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_d[None, :] * stride_kv_d,
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


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode == one query token per request, so the
# prefill kernel (which parallelizes over the query dim) leaves the GPU idle.
# This instead parallelizes over the selected top-k blocks, producing partials
# that the merge kernel combines (flash-decoding). All chunk counts depend only
# on shape constants so the grid is fixed within a cuda graph. Base-2
# (exp2/log2) softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
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
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q (== batch), num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, 2, 128, num_kv_heads, head_dim]
    t_ptr,  # topk_idx: [num_kv_heads, batch, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, batch, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, batch, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [batch]
    batch_size,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_kv,
    stride_kv_pos,
    stride_kv_h,
    stride_kv_d,
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
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (batch, chunk) together.
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % batch_size
    pid_c = pid_bc // batch_size
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk
    seq_len = tl.load(seq_lens + pid_b)
    # number of valid (non-padded) selected blocks for this request
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
        pos_mask = pos < seq_len  # decode query is the last token: attend all valid
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 0 * stride_kv_kv
            + off_n[None, :] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if FP8_KV_CACHE:
            # Triton/ROCm does not support fp8 as RHS for tl.dot here.
            k = k.to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + 1 * stride_kv_kv
            + off_n[:, None] * stride_kv_pos
            + pid_kh * stride_kv_h
            + off_d[None, :] * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if FP8_KV_CACHE:
            v = v.to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
    # empty chunks (chunk_start >= real_topk) keep lse_i = -inf -> weight 0 in merge
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
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, batch, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, batch, num_heads]
    out_ptr,  # merged out: [total_q (== batch), num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    valid = lse_max > float("-inf")
    weights = tl.where(valid, tl.exp2(lse - lse_max), tl.zeros_like(lse))
    weights = tl.where(valid, weights / tl.sum(weights, axis=0), weights)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    grid = (max_query_len, num_kv_heads, batch)
    _gqa_sparse_fwd_kernel[grid](
        q,
        kv_cache,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        FP8_KV_CACHE=_is_fp8_kv_cache_tensor(kv_cache),
        num_stages=1,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [batch, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, batch, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [batch, num_heads, head_dim]
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    batch, num_heads, head_dim = q.shape
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, batch * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, batch, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, batch, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (batch * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
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
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
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
        FP8_KV_CACHE=_is_fp8_kv_cache_tensor(kv_cache),
        num_stages=1,
    )
    merge_grid = (batch, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
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
        NUM_TOPK_CHUNKS=num_topk_chunks,
    )


# ---------------------------------------------------------------------------
# ASM paged-attention decode path (alternative to the Triton split-K decode).
#
# The lightning-indexer selects, per request, up to `topk` 128-token blocks.
# AITER's bf16 `pa_fwd_asm` only ships a blkSz=16 kernel, so the main KV cache
# is paged at 16 and each selected logical 128-block expands into its 8 physical
# 16-pages (physical = block_table[blk]*8 + j). We compact those selected pages
# into a dense per-request block_table + a context_lens giving the true number
# of valid KV tokens, then hand them to pa_fwd_asm.
#
# Only valid when per-rank num_kv_heads == 1 (topk is per-kv-head; ASM PA shares
# one block_table across all kv heads, so it cannot express per-kv-head
# selection). The caller asserts this.
#
# The partial (seq_len-containing) 128-block must be packed LAST so that
# pa_fwd_asm's tail mask (context_lens % 16) applies to it; all other selected
# blocks contribute full 8x16-page groups packed contiguously at the front.
# ---------------------------------------------------------------------------
# Physical KV page size for the ASM decode path. The bf16 gfx950 pa_fwd_asm
# kernels only ship a blkSz=16 variant (Gqa in {8,16}); the model's logical
# sparse block is 128, so each selected 128-block expands into
# PAGES_PER_SPARSE_BLOCK == 8 contiguous physical 16-pages.
ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE  # 8


@triton.jit
def _build_sparse_block_table_kernel(
    t_ptr,  # topk_idx: [1, batch, topk] int32, 0-indexed 128-blocks, -1 pad
    block_table_ptr,  # logical block_table [batch, max_blocks] int32 (128-granularity)
    seq_lens_ptr,  # [batch] int32
    sparse_bt_ptr,  # out: compacted 16-page block_table [batch, topk*8] int32
    sparse_ctx_ptr,  # out: compacted context_lens [batch] int32
    max_topk,
    sm_block_size: tl.constexpr,  # logical sparse block size (128)
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    asm_page_size: tl.constexpr,  # physical page size (16)
    stride_tn,
    stride_tk,
    stride_bt_b,
    stride_sbt_b,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    # logical 128-block containing the last valid token (the partial tail block).
    last_blk = (seq_len - 1) // sm_block_size
    bt_row = block_table_ptr + pid_b * stride_bt_b
    t_row = t_ptr + pid_b * stride_tn
    sbt_row = sparse_bt_ptr + pid_b * stride_sbt_b

    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(t_row + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    valid = blk >= 0
    is_tail = valid & (blk == last_blk)
    is_full = valid & (blk != last_blk)

    # Stable compaction in units of SPARSE BLOCKS: full blocks first (in
    # selection order), tail block last. Each sparse block then expands to
    # `pages_per_block` physical 16-pages.
    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)  # tail -> slot n_full

    # logical 128-page id of each selected block -> 8 physical 16-pages:
    #   physical = logical_id * pages_per_block + j   (matches block_convert)
    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * pages_per_block  # [BLOCK_SIZE_T]
    dst_base = slot * pages_per_block  # [BLOCK_SIZE_T]

    for j in range(pages_per_block):
        write_mask = valid & (dst_base + j < BLOCK_SIZE_T * pages_per_block)
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=write_mask)

    # true valid token count: full blocks contribute 128 each, tail the remainder.
    tail_tokens = seq_len - last_blk * sm_block_size
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * sm_block_size + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * sm_block_size, seq_len))
    tl.store(sparse_ctx_ptr + pid_b, ctx)


@torch.no_grad()
def minimax_m3_build_sparse_block_table(
    topk_idx: torch.Tensor,  # [1, batch, topk] int32 (num_kv_heads == 1)
    block_table: torch.Tensor,  # [batch, max_blocks] int32, logical 128-granularity
    seq_lens: torch.Tensor,  # [batch] int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact per-request selected 128-blocks into a dense 16-page block_table +
    context_lens for `pa_fwd_asm`.

    Each selected logical 128-block expands to its 8 physical 16-pages
    (``logical_id * 8 + j``, matching ``block_convert``). The partial tail block
    is packed last so pa_fwd_asm's tail mask (context_lens % 16) lands on it.

    Returns (sparse_bt [batch, topk*8] int32, sparse_ctx_lens [batch] int32).
    The compacted width is fixed (topk*8), so the grid is shape-constant
    (cudagraph-safe).
    """
    assert topk_idx.shape[0] == 1, "ASM PA decode requires num_kv_heads == 1"
    batch = topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    width = topk * PAGES_PER_SPARSE_BLOCK
    sparse_bt = torch.zeros((batch, width), dtype=torch.int32, device=topk_idx.device)
    sparse_ctx = torch.zeros((batch,), dtype=torch.int32, device=topk_idx.device)
    _build_sparse_block_table_kernel[(batch,)](
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        topk,
        SPARSE_BLOCK_SIZE,
        PAGES_PER_SPARSE_BLOCK,
        ASM_PAGE_SIZE,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@torch.no_grad()
def minimax_m3_sparse_attn_decode_asm(
    q: torch.Tensor,  # [batch, num_heads, head_dim==128]
    k_cache: torch.Tensor,  # SHUFFLE K [num_blocks, num_kv_heads, head_dim//x, 16, x]
    v_cache: torch.Tensor,  # SHUFFLE V [num_blocks, num_kv_heads, 16//x, head_dim, x]
    topk_idx: torch.Tensor,  # [num_kv_heads, batch, topk] int32
    block_table: torch.Tensor,  # [batch, max_blocks] int32, logical 128-granularity
    seq_lens: torch.Tensor,  # [batch] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [batch, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Block-sparse decode attention via the AITER Gluon paged-attention kernel.

    The lightning-indexer's selected logical 128-blocks are compacted into a
    dense PHYSICAL 16-page block_table (each 128-block -> 8 pages, tail packed
    last) + exact context_lens, then fed to the Gluon split-KV paged-attention
    decode kernel (``pa_decode_gluon``) over the page-16 SHUFFLE KV cache. The
    split-KV (flash-decoding) implementation is more efficient than the monolithic
    ASM kernel at low concurrency (few decode sequences), where it parallelizes
    over KV partitions to keep the GPU busy.

    Requires per-rank num_kv_heads == 1 (the indexer top-k is per-kv-head; one
    shared block_table cannot express per-kv-head selection) and head_dim == 128.
    """
    from atom.model_ops.base_attention import run_pa_decode_gluon
    from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

    assert num_kv_heads == 1, (
        "minimax_m3_sparse_attn_decode_asm requires per-rank num_kv_heads == 1;"
        f" got {num_kv_heads}. Use the Triton split-K decode path for GQA."
    )
    assert q.shape[-1] == 128, "Gluon paged-attention requires head_dim == 128."

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table(
        topk_idx, block_table, seq_lens
    )

    # Gluon split-KV decode setup (mirrors the standard MHA path in
    # attention_mha.py::paged_attention_triton). q is [batch, num_heads, 128];
    # one query token per sequence (max_seqlen_q == 1).
    num_seqs, num_q_heads_total, head_size = q.shape
    query_group_size = num_q_heads_total // num_kv_heads
    max_context_partition_num = get_recommended_splits(num_seqs, num_kv_heads)
    context_partition_size = 256
    intermediate_shape = (
        num_seqs,
        num_kv_heads,
        max_context_partition_num,
        query_group_size,
    )
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    temporary_output = torch.empty(
        *intermediate_shape, head_size, dtype=q.dtype, device=q.device
    )
    compute_type = torch.bfloat16  # bf16 KV cache (k/v_scale = None)
    run_pa_decode_gluon(
        output=output,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        context_lens=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        max_seqlen_q=1,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        q_scale=None,
        k_scale=k_scale,
        v_scale=v_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        sliding_window=-1,
        ps=True,
    )


# ---------------------------------------------------------------------------
# ASM paged-attention PREFILL path (per-token-as-decode).
#
# In M3 sparse attention each prefill query token attends its OWN per-token top-k
# 128-block set, so prefill == N independent length-1 "decodes". We build a
# PER-QUERY-TOKEN compacted 16-page block_table + per-token context_len (with the
# causal diagonal mask folded into the length), then drive pa_fwd_asm with
# qo_indptr=[0,1,...,total_q] (each token a length-1 segment). Verified: pa_fwd_asm
# honors per-token block_table/context_len indexing under qo_indptr.
#
# Causal: query token at absolute pos p sees keys k_abs <= p. So its effective
# length is p+1: full selected blocks below the self-block (p//128) contribute 128
# each; the self-block (packed LAST so pa_fwd_asm's tail mask lands on it)
# contributes p%128 + 1. Selected blocks above the self-block are causally invalid
# (the causal indexer should not pick them, but we mask defensively by excluding
# any block with blk > p//128).
# ---------------------------------------------------------------------------
@triton.jit
def _build_sparse_block_table_prefill_kernel(
    t_ptr,  # topk_idx: [1, total_q, topk] int32, 0-indexed 128-blocks, -1 pad
    block_table_ptr,  # logical block_table [batch, max_blocks] int32 (128-granularity)
    req_id_ptr,  # [total_q] int32: request index b of each query token (precomputed)
    abs_pos_ptr,  # [total_q] int32: absolute position p of each query token (precomputed)
    sparse_bt_ptr,  # out: compacted 16-page block_table [total_q, topk*8] int32
    sparse_ctx_ptr,  # out: compacted context_lens [total_q] int32
    max_topk,
    sm_block_size: tl.constexpr,  # logical sparse block size (128)
    pages_per_block: tl.constexpr,  # 16-pages per sparse block (8)
    stride_tn,
    stride_tk,
    stride_bt_b,
    stride_sbt_n,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)  # query token index (global)
    # req_id / abs_pos are layer-invariant and precomputed once in prepare_prefill
    # (numpy, no device sync), reused across all sparse layers -> no per-layer D2H.
    b = tl.load(req_id_ptr + pid_n)
    p = tl.load(abs_pos_ptr + pid_n)
    causal_len = p + 1
    self_blk = p // sm_block_size  # logical block containing this query token

    bt_row = block_table_ptr + b * stride_bt_b
    t_row = t_ptr + pid_n * stride_tn
    sbt_row = sparse_bt_ptr + pid_n * stride_sbt_n

    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(t_row + off_t * stride_tk, mask=off_t < max_topk, other=-1)
    # causal: drop any selected block strictly above the self-block.
    valid = (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)

    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)  # tail -> slot n_full

    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * pages_per_block
    dst_base = slot * pages_per_block

    for j in range(pages_per_block):
        write_mask = valid & (dst_base + j < BLOCK_SIZE_T * pages_per_block)
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=write_mask)

    # full blocks contribute 128 each; tail (self-block) contributes p%128 + 1.
    tail_tokens = causal_len - self_blk * sm_block_size
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * sm_block_size + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * sm_block_size, causal_len))
    tl.store(sparse_ctx_ptr + pid_n, ctx)


@torch.no_grad()
def minimax_m3_build_sparse_block_table_prefill(
    topk_idx: torch.Tensor,  # [1, total_q, topk] int32 (num_kv_heads == 1)
    block_table: torch.Tensor,  # [batch, max_blocks] int32, logical 128-granularity
    query_req_id: torch.Tensor,  # [total_q] int32, precomputed in prepare_prefill
    query_abs_pos: torch.Tensor,  # [total_q] int32, precomputed in prepare_prefill
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query-token compacted 16-page block_table + causal context_lens.

    Returns (sparse_bt [total_q, topk*8], sparse_ctx [total_q]). Each query token
    becomes a length-1 "request" for pa_fwd_asm; its causal cutoff (absolute pos
    p, so length p+1) is folded into context_len with the self-block packed last.

    ``query_req_id`` / ``query_abs_pos`` are layer-invariant and built ONCE in
    prepare_prefill (host numpy, no device sync) -> this per-layer build is fully
    on-device with zero D2H.
    """
    assert topk_idx.shape[0] == 1, "ASM PA prefill requires num_kv_heads == 1"
    total_q = topk_idx.shape[1]
    topk = topk_idx.shape[-1]
    device = topk_idx.device

    width = topk * PAGES_PER_SPARSE_BLOCK
    sparse_bt = torch.zeros((total_q, width), dtype=torch.int32, device=device)
    sparse_ctx = torch.zeros((total_q,), dtype=torch.int32, device=device)
    _build_sparse_block_table_prefill_kernel[(total_q,)](
        topk_idx,
        block_table,
        query_req_id,
        query_abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        SPARSE_BLOCK_SIZE,
        PAGES_PER_SPARSE_BLOCK,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@torch.no_grad()
def minimax_m3_sparse_attn_prefill_asm(
    q: torch.Tensor,  # [total_q, num_heads, head_dim==128]
    k_cache: torch.Tensor,  # SHUFFLE K [num_blocks, num_kv_heads, head_dim//x, 16, x]
    v_cache: torch.Tensor,  # SHUFFLE V [num_blocks, num_kv_heads, 16//x, head_dim, x]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] int32
    block_table: torch.Tensor,  # [batch, max_blocks] int32, logical 128-granularity
    query_req_id: (
        torch.Tensor | None
    ),  # [total_q] int32, precomputed in prepare_prefill
    query_abs_pos: (
        torch.Tensor | None
    ),  # [total_q] int32, precomputed in prepare_prefill
    qo_indptr: torch.Tensor | None,  # [total_q+1] int32, per-token CSR (precomputed)
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,  # [batch+1] int32, for the fallback
    prefix_lens: torch.Tensor | None = None,  # [batch] int32, for the fallback
) -> None:
    """Block-sparse PREFILL via AITER ASM pa_fwd_asm, per-token-as-decode.

    Each query token is a length-1 segment (qo_indptr=[0..total_q], max_qlen=1)
    with its own causal-capped block_table/context_len. The per-token metadata
    (query_req_id, query_abs_pos, qo_indptr) is layer-invariant and built once in
    prepare_prefill, so the hot path has zero host sync. Requires per-rank
    num_kv_heads == 1 and head_dim == 128.

    Fallback: if the precomputed metadata is None (e.g. spec-decode prefill paths
    that don't populate it), derive it on-device, SYNC-FREE, via searchsorted /
    arange (no .item(), no GPU repeat_interleave).
    """
    from atom.model_ops.base_attention import run_pa_fwd_asm

    assert num_kv_heads == 1, (
        "minimax_m3_sparse_attn_prefill_asm requires per-rank num_kv_heads == 1;"
        f" got {num_kv_heads}."
    )
    assert q.shape[-1] == 128, "ASM paged-attention requires head_dim == 128."

    total_q = q.shape[0]
    device = q.device
    if query_req_id is None or query_abs_pos is None:
        # Sync-free on-device derivation: req_id[n] = #(cu_seqlens_q[1:] <= n),
        # abs_pos[n] = prefix_lens[req] + (n - cu_seqlens_q[req]).
        assert cu_seqlens_q is not None and prefix_lens is not None
        pos = torch.arange(total_q, dtype=torch.int32, device=device)
        query_req_id = torch.searchsorted(
            cu_seqlens_q[1:].contiguous(), pos, right=True
        ).to(torch.int32)
        query_abs_pos = (
            prefix_lens[query_req_id] + (pos - cu_seqlens_q[query_req_id])
        ).to(torch.int32)
    if qo_indptr is None:
        qo_indptr = torch.arange(total_q + 1, dtype=torch.int32, device=device)

    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
        topk_idx, block_table, query_req_id, query_abs_pos
    )
    run_pa_fwd_asm(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=sparse_bt,
        context_lens=sparse_ctx,
        k_scale=k_scale,
        v_scale=v_scale,
        out=output,
        qo_indptr=qo_indptr,
        max_qlen=1,
        high_precision=0,
    )
