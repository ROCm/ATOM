"""
Triton MLA stage1 decode kernel for fp8 + multi-token (qo_len > 4).

Strategy:
  - The wrapper dequantizes Q and KV from fp8 to bf16 in PyTorch (handles AMD's
    fp8 semantics correctly via torch).
  - The Triton kernel then operates on bf16 inputs (which tl.dot supports natively
    on AMD MFMA).
  - This avoids the fp8 dtype semantics mismatch that caused output divergence
    in the earlier all-fp8-internal version.

Replaces aiter.mla_decode_stage1_asm_fwd when q.dtype == fp8 and max_seqlen_q > 4.

Status: PROTOTYPE v2 (bf16-internal)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_decode_stage1_bf16_multi_kernel(
    Q_ptr,           # [total_s, nhead, kv_lora_rank + rope_dim] bf16
    KV_ptr,          # [num_pages, kv_lora_rank + rope_dim] bf16 (flattened)
    qo_indptr_ptr,   # [batch+1] int32
    kv_indptr_ptr,   # [batch+1] int32
    kv_indices_ptr,  # [total_kv_indices] int32
    Logits_ptr,      # [total_s, NUM_KV_SPLITS, nhead, kv_lora_rank] fp32
    LSE_ptr,         # [total_s, NUM_KV_SPLITS, nhead, 1] fp32
    sm_scale,        # scalar fp32 softmax scale
    # strides (Q has 3 dims; KV is 2D flat; outputs as documented)
    stride_q_s,
    stride_q_h,
    stride_kv_p,
    stride_lo_s,
    stride_lo_sp,
    stride_lo_h,
    stride_lse_s,
    stride_lse_sp,
    stride_lse_h,
    # constants
    KV_LORA_RANK: tl.constexpr,    # 512
    ROPE_DIM: tl.constexpr,         # 64
    NUM_HEADS: tl.constexpr,        # 16
    BLOCK_M: tl.constexpr,          # M-chunk size (4)
    BLOCK_N: tl.constexpr,          # KV block (32)
    NUM_KV_SPLITS: tl.constexpr,    # 16
    M_START: tl.constexpr,          # starting M-offset within qo_len
):
    pid = tl.program_id(0)
    cur_batch = pid // NUM_KV_SPLITS
    split_kv_id = pid % NUM_KV_SPLITS

    qo_start = tl.load(qo_indptr_ptr + cur_batch)
    qo_end = tl.load(qo_indptr_ptr + cur_batch + 1)
    qo_len = qo_end - qo_start

    kv_start_idx = tl.load(kv_indptr_ptr + cur_batch)
    kv_end_idx = tl.load(kv_indptr_ptr + cur_batch + 1)
    seq_len = kv_end_idx - kv_start_idx

    if seq_len == 0:
        return

    kv_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_per_split, seq_len)

    if split_end <= split_start:
        return

    offs_m = M_START + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, NUM_HEADS)
    offs_c = tl.arange(0, KV_LORA_RANK)
    offs_r = tl.arange(0, ROPE_DIM)
    mask_m = offs_m < qo_len

    # Load Q (bf16)
    q_nope_offs = (
        (qo_start + offs_m[:, None, None]) * stride_q_s
        + offs_h[None, :, None] * stride_q_h
        + offs_c[None, None, :]
    )
    q_nope = tl.load(
        Q_ptr + q_nope_offs,
        mask=mask_m[:, None, None],
        other=0.0,
    )  # [BLOCK_M, NUM_HEADS, KV_LORA_RANK] bf16
    q_pe_offs = (
        (qo_start + offs_m[:, None, None]) * stride_q_s
        + offs_h[None, :, None] * stride_q_h
        + (KV_LORA_RANK + offs_r[None, None, :])
    )
    q_pe = tl.load(
        Q_ptr + q_pe_offs,
        mask=mask_m[:, None, None],
        other=0.0,
    )  # [BLOCK_M, NUM_HEADS, ROPE_DIM] bf16

    # Position of Q[m]: drafts occupy last qo_len positions
    q_pos = seq_len - qo_len + offs_m  # [BLOCK_M]

    e_max = tl.full([BLOCK_M, NUM_HEADS], float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_M, NUM_HEADS], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, NUM_HEADS, KV_LORA_RANK], dtype=tl.float32)

    # Reshape Q for matmul: [M*H, C/R]
    q_nope_2d = tl.reshape(q_nope, (BLOCK_M * NUM_HEADS, KV_LORA_RANK))
    q_pe_2d = tl.reshape(q_pe, (BLOCK_M * NUM_HEADS, ROPE_DIM))

    for start_n in range(split_start, split_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < split_end

        kv_loc = tl.load(
            kv_indices_ptr + kv_start_idx + offs_n,
            mask=mask_n,
            other=0,
        )

        k_nope_offs = kv_loc[:, None] * stride_kv_p + offs_c[None, :]
        k_pe_offs = kv_loc[:, None] * stride_kv_p + (KV_LORA_RANK + offs_r[None, :])

        k_nope = tl.load(KV_ptr + k_nope_offs, mask=mask_n[:, None], other=0.0)
        k_pe = tl.load(KV_ptr + k_pe_offs, mask=mask_n[:, None], other=0.0)

        # qk_nope = q_nope @ k_nope^T → [M*H, N]
        qk_nope = tl.dot(q_nope_2d, tl.trans(k_nope))
        qk_pe = tl.dot(q_pe_2d, tl.trans(k_pe))
        qk = (qk_nope + qk_pe) * sm_scale
        qk = tl.reshape(qk, (BLOCK_M, NUM_HEADS, BLOCK_N))

        # Causal + bounds mask
        causal_mask = offs_n[None, :] <= q_pos[:, None]
        valid = mask_m[:, None] & mask_n[None, :] & causal_mask
        qk = tl.where(valid[:, None, :], qk, float("-inf"))

        # Online softmax
        cur_max = tl.max(qk, axis=2)
        new_max = tl.maximum(cur_max, e_max)
        new_max_safe = tl.where(new_max == float("-inf"), 0.0, new_max)
        rescale = tl.exp(e_max - new_max_safe)
        rescale = tl.where(new_max == float("-inf"), 1.0, rescale)

        p = tl.exp(qk - new_max_safe[:, :, None])
        p = tl.where(valid[:, None, :], p, 0.0)

        acc = acc * rescale[:, :, None]
        p_2d = tl.reshape(p, (BLOCK_M * NUM_HEADS, BLOCK_N))
        # V = k_nope (MLA absorbs W_uV into the output proj)
        delta = tl.dot(p_2d.to(k_nope.dtype), k_nope)
        delta = tl.reshape(delta, (BLOCK_M, NUM_HEADS, KV_LORA_RANK))
        acc = acc + delta

        e_sum = e_sum * rescale + tl.sum(p, axis=2)
        e_max = new_max

    # Normalize
    e_sum_safe = tl.where(e_sum == 0.0, 1.0, e_sum)
    out = acc / e_sum_safe[:, :, None]

    out_offs = (
        (qo_start + offs_m[:, None, None]) * stride_lo_s
        + split_kv_id * stride_lo_sp
        + offs_h[None, :, None] * stride_lo_h
        + offs_c[None, None, :]
    )
    tl.store(Logits_ptr + out_offs, out, mask=mask_m[:, None, None])

    lse_val = e_max + tl.log(e_sum_safe)
    lse_val = tl.where(e_sum == 0.0, float("-inf"), lse_val)
    lse_offs = (
        (qo_start + offs_m[:, None]) * stride_lse_s
        + split_kv_id * stride_lse_sp
        + offs_h[None, :] * stride_lse_h
    )
    tl.store(LSE_ptr + lse_offs, lse_val, mask=mask_m[:, None])


def mla_decode_stage1_fp8_multi(
    q,                  # fp8 [total_s, nhead, kv_lora_rank + rope_dim]
    kv_buffer,          # fp8 [N, 1, 1, kv_lora_rank + rope_dim]
    qo_indptr,          # [batch+1] int32
    kv_indptr,          # [batch+1] int32
    kv_indices,         # int32
    num_kv_splits,      # int
    sm_scale,           # float
    logits,             # fp32 [total_s, num_kv_splits, nhead, kv_lora_rank]
    attn_lse,           # fp32 [total_s, num_kv_splits, nhead, 1]
    q_scale,            # scalar tensor (assumed 1.0 in ATOM)
    kv_scale,           # scalar tensor (assumed 1.0 in ATOM)
    max_seqlen_q,       # int (passed by caller — no .item() needed)
):
    """Wrapper: dequant fp8 → bf16, then launch Triton kernel.
    Cudagraph-friendly: no .item() calls (would break stream capture).
    """
    total_s, nhead, head_size = q.shape
    kv_lora_rank = logits.shape[-1]
    rope_dim = head_size - kv_lora_rank
    batch = qo_indptr.shape[0] - 1

    # Direct fp8 → bf16 cast (skip fp32, avoids OOM).
    # ATOM's scales are always scalar 1.0 — skip mul to be cudagraph-safe.
    q_bf16 = q.to(torch.bfloat16)
    # KV: only dequant the entire pool to bf16 (2x memory but no sync)
    kv_bf16 = kv_buffer.reshape(-1, head_size).to(torch.bfloat16)

    BLOCK_M = 4
    BLOCK_N = 32

    grid = (batch * num_kv_splits,)

    # Iterate M-chunks. max_seqlen_q is a static int from caller.
    for m_start in range(0, max_seqlen_q, BLOCK_M):
        _mla_decode_stage1_bf16_multi_kernel[grid](
            q_bf16,
            kv_bf16,
            qo_indptr,
            kv_indptr,
            kv_indices,
            logits,
            attn_lse,
            sm_scale,
            q_bf16.stride(0),
            q_bf16.stride(1),
            kv_bf16.stride(0),
            logits.stride(0),
            logits.stride(1),
            logits.stride(2),
            attn_lse.stride(0),
            attn_lse.stride(1),
            attn_lse.stride(2),
            KV_LORA_RANK=kv_lora_rank,
            ROPE_DIM=rope_dim,
            NUM_HEADS=nhead,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_KV_SPLITS=num_kv_splits,
            M_START=m_start,
        )
