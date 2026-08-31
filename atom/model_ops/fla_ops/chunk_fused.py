# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused single-kernel chunked Gated DeltaNet forward — Phase 1.

This file implements a fused version of the FLA-style chunked Gated DeltaNet
prefill that combines the h-recurrence (chunk_delta_h kernel) with the o-GEMM
(chunk_o kernel) into ONE Triton kernel. Intermediate per-chunk hidden
states (the `h` tensor of shape [B, NT, H, V, K]) never get materialized in
HBM — they live entirely in registers across the chunk loop.

Scope (Phase 1):
    * Prologue kernels (l2norm, cumsum, KKT, solve_tril, recompute_w_u) are
      reused unchanged from atom.model_ops.fla_ops.* — they produce per-chunk
      artifacts (g_cum, w, u) that the fused kernel consumes. Fusing those is
      a later phase.
    * State layout is fixed to vLLM's "[V, K]"-per-head convention so the
      output `final_state` can be written directly into ssm_state without a
      transpose. ATOM's [K, V] layout is not supported here.
    * Output buffer `o` is always allocated inside the wrapper. No inplace
      `o=` parameter — callers write into their target buffer themselves
      (matches the modeling-code pattern that existed before the inplace
      experiment).

Correctness reference: vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .op import exp
from .solve_tril import solve_tril
from .utils import use_cuda_graph
from .wy_fast import recompute_w_u_fwd

# ---------------------------------------------------------------------------
# Fused Phase-1 kernel: h-recurrence (vk layout) + per-chunk o-GEMM.
#
# This kernel is a port of the existing
# chunk_gated_delta_rule_fwd_kernel_h_blockdim64_vk (in chunk_delta_h.py)
# with the body of chunk_fwd_kernel_o (in chunk_o.py) inlined immediately
# AFTER each chunk's h-update — using the b_h1..b_h4 register tiles that
# would otherwise have been spilled to HBM as `h[c]`.
#
# Grid: (cdiv(V, BV), N * H)  — one program per (sequence, head, V-block).
# Each program walks the sequence's NT chunks sequentially because the
# h-recurrence is causal.
# ---------------------------------------------------------------------------

NUM_WARPS = [2, 4]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
        # BV=8, 16 added for long-sequence grid occupancy on small-H models.
        # At T=8192, H=16, BV=32 gives only V/BV * H = 4 * 16 = 64 programs
        # (~21% of MI300X CUs); BV=16 doubles that, BV=8 quadruples it. Triton
        # autotunes per (H, K, V, BT) so short-T shapes naturally pick larger
        # BV for better register efficiency while long-T shapes pick smaller
        # BV for grid parallelism.
        for BV in [8, 16, 32, 64]
    ],
    # Include T in the autotune key: the per-program work (chunk loop count
    # NT = cdiv(T, BT)) is proportional to T, and the best BV depends on T.
    # Short-T wants larger BV for register efficiency; long-T wants smaller
    # BV for grid occupancy (e.g., at T=8192, H=16 with BV=32 we have only
    # 64 programs — under 25% of MI300X CUs; BV=8 gets 256 programs, ~84%).
    # Without T in the key, Triton picks one config at first call (often a
    # tiny test shape) and reuses it for all later T, which catastrophically
    # underperforms on long sequences.
    key=["H", "K", "V", "BT", "T"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_fused_fwd_kernel_vk(
    # --- inputs (all needed by the recurrence) ---
    q,  # [B, T, Hg, K]
    k,  # [B, T, Hg, K]
    u,  # [B, T, H,  V]  — the "new v" from recompute_w_u
    w,  # [B, T, H,  K]  — the "corrected k weight" from recompute_w_u
    g,  # [B, T, H]      — cumulative log-decay
    h0,  # [N, H, V, K] fp32 or None
    # --- outputs ---
    ht,  # [N, H, V, K] fp32 or None  (final recurrent state)
    o,  # [B, T, H, V] — the chunk-attention output (same dtype as q)
    # --- ragged-batch metadata ---
    cu_seqlens,  # [N+1] int32 or None
    scale,  # 1/sqrt(K), runtime scalar
    # --- compile-time shapes ---
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    # --- heuristic flags ---
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # Grid axes: (V-block, N * H)
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    i_hg = i_h // (H // Hg)  # GQA: map v-head -> k-head

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos = i_n * T
        eos = bos + T
        NT = tl.cdiv(T, BT)

    # --- per-head register tiles for the recurrent state S_h ∈ [V, K] ---
    # Stored vLLM-style as [BV, K], split across 4 K-tiles of width 64.
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # --- base pointer offsets (per-head, per-sequence) ---
    # Layout: q,k are [B, T, Hg, K]; u,o are [B, T, H, V]; w is [B, T, H, K];
    # g is [B, T, H]. We pre-shift each pointer to the start of this head/sequence.
    q_p = q + (bos * Hg + i_hg) * K
    k_p = k + (bos * Hg + i_hg) * K
    u_p = u + (bos * H + i_h) * V
    w_p = w + (bos * H + i_h) * K
    o_p = o + (bos * H + i_h) * V
    g_p = g + bos * H + i_h
    stride_q = Hg * K
    stride_k = Hg * K
    stride_u = H * V
    stride_w = H * K
    stride_o = H * V
    stride_g = H

    if USE_INITIAL_STATE:
        h0_p = h0 + (i_n * H + i_h) * V * K
        _p_h0_1_0 = (i_v * BV) + tl.arange(0, BV)
        _p_h0_1_1 = (0) + tl.arange(0, 64)
        b_h1 += tl.load(
            h0_p + _p_h0_1_0[:, None] * (K) + _p_h0_1_1[None, :] * (1),
            mask=(_p_h0_1_0[:, None] < (V)) & (_p_h0_1_1[None, :] < (K)),
            other=0.0,
        ).to(tl.float32)
        if K > 64:
            _p_h0_2_0 = (i_v * BV) + tl.arange(0, BV)
            _p_h0_2_1 = (64) + tl.arange(0, 64)
            b_h2 += tl.load(
                h0_p + _p_h0_2_0[:, None] * (K) + _p_h0_2_1[None, :] * (1),
                mask=(_p_h0_2_0[:, None] < (V)) & (_p_h0_2_1[None, :] < (K)),
                other=0.0,
            ).to(tl.float32)
        if K > 128:
            _p_h0_3_0 = (i_v * BV) + tl.arange(0, BV)
            _p_h0_3_1 = (128) + tl.arange(0, 64)
            b_h3 += tl.load(
                h0_p + _p_h0_3_0[:, None] * (K) + _p_h0_3_1[None, :] * (1),
                mask=(_p_h0_3_0[:, None] < (V)) & (_p_h0_3_1[None, :] < (K)),
                other=0.0,
            ).to(tl.float32)
        if K > 192:
            _p_h0_4_0 = (i_v * BV) + tl.arange(0, BV)
            _p_h0_4_1 = (192) + tl.arange(0, 64)
            b_h4 += tl.load(
                h0_p + _p_h0_4_0[:, None] * (K) + _p_h0_4_1[None, :] * (1),
                mask=(_p_h0_4_0[:, None] < (V)) & (_p_h0_4_1[None, :] < (K)),
                other=0.0,
            ).to(tl.float32)

    # ----- main chunk loop -----
    # Each chunk runs the recurrence step (kernel 5 work) and emits its
    # contribution to o (kernel 6 work) using b_h_{1..4} already in registers.
    for i_t in range(NT):
        # === o-GEMM for this chunk ===
        # Build b_o ∈ [BT, BV] = Q_c · S_BV_prev^T   (the inter-chunk piece,
        # i.e. the prior state's contribution to the new attention output).
        # The recurrence below will then update S_BV with the chunk's K, V.
        #
        # vLLM's o kernel iterates K-blocks of [BK=64, BV] of h; here we have
        # the full h_BV[V_block, K] = (b_h1 | b_h2 | b_h3 | b_h4) in registers.
        # b_o += dot(b_q_k, b_h_k.T) for each K-tile k.
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        # Downcast b_h to the input dtype BEFORE the o-emit dot so that the
        # numerical regime matches vLLM upstream's chunk_fwd_kernel_o, which
        # loads `h` from HBM as bf16 (it does `tl.dot(b_q, tl.trans(b_h))`
        # where b_h is the bf16-loaded tile). The order matters: cast first,
        # THEN trans, so the cast happens in [BV, K]-tile shape and trans is
        # a pure metadata op on bf16 values — exactly mirroring "store as
        # bf16 then load and trans" in the unfused pipeline.
        b_h1_bf = b_h1.to(q_p.dtype.element_ty)
        b_h1_q = tl.trans(b_h1_bf)
        if K > 64:
            b_h2_bf = b_h2.to(q_p.dtype.element_ty)
            b_h2_q = tl.trans(b_h2_bf)
        if K > 128:
            b_h3_bf = b_h3.to(q_p.dtype.element_ty)
            b_h3_q = tl.trans(b_h3_bf)
        if K > 192:
            b_h4_bf = b_h4.to(q_p.dtype.element_ty)
            b_h4_q = tl.trans(b_h4_bf)

        # K-tile 1
        # b_k as (K, T) so that dot(b_q, b_k) -> [BT, BT].
        _p_q1_0 = (i_t * BT) + tl.arange(0, BT)
        _p_q1_1 = (0) + tl.arange(0, 64)
        b_q1 = tl.load(
            q_p + _p_q1_0[:, None] * (stride_q) + _p_q1_1[None, :] * (1),
            mask=(_p_q1_0[:, None] < (T)) & (_p_q1_1[None, :] < (K)),
            other=0.0,
        )
        _p_k1_0 = (0) + tl.arange(0, 64)
        _p_k1_1 = (i_t * BT) + tl.arange(0, BT)
        b_k1 = tl.load(
            k_p + _p_k1_0[:, None] * (1) + _p_k1_1[None, :] * (stride_k),
            mask=(_p_k1_0[:, None] < (K)) & (_p_k1_1[None, :] < (T)),
            other=0.0,
        )
        # b_h1_q is [64, BV] in input dtype — fold into b_o ∈ [BT, BV] via
        # b_q1 @ b_h1_q which is [BT, 64] @ [64, BV] = [BT, BV].
        b_o += tl.dot(b_q1, b_h1_q)
        b_A += tl.dot(b_q1, b_k1)

        if K > 64:
            _p_q2_0 = (i_t * BT) + tl.arange(0, BT)
            _p_q2_1 = (64) + tl.arange(0, 64)
            b_q2 = tl.load(
                q_p + _p_q2_0[:, None] * (stride_q) + _p_q2_1[None, :] * (1),
                mask=(_p_q2_0[:, None] < (T)) & (_p_q2_1[None, :] < (K)),
                other=0.0,
            )
            _p_k2_0 = (64) + tl.arange(0, 64)
            _p_k2_1 = (i_t * BT) + tl.arange(0, BT)
            b_k2 = tl.load(
                k_p + _p_k2_0[:, None] * (1) + _p_k2_1[None, :] * (stride_k),
                mask=(_p_k2_0[:, None] < (K)) & (_p_k2_1[None, :] < (T)),
                other=0.0,
            )
            b_o += tl.dot(b_q2, b_h2_q)
            b_A += tl.dot(b_q2, b_k2)
        if K > 128:
            _p_q3_0 = (i_t * BT) + tl.arange(0, BT)
            _p_q3_1 = (128) + tl.arange(0, 64)
            b_q3 = tl.load(
                q_p + _p_q3_0[:, None] * (stride_q) + _p_q3_1[None, :] * (1),
                mask=(_p_q3_0[:, None] < (T)) & (_p_q3_1[None, :] < (K)),
                other=0.0,
            )
            _p_k3_0 = (128) + tl.arange(0, 64)
            _p_k3_1 = (i_t * BT) + tl.arange(0, BT)
            b_k3 = tl.load(
                k_p + _p_k3_0[:, None] * (1) + _p_k3_1[None, :] * (stride_k),
                mask=(_p_k3_0[:, None] < (K)) & (_p_k3_1[None, :] < (T)),
                other=0.0,
            )
            b_o += tl.dot(b_q3, b_h3_q)
            b_A += tl.dot(b_q3, b_k3)
        if K > 192:
            _p_q4_0 = (i_t * BT) + tl.arange(0, BT)
            _p_q4_1 = (192) + tl.arange(0, 64)
            b_q4 = tl.load(
                q_p + _p_q4_0[:, None] * (stride_q) + _p_q4_1[None, :] * (1),
                mask=(_p_q4_0[:, None] < (T)) & (_p_q4_1[None, :] < (K)),
                other=0.0,
            )
            _p_k4_0 = (192) + tl.arange(0, 64)
            _p_k4_1 = (i_t * BT) + tl.arange(0, BT)
            b_k4 = tl.load(
                k_p + _p_k4_0[:, None] * (1) + _p_k4_1[None, :] * (stride_k),
                mask=(_p_k4_0[:, None] < (K)) & (_p_k4_1[None, :] < (T)),
                other=0.0,
            )
            b_o += tl.dot(b_q4, b_h4_q)
            b_A += tl.dot(b_q4, b_k4)

        # Apply chunk-relative gating to the inter-chunk attn output and the
        # intra-chunk A. (Identical to chunk_fwd_kernel_o.)
        if USE_G:
            _p_g_0 = (i_t * BT) + tl.arange(0, BT)
            b_g = tl.load(g_p + _p_g_0 * (stride_g), mask=(_p_g_0 < (T)), other=0.0)
            b_o = b_o * exp(b_g)[:, None]
            b_A = b_A * exp(b_g[:, None] - b_g[None, :])

        # Causal mask on A (lower-tri inside the chunk, zero outside).
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_A = tl.where(m_A, b_A, 0)

        # Load the chunk's "new v" (u) — it's the post-correction value the
        # recurrence and the o-write both consume. We pre-load it BEFORE
        # running the recurrence so we use the same bytes for both: the
        # o-emit needs u for the intra-chunk piece, the recurrence needs
        # the (g-decayed, gated) version for b_h.
        _p_u_0 = (i_t * BT) + tl.arange(0, BT)
        _p_u_1 = (i_v * BV) + tl.arange(0, BV)
        b_u = tl.load(
            u_p + _p_u_0[:, None] * (stride_u) + _p_u_1[None, :] * (1),
            mask=(_p_u_0[:, None] < (T)) & (_p_u_1[None, :] < (V)),
            other=0.0,
        )

        # Compose o_c: inter-chunk + intra-chunk causal, both scaled by 1/sqrt(K).
        b_o = b_o * scale + tl.dot(b_A.to(b_u.dtype), b_u) * scale

        # Write the chunk's output to HBM.
        _p_o_0 = (i_t * BT) + tl.arange(0, BT)
        _p_o_1 = (i_v * BV) + tl.arange(0, BV)
        tl.store(
            o_p + _p_o_0[:, None] * (stride_o) + _p_o_1[None, :] * (1),
            b_o.to(o_p.dtype.element_ty),
            mask=(_p_o_0[:, None] < (T)) & (_p_o_1[None, :] < (V)),
        )

        # === recurrence step: update S_BV ===
        # The mathematics matches chunk_gated_delta_rule_fwd_kernel_h_blockdim64_vk:
        #   1. b_v_new = u - w @ S_BV^T          (post-delta-correction value)
        #   2. b_v_new *= exp(g_last - g)        (per-token gate)
        #   3. b_h *= exp(g_last)                (per-chunk gate on prior state)
        #   4. b_h += k_c^T @ b_v_new            (rank-BT update)
        # Step 1: load w-chunk for each K-tile and accumulate b_v ∈ [BT, BV].
        _p_w1_0 = (i_t * BT) + tl.arange(0, BT)
        _p_w1_1 = (0) + tl.arange(0, 64)
        b_w1 = tl.load(
            w_p + _p_w1_0[:, None] * (stride_w) + _p_w1_1[None, :] * (1),
            mask=(_p_w1_0[:, None] < (T)) & (_p_w1_1[None, :] < (K)),
            other=0.0,
        )
        # b_h1 is [BV, 64]; w_chunk is [BT, 64]; want b_v ∈ [BT, BV] =
        # b_w @ b_h.T = [BT, 64] @ [64, BV].
        b_v = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
        if K > 64:
            _p_w2_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w2_1 = (64) + tl.arange(0, 64)
            b_w2 = tl.load(
                w_p + _p_w2_0[:, None] * (stride_w) + _p_w2_1[None, :] * (1),
                mask=(_p_w2_0[:, None] < (T)) & (_p_w2_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
        if K > 128:
            _p_w3_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w3_1 = (128) + tl.arange(0, 64)
            b_w3 = tl.load(
                w_p + _p_w3_0[:, None] * (stride_w) + _p_w3_1[None, :] * (1),
                mask=(_p_w3_0[:, None] < (T)) & (_p_w3_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w3, tl.trans(b_h3).to(b_w3.dtype))
        if K > 192:
            _p_w4_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w4_1 = (192) + tl.arange(0, 64)
            b_w4 = tl.load(
                w_p + _p_w4_0[:, None] * (stride_w) + _p_w4_1[None, :] * (1),
                mask=(_p_w4_0[:, None] < (T)) & (_p_w4_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w4, tl.trans(b_h4).to(b_w4.dtype))

        # b_v ← b_u - (w @ S^T)   — the corrected delta value
        b_v = b_u - b_v

        # Apply per-token gating to the corrected value (Step 2) and scale
        # the carried-over state by the chunk's last gate (Step 3).
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            # Reload g for this chunk; same b_g as we computed for o above
            # — but we let the compiler decide whether to re-load or reuse.
            # (Triton can CSE this when the inputs are constant.)
            _p_g_recur_0 = (i_t * BT) + tl.arange(0, BT)
            b_g_recur = tl.load(
                g_p + _p_g_recur_0 * (stride_g), mask=(_p_g_recur_0 < (T)), other=0.0
            )
            b_g_last = tl.load(g_p + last_idx * stride_g)
            mask_chunk = (i_t * BT + tl.arange(0, BT)) < T
            b_v = b_v * tl.where(mask_chunk, exp(b_g_last - b_g_recur), 0)[:, None]
            scale_last = exp(b_g_last)
            b_h1 *= scale_last
            if K > 64:
                b_h2 *= scale_last
            if K > 128:
                b_h3 *= scale_last
            if K > 192:
                b_h4 *= scale_last

        # Step 4: rank-BT update — b_h += b_k^T @ b_v_new for each K-tile.
        b_v_t = b_v.to(k_p.dtype.element_ty)
        # b_k1 already loaded above (in the o-GEMM step); reuse it.
        # b_k1 has shape [64, BT]; b_v_t has shape [BT, BV]; result is [64, BV].
        # We want to add into b_h1 ∈ [BV, 64], so transpose: trans([64, BV]) = [BV, 64].
        b_h1 += tl.trans(tl.dot(b_k1, b_v_t))
        if K > 64:
            b_h2 += tl.trans(tl.dot(b_k2, b_v_t))
        if K > 128:
            b_h3 += tl.trans(tl.dot(b_k3, b_v_t))
        if K > 192:
            b_h4 += tl.trans(tl.dot(b_k4, b_v_t))

    # ----- epilogue: write final recurrent state -----
    if STORE_FINAL_STATE:
        ht_p = ht + (i_n * H + i_h) * V * K
        _p_ht1_0 = (i_v * BV) + tl.arange(0, BV)
        _p_ht1_1 = (0) + tl.arange(0, 64)
        tl.store(
            ht_p + _p_ht1_0[:, None] * (K) + _p_ht1_1[None, :] * (1),
            b_h1.to(ht_p.dtype.element_ty),
            mask=(_p_ht1_0[:, None] < (V)) & (_p_ht1_1[None, :] < (K)),
        )
        if K > 64:
            _p_ht2_0 = (i_v * BV) + tl.arange(0, BV)
            _p_ht2_1 = (64) + tl.arange(0, 64)
            tl.store(
                ht_p + _p_ht2_0[:, None] * (K) + _p_ht2_1[None, :] * (1),
                b_h2.to(ht_p.dtype.element_ty),
                mask=(_p_ht2_0[:, None] < (V)) & (_p_ht2_1[None, :] < (K)),
            )
        if K > 128:
            _p_ht3_0 = (i_v * BV) + tl.arange(0, BV)
            _p_ht3_1 = (128) + tl.arange(0, 64)
            tl.store(
                ht_p + _p_ht3_0[:, None] * (K) + _p_ht3_1[None, :] * (1),
                b_h3.to(ht_p.dtype.element_ty),
                mask=(_p_ht3_0[:, None] < (V)) & (_p_ht3_1[None, :] < (K)),
            )
        if K > 192:
            _p_ht4_0 = (i_v * BV) + tl.arange(0, BV)
            _p_ht4_1 = (192) + tl.arange(0, 64)
            tl.store(
                ht_p + _p_ht4_0[:, None] * (K) + _p_ht4_1[None, :] * (1),
                b_h4.to(ht_p.dtype.element_ty),
                mask=(_p_ht4_0[:, None] < (V)) & (_p_ht4_1[None, :] < (K)),
            )


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused chunked Gated DeltaNet forward — Phase 1.

    Reuses the existing prologue kernels (l2norm, cumsum, KKT, solve_tril,
    recompute_w_u) and fuses the h-recurrence + o-GEMM into a single kernel
    that keeps the chunk-state `b_h` in registers across chunks.

    Inputs / outputs match the FLA reference contract; see module docstring
    for scope notes.
    """
    # --- input validation (mirrors the FLA reference's checks) ---
    assert q.dtype == k.dtype == v.dtype, "q/k/v must share dtype"
    assert q.dtype != torch.float32, (
        "Use bfloat16 / float16; chunk_gated_delta_rule_fused does not "
        "support fp32 inputs."
    )
    assert beta.dim() == 3, f"beta must be [B, T, H]; got shape {tuple(beta.shape)}"

    if cu_seqlens is not None:
        assert q.shape[0] == 1, (
            f"cu_seqlens requires batch=1 (got B={q.shape[0]}); flatten "
            f"variable-length inputs before calling."
        )
        N = cu_seqlens.numel() - 1
        if initial_state is not None:
            assert initial_state.shape[0] == N, (
                f"initial_state.shape[0]={initial_state.shape[0]} != "
                f"len(cu_seqlens)-1={N}"
            )
    else:
        N = q.shape[0]

    # --- optional in-kernel l2norm on q, k ---
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # --- prologue (reused unchanged from the existing 7-kernel path) ---
    g_cum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g=g_cum, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g_cumsum=g_cum, cu_seqlens=cu_seqlens
    )

    # --- output buffers ---
    # o is [B, T, H_v, V], same dtype as q (matches FLA reference).
    o = torch.empty_like(v)
    final_state = (
        k.new_empty(N, v.shape[-2], v.shape[-1], k.shape[-1], dtype=torch.float32)
        if output_final_state
        else None
    )

    # --- launch the fused kernel ---
    B, T, Hg, K = q.shape
    H = v.shape[-2]
    V = v.shape[-1]
    BT = 64  # algorithmic chunk size (must match the prologue's chunk_size)
    assert (
        K <= 256
    ), f"chunk_fused: K must be <= 256 (got {K}); kernel only has 4 K-tiles."

    # N (# sequences) is already computed above based on cu_seqlens vs B.

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_fused_fwd_kernel_vk[grid](
        q=q,
        k=k,
        u=u,
        w=w,
        g=g_cum,
        h0=initial_state,
        ht=final_state,
        o=o,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o, final_state
