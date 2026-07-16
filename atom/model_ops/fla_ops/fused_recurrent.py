# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

import triton
import triton.language as tl

from .op import exp


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv

    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * HV + i_hv) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_init_state_token
            )
        else:
            p_h0 = h0 + bos * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BK, BV]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])
        # [BV]
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BK, BV]
        b_h += b_k[:, None] * b_v[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            p_ht = (
                ht
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                * stride_final_state_token
            )
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


@triton.jit
def gdn_decode_update_lossy_fast_fwd_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    out,
    state,
    state_indices,
    scale: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HEADS_PER_V: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n = i_nh // HV
    i_hv = i_nh - i_n * HV
    i_h = i_hv // HEADS_PER_V

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    state_idx = tl.load(state_indices + i_n).to(tl.int64)
    if state_idx < 0:
        # Padded / idle slot (e.g. PAD_SLOT_ID = -1 from SGLang's
        # mamba_cache_indices). Skip the state load/store and write zeros so
        # downstream ops that consume the full out buffer do not see
        # uninitialized memory.
        out_offsets = (i_n * HV + i_hv) * V + o_v
        tl.store(
            out + out_offsets,
            tl.zeros([BV], dtype=out.dtype.element_ty),
            mask=mask_v,
        )
        return

    state_base = ((state_idx * HV + i_hv) * K) * V
    state_offsets = state_base + o_k[:, None] * V + o_v[None, :]
    h = tl.load(
        state + state_offsets,
        mask=mask_h,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)

    q_offsets = (i_n * H + i_h) * K + o_k
    k_offsets = (i_n * H + i_h) * K + o_k
    v_offsets = (i_n * HV + i_hv) * V + o_v
    q_vec = tl.load(
        q + q_offsets,
        mask=mask_k,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    k_vec = tl.load(
        k + k_offsets,
        mask=mask_k,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    v_vec = tl.load(
        v + v_offsets,
        mask=mask_v,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)

    x = tl.load(a + i_n * HV + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(
        tl.float32
    )
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    gate = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    beta_val = tl.sigmoid(tl.load(b + i_n * HV + i_hv).to(tl.float32))

    q_vec = q_vec * tl.rsqrt(tl.sum(q_vec * q_vec, axis=0) + 1.0e-6)
    k_vec = k_vec * tl.rsqrt(tl.sum(k_vec * k_vec, axis=0) + 1.0e-6)
    q_vec = q_vec * scale

    h = h * tl.exp(gate)
    v_vec = (v_vec - tl.sum(h * k_vec[:, None], axis=0)) * beta_val
    h = h + k_vec[:, None] * v_vec[None, :]
    out_vec = tl.sum(h * q_vec[:, None], axis=0)

    out_offsets = (i_n * HV + i_hv) * V + o_v
    tl.store(out + out_offsets, out_vec.to(out.dtype.element_ty), mask=mask_v)
    tl.store(
        state + state_offsets,
        h.to(state.dtype.element_ty),
        mask=mask_h,
        cache_modifier=".cg",
    )


def gdn_decode_update_lossy_fast(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate decode fast path that fuses gating and recurrent update.

    This path updates ``initial_state`` in place and returns the same tensor as
    the final state. The kernel expects a contiguous state cache with layout
    ``[slot, value_head, key_dim, value_dim]``.
    """
    if beta != 1.0 or threshold != 20.0:
        raise ValueError(
            "gdn_decode_update_lossy_fast supports beta=1.0 and threshold=20.0"
        )
    if not use_qk_l2norm_in_kernel:
        raise ValueError("gdn_decode_update_lossy_fast requires QK L2 norm in kernel")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    A_log = A_log.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    dt_bias = dt_bias.contiguous()
    ssm_state_indices = ssm_state_indices.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert B == 1, "decode fast path expects B == 1"
    assert a.shape == (T, HV), "decode fast path expects a shaped [T, HV]"
    assert b.shape == (T, HV), "decode fast path expects b shaped [T, HV]"
    if HV < H or HV % H != 0:
        raise ValueError(
            "decode fast path expects value heads to be a multiple of heads"
        )
    if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, K, V):
        raise ValueError(
            "decode fast path expects initial_state shaped "
            f"[num_slots, {HV}, {K}, {V}]"
        )
    if not initial_state.is_contiguous():
        raise ValueError("decode fast path expects contiguous initial_state")

    BK = triton.next_power_of_2(K)
    BV = 64
    out = torch.empty_like(v).squeeze(0)
    grid = (triton.cdiv(K, BK), triton.cdiv(V, BV), T * HV)
    gdn_decode_update_lossy_fast_fwd_kernel[grid](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        out,
        initial_state,
        ssm_state_indices,
        scale,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        HV // H,
        num_warps=4,
        num_stages=1,
    )
    return out.unsqueeze(0), initial_state


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
