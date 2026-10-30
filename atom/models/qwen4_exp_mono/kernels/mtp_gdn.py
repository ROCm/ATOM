# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""q4 convolution and GDN gates in one launch, preserving Native rollback.

The convolution device function remains the sole owner of history updates.
Each request's first channel tile also publishes all four rows of g/beta.
ReplaySSM follows on the same stream; this kernel never advances write_pos.
"""

import triton
import triton.language as tl

from atom.model_ops.mamba_ops.causal_conv1d import _causal_conv1d_update_kernel


@triton.jit
def _prepare(
    X,
    W,
    STATE,
    SLOTS,
    ACCEPTED,
    CU,
    Q,
    K,
    V,
    A,
    B,
    ALOG,
    BIAS,
    G,
    BETA,
    REQUESTS: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    HEADS: tl.constexpr,
    STATE_SLOTS: tl.constexpr,
    XS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    ST: tl.constexpr,
    IS: tl.constexpr,
):
    _causal_conv1d_update_kernel(
        x_ptr=X,
        w_ptr=W,
        bias_ptr=W,
        conv_state_ptr=STATE,
        conv_state_indices_ptr=SLOTS,
        conv_state_indices_in_ptr=SLOTS,
        num_accepted_tokens_ptr=ACCEPTED,
        query_start_loc_ptr=CU,
        block_idx_last_scheduled_token=SLOTS,
        initial_state_idx=SLOTS,
        query_ptr=Q,
        key_ptr=K,
        value_ptr=V,
        batch=REQUESTS,
        dim=2 * K_DIM + V_DIM,
        seqlen=4,
        k_dim_size=K_DIM,
        v_dim_size=V_DIM,
        state_len=6,
        num_cache_lines=STATE_SLOTS,
        stride_x_seq=0,
        stride_x_dim=1,
        stride_x_token=XS,
        stride_w_dim=4,
        stride_w_width=1,
        stride_conv_state_seq=SS,
        stride_conv_state_dim=SD,
        stride_conv_state_tok=ST,
        stride_state_indices=IS,
        stride_q_seq=K_DIM,
        stride_q_token=1,
        stride_q_dim=1,
        stride_k_seq=K_DIM,
        stride_k_token=1,
        stride_k_dim=1,
        stride_v_seq=V_DIM,
        stride_v_token=1,
        stride_v_dim=1,
        pad_slot_id=-1,
        HAS_BIAS=False,
        KERNEL_WIDTH=4,
        SILU_ACTIVATION=True,
        IS_VARLEN=True,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=True,
        NP2_STATELEN=8,
        USE_PAD_SLOT=True,
        BLOCK_N=256,
    )
    if tl.program_id(1) == 0:
        request = tl.program_id(0)
        first = tl.load(CU + request)
        end = tl.load(CU + request + 1)
        row = first + tl.arange(0, 4)
        head = tl.arange(0, triton.next_power_of_2(HEADS))
        mask = (row[:, None] < end) & (head[None, :] < HEADS)
        a = tl.load(A + row[:, None] * AS + head[None, :], mask, 0).to(tl.float32)
        b = tl.load(B + row[:, None] * BS + head[None, :], mask, 0).to(tl.float32)
        x = a + tl.load(BIAS + head, head < HEADS, 0)[None, :].to(tl.float32)
        softplus = tl.where(x <= 20, tl.log(1 + tl.exp(x)), x)
        g = (
            -tl.exp(tl.load(ALOG + head, head < HEADS, 0)[None, :].to(tl.float32))
            * softplus
        )
        tl.store(G + row[:, None] * HEADS + head[None, :], g, mask)
        tl.store(BETA + row[:, None] * HEADS + head[None, :], tl.sigmoid(b), mask)


def prepare(impl, mixed_qkv, a, b, cache, metadata, workspace):
    state = cache.k_cache
    if state.shape[1] != impl.conv1d.weight.shape[0]:
        state = state.transpose(-1, -2)
    # Native convolution casts the input to the history dtype before arithmetic.
    x = mixed_qkv.to(state.dtype)
    q, k, v, g, beta = workspace
    slots = metadata.spec_state_indices_tensor[:, 0]
    _prepare[
        (metadata.num_spec_decodes, triton.cdiv(impl.key_dim * 2 + impl.value_dim, 256))
    ](
        x,
        impl.conv1d.weight,
        state,
        slots,
        metadata.num_accepted_tokens,
        metadata.spec_query_start_loc,
        q,
        k,
        v,
        a,
        b,
        impl.A_log,
        impl.dt_bias,
        g,
        beta,
        metadata.num_spec_decodes,
        impl.key_dim,
        impl.value_dim,
        impl.num_v_heads,
        state.shape[0],
        x.stride(0),
        a.stride(0),
        b.stride(0),
        *state.stride(),
        slots.stride(0),
        num_warps=4,
    )
    return q, k, v, g, beta
