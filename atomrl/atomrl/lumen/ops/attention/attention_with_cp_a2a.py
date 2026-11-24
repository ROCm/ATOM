# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Context-parallelism (all-to-all) attention variants.

Refer to the paper `DeepSpeed Ulysses <https://arxiv.org/abs/2309.14509>`_
for the underlying communication pattern.

When ``use_sdma=True`` the all-to-all exchanges are routed through
mori's SDMA engine (via :class:`~atomrl.lumen.ops.sdma.SdmaAll2all`),
freeing the compute units for overlapping work.
"""

import threading
from functools import lru_cache
from typing import Optional

import torch

from atomrl.lumen.core.grad_quant import quantize_grad_tensor
from atomrl.lumen.ops.sdma import is_sdma_available

if is_sdma_available():
    from atomrl.lumen.ops.sdma import SdmaAll2all


from aiter.ops.mha import flash_attn_func

from atomrl.lumen.kernels.attention.attention_impl import (
    attention_backward as triton_fp8_backward,
)
from atomrl.lumen.kernels.attention.attention_impl import (
    attention_forward as triton_fp8_forward,
)
from atomrl.lumen.kernels.attention.attention_impl import (
    attention_mxfp8_backward as triton_mxfp8_backward,
)
from atomrl.lumen.kernels.attention.attention_impl import (
    attention_mxfp8_forward as triton_mxfp8_forward,
)

# ---------------------------------------------------------------------------
# A2A helper — reshapes tensors between "local tokens / all heads" and
# "all tokens / local heads" representations.
# ---------------------------------------------------------------------------


@lru_cache
def get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
    return AttentionCPA2AHelper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)


class AttentionCPA2AHelper:
    """Transpose tensors between local-token and local-head layouts for CP A2A."""

    def __init__(self, b, s, h_q, h_kv, d_qk, d_v, seq_dim, n):
        assert seq_dim == 1, "only_support bshd yet"
        self.seq_dim = seq_dim
        self.qkv_shape_traits = (
            (n, b, s, h_q, d_qk),
            (n, b, s, h_kv, d_qk),
            (n, b, s, h_kv, d_v),
        )
        self.o_shape_traits = (n, b, s, h_q, d_v)
        self.combine_splits = (
            b * s * h_q * d_qk // n // n,
            b * s * h_kv * d_qk // n // n,
            b * s * h_kv * d_v // n // n,
        )

    def combine_qkv_before_a2a(self, q, k, v):
        q, k, v = (
            x.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )
        total = q.shape[1] + k.shape[1] + v.shape[1]
        out = torch.empty((q.shape[0], total), dtype=q.dtype, device=q.device)
        c0 = q.shape[1]
        c1 = c0 + k.shape[1]
        out[:, :c0] = q
        out[:, c0:c1] = k
        out[:, c1:] = v
        return out

    def splits_qkv_after_a2a(self, qkv):
        q, k, v = torch.split(qkv, self.combine_splits, dim=1)
        q, k, v = (
            x.view(n, b, s // n, h // n, d)
            .movedim(0, 1)
            .contiguous()
            .view(b, s, h // n, d)
            for x, (n, b, s, h, d) in zip((q, k, v), self.qkv_shape_traits)
        )
        return q, k, v

    def reshape_o_before_a2a(self, o):
        n, b, s, h, d = self.o_shape_traits
        return o.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous()

    def reshape_o_after_a2a(self, o):
        n, b, s, h, d = self.o_shape_traits
        return o.movedim(0, -3).contiguous().view(b, s // n, h, d)

    def reshape_do_before_a2a(self, d_o):
        n, b, s, h, d = self.o_shape_traits
        return d_o.view(b, s // n, n, h // n, d).movedim(-3, 0).contiguous()

    def reshape_do_after_a2a(self, d_o):
        n, b, s, h, d = self.o_shape_traits
        return d_o.movedim(0, 1).contiguous().view(b, s, h // n, d)

    def combine_dqkv_before_a2a(self, dq, dk, dv):
        dq, dk, dv = (
            x.view(b, n, s // n, h // n, d).movedim(1, 0).contiguous().view(n, -1)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        total = dq.shape[1] + dk.shape[1] + dv.shape[1]
        out = torch.empty((dq.shape[0], total), dtype=dq.dtype, device=dq.device)
        c0 = dq.shape[1]
        c1 = c0 + dk.shape[1]
        out[:, :c0] = dq
        out[:, c0:c1] = dk
        out[:, c1:] = dv
        return out

    def split_dqkv_after_a2a(self, dqkv):
        dq, dk, dv = torch.split(dqkv, self.combine_splits, dim=1)
        dq, dk, dv = (
            x.view(n, b, s // n, h // n, d)
            .movedim(0, -3)
            .contiguous()
            .view(b, s // n, h, d)
            for x, (n, b, s, h, d) in zip((dq, dk, dv), self.qkv_shape_traits)
        )
        return dq, dk, dv


# ---------------------------------------------------------------------------
# Differentiable all-to-all wrapper
# ---------------------------------------------------------------------------


class _A2ASingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        out = torch.empty_like(x)
        torch.distributed.all_to_all_single(out, x, group=group, async_op=False)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        torch.distributed.all_to_all_single(
            grad_input, grad_output, group=ctx.group, async_op=False
        )
        return grad_input, None


# ---------------------------------------------------------------------------
# Cached SDMA A2A handle (process-level, keyed by capacity)
# ---------------------------------------------------------------------------

_sdma_a2a_handle: Optional["SdmaAll2all"] = None
_sdma_a2a_lock = threading.Lock()


def _get_sdma_a2a() -> "SdmaAll2all":
    """Return or create the process-level SdmaAll2all handle."""
    global _sdma_a2a_handle
    if _sdma_a2a_handle is None:
        with _sdma_a2a_lock:
            if _sdma_a2a_handle is None:
                _sdma_a2a_handle = SdmaAll2all()
    return _sdma_a2a_handle


def _sdma_a2a_exchange(src: torch.Tensor, dst: torch.Tensor) -> None:
    """All-to-all *src* into *dst* using mori SDMA."""
    handle = _get_sdma_a2a()
    handle(src, dst)


def reset_sdma_a2a_cache() -> None:
    """Release the process-level SdmaAll2all handle.

    Must be called before ``SdmaContext.reset()`` / ``shmem_finalize()`` so
    that the C++ SDMA handle destructor runs while shmem is still alive.
    """
    global _sdma_a2a_handle
    with _sdma_a2a_lock:
        _sdma_a2a_handle = None


# ---------------------------------------------------------------------------
# Shared A2A communication wrappers
# ---------------------------------------------------------------------------


def _a2a_pre_forward(q, k, v, cp_group, use_sdma=False):
    """A2A scatter: local tokens -> local heads."""
    n = cp_group.size()
    b, s_local, h_q, d_qk = q.shape
    _, _, h_kv, d_v = v.shape
    s = s_local * n
    assert h_q % n == 0 and h_kv % n == 0
    seq_dim = 1
    attn_helper = get_attention_cp_a2a_helper(b, s, h_q, h_kv, d_qk, d_v, seq_dim, n)

    qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
    qkv_out = torch.empty_like(qkv)
    if use_sdma:
        _sdma_a2a_exchange(qkv, qkv_out)
    else:
        torch.distributed.all_to_all_single(
            qkv_out, qkv, group=cp_group, async_op=False
        )
    q_lh, k_lh, v_lh = attn_helper.splits_qkv_after_a2a(qkv_out)
    return q_lh, k_lh, v_lh, attn_helper, seq_dim


def _a2a_post_forward(output_local_heads, attn_helper, cp_group, use_sdma=False):
    """A2A gather: local heads -> local tokens."""
    output_local_heads = attn_helper.reshape_o_before_a2a(output_local_heads)
    output_local_tokens = torch.empty_like(output_local_heads)
    if use_sdma:
        _sdma_a2a_exchange(output_local_heads, output_local_tokens)
    else:
        torch.distributed.all_to_all_single(
            output_local_tokens,
            output_local_heads,
            group=cp_group,
            async_op=False,
        )
    return attn_helper.reshape_o_after_a2a(output_local_tokens)


def _a2a_pre_backward(dout, attn_helper, cp_group, use_sdma=False):
    """A2A scatter grad: local tokens -> local heads."""
    dout = attn_helper.reshape_do_before_a2a(dout)
    dout_local_heads = torch.empty_like(dout)
    if use_sdma:
        _sdma_a2a_exchange(dout, dout_local_heads)
    else:
        torch.distributed.all_to_all_single(dout_local_heads, dout, group=cp_group)
    return attn_helper.reshape_do_after_a2a(dout_local_heads)


def _a2a_post_backward(dq, dk, dv, attn_helper, cp_group, use_sdma=False):
    """A2A gather grad: local heads -> local tokens."""
    dqkv = attn_helper.combine_dqkv_before_a2a(dq, dk, dv)
    dqkv_out = torch.empty_like(dqkv)
    if use_sdma:
        _sdma_a2a_exchange(dqkv, dqkv_out)
    else:
        torch.distributed.all_to_all_single(dqkv_out, dqkv, group=cp_group)
    return attn_helper.split_dqkv_after_a2a(dqkv_out)


# ---------------------------------------------------------------------------
# CP Autograd Functions
# ---------------------------------------------------------------------------


class AttentionTritonFunctionCPA2A(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad,
        use_fp8,
        cp_group,
        grad_quant_type=None,
        use_sdma=False,
    ):
        assert bias is None
        q_lh, k_lh, v_lh, attn_helper, _seq_dim = _a2a_pre_forward(
            q,
            k,
            v,
            cp_group,
            use_sdma=use_sdma,
        )

        (
            output,
            softmax_lse,
            exp_scores,
            q_lh,
            k_lh,
            v_lh,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            rng_state,
        ) = triton_fp8_forward(
            q_lh,
            k_lh,
            v_lh,
            use_fp8,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
        )

        if is_grad:
            ctx.save_for_backward(
                q_lh,
                k_lh,
                v_lh,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = use_fp8
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q_lh.shape[1]
            ctx.max_seqlens_k = k_lh.shape[1]
            ctx.attn_helper = attn_helper
            ctx.cp_group = cp_group
            ctx.grad_quant_type = grad_quant_type
            ctx.use_sdma = use_sdma

        output_tokens = _a2a_post_forward(
            output,
            attn_helper,
            cp_group,
            use_sdma=use_sdma,
        )

        result = [output_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            o,
            softmax_lse,
            alibi_slopes,
            bias,
            q_scale,
            k_scale,
            v_scale,
            rng_state,
        ) = ctx.saved_tensors
        assert bias is None

        dout_lh = _a2a_pre_backward(
            dout,
            ctx.attn_helper,
            ctx.cp_group,
            use_sdma=ctx.use_sdma,
        )
        dq, dk, dv = triton_fp8_backward(
            dout_lh,
            q,
            k,
            v,
            o,
            q_scale,
            k_scale,
            v_scale,
            ctx.p_scale,
            softmax_lse,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.sm_scale,
            ctx.causal,
            alibi_slopes,
            ctx.use_fp8,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
        )
        dq_t, dk_t, dv_t = _a2a_post_backward(
            dq,
            dk,
            dv,
            ctx.attn_helper,
            ctx.cp_group,
            use_sdma=ctx.use_sdma,
        )
        gqt = ctx.grad_quant_type
        dq_t = quantize_grad_tensor(dq_t, gqt)
        dk_t = quantize_grad_tensor(dk_t, gqt)
        dv_t = quantize_grad_tensor(dv_t, gqt)
        return (
            dq_t,
            dk_t,
            dv_t,
        ) + (None,) * 13


class AttentionTritonMXFP8FunctionCPA2A(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        return_lse,
        return_softmax,
        is_grad,
        use_mxfp8,
        cp_group,
        block_m_fwd=64,
        block_n_fwd=64,
        block_m_dq_bwd=64,
        block_n_dq_bwd=64,
        block_m_dkv_bwd=64,
        block_n_dkv_bwd=64,
        quant_block_size=32,
        grad_quant_type=None,
        use_sdma=False,
    ):
        assert bias is None
        q_lh, k_lh, v_lh, attn_helper, _seq_dim = _a2a_pre_forward(
            q,
            k,
            v,
            cp_group,
            use_sdma=use_sdma,
        )

        (
            output,
            softmax_lse,
            exp_scores,
            q_lh,
            k_lh,
            v_lh,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            rng_state,
        ) = triton_mxfp8_forward(
            q_lh,
            k_lh,
            v_lh,
            use_mxfp8,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
            block_m_fwd,
            block_n_fwd,
            quant_block_size,
        )

        if is_grad:
            ctx.save_for_backward(
                q_lh,
                k_lh,
                v_lh,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale,
                k_scale,
                v_scale,
                rng_state,
            )
            ctx.use_mxfp8 = use_mxfp8
            ctx.p_scale = p_scale
            ctx.sm_scale = softmax_scale
            ctx.causal = causal
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.block_m_dq_bwd = block_m_dq_bwd
            ctx.block_n_dq_bwd = block_n_dq_bwd
            ctx.block_m_dkv_bwd = block_m_dkv_bwd
            ctx.block_n_dkv_bwd = block_n_dkv_bwd
            ctx.quant_block_size = quant_block_size
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q_lh.shape[1]
            ctx.max_seqlens_k = k_lh.shape[1]
            ctx.attn_helper = attn_helper
            ctx.cp_group = cp_group
            ctx.grad_quant_type = grad_quant_type
            ctx.use_sdma = use_sdma

        output_tokens = _a2a_post_forward(
            output,
            attn_helper,
            cp_group,
            use_sdma=use_sdma,
        )

        result = [output_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            o,
            softmax_lse,
            alibi_slopes,
            bias,
            q_scale,
            k_scale,
            v_scale,
            rng_state,
        ) = ctx.saved_tensors
        assert bias is None

        dout_lh = _a2a_pre_backward(
            dout,
            ctx.attn_helper,
            ctx.cp_group,
            use_sdma=ctx.use_sdma,
        )
        dq, dk, dv = triton_mxfp8_backward(
            dout_lh,
            q,
            k,
            v,
            o,
            softmax_lse,
            q_scale,
            k_scale,
            v_scale,
            ctx.sm_scale,
            ctx.p_scale,
            alibi_slopes,
            ctx.causal,
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            ctx.max_seqlens_q,
            ctx.max_seqlens_k,
            ctx.use_mxfp8,
            ctx.block_m_dq_bwd,
            ctx.block_n_dq_bwd,
            ctx.block_m_dkv_bwd,
            ctx.block_n_dkv_bwd,
            ctx.quant_block_size,
            rng_state=rng_state,
            dropout_p=ctx.dropout_p,
            bias=bias,
            window_size=ctx.window_size,
        )
        dq_t, dk_t, dv_t = _a2a_post_backward(
            dq,
            dk,
            dv,
            ctx.attn_helper,
            ctx.cp_group,
            use_sdma=ctx.use_sdma,
        )
        gqt = ctx.grad_quant_type
        dq_t = quantize_grad_tensor(dq_t, gqt)
        dk_t = quantize_grad_tensor(dk_t, gqt)
        dv_t = quantize_grad_tensor(dv_t, gqt)
        return (
            dq_t,
            dk_t,
            dv_t,
        ) + (None,) * 20


# ---------------------------------------------------------------------------
# Aiter CP variant — uses differentiable _A2ASingle so autograd flows
# through flash_attn_func naturally (no custom backward needed).
# ---------------------------------------------------------------------------


class _SdmaA2ASingle(torch.autograd.Function):
    """Differentiable all-to-all via mori SDMA (mirrors _A2ASingle)."""

    @staticmethod
    def forward(ctx, x, cp_group):
        ctx.cp_group = cp_group
        out = torch.empty_like(x)
        _sdma_a2a_exchange(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        _sdma_a2a_exchange(grad_output, grad_input)
        return grad_input, None


class AttentionAiterFunctionCPA2A:
    @staticmethod
    def apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        cp_group,
        use_sdma=False,
    ):
        assert bias is None
        n = cp_group.size()
        b, s_local, h_q, d_qk = q.shape
        _, _, h_kv, d_v = v.shape
        s = s_local * n
        assert h_q % n == 0 and h_kv % n == 0
        seq_dim = 1
        attn_helper = get_attention_cp_a2a_helper(
            b, s, h_q, h_kv, d_qk, d_v, seq_dim, n
        )

        qkv = attn_helper.combine_qkv_before_a2a(q, k, v)
        a2a_fn = _SdmaA2ASingle.apply if use_sdma else _A2ASingle.apply
        qkv_out = a2a_fn(qkv, cp_group)
        q_lh, k_lh, v_lh = attn_helper.splits_qkv_after_a2a(qkv_out)

        _return_softmax = return_softmax and dropout_p > 0
        attn_result = flash_attn_func(
            q_lh,
            k_lh,
            v_lh,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=True,
            return_attn_probs=_return_softmax,
        )

        if _return_softmax:
            output_lh, softmax_lse, S_dmask = attn_result
        else:
            output_lh, softmax_lse = attn_result
            S_dmask = None

        output_lh = attn_helper.reshape_o_before_a2a(output_lh)
        output_tokens = a2a_fn(output_lh, cp_group)
        output_tokens = attn_helper.reshape_o_after_a2a(output_tokens)

        result = [output_tokens]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)
        return result[0] if len(result) == 1 else tuple(result)
