# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

import logging
import os as _os

import torch

logger = logging.getLogger(__name__)

from aiter.ops.mha import flash_attn_func

from atomrl.lumen.core.grad_quant import quantize_grad_tensor
from atomrl.lumen.kernels.attention.attention_impl import (
    FIXED_BLOCK_M,
    attention_triton_forward_impl,
)
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
from atomrl.lumen.ops.attention.attention_with_cp_p2p import attention_cp_p2p

__all__ = ["attention", "attention_fp8_quant"]

_FP8_BACKENDS = ("aiter_triton_fp8", "aiter_csrc_fp8", "aiter_asm_fp8")

_QUANT_TYPE_ALIAS = {
    "fp8_blockwise": "blockwise",
}


# ---------------------------------------------------------------------------
# torch.compile support
# ---------------------------------------------------------------------------
def _mark_allow_in_graph(*classes):
    try:
        from torch._dynamo import allow_in_graph

        for cls in classes:
            allow_in_graph(cls)
    except RuntimeError:
        pass


def _extract_triton_lse(lse_raw, seqlen_q):
    """Extract contiguous LSE from the Triton kernel's interleaved buffer.

    The kernel allocates (B, H, SQ*2) and writes each block's LSE followed
    by BLOCK_M slots reserved for backward delta:
        [LSE_blk0(BM) | delta_blk0(BM) | LSE_blk1(BM) | delta_blk1(BM) | ...]

    Requires seqlen_q to be a multiple of FIXED_BLOCK_M (kernel constraint).
    """
    B, H, _total = lse_raw.shape
    num_blocks = (seqlen_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M
    reshaped = lse_raw.reshape(B, H, num_blocks, 2 * FIXED_BLOCK_M)
    lse_clean = (
        reshaped[:, :, :, :FIXED_BLOCK_M]
        .contiguous()
        .reshape(B, H, num_blocks * FIXED_BLOCK_M)
    )
    return lse_clean[:, :, :seqlen_q]


# ---------------------------------------------------------------------------
# Autograd Functions
# ---------------------------------------------------------------------------


class AttentionTritonFunction(torch.autograd.Function):
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
        is_grad_enabled,
        use_fp8,
        grad_quant_type=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (
            output,
            softmax_lse,
            exp_scores,
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            rng_state,
        ) = triton_fp8_forward(
            q,
            k,
            v,
            use_fp8,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            bias,
            alibi_slopes,
            return_softmax,
            force_triton=True,
        )

        if is_grad:
            ctx.save_for_backward(
                q,
                k,
                v,
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
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(_extract_triton_lse(softmax_lse, q.shape[1]))
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
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

        dq, dk, dv = triton_fp8_backward(
            do,
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
            force_triton=True,
        )
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class AttentionTritonMXFP8Function(torch.autograd.Function):
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
        is_grad_enabled,
        use_mxfp8,
        block_m_fwd=64,
        block_n_fwd=64,
        block_m_dq_bwd=64,
        block_n_dq_bwd=64,
        block_m_dkv_bwd=64,
        block_n_dkv_bwd=64,
        quant_block_size=32,
        grad_quant_type=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        (
            output,
            softmax_lse,
            exp_scores,
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            rng_state,
        ) = triton_mxfp8_forward(
            q,
            k,
            v,
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
                q,
                k,
                v,
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
            ctx.layout = "bshd"
            ctx.block_m_dq_bwd = block_m_dq_bwd
            ctx.block_n_dq_bwd = block_n_dq_bwd
            ctx.block_m_dkv_bwd = block_m_dkv_bwd
            ctx.block_n_dkv_bwd = block_n_dkv_bwd
            ctx.quant_block_size = quant_block_size
            ctx.cu_seqlens_q = torch.tensor(0, device="cuda")
            ctx.cu_seqlens_k = torch.tensor(0, device="cuda")
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type

        result = [output]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
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

        dq, dk, dv = triton_mxfp8_backward(
            do,
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
        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return (
            dq,
            dk,
            dv,
        ) + (None,) * 18


class AttentionTritonBlockwise2DFunction(torch.autograd.Function):
    """Autograd function for blockwise2d FP8 attention.

    Uses 2D block quantization to cache Q/K/V with scale shape
    ``[B, H, S//bm, D//bn]`` via a :class:`Blockwise2DScaleManager`.
    The actual kernel still uses 1D block scales (``quantize_block_fp8``
    for Q/K, per-tensor for V) since the underlying Triton kernel expects
    that layout.  The 2D cache enables backward to fully reuse forward's
    quantized tensors and to persist the dO scale across iterations.
    """

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
        is_grad_enabled,
        grad_quant_type=None,
        scale_manager=None,
    ):
        from atomrl.lumen.quantize.scaling_manager import (
            Blockwise2DScaleManager,
            ScalingManager,
        )

        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        fp8_dt = None
        try:
            from atomrl.lumen.core.float8 import float8_e4m3

            fp8_dt = float8_e4m3
        except ImportError:
            from atomrl.lumen.quantize.config import _get_float8_e4m3

            fp8_dt = _get_float8_e4m3()

        if scale_manager is not None and isinstance(
            scale_manager, Blockwise2DScaleManager
        ):
            scale_manager.quantize_and_cache(q, k, v, fp8_dt)

        q_fp8, q_scale_1d = ScalingManager.quantize_block_fp8(q, FIXED_BLOCK_M, fp8_dt)
        k_fp8, k_scale_1d = ScalingManager.quantize_block_fp8(k, FIXED_BLOCK_M, fp8_dt)
        v_fp8, v_scale_1d, p_scale = ScalingManager.quantize_v_fp8(v, fp8_dt)

        output, softmax_lse, exp_scores = attention_triton_forward_impl(
            q_fp8,
            k_fp8,
            v_fp8,
            p_scale,
            q_scale_1d,
            k_scale_1d,
            v_scale_1d,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0],
            window_size[1],
            bias,
            alibi_slopes,
            return_softmax,
            True,
        )
        rng_state = None

        if is_grad:
            ctx.save_for_backward(
                q_fp8,
                k_fp8,
                v_fp8,
                output,
                softmax_lse,
                alibi_slopes,
                bias,
                q_scale_1d,
                k_scale_1d,
                v_scale_1d,
                rng_state,
            )
            ctx.sm_scale = softmax_scale
            ctx.p_scale = p_scale
            ctx.causal = causal
            ctx.use_fp8 = True
            ctx.dropout_p = dropout_p
            ctx.window_size = window_size
            ctx.cu_seqlens_q = 0
            ctx.cu_seqlens_k = 0
            ctx.max_seqlens_q = q.shape[1]
            ctx.max_seqlens_k = k.shape[1]
            ctx.grad_quant_type = grad_quant_type
            ctx.scale_manager = scale_manager

        result = [output]
        if return_lse:
            result.append(_extract_triton_lse(softmax_lse, q.shape[1]))
        if return_softmax:
            result.append(exp_scores)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
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

        dq, dk, dv = triton_fp8_backward(
            do,
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
            force_triton=True,
        )

        scale_mgr = ctx.scale_manager
        if scale_mgr is not None:
            from atomrl.lumen.quantize.scaling_manager import Blockwise2DScaleManager

            if isinstance(scale_mgr, Blockwise2DScaleManager):
                do_abs_max = do.abs().amax()
                fp8_max = torch.finfo(q.dtype).max if q.is_floating_point() else 448.0
                do_scale = fp8_max / torch.where(
                    do_abs_max == 0,
                    torch.tensor(fp8_max, device=do.device),
                    do_abs_max,
                )
                scale_mgr.cache_do_scale(do_scale)

        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_LUMEN_FP8_ATTN_BWD = _os.environ.get("LUMEN_FP8_ATTN_BWD", "0") == "1"


class AttentionCsrcFP8BwdFunction(torch.autograd.Function):
    """CK csrc forward + Triton FP8 backward.

    Uses AITER's highly optimized CK forward kernel (BF16), but routes the
    backward through Lumen's Triton FP8 backward kernels which quantize dO,
    Q, K to FP8 blockwise for reduced memory bandwidth.

    Controlled by ``LUMEN_FP8_ATTN_BWD=1``.
    """

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
        deterministic,
        grad_quant_type,
    ):
        result = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=True,
        )
        if isinstance(result, tuple):
            out, lse = result[0], result[1]
        else:
            out = result
            lse = None

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.grad_quant_type = grad_quant_type
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors

        from atomrl.lumen.kernels.attention.attention_impl import (
            attention_backward as triton_fp8_bwd,
        )
        from atomrl.lumen.kernels.attention.attention_impl import (
            get_f8_fwd_dtype,
        )
        from atomrl.lumen.quantize.scaling_manager import ScalingManager

        B, S_Q, H_Q, _D = q.shape
        _, S_K, _H_K, _ = k.shape

        fp8_dt = get_f8_fwd_dtype()
        q_fp8, q_scale = ScalingManager.quantize_block_fp8(q, FIXED_BLOCK_M, fp8_dt)
        k_fp8, k_scale = ScalingManager.quantize_block_fp8(k, FIXED_BLOCK_M, fp8_dt)
        v_fp8, v_scale, p_scale = ScalingManager.quantize_v_fp8(v, fp8_dt)

        if lse is not None and lse.dim() == 3:
            seqlen_q = q.shape[1]
            num_blocks = (seqlen_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M
            lse_interleaved = torch.zeros(
                B,
                H_Q,
                num_blocks * 2 * FIXED_BLOCK_M,
                dtype=torch.float32,
                device=q.device,
            )
            for blk in range(num_blocks):
                start = blk * FIXED_BLOCK_M
                end = min(start + FIXED_BLOCK_M, seqlen_q)
                lse_interleaved[
                    :,
                    :,
                    blk * 2 * FIXED_BLOCK_M : blk * 2 * FIXED_BLOCK_M + (end - start),
                ] = lse[:, :, start:end]
            lse_for_bwd = lse_interleaved
        else:
            lse_for_bwd = lse

        dq, dk, dv = triton_fp8_bwd(
            do,
            q_fp8,
            k_fp8,
            v_fp8,
            o,
            q_scale,
            k_scale,
            v_scale,
            p_scale,
            lse_for_bwd,
            0,
            0,
            S_Q,
            S_K,
            ctx.softmax_scale,
            ctx.causal,
            alibi_slopes=None,
            use_fp8=True,
            dropout_p=ctx.dropout_p,
            force_triton=True,
        )

        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)

        return dq, dk, dv, None, None, None, None, None, None, None, None


class AttentionCsrcSeqMajorOutFunction(torch.autograd.Function):
    """CK csrc attention whose output lands in a seq-major ``[s, b, h, d]`` buffer.

    The CK forward takes its output buffer as an argument and reads the batch,
    sequence and head strides from it, requiring only ``stride(-1) == 1``.
    Handing it a transposed view of a ``[s, b, h, d]`` allocation therefore costs
    the kernel nothing, and saves a caller whose consumer is seq-major (Megatron's
    residual stream) a full read+write pass over the output.

    ``flash_attn_func`` does not expose that buffer, so this reimplements the
    autograd glue around the same forward and backward entry points.  It is
    restricted to the plain dense case -- no bias, alibi, dropout or padding --
    which is all the layout shortcut is worth; everything else keeps using
    ``flash_attn_func``.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        deterministic,
        grad_quant_type,
    ):
        from aiter.ops.mha import _flash_attn_forward

        batch, seqlen_q, nhead_q, _ = q.shape
        hdim_v = v.shape[3]
        out = torch.empty(
            (seqlen_q, batch, nhead_q, hdim_v), dtype=q.dtype, device=q.device
        ).permute(1, 0, 2, 3)

        out, softmax_lse, _, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            0.0,
            softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=True,
            return_softmax=False,
            out=out,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.grad_quant_type = grad_quant_type
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        from aiter.ops.mha import _flash_attn_backward

        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            None,
            0.0,
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            None,
            None,
            ctx.deterministic,
            rng_state,
        )

        gqt = ctx.grad_quant_type
        dq = quantize_grad_tensor(dq, gqt)
        dk = quantize_grad_tensor(dk, gqt)
        dv = quantize_grad_tensor(dv, gqt)
        return dq, dk, dv, None, None, None, None, None


_mark_allow_in_graph(
    AttentionTritonFunction,
    AttentionTritonMXFP8Function,
    AttentionTritonBlockwise2DFunction,
    AttentionCsrcFP8BwdFunction,
    AttentionCsrcSeqMajorOutFunction,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attention(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "auto",
    cp_param_bundle=None,
    grad_quant_type: str | None = None,
    seq_major_out: bool = False,
):
    """Dense attention over ``[b, s, h, d]`` operands.

    ``seq_major_out`` asks for the output to be allocated so that
    ``out.permute(1, 0, 2, 3)`` is contiguous.  Callers feeding a seq-major
    consumer can then transpose the result for free instead of copying it.  It
    is a hint: backends that cannot honour it return the usual layout.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Normalize alibi_slopes to (B, H) so Triton pointer arithmetic
    # (off_z * stride_az + off_h * stride_ah) never reads out of bounds.
    # AITER kernels expect (batch, nheads); (1, H) with stride (H, 1)
    # would OOB for off_z > 0.  expand() sets stride(0)=0, no copy.
    if alibi_slopes is not None:
        batch = q.shape[0]
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        elif alibi_slopes.shape[0] == 1 and batch > 1:
            alibi_slopes = alibi_slopes.expand(batch, -1)

    # AITER is a required runtime dependency of the release image.
    if backend_type == "auto":
        backend_type = "aiter_csrc"

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionAiterFunctionCPA2A,
            AttentionTritonFunctionCPA2A,
        )
        from atomrl.lumen.ops.sdma import is_sdma_available

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        cp_use_sdma = cp_param_bundle.get("use_sdma", False) and is_sdma_available()
        if backend_type == "aiter_csrc" and cp_comm_type == "a2a":
            return AttentionAiterFunctionCPA2A.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                cp_group,
                cp_use_sdma,
            )
        elif backend_type == "aiter_triton" and cp_comm_type == "a2a":
            return AttentionTritonFunctionCPA2A.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                cp_group,
                grad_quant_type,
                cp_use_sdma,
            )
        elif cp_comm_type == "p2p":
            cp_size = cp_group.size()
            cp_rank = cp_group.rank()

            def _make_attn_fn():
                _dropout_p = dropout_p
                _window_size = window_size
                _bias = bias
                _alibi_slopes = alibi_slopes
                _backend_type = backend_type
                _grad_quant_type = grad_quant_type
                _deterministic = deterministic

                def attn_fn(q_chunk, k_chunk, v_chunk, causal, softmax_scale):
                    # Backends return LSE as (B, H, S); _online_softmax_update
                    # expects (B, S, H) to broadcast with output (B, S, H, D).
                    if _backend_type == "aiter_csrc":
                        out, lse = flash_attn_func(
                            q_chunk,
                            k_chunk,
                            v_chunk,
                            dropout_p=_dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal,
                            window_size=_window_size,
                            bias=_bias,
                            alibi_slopes=_alibi_slopes,
                            deterministic=_deterministic,
                            return_lse=True,
                            return_attn_probs=False,
                        )
                        return out, lse.transpose(1, 2).contiguous()
                    else:
                        result = AttentionTritonFunction.apply(
                            q_chunk,
                            k_chunk,
                            v_chunk,
                            _dropout_p,
                            softmax_scale,
                            causal,
                            _window_size,
                            _bias,
                            _alibi_slopes,
                            True,
                            False,
                            torch.is_grad_enabled(),
                            False,
                            _grad_quant_type,
                        )
                        return result[0], result[1].transpose(1, 2).contiguous()

                return attn_fn

            attn_fn = _make_attn_fn()
            out = attention_cp_p2p(
                q,
                k,
                v,
                cp_group,
                cp_size,
                cp_rank,
                attn_fn,
                causal,
                softmax_scale,
            )
            if return_lse:
                raise NotImplementedError(
                    "return_lse not supported for cp_comm_type='p2p' yet"
                )
            return out
        else:
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )

    # Single-GPU path: prefer CK csrc, fall back to Triton.
    if backend_type == "aiter_csrc":
        _return_softmax = return_attn_probs and dropout_p > 0
        _needs_grad = torch.is_grad_enabled() and any(
            t.requires_grad for t in [q, k, v]
        )
        _internal_return_lse = return_lse or _needs_grad

        # CK csrc requires bias as 2D [sq, sk] — one bias broadcast across
        # all batches and heads.  Squeeze (1,1,sq,sk) silently; route
        # per-head / per-batch bias directly to Triton (csrc can't handle it).
        _csrc_bias = bias
        _csrc_bias_unsupported = False
        if _csrc_bias is not None and _csrc_bias.dim() == 4:
            if _csrc_bias.shape[0] == 1 and _csrc_bias.shape[1] == 1:
                _csrc_bias = _csrc_bias.squeeze(0).squeeze(0)
            else:
                _csrc_bias_unsupported = True
        if _csrc_bias is not None and _csrc_bias.dim() == 3:
            _csrc_bias_unsupported = True

        if _csrc_bias_unsupported:
            return AttentionTritonFunction.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                grad_quant_type,
            )

        if (
            _LUMEN_FP8_ATTN_BWD
            and _needs_grad
            and _csrc_bias is None
            and alibi_slopes is None
        ):
            return AttentionCsrcFP8BwdFunction.apply(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                _csrc_bias,
                alibi_slopes,
                deterministic,
                grad_quant_type,
            )

        # Seq-major output is only a layout choice for the buffer CK writes, so
        # it is available whenever nothing else needs the generic wrapper: the
        # head dims must not need padding, and bias / alibi / dropout / probs
        # all route through code paths flash_attn_func owns.
        _seq_major_eligible = (
            seq_major_out
            and dropout_p == 0.0
            and _csrc_bias is None
            and alibi_slopes is None
            and not return_lse
            and not _return_softmax
            and q.shape[3] % 8 == 0
            and v.shape[3] % 8 == 0
        )
        if _seq_major_eligible:
            return AttentionCsrcSeqMajorOutFunction.apply(
                q,
                k,
                v,
                softmax_scale,
                causal,
                window_size,
                deterministic,
                grad_quant_type,
            )

        try:
            result = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                bias=_csrc_bias,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_lse=_internal_return_lse,
                return_attn_probs=_return_softmax,
            )
            if _internal_return_lse and not return_lse and isinstance(result, tuple):
                if len(result) == 2:
                    result = result[0]
                elif len(result) == 3:
                    # (out, lse, S_dmask) -> (out, S_dmask)
                    result = (result[0], result[2])
            return result
        except RuntimeError as _aiter_err:
            logger.warning(
                "attention: aiter csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _aiter_err,
            )
            return AttentionTritonFunction.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                False,
                grad_quant_type,
            )
    elif backend_type == "aiter_triton":
        return AttentionTritonFunction.apply(
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
            return_attn_probs,
            torch.is_grad_enabled(),
            False,
            grad_quant_type,
        )
    else:
        raise NotImplementedError(f"backend_type {backend_type} not supported")


def attention_fp8_quant(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    backend_type: str = "aiter_triton",
    cp_param_bundle=None,
    quant_type: str = "blockwise",
    block_m_fwd: int = 64,
    block_n_fwd: int = 64,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
    grad_quant_type: str | None = None,
    fp8_mha: bool = False,
    scale_manager=None,
):
    assert backend_type in (
        "aiter_triton",
        "aiter_triton_fp8",
        "aiter_csrc_fp8",
        "aiter_asm_fp8",
    ), (
        f"attention_fp8_quant only supports aiter_triton/aiter_triton_fp8/"
        f"aiter_csrc_fp8/aiter_asm_fp8 backends, got {backend_type}"
    )

    # Normalise legacy quant_type names
    quant_type = _QUANT_TYPE_ALIAS.get(quant_type, quant_type)

    # "none" means no FP8 quantisation — delegate to bf16 attention
    if quant_type == "none":
        _effective_backend = "aiter_csrc"
        return attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            backend_type=_effective_backend,
            cp_param_bundle=cp_param_bundle,
            grad_quant_type=grad_quant_type,
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if alibi_slopes is not None:
        batch = q.shape[0]
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        elif alibi_slopes.shape[0] == 1 and batch > 1:
            alibi_slopes = alibi_slopes.expand(batch, -1)

    # Context-parallelism path
    if cp_param_bundle is not None:
        assert "cp_group" in cp_param_bundle
        assert "cp_comm_type" in cp_param_bundle
        from atomrl.lumen.ops.attention.attention_with_cp_a2a import (
            AttentionTritonFunctionCPA2A,
            AttentionTritonMXFP8FunctionCPA2A,
        )
        from atomrl.lumen.ops.sdma import is_sdma_available

        cp_group = cp_param_bundle["cp_group"]
        cp_comm_type = cp_param_bundle["cp_comm_type"]
        cp_use_sdma = cp_param_bundle.get("use_sdma", False) and is_sdma_available()
        if cp_comm_type != "a2a":
            raise NotImplementedError(
                f"not supported backend_type {backend_type} cp_comm_type {cp_comm_type} yet"
            )
        if quant_type == "mxfp8":
            return AttentionTritonMXFP8FunctionCPA2A.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                block_m_fwd,
                block_n_fwd,
                block_m_dq_bwd,
                block_n_dq_bwd,
                block_m_dkv_bwd,
                block_n_dkv_bwd,
                quant_block_size,
                grad_quant_type,
                cp_use_sdma,
            )
        elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
            return AttentionTritonFunctionCPA2A.apply(
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
                return_attn_probs,
                torch.is_grad_enabled(),
                True,
                cp_group,
                grad_quant_type,
                cp_use_sdma,
            )
        else:
            raise NotImplementedError(f"not supported quant_type {quant_type}")

    # Single-GPU path
    if quant_type == "mxfp8":
        return AttentionTritonMXFP8Function.apply(
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
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            block_m_fwd,
            block_n_fwd,
            block_m_dq_bwd,
            block_n_dq_bwd,
            block_m_dkv_bwd,
            block_n_dkv_bwd,
            quant_block_size,
            grad_quant_type,
        )
    elif quant_type == "blockwise2d":
        return AttentionTritonBlockwise2DFunction.apply(
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
            return_attn_probs,
            torch.is_grad_enabled(),
            grad_quant_type,
            scale_manager,
        )
    elif quant_type in ("blockwise", "dynamic", "delayed", "per_token"):
        return AttentionTritonFunction.apply(
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
            return_attn_probs,
            torch.is_grad_enabled(),
            True,
            grad_quant_type,
        )
    else:
        raise NotImplementedError(
            f"not supported quant_type {quant_type} backend_type {backend_type} yet"
        )
