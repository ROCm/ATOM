# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Unified attention kernel dispatcher.

Selects between aiter's C++/asm backend and Triton kernels based on the
environment variable ``LUMEN_ATTN_KERNEL_BACKEND``:

    LUMEN_ATTN_KERNEL_BACKEND=auto    # (default) prefer csrc, fallback triton
    LUMEN_ATTN_KERNEL_BACKEND=csrc    # force csrc, fallback triton if missing
    LUMEN_ATTN_KERNEL_BACKEND=triton  # always use triton

**Extending with new aiter kernels**

When the supported AITER API adds a new C++ kernel (e.g. FP8 csrc forward),
three steps:

1. Add the symbol to the explicit contract in
   :func:`_initialize_aiter_csrc`.
2. Write a ``@_torch_custom_op_wrapper`` implementation in Section 4.
3. Add a branch in the corresponding dispatch function in Section 6.
"""

import logging
import math
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.attention.block_scale_fp8_attention_kernel import (
    DEBUG,
    FIXED_BLOCK_M,
    FIXED_BLOCK_N,
    USE_FP8E5M2_BWD,
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    philox_offset,
    philox_seed,
)
from aiter.ops.triton._triton_kernels.attention.mxfp8_attention_kernel import (
    _bwd_kernel_dkdv_mxfp8,
    _bwd_kernel_dq_mxfp8,
    _bwd_preprocess_use_o_mxfp8,
    attn_fwd_mxfp8,
)
from torch._library import wrap_triton

from atomrl.lumen.core.float8 import float8_e4m3, float8_e5m2

# ── quantization helpers ────────────────────────────────────────────────
# All quantization is handled directly through ScalingManager static methods:
#   - ScalingManager.quantize_per_tensor_fp8
#   - ScalingManager.quantize_block_fp8
#   - ScalingManager.quantize_v_fp8
#   - ScalingManager.quantize_block_mxfp8
#   - ScalingManager.compute_p_scale_mxfp8
from atomrl.lumen.quantize.scaling_manager import ScalingManager

logger = logging.getLogger(__name__)
_torch_custom_op_wrapper = torch.library.custom_op

if attn_fwd is None:
    logger.warning(
        "aiter block_scale_fp8_attention_kernel unavailable; FP8 Triton attention disabled "
        "(csrc and BF16 backends unaffected)"
    )
if attn_fwd_mxfp8 is None:
    logger.warning(
        "aiter mxfp8_attention_kernel unavailable; MXFP8 Triton attention disabled"
    )


###########################################################################
# 1. Configuration
###########################################################################

_BACKEND_PREF = os.environ.get("LUMEN_ATTN_KERNEL_BACKEND", "auto")


###########################################################################
# 2. Backend capability detection
###########################################################################

_aiter_mha = None
_CSRC_OPS: dict[str, bool] = {}


def _get_aiter_mha():
    global _aiter_mha
    if _aiter_mha is None:
        import aiter.ops.mha as _mod

        _aiter_mha = _mod
    return _aiter_mha


def _initialize_aiter_csrc() -> None:
    """Register the C++/ASM operations provided by the installed AITER build."""
    if _BACKEND_PREF not in ("auto", "csrc"):
        return
    _get_aiter_mha()
    _CSRC_OPS.update(
        {
            "flash_attn_fwd": True,
            "flash_attn_bwd": True,
            "flash_attn_fp8_pertensor_fwd": True,
        }
    )


_initialize_aiter_csrc()


def csrc_available(op_name: str) -> bool:
    """Check whether a specific csrc operation is available."""
    return _CSRC_OPS.get(op_name, False)


###########################################################################
# 3. Shared helpers / constants / quantization
###########################################################################

fwd_torch_dtype: tl.constexpr = torch.bfloat16
bwd_torch_dtype: tl.constexpr = torch.bfloat16


def get_f8_fwd_dtype():
    return float8_e4m3


def get_f8_bwd_dtype():
    return float8_e5m2 if USE_FP8E5M2_BWD else float8_e4m3


F8_FWD_MAX: tl.constexpr = torch.finfo(get_f8_fwd_dtype()).max
F8_BWD_MAX: tl.constexpr = torch.finfo(get_f8_bwd_dtype()).max


def round_multiple(x, m):
    return (x + m - 1) // m * m


def is_cdna4() -> bool:
    """True on gfx950, the only arch the MXFP8 Triton kernels support."""
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def get_tl_f8_bwd_dtype():
    """Triton dtype of the FP8 operands in the backward kernels."""
    fnuz = not is_cdna4()
    if USE_FP8E5M2_BWD:
        return tl.float8e5b16 if fnuz else tl.float8e5
    return tl.float8e4b8 if fnuz else tl.float8e4nv


def get_padded_head_dim(head_size: int) -> int:
    """Head dim rounded up to a power of two, at least 16 (the kernels' minimum tile)."""
    return max(1 << (head_size - 1).bit_length(), 16)


def get_shape_from_layout(
    q,
    k,
    v,
    layout,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    """``(batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k)``."""
    if layout == "bhsd":
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
        head_size_v = v.shape[3]
    elif layout == "bshd":
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
        head_size_v = v.shape[3]
    elif layout == "thd":
        batch_q, nheads_q, head_size_q = len(cu_seqlens_q) - 1, q.shape[1], q.shape[2]
        batch_k, nheads_k, head_size_k = len(cu_seqlens_k) - 1, k.shape[1], k.shape[2]
        head_size_v = v.shape[2]
    else:
        raise ValueError(f"unsupported attention layout {layout!r}")

    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return (
        batch_q,
        nheads_q,
        nheads_k,
        head_size_q,
        head_size_v,
        max_seqlen_q,
        max_seqlen_k,
    )


def get_strides_from_layout(q, layout):
    """``(batch, head, seq, dim)`` strides of ``q`` for the given layout."""
    if layout == "thd":
        return (0, q.stride(1), q.stride(0), q.stride(2))
    if layout == "bhsd":
        return (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    if layout == "bshd":
        return (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    raise ValueError(f"unsupported attention layout {layout!r}")


def _mxfp8_flag_args(kernel, use_mxfp8):
    """``(use_mxfp8,)`` for an MXFP8 kernel that still declares the switch, else ``()``.

    Older aiter MXFP8 kernels took ``use_mxfp8`` (False pushed BF16 through the
    same kernel). Current aiter dropped it: those kernels are always MXFP8. Both
    callers in :mod:`atomrl.lumen.ops.attention` pass ``True``, so the only thing lost
    is a mode nothing selects; fail loudly if that ever stops being true rather
    than silently running MXFP8 on data the caller left unquantized.
    """
    if "use_mxfp8" in kernel.arg_names:
        return (use_mxfp8,)
    assert use_mxfp8, (
        "this aiter's MXFP8 kernels have no use_mxfp8 switch; the Triton MXFP8 path "
        "cannot run on unquantized inputs"
    )
    return ()


def _mxfp8_flag_kwargs(kernel, use_mxfp8):
    """Keyword form of :func:`_mxfp8_flag_args`."""
    return {"use_mxfp8": use_mxfp8} if _mxfp8_flag_args(kernel, use_mxfp8) else {}


###########################################################################
# 4. Low-level kernel implementations (torch custom_ops)
###########################################################################

# ── 4.1  csrc (aiter C++/asm) ──────────────────────────────────────────

if csrc_available("flash_attn_fwd"):

    @_torch_custom_op_wrapper(
        "lumen::attention_aiter_csrc_forward_impl",
        mutates_args=(),
        device_types="cuda",
    )
    def attention_aiter_csrc_forward_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        return_lse: bool,
        return_softmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        _mha = _get_aiter_mha()
        _unit = torch.ones(1, device=q.device, dtype=torch.float32)
        out_padded, softmax_lse, S_dmask, rng_state = _mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            sink_size=0,
            bias=bias,
            alibi_slopes=alibi_slopes,
            q_descale=_unit,
            k_descale=_unit,
            v_descale=_unit,
            return_lse=return_lse,
            return_softmax=return_softmax,
            how_v3_bf16_cvt=2,  # rtz: faster than rtna on gfx942; same kernel, different pre-compiled variant
        )
        return out_padded, softmax_lse, S_dmask, rng_state

    @attention_aiter_csrc_forward_impl.register_fake
    def _attention_aiter_csrc_forward_impl_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        return_lse: bool,
        return_softmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _mha = _get_aiter_mha()
        q, k, v = [_mha.maybe_contiguous(x) for x in (q, k, v)]
        batch_size, seqlen_q, num_heads, _head_size = q.shape
        seqlen_k = k.shape[1]
        out = torch.empty_like(q)
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q),
            dtype=torch.float32,
            device=q.device,
            layout=q.layout,
        )
        p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
        if return_softmax:
            p = torch.empty(
                (
                    batch_size,
                    num_heads,
                    round_multiple(seqlen_q, 128),
                    round_multiple(seqlen_k, 128),
                ),
                dtype=q.dtype,
                device=q.device,
                layout=q.layout,
            )
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
        return out, softmax_lse, p, rng_state


if csrc_available("flash_attn_bwd"):

    @_torch_custom_op_wrapper(
        "lumen::attention_aiter_csrc_backward_impl",
        mutates_args=("dq", "dk", "dv"),
        device_types="cuda",
    )
    def attention_aiter_csrc_backward_impl(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor | None,
        dk: torch.Tensor | None,
        dv: torch.Tensor | None,
        dbias: torch.Tensor | None,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        rng_state: torch.Tensor | None = None,
        is_v3_atomic_fp32: bool | None = True,
        how_v3_bf16_cvt: int | None = 1,
    ) -> torch.Tensor:
        _mha = _get_aiter_mha()
        return _mha._flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dbias,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            bias,
            alibi_slopes,
            deterministic,
            rng_state,
            is_v3_atomic_fp32,
            how_v3_bf16_cvt,
        )

    @attention_aiter_csrc_backward_impl.register_fake
    def _attention_aiter_csrc_backward_impl_fake(
        dout: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        softmax_lse: torch.Tensor,
        dq: torch.Tensor | None,
        dk: torch.Tensor | None,
        dv: torch.Tensor | None,
        dbias: torch.Tensor | None,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
        bias: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        deterministic: bool,
        rng_state: torch.Tensor | None = None,
        is_v3_atomic_fp32: bool | None = True,
        how_v3_bf16_cvt: int | None = 1,
    ) -> torch.Tensor:
        _mha = _get_aiter_mha()
        dout, q, k, v, out = [_mha.maybe_contiguous(x) for x in (dout, q, k, v, out)]
        if dq is None:
            dq = torch.empty_like(q)
        if dk is None:
            dk = torch.empty_like(k)
        if dv is None:
            dv = torch.empty_like(v)
        batch_size, seqlen_q, num_heads, _ = q.shape
        softmax_d = torch.empty(
            (batch_size, num_heads, round_multiple(seqlen_q, 128)),
            device=q.device,
            dtype=torch.float32,
        )
        return softmax_d


# ── 4.1b  csrc – FP8 per-tensor forward (forward-only, no csrc bwd) ───

if csrc_available("flash_attn_fp8_pertensor_fwd"):

    @_torch_custom_op_wrapper(
        "lumen::attention_aiter_csrc_fp8_pertensor_forward_impl",
        mutates_args=(),
        device_types="cuda",
    )
    def attention_aiter_csrc_fp8_pertensor_forward_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _mha = _get_aiter_mha()
        out_padded, softmax_lse, _, _ = _mha._flash_attn_forward(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            sink_size=0,
            bias=None,
            alibi_slopes=None,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            return_lse=True,
            return_softmax=False,
            how_v3_bf16_cvt=2,  # rtz: faster than rtna on gfx942; applies to bf16 kernel variant selection
        )
        return out_padded, softmax_lse

    @attention_aiter_csrc_fp8_pertensor_forward_impl.register_fake
    def _attention_aiter_csrc_fp8_pertensor_forward_impl_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        window_size_left: int,
        window_size_right: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seqlen_q, num_heads, head_size_v = v.shape
        o_shape = list(q.shape)
        o_shape[-1] = head_size_v
        out = torch.empty(o_shape, device=q.device, dtype=fwd_torch_dtype)
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q),
            dtype=torch.float32,
            device=q.device,
        )
        return out, softmax_lse


# ── 4.2  Triton – FP8 blockwise ────────────────────────────────────────


@_torch_custom_op_wrapper(
    "lumen::attention_triton_forward_impl",
    mutates_args=(),
    device_types="cuda",
)
def attention_triton_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    return_softmax: bool,
    use_fp8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert q_descale.is_contiguous()
    assert k_descale.is_contiguous()

    layout = "bshd"
    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1]
    max_seqlens_k = k.shape[1]
    return_scores = return_softmax
    use_exp2 = True
    if DEBUG:
        print()
        print("attention_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)
        print("use_fp8:", use_fp8)

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_fp8 else q.dtype,
        requires_grad=True,
    )

    is_varlen = layout == "thd"

    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, _seqlen_q, _seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
        )
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    grid = (triton.cdiv(max_seqlens_q, FIXED_BLOCK_M), nheads_q, batch)

    if return_scores:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_strides = (
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            scores.stride(3),
        )
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    if is_varlen:
        softmax_lse = torch.empty(
            (q.shape[0] * 2, nheads_q), device=q.device, dtype=torch.float32
        )
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        _lse_cols = (
            ((max_seqlens_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M) * 2 * FIXED_BLOCK_M
        )
        softmax_lse = torch.empty(
            (batch, nheads_q, _lse_cols), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        elif alibi_slopes.shape[0] == 1 and batch > 1:
            alibi_slopes = alibi_slopes.expand(batch, -1)
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_fp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = (
            q_descale.stride(0),
            q_descale.stride(1),
            q_descale.stride(2),
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = (
            k_descale.stride(0),
            k_descale.stride(1),
            k_descale.stride(2),
        )
        padded_kscale_block_num = 1 << (stride_kdescale_h - 1).bit_length()
    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m = None, None, None
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m = None, None, None
        padded_kscale_block_num = None

    wrap_triton(attn_fwd)[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_descale,
        k_descale,
        v_scale,
        use_fp8,
        softmax_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        padded_kscale_block_num,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=not bias is None,
        USE_ALIBI=not alibi_slopes is None,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
    )

    return o, softmax_lse, exp_scores


@attention_triton_forward_impl.register_fake
def _attention_triton_forward_impl_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    return_softmax: bool,
    use_fp8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=torch.bfloat16 if use_fp8 else q.dtype,
        requires_grad=True,
    )
    batch_q, max_seqlen_q, nheads_q, _head_size_q = q.shape
    _batch_k, max_seqlen_k, _nheads_k, _head_size_k = k.shape
    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)
    _lse_cols = (
        ((max_seqlen_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M) * 2 * FIXED_BLOCK_M
    )
    softmax_lse = torch.empty(
        (batch_q, nheads_q, _lse_cols), device=q.device, dtype=torch.float32
    )
    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "lumen::attention_triton_backward_impl",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def attention_triton_backward_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse_delta: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: torch.Tensor | None,
    use_fp8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    use_exp2 = True
    layout = "bshd"
    sequence_parallel = True
    do = do.contiguous()

    if DEBUG:
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse_delta, softmax_lse_delta.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("softmax_scale:", softmax_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("sequence_parallel:", sequence_parallel)

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
    )
    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    assert head_size_qk >= 32 and head_size_v >= 32
    assert head_size_qk % 2 == 0 and head_size_v % 2 == 0
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    copy_back = {"dq": False, "dk": False, "dv": False}

    if dq is None:
        dq = torch.zeros_like(q, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True
        dq.zero_()
    stride_dq_all = dq.stride()[0]

    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True
        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse_delta.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse_delta.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = (
            softmax_lse_delta.stride()
        )

    if use_fp8:
        do_fp8 = torch.empty_like(do, dtype=get_f8_bwd_dtype())
        _shape = (batch, nheads_q, triton.cdiv(max_seqlen_q, FIXED_BLOCK_M))
        do_scale = torch.empty(_shape, dtype=torch.float32, device=q.device)
        stride_descalez, stride_descaleh, stride_descalem = do_scale.stride()
        stride_qscalez, stride_qscaleh, stride_qscalem = q_scale.stride()
        stride_kscalez, stride_kscaleh, stride_kscalem = k_scale.stride()
        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh - 1).bit_length()
    else:
        do_fp8 = None
        do_scale = None
        stride_descalez, stride_descaleh, stride_descalem = None, None, None
        stride_qscalez, stride_qscaleh, stride_qscalem = None, None, None
        stride_kscalez, stride_kscaleh, stride_kscalem = None, None, None
        padded_doscale_block_num, padded_qscale_block_num, padded_kscale_block_num = (
            None,
            None,
            None,
        )

    grid_prebwd = (triton.cdiv(max_seqlen_q, FIXED_BLOCK_M), batch_headsize_q)
    wrap_triton(_bwd_preprocess_use_o)[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        softmax_lse_delta,
        use_fp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        BLOCK_M=FIXED_BLOCK_M,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        F8_BWD_MAX=F8_BWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_scale", softmax_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse_delta, softmax_lse_delta.shape)
        print(
            "stride_qz, stride_qh, stride_qm, stride_qk:",
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
        )
        print(
            "stride_kz, stride_kh, stride_kn, stride_kk:",
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
        )
        print(
            "stride_vz, stride_vh, stride_vn, stride_vk:",
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
        )
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL_QK:", padded_d_model_qk)
        print("BLOCK_DMODEL_V:", padded_d_model_v)
        print("SEQUENCE_PARALLEL:", sequence_parallel)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    log_p_scale = math.log(p_scale)
    num_block_m = triton.cdiv(max_seqlen_q, FIXED_BLOCK_M)
    grid_bwd = (
        batch_headsize_q,
        triton.cdiv(max_seqlen_q, FIXED_BLOCK_M) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dq)[grid_bwd](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if use_fp8:
        n_groups = nheads_q // nheads_k
        padded_doscale_block_num = 1 << (stride_descaleh - 1).bit_length()
        padded_qscale_block_num = 1 << (stride_qscaleh * n_groups - 1).bit_length()
        padded_kscale_block_num = 1 << (stride_kscaleh * n_groups - 1).bit_length()
    else:
        padded_doscale_block_num = None
        padded_qscale_block_num = None
        padded_kscale_block_num = None

    grid_bwd_dkdv = (
        batch_headsize_k,
        triton.cdiv(max_seqlen_k, FIXED_BLOCK_N) if sequence_parallel else 1,
    )
    wrap_triton(_bwd_kernel_dkdv)[grid_bwd_dkdv](
        q,
        k,
        v,
        softmax_scale,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        do_scale,
        o,
        do_fp8 if use_fp8 else do,
        dq,
        dk,
        dv,
        softmax_lse_delta,
        stride_dq_all,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_descalez,
        stride_descaleh,
        stride_descalem,
        stride_qscalez,
        stride_qscaleh,
        stride_qscalem,
        stride_kscalez,
        stride_kscaleh,
        stride_kscalem,
        padded_doscale_block_num,
        padded_qscale_block_num,
        padded_kscale_block_num,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=FIXED_BLOCK_M,
        BLOCK_N=FIXED_BLOCK_N,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=F8_FWD_MAX,
    )

    if DEBUG:
        print("####################################################")
        print("_bwd_kernel outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("copy_back:", copy_back)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv


@attention_triton_backward_impl.register_fake
def _attention_triton_backward_impl_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: float,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    cu_seqlens_q: int,
    cu_seqlens_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: torch.Tensor | None,
    use_fp8: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q, dtype=bwd_torch_dtype),
        torch.empty_like(k, dtype=bwd_torch_dtype),
        torch.empty_like(v, dtype=bwd_torch_dtype),
    )


# ── 4.3  Triton – MXFP8 ───────────────────────────────────────────────


@_torch_custom_op_wrapper(
    "lumen::attention_mxfp8_forward_triton_impl",
    mutates_args=(),
    device_types="cuda",
)
def attention_mxfp8_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: int,
    sm_scale: float,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: torch.Tensor | None,
    dropout_p: float,
    return_softmax: bool,
    use_mxfp8: bool,
    block_m: int = 64,
    block_n: int = 64,
    quant_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."
    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    layout = "bshd"
    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1]
    max_seqlens_k = k.shape[1]
    return_scores = return_softmax
    use_exp2 = True
    quant_size: int = 32

    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        print()
        print("attention_forward_triton_impl")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("bias:", bias)
        print("dropout_p:", dropout_p)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlens_q:", max_seqlens_q)
        print("max_seqlens_k:", max_seqlens_k)
        print("return_scores:", return_scores)
        print("use_exp2:", use_exp2)
        print("use_mxfp8:", use_mxfp8)
        print("BLOCK_M:", block_m)
        print("BLOCK_N:", block_n)
        print("QUANT_BLOCK_SIZE:", quant_block_size)

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_mxfp8 else q.dtype,
        requires_grad=True,
    )

    is_varlen = layout == "thd"

    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, _seqlen_q, _seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
        )
    )

    assert (
        quant_block_size % quant_size == 0
    ), "quant block must be divided by quant size"
    assert (
        block_m % quant_block_size == 0
    ), "block M in fwd must be divided by quant size"
    assert (
        block_n % quant_block_size == 0
    ), "block N in fwd must be divided by quant size"

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)
    assert (
        padded_d_model_qk % quant_block_size == 0
    ), "padded_d_model_qk must be divided by quant size"
    assert (
        padded_d_model_v % quant_block_size == 0
    ), "padded_d_model_v must be divided by quant size"

    grid = (triton.cdiv(max_seqlens_q, block_m), nheads_q, batch)

    if return_scores:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_scaled_shifted = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_strides = (
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            scores.stride(3),
        )
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    if return_scores:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    if is_varlen:
        softmax_lse = torch.empty(
            (q.shape[0], nheads_q), device=q.device, dtype=torch.float32
        )
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty(
            (batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    if bias is not None:
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if alibi_slopes is not None:
        if alibi_slopes.dim() == 1:
            alibi_slopes = alibi_slopes.unsqueeze(0).expand(batch, -1)
        elif alibi_slopes.shape[0] == 1 and batch > 1:
            alibi_slopes = alibi_slopes.expand(batch, -1)
        alibi_strides = (alibi_slopes.stride(0), alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    if use_mxfp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m, stride_qdescale_d = (
            get_strides_from_layout(q_scale, layout)
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m, stride_kdescale_d = (
            get_strides_from_layout(k_scale, layout)
        )
        stride_vdescale_z, stride_vdescale_h, stride_vdescale_m, stride_vdescale_d = (
            get_strides_from_layout(v_scale, layout)
        )
    else:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m, stride_qdescale_d = (
            None,
            None,
            None,
            None,
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m, stride_kdescale_d = (
            None,
            None,
            None,
            None,
        )
        stride_vdescale_z, stride_vdescale_h, stride_vdescale_m, stride_vdescale_d = (
            None,
            None,
            None,
            None,
        )

    kernel_kwargs = {}
    if padded_d_model_qk % 128 == 0 and block_n % 128 == 0:
        kernel_kwargs["matrix_instr_nonkdim"] = 16

    wrap_triton(attn_fwd_mxfp8)[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_scale,
        k_scale,
        v_scale,
        *_mxfp8_flag_args(attn_fwd_mxfp8, use_mxfp8),
        sm_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_qdescale_d,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        stride_kdescale_d,
        stride_vdescale_z,
        stride_vdescale_h,
        stride_vdescale_m,
        stride_vdescale_d,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=not bias is None,
        USE_ALIBI=not alibi_slopes is None,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    return o, softmax_lse, exp_scores


@attention_mxfp8_forward_triton_impl.register_fake
def fake_attention_mxfp8_forward_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: int,
    sm_scale: float,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: torch.Tensor | None,
    dropout_p: float,
    return_softmax: bool,
    use_mxfp8: bool,
    block_m: int = 64,
    block_n: int = 64,
    quant_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=fwd_torch_dtype if use_mxfp8 else q.dtype,
        requires_grad=True,
    )
    batch_q, max_seqlen_q, nheads_q, _head_size_q = q.shape
    _batch_k, max_seqlen_k, _nheads_k, _head_size_k = k.shape
    if return_softmax:
        exp_scores = torch.zeros(
            (batch_q, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)
    _lse_cols = (
        ((max_seqlen_q + FIXED_BLOCK_M - 1) // FIXED_BLOCK_M) * 2 * FIXED_BLOCK_M
    )
    softmax_lse = torch.empty(
        (batch_q, nheads_q, _lse_cols), device=q.device, dtype=torch.float32
    )
    return o, softmax_lse, exp_scores


@_torch_custom_op_wrapper(
    "lumen::attention_triton_mxfp8_backward_triton_impl",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def attention_triton_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    p_scale: int,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    cu_seqlens_q: int | None,
    cu_seqlens_k: int | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    use_mxfp8: bool,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    assert (
        window_size_left == -1 and window_size_right == -1
    ), "in triton attn kernel, window_size_left and window_size_right must be -1."

    use_exp2 = True
    layout = "bshd"
    do = do.contiguous()
    quant_size: int = 32

    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        print("####################################################")
        print("attention_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("causal:", causal)
        print("layout:", layout)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("use_exp2:", use_exp2)
        print("use_mxfp8:", use_mxfp8)
        print("block_m_dq_bwd:", block_m_dq_bwd)
        print("block_n_dq_bwd:", block_n_dq_bwd)
        print("block_m_dkv_bwd:", block_m_dkv_bwd)
        print("block_n_dkv_bwd:", block_n_dkv_bwd)
        print("quant_block_size:", quant_block_size)

    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    if not do.is_contiguous():
        do = do.contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()
    assert q_scale.is_contiguous()
    assert k_scale.is_contiguous()

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
    )

    assert (
        quant_block_size % quant_size == 0
    ), "quant block must be divided by quant size"
    assert (
        block_m_dq_bwd % quant_block_size == 0
    ), "block M in dq bwd must be divided by quant size"
    assert (
        block_m_dkv_bwd % quant_block_size == 0
    ), "block M in dkv bwd must be divided by quant size"
    assert (
        block_n_dq_bwd % quant_block_size == 0
    ), "block N in dq bwd must be divided by quant size"
    assert (
        block_n_dkv_bwd % quant_block_size == 0
    ), "block N in dkv bwd must be divided by quant size"

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    assert head_size_qk >= 32 and head_size_v >= 32
    assert head_size_qk % 2 == 0 and head_size_v % 2 == 0
    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)
    assert (
        padded_d_model_qk % quant_block_size == 0
    ), "padded_d_model_qk must be divided by quant size"
    assert (
        padded_d_model_v % quant_block_size == 0
    ), "padded_d_model_v must be divided by quant size"

    copy_back = {"dq": False, "dk": False, "dv": False}

    if dq is None:
        dq = torch.zeros_like(q, dtype=bwd_torch_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True
        dq.zero_()
    dq.stride()[0]

    if (dk is None) or (dv is None):
        dk = torch.zeros_like(k, dtype=bwd_torch_dtype)
        dv = torch.zeros_like(v, dtype=bwd_torch_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True
        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    if DEBUG:
        print("copy_back:", copy_back)

    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = (
            softmax_lse.stride()
        )

    if use_mxfp8:
        m_blocks_q = triton.cdiv(max_seqlen_q, quant_block_size)
        dv_blocks = triton.cdiv(head_size_v, quant_block_size)
        if layout == "bhsd":
            _shape = (batch, nheads_q, m_blocks_q, dv_blocks)
        elif layout == "bshd":
            _shape = (batch, m_blocks_q, nheads_q, dv_blocks)
        elif layout == "thd":
            _shape = (q_scale.shape[0], q_scale.shape[1], dv_blocks)
        else:
            raise AssertionError(f"Got unsupported layout for do_scale: {layout}")
        do_fp8 = torch.empty_like(do, dtype=get_f8_bwd_dtype())
        do_scale = torch.empty(_shape, dtype=torch.uint8, device=q.device)
        stride_dodescalez, stride_dodescaleh, stride_dodescalem, stride_dodescaled = (
            get_strides_from_layout(do_scale, layout)
        )
        stride_qdescalez, stride_qdescaleh, stride_qdescalem, stride_qdescaled = (
            get_strides_from_layout(q_scale, layout)
        )
        stride_kdescalez, stride_kdescaleh, stride_kdescalem, stride_kdescaled = (
            get_strides_from_layout(k_scale, layout)
        )
        stride_vdescalez, stride_vdescaleh, stride_vdescalem, stride_vdescaled = (
            get_strides_from_layout(v_scale, layout)
        )
    else:
        do_fp8 = None
        do_scale = None
        stride_dodescalez, stride_dodescaleh, stride_dodescalem, stride_dodescaled = (
            None,
            None,
            None,
            None,
        )
        stride_qdescalez, stride_qdescaleh, stride_qdescalem, stride_qdescaled = (
            None,
            None,
            None,
            None,
        )
        stride_kdescalez, stride_kdescaleh, stride_kdescalem, stride_kdescaled = (
            None,
            None,
            None,
            None,
        )
        stride_vdescalez, stride_vdescaleh, stride_vdescalem, stride_vdescaled = (
            None,
            None,
            None,
            None,
        )

    preprocess_o_block = min(max_seqlen_q, 64)
    preprocess_o_block = max(preprocess_o_block, quant_block_size)
    grid_prebwd = (triton.cdiv(max_seqlen_q, preprocess_o_block), batch_headsize_q)
    wrap_triton(_bwd_preprocess_use_o_mxfp8)[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        delta,
        *_mxfp8_flag_args(_bwd_preprocess_use_o_mxfp8, use_mxfp8),
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=preprocess_o_block,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
    )

    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        print("####################################################")
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print(
            "stride_qz, stride_qh, stride_qm, stride_qk:",
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
        )
        print(
            "stride_kz, stride_kh, stride_kn, stride_kk:",
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
        )
        print(
            "stride_vz, stride_vh, stride_vn, stride_vk:",
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
        )
        print("batch_q:", batch)
        print("heads_q:", nheads_q)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("BLOCK_DMODEL_QK:", padded_d_model_qk)
        print("BLOCK_DMODEL_V:", padded_d_model_v)
        print("CAUSAL:", causal)
        print("USE_EXP2:", use_exp2)

    num_block_m = triton.cdiv(max_seqlen_q, block_m_dq_bwd)
    grid_bwd = (batch_headsize_q, num_block_m)
    kernel_kwargs = {}
    if (
        block_n_dq_bwd % 128 == 0
        and padded_d_model_qk % 128 == 0
        and padded_d_model_v % 128 == 0
    ):
        kernel_kwargs["matrix_instr_nonkdim"] = 16

    p_scale_t = math.pow(2.0, int(p_scale - 127))
    log_p_scale = math.log(p_scale_t)

    wrap_triton(_bwd_kernel_dq_mxfp8)[grid_bwd](
        q,
        k,
        v,
        sm_scale,
        p_scale,
        log_p_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dq_bwd,
        BLOCK_N=block_n_dq_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        **_mxfp8_flag_kwargs(_bwd_kernel_dq_mxfp8, use_mxfp8),
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    if (
        block_m_dkv_bwd % 128 == 0
        and padded_d_model_qk % 128 == 0
        and padded_d_model_v % 128 == 0
    ):
        kernel_kwargs["matrix_instr_nonkdim"] = 16
    else:
        kernel_kwargs = {}

    grid_bwd_dkdv = (
        batch_headsize_k,
        triton.cdiv(max_seqlen_k, block_n_dkv_bwd),
    )
    wrap_triton(_bwd_kernel_dkdv_mxfp8)[grid_bwd_dkdv](
        q,
        k,
        v,
        p_scale,
        log_p_scale,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dkv_bwd,
        BLOCK_N=block_n_dkv_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        **_mxfp8_flag_kwargs(_bwd_kernel_dkdv_mxfp8, use_mxfp8),
        F8_BWD_DTYPE=get_tl_f8_bwd_dtype(),
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        print("####################################################")
        print("_bwd_kernel scales")
        print("q_scale:", q_scale, q_scale.shape, q_scale.dtype)
        print("k_scale:", k_scale, k_scale.shape, k_scale.dtype)
        print("v_scale:", v_scale, v_scale.shape, v_scale.dtype)
        print("####################################################")
        print("attention_prefill_backward_triton_new_impl outputs")
        print("dq:", dq, dq.shape, dq.dtype)
        print("dk:", dk, dk.shape, dk.dtype)
        print("dv:", dv, dv.shape, dv.dtype)
        print("copy_back:", copy_back)

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv


@attention_triton_mxfp8_backward_triton_impl.register_fake
def fake_attention_triton_mxfp8_backward_triton_impl(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    p_scale: int,
    alibi_slopes: torch.Tensor | None,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    cu_seqlens_q: int | None,
    cu_seqlens_k: int | None,
    max_seqlen_q: int | None,
    max_seqlen_k: int | None,
    use_mxfp8: bool,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q, dtype=fwd_torch_dtype),
        torch.empty_like(k, dtype=fwd_torch_dtype),
        torch.empty_like(v, dtype=fwd_torch_dtype),
    )


###########################################################################
# 5. High-level dispatch API
#
# These are the public entry-points consumed by attention_utils.py (and
# transitively by attention.py, attention_with_cp_a2a.py, etc.).
#
# Each function resolves the best backend at call time:
#   csrc (C++/asm)  →  triton (fallback, always available)
#
# To support a newly-added aiter csrc kernel (e.g. FP8 csrc forward):
#   1. Add the symbol to _initialize_aiter_csrc above.
#   2. Register the custom_op in Section 4.
#   3. Add a branch below (see the "future" comments).
###########################################################################


def attention_forward(
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
    force_triton=False,
):
    """Dispatched attention forward (FP8 blockwise / non-quantized).

    Backend priority for non-quantized: csrc → triton.
    Backend priority for FP8 (inference-only): csrc per-tensor → triton per-block.
    Backend priority for FP8 (training):       triton per-block.

    When *force_triton* is True, csrc branches are skipped unconditionally.
    """
    # ── non-quantized: try csrc, fall back to triton on rejection ───
    if (
        not force_triton
        and not use_fp8
        and _BACKEND_PREF in ("auto", "csrc")
        and csrc_available("flash_attn_fwd")
    ):
        try:
            out, softmax_lse, exp_scores, rng_state = attention_aiter_csrc_forward_impl(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                bias,
                alibi_slopes,
                True,
                return_softmax,
            )
            q_scale = torch.tensor([1.0], device=q.device)
            k_scale = torch.tensor([1.0], device=q.device)
            v_scale = torch.tensor([1.0], device=q.device)
            p_scale = 1.0
            return (
                out,
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
            )
        except RuntimeError as _csrc_err:
            # Known trigger: CP A2A scatter can produce nheads==1 per rank,
            # which the CK fmha_fwd kernel rejects.  Safe to fall through
            # to the Triton path below — numerically equivalent.
            logger.warning(
                "attention_forward: aiter csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _csrc_err,
            )

    # ── FP8 per-tensor csrc (forward-only, inference) ───────────────
    # aiter has a C++ per-tensor FP8 forward but no matching backward,
    # so we only use it when none of the inputs require grad.
    _no_grad_needed = not (q.requires_grad or k.requires_grad or v.requires_grad)
    if (
        not force_triton
        and use_fp8
        and _no_grad_needed
        and csrc_available("flash_attn_fp8_pertensor_fwd")
    ):
        try:
            fp8_dt = get_f8_fwd_dtype()
            q_fp8, q_descale = ScalingManager.quantize_per_tensor_fp8(q, fp8_dt)
            k_fp8, k_descale = ScalingManager.quantize_per_tensor_fp8(k, fp8_dt)
            v_fp8, v_descale = ScalingManager.quantize_per_tensor_fp8(v, fp8_dt)
            out, softmax_lse = attention_aiter_csrc_fp8_pertensor_forward_impl(
                q_fp8,
                k_fp8,
                v_fp8,
                q_descale,
                k_descale,
                v_descale,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
            )
            exp_scores = torch.empty([], device=q.device, dtype=torch.float32)
            return (
                out,
                softmax_lse,
                exp_scores,
                q_fp8,
                k_fp8,
                v_fp8,
                q_descale,
                k_descale,
                v_descale,
                1.0,
                None,
            )
        except RuntimeError as _csrc_err:
            logger.warning(
                "attention_forward: aiter FP8 per-tensor csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _csrc_err,
            )

    # ── triton per-block fallback (supports both fwd & bwd) ─────────
    # Reached when: force_triton=True, csrc unavailable, OR csrc raised
    # RuntimeError for an unsupported config (e.g. nheads==1 after scatter).
    if use_fp8:
        fp8_dt = get_f8_fwd_dtype()
        q, q_scale = ScalingManager.quantize_block_fp8(q, FIXED_BLOCK_M, fp8_dt)
        k, k_scale = ScalingManager.quantize_block_fp8(k, FIXED_BLOCK_M, fp8_dt)
        v, v_scale, p_scale = ScalingManager.quantize_v_fp8(v, fp8_dt)
    else:
        q_scale = torch.tensor([1.0], device=q.device)
        k_scale = torch.tensor([1.0], device=q.device)
        v_scale = torch.tensor([1.0], device=q.device)
        p_scale = 1.0
    output, softmax_lse, exp_scores = attention_triton_forward_impl(
        q,
        k,
        v,
        p_scale,
        q_scale,
        k_scale,
        v_scale,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        bias,
        alibi_slopes,
        return_softmax,
        use_fp8,
    )
    return (
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
        None,
    )


def attention_backward(
    do,
    q,
    k,
    v,
    o,
    q_scale,
    k_scale,
    v_scale,
    p_scale,
    softmax_lse,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlens_q,
    max_seqlens_k,
    sm_scale,
    causal,
    alibi_slopes,
    use_fp8,
    rng_state=None,
    dropout_p=0.0,
    bias=None,
    window_size=(-1, -1),
    deterministic=True,
    force_triton=False,
):
    """Dispatched attention backward (FP8 blockwise / non-quantized).

    Backend priority for non-quantized: csrc → triton.
    Backend priority for FP8:           (future csrc fp8) → triton fp8.

    When *force_triton* is True, csrc branches are skipped unconditionally.
    """
    # ── non-quantized: try csrc, fall back to triton on rejection ───
    if (
        not force_triton
        and not use_fp8
        and _BACKEND_PREF in ("auto", "csrc")
        and csrc_available("flash_attn_bwd")
    ):
        try:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            attention_aiter_csrc_backward_impl(
                do,
                q,
                k,
                v,
                o,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                dropout_p,
                sm_scale,
                causal,
                window_size[0],
                window_size[1],
                bias,
                alibi_slopes,
                deterministic,
                rng_state,
            )
            return dq, dk, dv
        except RuntimeError as _csrc_err:
            logger.warning(
                "attention_backward: aiter csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _csrc_err,
            )

    # ── future: FP8 csrc ────────────────────────────────────────────
    # if use_fp8 and csrc_available("flash_attn_fp8_bwd"):
    #     return _csrc_fp8_backward(...)

    # ── triton fallback (always available for non-ALiBi configs) ────
    if alibi_slopes is not None:
        raise NotImplementedError(
            "ALiBi backward is not supported by the Triton attention kernels. "
            "Install AITER with CK support (csrc backend) or remove alibi_slopes."
        )
    return attention_triton_backward_impl(
        do,
        q,
        k,
        v,
        o,
        q_scale,
        k_scale,
        v_scale,
        p_scale,
        softmax_lse,
        None,
        None,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        sm_scale,
        causal,
        -1,
        -1,
        alibi_slopes,
        use_fp8,
    )


def attention_mxfp8_forward(
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
):
    """Dispatched MXFP8 attention forward.

    Backend priority for non-quantized: csrc → triton.
    Backend priority for MXFP8:         (future csrc mxfp8) → triton mxfp8.
    """
    # ── non-quantized: try csrc, fall back to triton on rejection ───
    if (
        not use_mxfp8
        and _BACKEND_PREF in ("auto", "csrc")
        and csrc_available("flash_attn_fwd")
    ):
        try:
            out, softmax_lse, exp_scores, rng_state = attention_aiter_csrc_forward_impl(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size[0],
                window_size[1],
                bias,
                alibi_slopes,
                True,
                return_softmax,
            )
            q_scale = torch.scalar_tensor(1.0, device=q.device)
            k_scale = torch.scalar_tensor(1.0, device=q.device)
            v_scale = torch.scalar_tensor(1.0, device=q.device)
            p_scale = 127
            return (
                out,
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
            )
        except RuntimeError as _csrc_err:
            logger.warning(
                "attention_mxfp8_forward: aiter csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _csrc_err,
            )

    # ── future: MXFP8 csrc ─────────────────────────────────────────
    # if use_mxfp8 and csrc_available("flash_attn_mxfp8_fwd"):
    #     return _csrc_mxfp8_forward(...)

    # ── triton fallback (only path for MXFP8 until csrc is ready) ──
    assert is_cdna4(), "mxfp8 is only supported by gfx950 and newer version"
    if use_mxfp8:
        fp8_dt = get_f8_fwd_dtype()
        q, q_scale = ScalingManager.quantize_block_mxfp8(
            q,
            quant_block_size,
            "bshd",
            is_2d_block=True,
            float8_dtype=fp8_dt,
            cu_seqlens=0,
            max_seqlens=q.shape[1],
        )
        k, k_scale = ScalingManager.quantize_block_mxfp8(
            k,
            quant_block_size,
            "bshd",
            is_2d_block=True,
            float8_dtype=fp8_dt,
            cu_seqlens=0,
            max_seqlens=k.shape[1],
        )
        v, v_scale = ScalingManager.quantize_block_mxfp8(
            v,
            quant_block_size,
            "bshd",
            is_2d_block=True,
            float8_dtype=fp8_dt,
            cu_seqlens=0,
            max_seqlens=k.shape[1],
        )
        p_scale = ScalingManager.compute_p_scale_mxfp8(fp8_dt)
    else:
        q_scale = torch.scalar_tensor(1.0, device=q.device)
        k_scale = torch.scalar_tensor(1.0, device=q.device)
        v_scale = torch.scalar_tensor(1.0, device=q.device)
        p_scale = 127

    output, softmax_lse, exp_scores = attention_mxfp8_forward_triton_impl(
        q=q,
        k=k,
        v=v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        p_scale=p_scale,
        sm_scale=softmax_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        bias=bias,
        dropout_p=dropout_p,
        return_softmax=return_softmax,
        use_mxfp8=use_mxfp8,
        block_m=block_m_fwd,
        block_n=block_n_fwd,
        quant_block_size=quant_block_size,
    )
    return (
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
        None,
    )


def attention_mxfp8_backward(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    q_scale,
    k_scale,
    v_scale,
    sm_scale,
    p_scale,
    alibi_slopes,
    causal,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlens_q,
    max_seqlens_k,
    use_mxfp8,
    block_m_dq_bwd,
    block_n_dq_bwd,
    block_m_dkv_bwd,
    block_n_dkv_bwd,
    quant_block_size,
    rng_state=None,
    dropout_p=0.0,
    bias=None,
    window_size=(-1, -1),
    deterministic=True,
):
    """Dispatched MXFP8 attention backward.

    Backend priority for non-quantized: csrc → triton.
    Backend priority for MXFP8:         (future csrc mxfp8) → triton mxfp8.
    """
    # ── non-quantized: try csrc, fall back to triton on rejection ───
    if (
        not use_mxfp8
        and _BACKEND_PREF in ("auto", "csrc")
        and csrc_available("flash_attn_bwd")
    ):
        try:
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            attention_aiter_csrc_backward_impl(
                do,
                q,
                k,
                v,
                o,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                dropout_p,
                sm_scale,
                causal,
                window_size[0],
                window_size[1],
                bias,
                alibi_slopes,
                deterministic,
                rng_state,
            )
            return dq, dk, dv
        except RuntimeError as _csrc_err:
            logger.warning(
                "attention_mxfp8_backward: aiter csrc rejected "
                "(q=%s, k=%s, causal=%s): %s — falling back to Triton",
                tuple(q.shape),
                tuple(k.shape),
                causal,
                _csrc_err,
            )

    # ── future: MXFP8 csrc ─────────────────────────────────────────
    # if use_mxfp8 and csrc_available("flash_attn_mxfp8_bwd"):
    #     return _csrc_mxfp8_backward(...)

    # ── triton fallback ─────────────────────────────────────────────
    if alibi_slopes is not None:
        raise NotImplementedError(
            "ALiBi backward is not supported by the Triton MXFP8 attention kernels. "
            "Install AITER with CK support (csrc backend) or remove alibi_slopes."
        )
    return attention_triton_mxfp8_backward_triton_impl(
        do=do,
        q=q,
        k=k,
        v=v,
        o=o,
        softmax_lse=softmax_lse,
        dq=None,
        dk=None,
        dv=None,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        p_scale=p_scale,
        alibi_slopes=alibi_slopes,
        causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlens_q,
        max_seqlen_k=max_seqlens_k,
        use_mxfp8=use_mxfp8,
        block_m_dq_bwd=block_m_dq_bwd,
        block_n_dq_bwd=block_n_dq_bwd,
        block_m_dkv_bwd=block_m_dkv_bwd,
        block_n_dkv_bwd=block_n_dkv_bwd,
        quant_block_size=quant_block_size,
    )
