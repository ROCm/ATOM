# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MiniMax-M3.1 as a TP1 P/D prefill producer: the decoder layer on fused
kernels, with the bits of the generic layer. The P/D router asks a producer for
one token, so its steps are prefills; a client asking it for more decodes here too.

A layer runs:

* its input norm as one kernel: the previous MoE layer's fold
  ``routed * 2 + shared``, the residual add, Gemma RMSNorm and the MXFP8 quant
  of the qkv input (``norm_mxfp8``);
* its MXFP8 GEMMs, from 1024 rows on the gfx950 8-wave kernel with preshuffled
  activation and weight scales where aiter's tuned row is that kernel
  (``psc_layout``: the scale producer and the GEMM agree on it by shape);
* o_proj's activation quant as a HIP kernel (``linear_bf16``);
* a MoE layer's post-attention norm as one kernel: bf16 for the router and the
  routed experts, MXFP8 for the shared expert (``norm_moe_input``);
* the shared expert with SwiGLU-OAI fused into its MXFP8 quant, and the routed
  experts (``moe_ffn``). The pair (routed sum, shared output) goes to the next
  layer, whose input norm folds it; the stage's last layer and DSpark taps fold
  it themselves (``materialize``).

Every op is an opaque custom op: the model runs under torch.compile even with
--enforce-eager. Kernels: aiter's ``m3_tp1_prefill`` (``norm_mx``,
``mx8_quant``).
"""

import functools
import logging

import torch
from torch import nn
from torch.library import Library

from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

try:
    from aiter.ops import m3_tp1_prefill as _K
except ImportError:  # an aiter without the m3_tp1 ops
    _K = None

logger = logging.getLogger("atom")

_LIB = Library("atom_m3tp1", "FRAGMENT")
_FP8 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BF16 = torch.bfloat16
# The hidden size the fused norms are built and verified for (MiniMax-M3.1).
_HIDDEN = 6144
# Preshuffled scales from this M: below it the tuned rows are bmm / mxpsh kernels.
_PSC_MIN_M = 1024


def kernels_available() -> bool:
    return _K is not None and bool(_K.is_supported())


def _deployment(atom_config) -> bool:
    """The switch is on for a TP1 P/D prefill producer."""
    from atom.kv_transfer.disaggregation.pd_producer import pd_producer_configured

    return (
        bool(envs.ATOM_M3_TP1_PREFILL)
        and atom_config.tensor_parallel_size == 1
        and pd_producer_configured(atom_config)
    )


def path_enabled(atom_config) -> bool:
    """`_deployment` with aiter's ops (a compile cache key factor)."""
    return _deployment(atom_config) and kernels_available()


def model_enabled(atom_config, hidden_size: int) -> bool:
    """Whether a MiniMax-M3 model runs its stage's layers here: `path_enabled`,
    hidden 6144, and no DP, DP attention, PCP, Ulysses SP, indexer CP or plugin
    mode (whose attention forward is patched). Logs when only the ops miss."""
    from atom.distributed.indexer_cp import indexer_cp_enabled
    from atom.plugin.prepare import is_plugin_mode

    if not _deployment(atom_config):
        return False
    if not kernels_available():
        need = "aiter's m3_tp1_prefill ops" if _K is None else "a gfx950 GPU"
        logger.info(f"MiniMax-M3 TP1 prefill path off: it needs {need}")
        return False
    return (
        hidden_size == _HIDDEN
        and atom_config.parallel_config.data_parallel_size == 1
        and not atom_config.enable_dp_attention
        and atom_config.prefill_context_parallel_size == 1
        and atom_config.sequence_parallel_size == 1
        and not indexer_cp_enabled()
        and not is_plugin_mode()
    )


def linear_supported(linear) -> bool:
    """A native MXFP8 1x32 linear without bias, its weight stored preshuffled
    for aiter's group32 GEMM (gfx950: ATOM_GROUP32_WEIGHT_PRESHUFFLE)."""
    return (
        getattr(linear, "native_a8_group_rows", None) == 1
        and not getattr(linear, "runs_bf16", False)
        and getattr(linear, "params_dtype", None) == _FP8
        and getattr(linear, "bias", None) is None
        and bool(envs.ATOM_GROUP32_WEIGHT_PRESHUFFLE)
    )


def _module(name: str) -> nn.Module:
    from atom.config import get_current_atom_config

    return get_current_atom_config().compilation_config.static_forward_context[name]


# ---------------------------------------------------------------------------
# MXFP8 linears
# ---------------------------------------------------------------------------


def scale_rows(m: int) -> int:
    """Rows of an activation's e8m0 scales: padded to 256, as the w4 layout is."""
    return (m + 255) // 256 * 256


def shuffle_scale_w4(scale: torch.Tensor) -> torch.Tensor:
    """Row-major e8m0 [rows, K/32] -> the 1-D shuffle_scale_w4 layout over rows
    padded to 256 with 127, which the 8-wave GEMM loads straight into VGPRs."""
    rows, gpr = scale.shape
    u = scale.contiguous().view(torch.uint8)
    if scale_rows(rows) != rows:
        u = torch.cat([u, u.new_full((scale_rows(rows) - rows, gpr), 127)])
    u = u.view(-1, 2, 16, gpr // 8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return u.contiguous().view(-1).view(_E8M0)


def want_w4_scales(*linears) -> None:
    """Have these linears keep a `shuffle_scale_w4` copy of their weight scales
    from load on (``weight_scale_w4``, N x K/32 bytes each)."""
    for linear in linears:
        linear.mxfp8_w4_scales = True


@functools.lru_cache(maxsize=8192)
def _psc_row(m: int, n: int, k: int) -> tuple[int, int] | None:
    """(waves_per_eu, xcd_swizzle) of aiter's tuned row for this GEMM when it is
    an unsplit 8-wave MXFP8 tile, else None."""
    from aiter.ops import gemm_op_a8w8 as op
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import parse_mxfp8_8wave_kernel_name

    row = op.get_mxscale_bpreshuffle_config(m, n, k, "1x32")
    cfg = None if row is None else parse_mxfp8_8wave_kernel_name(str(row["kernelName"]))
    if cfg is None or cfg["n_split"] != 1:
        return None
    return cfg["waves_per_eu"], cfg["xcd_swizzle"]


def psc_layout(m: int, n: int, k: int) -> bool:
    """Whether an [m, k] activation for an n-row MXFP8 weight carries its scales
    in the shuffle_scale_w4 layout rather than row-major."""
    return m >= _PSC_MIN_M and k % 256 == 0 and _psc_row(m, n, k) is not None


def _empty_mxfp8(m: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 [m, k] and its e8m0 scale bytes [scale_rows(m), k/32]."""
    return (
        torch.empty((m, k), dtype=_FP8, device=device),
        torch.empty((scale_rows(m), k // 32), dtype=torch.uint8, device=device),
    )


def gemm_mxfp8(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_w4: torch.Tensor,
) -> torch.Tensor:
    """[m, k] e4m3 x the preshuffled MXFP8 weight -> bf16 [m, n]; ``xs`` holds
    the scales in the layout ``psc_layout`` names for the shape."""
    m, k = xq.shape
    n = weight.shape[0]
    if psc_layout(m, n, k):
        from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import flydsl_mxfp8_8wave_gemm

        waves_per_eu, xcd_swizzle = _psc_row(m, n, k)
        return flydsl_mxfp8_8wave_gemm(
            xq,
            weight,
            xs.view(-1).view(_E8M0),
            weight_scale_w4,
            torch.empty((m, n), dtype=_BF16, device=xq.device),
            waves_per_eu=waves_per_eu,
            xcd_swizzle=xcd_swizzle,
        )
    from aiter import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(
        xq, weight, xs[:m].view(_E8M0), weight_scale, dtype=_BF16, isBpreshuffled=True
    )


def quant_mxfp8(x: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`quantize_fp8` of a contiguous [m, k] bf16 activation for an n-row weight,
    as a HIP kernel."""
    m, k = x.shape
    xq, xs = _empty_mxfp8(m, k, x.device)
    if m:
        _K.mx8_quant(x, xq, xs, psc_layout(m, n, k))
    return xq, xs


def _linear_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_w4: torch.Tensor,
) -> torch.Tensor:
    xq, xs = quant_mxfp8(x.contiguous(), weight.shape[0])
    return gemm_mxfp8(xq, xs, weight, weight_scale, weight_scale_w4)


def _linear_bf16_fake(x, weight, weight_scale, weight_scale_w4):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=_BF16)


def _linear_q(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_w4: torch.Tensor,
) -> torch.Tensor:
    return gemm_mxfp8(xq, xs, weight, weight_scale, weight_scale_w4)


def _linear_q_fake(xq, xs, weight, weight_scale, weight_scale_w4):
    return xq.new_empty((xq.shape[0], weight.shape[0]), dtype=_BF16)


linear_bf16 = direct_register_custom_op(
    "linear_bf16", _linear_bf16, [], _linear_bf16_fake, target_lib=_LIB
)
linear_q = direct_register_custom_op(
    "linear_q", _linear_q, [], _linear_q_fake, target_lib=_LIB
)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def _norm_mxfp8(
    x: torch.Tensor,
    shared: torch.Tensor | None,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    n_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add + Gemma RMSNorm + MXFP8 quant for an n_out-row weight ->
    (residual out, e4m3, e8m0). With ``shared``, ``x`` is a MoE layer's routed
    sum and the added hidden state is ``x * 2 + shared``."""
    m, h = residual.shape
    res_out = torch.empty_like(residual)
    xq, xs = _empty_mxfp8(m, h, residual.device)
    if m:
        _K.norm_mx(
            _K.NORM_INPUT_MX8 if shared is None else _K.NORM_FOLD_MX8,
            x.contiguous(),
            shared,
            residual.contiguous(),
            weight,
            res_out,
            None,
            xq,
            xs,
            None,
            None,
            eps,
            m,
            h,
            psc_layout(m, n_out, h),
        )
    return res_out, xq, xs


def _norm_mxfp8_fake(x, shared, residual, weight, eps, n_out):
    m, h = residual.shape
    return (
        torch.empty_like(residual),
        residual.new_empty((m, h), dtype=_FP8),
        residual.new_empty((scale_rows(m), h // 32), dtype=torch.uint8),
    )


norm_mxfp8 = direct_register_custom_op(
    "norm_mxfp8", _norm_mxfp8, [], _norm_mxfp8_fake, target_lib=_LIB
)


def _norm_moe_input(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    n_out: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residual add + Gemma RMSNorm -> (bf16 for the router and the routed
    experts, residual out, MXFP8 e4m3 and e8m0 for the shared expert's n_out-row
    gate_up)."""
    m, h = residual.shape
    normed = torch.empty_like(residual)
    res_out = torch.empty_like(residual)
    q8, s8 = _empty_mxfp8(m, h, residual.device)
    if m:
        _K.norm_mx(
            _K.NORM_POST_BF16_MX8,
            x.contiguous(),
            None,
            residual.contiguous(),
            weight,
            res_out,
            normed,
            q8,
            s8,
            None,
            None,
            eps,
            m,
            h,
            psc_layout(m, n_out, h),
        )
    return normed, res_out, q8, s8


def _norm_moe_input_fake(x, residual, weight, eps, n_out):
    m, h = residual.shape
    return (
        torch.empty_like(residual),
        torch.empty_like(residual),
        residual.new_empty((m, h), dtype=_FP8),
        residual.new_empty((scale_rows(m), h // 32), dtype=torch.uint8),
    )


norm_moe_input = direct_register_custom_op(
    "norm_moe_input", _norm_moe_input, [], _norm_moe_input_fake, target_lib=_LIB
)


def _materialize(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """A MoE layer's hidden output ``bf16(routed * 2 + shared)`` (the stage's last
    layer, DSpark taps)."""
    m, h = routed.shape
    out = torch.empty_like(routed)
    if m:
        _K.norm_mx(
            _K.NORM_FOLD_H,
            routed,
            shared,
            None,
            None,
            None,
            out,
            None,
            None,
            None,
            None,
            0.0,
            m,
            h,
            False,
        )
    return out


def _materialize_fake(routed, shared):
    return torch.empty_like(routed)


materialize = direct_register_custom_op(
    "materialize", _materialize, [], _materialize_fake, target_lib=_LIB
)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


def mlp_forward(mlp, q8: torch.Tensor, s8: torch.Tensor) -> torch.Tensor:
    """MiniMaxM3MLP on an MXFP8 input: gate_up -> SwiGLU+MXFP8 -> down."""
    from atom.model_ops.minimax_m3.tp1_swiglu import (
        swiglu_oai_mxfp8,
        swiglu_oai_mxfp8_w4,
    )

    gu, dn = mlp.gate_up_proj, mlp.down_proj
    gate_up = gemm_mxfp8(q8, s8, gu.weight, gu.weight_scale, gu.weight_scale_w4)
    m, inter = gate_up.shape[0], gate_up.shape[1] // 2
    swiglu = (
        swiglu_oai_mxfp8_w4
        if psc_layout(m, dn.weight.shape[0], inter)
        else swiglu_oai_mxfp8
    )
    y, s = swiglu(gate_up, mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit)
    return gemm_mxfp8(y, s, dn.weight, dn.weight_scale, dn.weight_scale_w4)


def _mlp(q8: torch.Tensor, s8: torch.Tensor, layer_name: str) -> torch.Tensor:
    return mlp_forward(_module(layer_name), q8, s8)


def _mlp_fake(q8, s8, layer_name):
    return q8.new_empty(q8.shape, dtype=_BF16)


mlp = direct_register_custom_op("mlp", _mlp, [], _mlp_fake, target_lib=_LIB)


def _moe_ffn(
    x: torch.Tensor,
    q8: torch.Tensor,
    s8: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One MoE layer's (routed experts' top-k sum, shared expert's output); the
    routed scale (2) and the sum are the consumer's."""
    moe = _module(layer_name)
    shared = mlp_forward(moe.shared_experts, q8, s8)
    return moe.experts(hidden_states=x, router_logits=router_logits), shared


def _moe_ffn_fake(x, q8, s8, router_logits, layer_name):
    return torch.empty_like(x), torch.empty_like(x)


moe_ffn = direct_register_custom_op(
    "moe_ffn", _moe_ffn, [], _moe_ffn_fake, target_lib=_LIB
)
