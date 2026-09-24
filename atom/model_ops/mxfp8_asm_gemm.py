# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MXFP8 ASM GEMM for three DeepSeek-V4 attention GEMMs at M >= 512.

Opt-in via ATOM_DSV4_USE_GFX1250_MXFP8_ASM_GEMM=1 (default off: nothing here runs).

    attn.wq_b          (N, K) = (65536, 1536)   A from q_norm (fused RMSNorm + quant)
    attn.indexer.wq_b  (N, K) = ( 8192, 1536)   same q_norm output
    attn.wo_b          (N, K) = ( 7168, 16384)  A is BF16, quantized here

Only these attention-TP1 shapes are flagged, at load. At run time a physical
M >= 512 takes the ASM GEMM (aiter.gemm_a8w8_mxfp8, kernel/split-K from aiter's
tuned CSV) with MXFP8 1x32 activations; a smaller M runs exactly today's FlyDSL
blockscale path (same kernels, same scale bytes).

The M test sits inside opaque custom ops, so torch.compile (one graph for a
dynamic num_tokens) never sees it, and their output shapes are one formula for
every M. q_norm therefore always returns a (pad32(M), K/32) e8m0 buffer: the ASM
A-scale layout (shuffle_mxfp8fp4_scale bytes) for M >= 512, and today's
column-major per_1x128 scale in its first M*K/128 bytes otherwise. Its two
consumers (attn.wq_b, attn.indexer.wq_b) are flagged together with it.
"""

import torch
from aiter import QuantType, dtypes
from aiter.jit.utils.torch_guard import torch_compile_guard

# Newer aiter APIs are imported where used, so importing this module (linear.py
# and layernorm.py do) needs nothing beyond what ATOM already requires.

ASM_MIN_M = 512
WQ_B_SHAPE = (65536, 1536)
INDEXER_WQ_B_SHAPE = (8192, 1536)
WO_B_SHAPE = (7168, 16384)


def asm_weight_scale(weight_scale: torch.Tensor, n: int) -> torch.Tensor:
    """128x128 e8m0 block scale [N/128, K/128] -> ASM B-scale [N, K/32]."""
    from aiter.ops.shuffle import shuffle_mxfp8fp4_scale

    ws = weight_scale.view(torch.uint8)
    ws = ws.repeat_interleave(128, dim=0)[:n].repeat_interleave(4, dim=1)
    return shuffle_mxfp8fp4_scale(ws.contiguous()).view(dtypes.fp8_e8m0)


def _rms_quant_fake(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    out = torch.empty((m, k), dtype=dtypes.fp8, device=x.device)
    m_pad = (m + 31) // 32 * 32
    scale = torch.empty((m_pad, k // 32), dtype=dtypes.fp8_e8m0, device=x.device)
    return out, scale


@torch_compile_guard(gen_fake=_rms_quant_fake, mutates_args=[])
def dsv4_mxfp8_asm_rms_quant(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import rmsnorm_quant

    out, scale = _rms_quant_fake(x, weight, eps)
    m, k = x.shape
    if m >= ASM_MIN_M:
        rmsnorm_quant(out, x, scale, weight, eps, 32, False, scale_layout_m32k4=True)
    else:  # today's per_1x128 column-major scale, in the front of the buffer
        g = k // 128
        rmsnorm_quant(
            out, x, scale.view(-1)[: m * g].view(m, g), weight, eps, 128, True
        )
    return out, scale


def _gemm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_asm: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((x.shape[0], weight.shape[0]), dtype=dtype, device=x.device)


@torch_compile_guard(gen_fake=_gemm_fake, mutates_args=[])
def dsv4_mxfp8_asm_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_asm: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """x_scale None: BF16 x (wo_b), quantized here; else q_norm's (x, buffer)."""
    from aiter import gemm_a8w8_blockscale_bpreshuffle
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8
    from aiter.ops.quant import per_group_quant_hip

    m, k = x.shape
    if m >= ASM_MIN_M:
        if x_scale is None:
            x, x_scale = per_group_quant_hip(
                x.contiguous(),
                quant_dtype=dtypes.fp8,
                group_size=32,
                scale_type=dtypes.fp8_e8m0,
                scale_layout_m32k4=True,
            )
        return gemm_a8w8_mxfp8(
            x, weight, x_scale, weight_scale_asm, dtype=dtype, a_preshuffle=False
        )
    if x_scale is None:  # today's LinearBase per_1x128 quant (transposed, e8m0)
        x, x_scale = per_group_quant_hip(
            x,
            quant_dtype=dtypes.fp8,
            group_size=128,
            transpose_scale=True,
            scale_type=dtypes.fp8_e8m0,
        )
    else:
        g = k // 128
        x_scale = x_scale.view(-1)[: m * g].view(m, g)
    return gemm_a8w8_blockscale_bpreshuffle(x, weight, x_scale, weight_scale, dtype)


def _linear_ok(layer, shape: tuple[int, int]) -> bool:
    ws = getattr(layer, "weight_scale", None)
    return (
        ws is not None
        and layer.quant_type.value == QuantType.per_1x128.value
        and layer.weight.dtype == dtypes.fp8
        and tuple(layer.weight.shape) == shape
        and ws.dtype == dtypes.fp8_e8m0
        and layer.bias is None
        and getattr(layer, "input_scale", None) is None
    )


def setup(attn) -> None:
    """Flag DeepseekV4Attention's eligible layers; runs from its load hook.

    Parents run before children, so each flagged linear then builds its ASM
    weight scale in its own load hook. q_norm, wq_b and indexer.wq_b are flagged
    together or not at all; wo_b is independent.
    """
    import aiter.ops.quant as aiter_quant

    from atom.utils import envs

    if not (
        attn._is_gfx1250
        and envs.ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE
        and envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        and getattr(aiter_quant, "SCALE_LAYOUT_M32K4_SUPPORTED", False)
    ):
        return
    qc = attn.q_norm.quant_config
    if qc is not None and qc.online_quant:
        return
    q_norm, idx = attn.q_norm, getattr(attn.indexer, "wq_b", None)
    if (
        q_norm.use_fused_quant
        and q_norm._aiter_transpose_scale
        and not q_norm.fused_allreduce
        and _linear_ok(attn.wq_b, WQ_B_SHAPE)
        and (idx is None or _linear_ok(idx, INDEXER_WQ_B_SHAPE))
    ):
        for layer in (q_norm, attn.wq_b, idx):
            if layer is not None:
                layer._mxfp8_asm = True
    if _linear_ok(attn.wo_b, WO_B_SHAPE):
        attn.wo_b._mxfp8_asm = True
