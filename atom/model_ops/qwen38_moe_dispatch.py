"""Shape-specific gfx942 EP MoE dispatch for channel-scaled FP8 weights."""

import os

import torch
from aiter import ActivationType, QuantType
from aiter.jit.utils.chip_info import get_gfx

_ENABLE_SMALL_DECODE = os.environ.get("ATOM_QWEN38_MOE_SMALL_DECODE", "1") == "1"


def maybe_small_decode(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None,
    activation: ActivationType,
    quant_type: QuantType,
    doweight_stage1: bool,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    swiglu_limit: float | None,
    gate_mode: str,
    scheduled_tokens: torch.Tensor | None,
) -> torch.Tensor | None:
    # Real EP routes favor native decode at these verification sizes. Larger
    # batches, prefill, and unsupported contracts keep the existing CK/ASM
    # dispatcher. Weight storage is shuffled by the compressed-FP8 loader.
    if not _ENABLE_SMALL_DECODE:
        return None
    if (
        hidden_states.shape[0] not in (3, 6, 12)
        or hidden_states.shape[1] != 2560
        or tuple(w1.shape) != (257, 1280, 2560)
        or tuple(w2.shape) != (257, 2560, 640)
        or tuple(topk_ids.shape) != (hidden_states.shape[0], 12)
        or topk_weights.shape != topk_ids.shape
        or topk_weights.dtype != torch.float32
        or topk_ids.dtype != torch.int32
        or hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.float8_e4m3fnuz
        or w2.dtype != w1.dtype
        or quant_type != QuantType.per_Token
        or activation != ActivationType.Silu
        or doweight_stage1
        or a1_scale is not None
        or a2_scale is not None
        or swiglu_limit not in (None, 0.0)
        or gate_mode != "separated"
        or scheduled_tokens is not None
        or expert_mask is None
        or tuple(expert_mask.shape) != (514,)
        or expert_mask.device != hidden_states.device
        or expert_mask.dtype != torch.int32
        or w1_scale is None
        or w2_scale is None
        or tuple(w1_scale.shape) != (257, 1280, 1)
        or tuple(w2_scale.shape) != (257, 2560, 1)
        or w1_scale.dtype != torch.float32
        or w2_scale.dtype != torch.float32
        or get_gfx() != "gfx942"
    ):
        return None
    if not all(
        tensor.is_contiguous()
        for tensor in (
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
            expert_mask,
        )
    ):
        return None

    from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942

    # The sort maps global routes to local experts and drops remote routes.
    # Keep the original ids/weights: no remap allocation, kernel, or host sync.
    return run_flydsl_moe_gfx942(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        quant_type,
        w1_scale,
        w2_scale,
        expert_mask,
        None,
        0,
        "16_16_16_False:default:0",
        swiglu_limit,
    )
