# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bind the loaded Qwen tensors in place; Mono copies or relayouts none of them.

The production checkpoint uses BF16 hyper-connection/router weights and
``shuffle_weight((16, 16))`` FP8 expert weights with FP32 128x128 block
scales.  Keep this contract here rather than letting the kernels infer a
layout from a pointer: a different checkpoint layout must select a different
build instead of entering the same kernel through a runtime branch.
"""

from dataclasses import dataclass

import torch

from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.plan import LayerTask, TaskPlan
from atom.models.qwen4_exp_mono.shapes import (
    EXPERTS,
    FP8_BLOCK,
    GDN_CONV_DIM,
    GDN_CONV_WIDTH,
    GDN_HEAD_DIM,
    GDN_KEY_HEADS,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    HC,
    HC_RANK,
    HIDDEN,
    INTERMEDIATE,
    QSA_HEAD_DIM,
    QSA_INDEX_HEAD_DIM,
    QSA_INDEX_QK_SIZE,
    QSA_ROTARY_DIM,
    ROUTER_OUTPUTS,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.quant_spec import QuantType


@dataclass(frozen=True)
class HcWeights:
    norm: torch.Tensor
    down_inject: torch.Tensor
    up: torch.Tensor


@dataclass(frozen=True)
class MoeWeights:
    router: torch.Tensor
    shared_w13: torch.Tensor
    shared_w2: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    expert_map: torch.Tensor | None


@dataclass(frozen=True)
class GdnWeights:
    """Zero-copy views of one linear-attention layer's BF16 GDN tensors."""

    qkvzba: torch.Tensor
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    out: torch.Tensor


@dataclass(frozen=True)
class QsaWeights:
    """QSA K1/K2 tensors; K/V caches live in the per-step view, not here."""

    qkv: torch.Tensor
    o_proj: torch.Tensor
    index_qk: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    index_q_norm: torch.Tensor
    index_k_norm: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    index_cos: torch.Tensor
    index_sin: torch.Tensor


@dataclass(frozen=True)
class LayerWeights:
    attn_hc: HcWeights
    mlp_hc: HcWeights
    moe: MoeWeights
    gdn: GdnWeights | None
    qsa: QsaWeights | None


def _tensor(
    module,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    shuffled: bool = False,
) -> torch.Tensor:
    value = module
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            raise MonoUnsupported(f"no {name}")
    if tuple(value.shape) != shape:
        raise MonoUnsupported(f"{name} is {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise MonoUnsupported(f"{name} is {value.dtype}, expected {dtype}")
    if not value.is_contiguous():
        raise MonoUnsupported(f"{name} is not contiguous")
    if shuffled and not getattr(value, "is_shuffled", False):
        raise MonoUnsupported(f"{name} is not shuffle_weight(16,16) storage")
    if value.device.type != "meta" and value.data_ptr() % 16:
        raise MonoUnsupported(f"{name} is not 16-byte aligned")
    return value


def _bind_tensors(module, specs, dtype=torch.bfloat16, *, shuffled=False):
    """Resolve one homogeneous native storage group without copying tensors."""
    return {
        field: _tensor(module, path, shape, dtype, shuffled=shuffled)
        for field, (path, shape) in specs.items()
    }


def _bind_hc(module, *, has_inject=True) -> HcWeights:
    hc_dim = HC * HIDDEN
    return HcWeights(
        **_bind_tensors(
            module,
            {
                "norm": ("hc_norm.weight", (hc_dim,)),
                "down_inject": (
                    "fused_w_cat",
                    (HC_RANK + (HC if has_inject else 0), hc_dim),
                ),
                "up": ("input_mix_weight_up.weight", (hc_dim, HC_RANK)),
            },
        )
    )


def _bind_gdn(layer, task: LayerTask, dims: Dims) -> GdnWeights | None:
    if not task.is_gdn:
        return None
    attn = getattr(layer, "linear_attn", None)
    if attn is None:
        raise MonoUnsupported("GDN layer has no linear_attn")
    if getattr(attn, "quantized_inputs", True):
        raise MonoUnsupported("Qwen mono GDN requires the packed BF16 projection")
    if (
        getattr(attn, "num_k_heads", None),
        getattr(attn, "num_v_heads", None),
        getattr(attn, "head_k_dim", None),
        getattr(attn, "head_v_dim", None),
        getattr(attn, "conv_kernel_size", None),
    ) != (
        GDN_KEY_HEADS,
        GDN_VALUE_HEADS,
        GDN_HEAD_DIM,
        GDN_HEAD_DIM,
        GDN_CONV_WIDTH,
    ):
        raise MonoUnsupported("Qwen mono GDN dimensions do not match the S1 build")
    if getattr(attn, "conv_dim", None) != GDN_CONV_DIM:
        raise MonoUnsupported("Qwen mono GDN convolution width does not match")
    if getattr(attn, "value_dim", None) != GDN_VALUE_DIM:
        raise MonoUnsupported("Qwen mono GDN value width does not match")
    return GdnWeights(
        **_bind_tensors(
            attn,
            {
                "qkvzba": ("in_proj_qkvzba.weight", (dims.gdn_proj_dim, HIDDEN)),
                "conv": ("conv1d.weight", (dims.gdn_conv_dim, 1, GDN_CONV_WIDTH)),
                "a_log": ("A_log", (dims.gdn_value_heads,)),
                "dt_bias": ("dt_bias", (dims.gdn_value_heads,)),
                "norm": ("norm.weight", (GDN_HEAD_DIM,)),
                "out": ("out_proj.weight", (HIDDEN, dims.gdn_value_dim)),
            },
        )
    )


def _bind_projection(module, shape: tuple[int, int]) -> torch.Tensor:
    """Bind the BF16 QSA storage used by the production checkpoint."""

    return _tensor(module, "weight", shape, torch.bfloat16)


def _bind_rope(rope, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    caches = []
    for leaf in ("cos_cache", "sin_cache"):
        value = getattr(rope, leaf, None)
        if not isinstance(value, torch.Tensor):
            raise MonoUnsupported(f"Qwen mono {name} has no {leaf}")
        if value.dtype != torch.bfloat16:
            raise MonoUnsupported(f"Qwen mono {name}.{leaf} is {value.dtype}")
        if (
            tuple(value.shape[1:])
            not in ((QSA_ROTARY_DIM // 2,), (1, 1, QSA_ROTARY_DIM // 2))
            or not value.is_contiguous()
        ):
            raise MonoUnsupported(
                f"Qwen mono {name}.{leaf} is {tuple(value.shape)} {value.stride()}"
            )
        # Native RoPE uses [positions, 1, 1, rotary/2]; both admitted views
        # have exactly rotary/2 contiguous values per position.
        caches.append(value)
    return caches[0], caches[1]


def _bind_qsa(layer, task: LayerTask, dims: Dims) -> QsaWeights | None:
    if task.is_gdn:
        return None
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise MonoUnsupported("QSA layer has no self_attn")
    indexer = getattr(attn, "indexer", None)
    if indexer is None:
        raise MonoUnsupported("QSA layer has no indexer")
    cos, sin = _bind_rope(attn.rotary_emb, "QSA RoPE")
    index_cos, index_sin = _bind_rope(indexer.rotary_emb, "QSA index RoPE")
    return QsaWeights(
        qkv=_bind_projection(attn.qkv_proj, (dims.qsa_qkv_size, HIDDEN)),
        o_proj=_bind_projection(attn.o_proj, (HIDDEN, dims.qsa_q_size)),
        index_qk=_bind_projection(
            indexer.index_qk_proj,
            (QSA_INDEX_QK_SIZE, HIDDEN),
        ),
        q_norm=_tensor(attn, "q_norm.weight", (QSA_HEAD_DIM,), torch.bfloat16),
        k_norm=_tensor(attn, "k_norm.weight", (QSA_HEAD_DIM,), torch.bfloat16),
        index_q_norm=_tensor(
            indexer, "q_layernorm.weight", (QSA_INDEX_HEAD_DIM,), torch.bfloat16
        ),
        index_k_norm=_tensor(
            indexer, "k_layernorm.weight", (QSA_INDEX_HEAD_DIM,), torch.bfloat16
        ),
        cos=cos,
        sin=sin,
        index_cos=index_cos,
        index_sin=index_sin,
    )


def _bind_expert_map(experts, dims: Dims) -> torch.Tensor | None:
    expert_map = getattr(experts, "expert_map", None)
    if dims.tp == 1:
        if expert_map is not None:
            raise MonoUnsupported("TP1 routed experts unexpectedly have expert_map")
        return None
    if not getattr(experts, "use_ep", False):
        raise MonoUnsupported("TP8 routed experts are not expert-owned")
    if getattr(experts, "ep_size", None) != dims.tp:
        raise MonoUnsupported("TP8 routed expert EP size does not match TP")
    if getattr(experts, "tp_size", None) != 1:
        raise MonoUnsupported("TP8 routed experts still shard their intermediate")
    if getattr(experts, "local_num_experts", None) != dims.local_experts:
        raise MonoUnsupported(
            "TP8 local expert count is "
            f"{getattr(experts, 'local_num_experts', None)}, "
            f"expected {dims.local_experts}"
        )
    if not isinstance(expert_map, torch.Tensor):
        raise MonoUnsupported("TP8 routed experts have no expert_map")
    if (
        tuple(expert_map.shape) not in ((EXPERTS,), (EXPERTS + 1,))
        or expert_map.dtype != torch.int32
        or not expert_map.is_contiguous()
    ):
        raise MonoUnsupported(
            "TP8 expert_map is "
            f"{tuple(expert_map.shape)} {expert_map.dtype} {expert_map.stride()}"
        )
    # Native AITER appends one non-local-token sentinel.  Mono routing only
    # indexes the 512 global routed expert ids, so bind the exact ABI prefix.
    return expert_map[:EXPERTS]


def _bind_moe(layer, dims: Dims) -> MoeWeights:
    """Validate native shared/owned expert storage and bind it without copying."""

    moe = layer.mlp
    experts = moe.experts
    if moe.fuse_shared or moe.shared_expert is None:
        raise MonoUnsupported("Qwen mono requires the separate BF16 shared expert")
    if experts.num_fused_shared_experts != 0:
        raise MonoUnsupported(
            "routed expert storage unexpectedly contains "
            f"{experts.num_fused_shared_experts} shared slots"
        )
    if experts.intermediate_size_per_partition != INTERMEDIATE:
        raise MonoUnsupported(
            "expert intermediate is "
            f"{experts.intermediate_size_per_partition}, expected {INTERMEDIATE}"
        )
    quant = experts.quant_method
    if not getattr(quant, "block_quant", False):
        raise MonoUnsupported("expert weights are not block-quantized")
    if getattr(quant, "quant_type", None) != QuantType.per_1x128:
        raise MonoUnsupported("expert weights are not FP8 128x128 block-scale")
    if getattr(experts, "w13_input_scale", None) is not None:
        raise MonoUnsupported("G1 activation quantization is not dynamic")
    if getattr(experts, "w2_input_scale", None) is not None:
        raise MonoUnsupported("G2 activation quantization is not dynamic")

    bf16 = _bind_tensors(
        moe,
        {
            "router": ("gate.weight", (ROUTER_OUTPUTS, HIDDEN)),
            "shared_w13": (
                "shared_expert.gate_up_proj.weight",
                (2 * dims.shared_intermediate, HIDDEN),
            ),
            "shared_w2": (
                "shared_expert.down_proj.weight",
                (HIDDEN, dims.shared_intermediate),
            ),
        },
    )
    expert_shapes = {
        "w13": (dims.local_experts, 2 * dims.routed_intermediate, HIDDEN),
        "w2": (dims.local_experts, HIDDEN, dims.routed_intermediate),
    }
    fp8 = _bind_tensors(
        moe,
        {
            name: (f"experts.{name}_weight", shape)
            for name, shape in expert_shapes.items()
        },
        torch.float8_e4m3fn,
        shuffled=True,
    )
    scales = _bind_tensors(
        moe,
        {
            name
            + "_scale": (
                f"experts.{name}_weight_scale",
                (count, rows // FP8_BLOCK, cols // FP8_BLOCK),
            )
            for name, (count, rows, cols) in expert_shapes.items()
        },
        torch.float32,
    )
    return MoeWeights(
        **bf16, **fp8, **scales, expert_map=_bind_expert_map(experts, dims)
    )


def bind_layer(layer, task: LayerTask, tp: int = 1) -> LayerWeights:
    dims = Dims(tp)
    attn_hc = _bind_hc(layer.attn_hyper_connection)
    mlp_hc = _bind_hc(layer.mlp_hyper_connection)
    gdn = _bind_gdn(layer, task, dims)
    qsa = _bind_qsa(layer, task, dims)

    moe = _bind_moe(layer, dims)
    return LayerWeights(attn_hc, mlp_hc, moe, gdn, qsa)


def bind_model(model, plan: TaskPlan, tp: int = 1) -> tuple[LayerWeights, ...]:
    layers = model.model.layers
    return tuple(
        bind_layer(layer, task, tp)
        for layer, task in zip(layers, plan.layers, strict=True)
    )
