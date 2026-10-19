# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The loaded layer tensors the mono kernels read in place.

Nothing is copied or re-laid-out: a kernel reads a parameter where the original
model keeps it, so a layout the kernels were not written for must be refused here
rather than read wrong. Every tensor is checked against the shape and dtype the
kernels assume for the rank's shard.
"""

from dataclasses import dataclass

import torch

from atom.models.deepseek_v41.config import AttentionMode
from atom.mono.plan.shard import Shard
from atom.mono.runtime.consensus import MonoUnsupported


@dataclass(frozen=True)
class LayerWeights:
    """One decoder block's tensors, by the parameter name the block gives them."""

    layer_id: int
    mode: AttentionMode
    tensors: dict[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]


F32, BF16 = torch.float32, torch.bfloat16
FP8, E8M0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
FP4X2 = torch.float4_e2m1fn_x2
# e8m0 scales per 32x32 block of an FP8 weight
BLOCK = 32
# a routed expert's intermediate width is padded to the 128-k MXFP4 step
MOE_PAD = 128


def _fp8(rows: int, cols: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        "weight": ((rows, cols), FP8),
        "weight_scale": ((rows // BLOCK, cols // BLOCK), E8M0),
    }


def expected(
    config, mode: AttentionMode, tp: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """(shape, dtype) of the TP ``tp`` shard of every tensor a layer in ``mode``
    has, by parameter name."""
    hidden, hc = config.hidden_size, config.hc_mult
    shard = Shard(tp)
    heads = shard.split(config.num_attention_heads, "query heads")
    groups = shard.split(config.o_groups, "wo_a groups")
    q_rank, head_dim, o_rank = config.q_lora_rank, config.head_dim, config.o_lora_rank
    experts = config.n_routed_experts
    inter = shard.split(config.moe_intermediate_size, "routed intermediate")
    inter_padded = -(-inter // MOE_PAD) * MOE_PAD
    mix = hc * (hc + 2)
    table = {
        "attn_norm.weight": ((hidden,), BF16),
        "ffn_norm.weight": ((hidden,), BF16),
        "attn.attn_sink": ((heads,), F32),
        "attn.q_norm.weight": ((q_rank,), BF16),
        "attn.kv_norm.weight": ((head_dim,), BF16),
        "ffn.gate.weight": ((experts, hidden), BF16),
        "ffn.gate.e_score_correction_bias": ((experts,), F32),
        # MXFP4, K packed two a byte; scales in aiter's shuffle_scale order
        "ffn.experts.w13_weight": ((experts, 2 * inter_padded, hidden // 2), FP4X2),
        "ffn.experts.w2_weight": ((experts, hidden, inter_padded // 2), FP4X2),
    }
    linears = {
        "attn.wqkv_a": (q_rank + head_dim, hidden),
        "attn.wq_b": (heads * head_dim, q_rank),
        "attn.wo_a": (
            groups * o_rank,
            config.num_attention_heads * head_dim // config.o_groups,
        ),
        "attn.wo_b": (hidden, groups * o_rank),
        "ffn.shared_experts.gate_up_proj": (2 * inter, hidden),
        "ffn.shared_experts.w2": (hidden, inter),
    }
    if mode in (AttentionMode.FULL, AttentionMode.REINDEX):
        index_heads, index_dim = config.index_n_heads, config.index_head_dim
        linears["attn.indexer.wq_b"] = (index_heads * index_dim, q_rank)
        table["attn.indexer.weights_proj.weight"] = ((index_heads, hidden), BF16)
    for prefix, (rows, cols) in linears.items():
        for suffix, entry in _fp8(rows, cols).items():
            table[f"{prefix}.{suffix}"] = entry
    for sublayer in ("attn", "ffn"):
        table[f"hc_{sublayer}_fn"] = ((mix, hc * hidden), F32)
        table[f"hc_{sublayer}_base"] = ((mix,), F32)
        table[f"hc_{sublayer}_scale"] = ((3,), F32)
    return table


def bind_layer(block, config, tp: int) -> LayerWeights:
    """``block``'s parameters, each checked against ``expected``."""
    spec = block.attn.spec
    method = block.ffn.experts.quant_method
    # K2b runs the original path's A8W4: w13 gate / up interleaved a 16-row
    # tile, MXFP8 activations
    if not (
        getattr(method, "is_guinterleave", False)
        and getattr(method, "fp8_activations", False)
    ):
        raise MonoUnsupported(
            f"layer {spec.layer_id}: routed experts not A8W4 (interleaved gate/up "
            "rows, MXFP8 activations)"
        )
    params = dict(block.named_parameters())
    for name, (shape, dtype) in expected(config, spec.mode, tp).items():
        tensor = params.get(name)
        if tensor is None:
            raise MonoUnsupported(f"layer {spec.layer_id}: no {name}")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise MonoUnsupported(
                f"layer {spec.layer_id}: {name} is {tuple(tensor.shape)} "
                f"{tensor.dtype}, expected {shape} {dtype}"
            )
        if not tensor.is_contiguous():
            raise MonoUnsupported(f"layer {spec.layer_id}: {name} not contiguous")
    return LayerWeights(spec.layer_id, spec.mode, params)


def bind_model(model, tp: int) -> list[LayerWeights]:
    """Every decoder block of the V4.1 runtime model, in layer order."""
    return [bind_layer(block, model.config, tp) for block in model.layers]
