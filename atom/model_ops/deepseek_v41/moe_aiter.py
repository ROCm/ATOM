# SPDX-License-Identifier: MIT
"""V4.1 adaptation of existing AITER A8W4 expert GEMMs and top-k sorting.

AITER owns device kernels. This adapter owns native weight views, EP metadata,
V4.1's weighted activation boundary and the FP32 local expert sum. The model
owns router selection, shared experts and communication.
"""

import torch
from aiter.ops.triton.fusions.fused_routing_from_topk import fused_routing_from_topk
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData

from atom.model_ops.blockscale import quantize_fp8

from .moe import weighted_swiglu

# Bounds intermediates and stays within AITER's 4096-route sorter budget for
# the published top-6 model. Splitting tokens never splits an expert's K axis.
MAX_TOKENS = 512


def _metadata(hist, gates):
    """Build AITER's block16 expert schedule without host counts or scratch state."""
    zero = torch.zeros(1, dtype=torch.int32, device=hist.device)
    raw = torch.cat((zero, hist.cumsum(0, dtype=torch.int32)))
    tiles = torch.cat((zero, ((hist + 15) // 16).cumsum(0, dtype=torch.int32)))
    experts = hist.numel()
    capacity = (
        gates if gates <= experts else experts - 1 - ((experts - gates - 1) // 16)
    )
    tile = torch.arange(capacity, dtype=torch.int32, device=hist.device)
    expert = torch.searchsorted(
        tiles[1:].contiguous(), tile, right=True, out_int32=True
    )
    block = ((tile - tiles[expert]) << 16) + expert
    block = torch.where(tile < tiles[-1], block, -1)
    return ExptData(hist, raw, tiles, block)


def _assemble(modules, projections, attribute):
    """Move native bytes into an arena; original parameters become shared views."""
    tensors = [
        [getattr(getattr(module, name), attribute) for name in projections]
        for module in modules
    ]
    rows = sum(tensor.shape[0] for tensor in tensors[0])
    sample = tensors[0][0]
    arena = torch.empty(
        (len(modules), rows, sample.shape[1]), device=sample.device, dtype=torch.uint8
    )
    for expert, parameters in enumerate(tensors):
        begin = 0
        for parameter in parameters:
            end = begin + parameter.shape[0]
            view = arena[expert, begin:end]
            view.copy_(parameter.view(torch.uint8))
            parameter.data = view.view(parameter.dtype)
            begin = end
    return arena.transpose(-1, -2)


class AiterExperts:
    max_tokens = MAX_TOKENS

    def __init__(self, experts, num_experts):
        names = sorted(int(name) for name in experts)
        if (
            not names
            or names != list(range(names[0], names[0] + len(names)))
            or names[0] < 0
            or names[-1] >= num_experts
        ):
            raise ValueError("AITER experts require contiguous whole-expert ownership")
        self.first, self.count = names[0], len(names)
        modules = [experts[str(i)] for i in names]
        sample = modules[0]
        self.hidden = sample.w1.weight.shape[1] * 2
        self.intermediate = sample.w1.weight.shape[0]
        self.limit = sample.swiglu_limit
        device = sample.w1.weight.device
        if self.hidden % 32 or self.intermediate % 32:
            raise ValueError("AITER experts require group32-aligned dimensions")
        for expert in modules:
            if expert.swiglu_limit != self.limit:
                raise ValueError("Expert activation limits must agree")
            for name, shape in (
                ("w1", (self.intermediate, self.hidden // 2)),
                ("w3", (self.intermediate, self.hidden // 2)),
                ("w2", (self.hidden, self.intermediate // 2)),
            ):
                projection = getattr(expert, name)
                if (
                    projection.weight.dtype != torch.float4_e2m1fn_x2
                    or tuple(projection.weight.shape) != shape
                    or not projection.weight.is_contiguous()
                    or projection.weight_scale.dtype != torch.float8_e8m0fnu
                    or not projection.weight_scale.is_contiguous()
                    or tuple(projection.weight_scale.shape)
                    != (shape[0], shape[1] // 16)
                    or projection.weight.device != device
                    or projection.weight_scale.device != device
                ):
                    raise ValueError(
                        "AITER experts require native contiguous W4A8 group32 weights"
                    )
        # Remote routes sort into a final sentinel bin, which is excluded from
        # GEMM metadata. They consume no expert GEMM work on this rank.
        self.expert_map = torch.full(
            (num_experts,), self.count, device=device, dtype=torch.int32
        )
        self.expert_map[self.first : self.first + self.count] = torch.arange(
            self.count, device=device, dtype=torch.int32
        )
        self.up = _assemble(modules, ("w1", "w3"), "weight")
        self.up_scale = _assemble(modules, ("w1", "w3"), "weight_scale")
        self.down = _assemble(modules, ("w2",), "weight")
        self.down_scale = _assemble(modules, ("w2",), "weight_scale")

    def __call__(self, hidden, weights, indices):
        tokens, dim = hidden.shape
        topk = indices.shape[-1]
        if (
            dim != self.hidden
            or hidden.dtype != torch.bfloat16
            or weights.shape != indices.shape
            or indices.shape[0] != tokens
            or not 1 <= topk <= self.expert_map.numel()
        ):
            raise ValueError("Invalid AITER expert inputs")
        if not tokens:
            return torch.zeros_like(hidden, dtype=torch.float32)
        chunk = min(MAX_TOKENS, 4096 // topk)
        if tokens > chunk:
            return torch.cat(
                [
                    self(
                        hidden[i : i + chunk],
                        weights[i : i + chunk],
                        indices[i : i + chunk],
                    )
                    for i in range(0, tokens, chunk)
                ]
            )
        gates = tokens * topk
        hist, gather, scatter, sorted_weights = fused_routing_from_topk(
            weights.contiguous(),
            indices.to(torch.int32).contiguous(),
            self.count + 1,
            self.expert_map,
        )
        hist = hist[: self.count]
        metadata = _metadata(hist, gates)
        routing = RoutingData(16, sorted_weights, hist, self.count, topk, metadata)
        x, scales = quantize_fp8(hidden)
        up = moe_gemm_a8w4(
            x,
            self.up,
            scales.view(torch.uint8),
            self.up_scale,
            routing_data=routing,
            gather_indx=gather,
            out_dtype=torch.bfloat16,
        )
        live = torch.arange(gates, device=hidden.device) < metadata.token_offs_raw[-1]
        up = torch.where(live[:, None], up, 0)
        activation = weighted_swiglu(
            up[:, : self.intermediate],
            up[:, self.intermediate :],
            torch.where(live, sorted_weights, 0)[:, None],
            limit=self.limit,
        )
        x, scales = quantize_fp8(activation)
        down = moe_gemm_a8w4(
            x,
            self.down,
            scales.view(torch.uint8),
            self.down_scale,
            routing_data=routing,
            out_dtype=torch.bfloat16,
        )
        valid = (indices >= self.first) & (indices < self.first + self.count)
        safe = torch.where(valid.flatten(), scatter, 0).long()
        output = torch.where(valid.flatten()[:, None], down[safe], 0).view(
            tokens, topk, dim
        )
        order = indices.argsort(-1)
        output = output.gather(1, order[:, :, None].expand(-1, -1, dim))
        result = torch.zeros_like(hidden, dtype=torch.float32)
        for choice in range(topk):
            result = result + output[:, choice].float()
        return result
