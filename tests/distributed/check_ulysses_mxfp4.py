# SPDX-License-Identifier: MIT
"""Compare SP input prequantization with BF16 gather followed by real MoE.

Run with torchrun --nproc-per-node=4 on gfx950. Includes changed-input graph
replay and a small-batch fallback check. No model checkpoint is required.
"""

import os

import torch
from aiter import ActivationType, QuantType, dtypes, get_hip_quant
from aiter.dist import parallel_state as ps
from aiter.fused_moe import fused_moe

from atom.distributed import ulysses_sp as sp
from atom.model_ops.moe import Mxfp4MoEMethod


def same_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ps.init_distributed_environment(world_size=world, rank=rank)
    ps.initialize_model_parallel(prefill_context_model_parallel_size=world)
    sp.set_sp_world_size(world)
    torch.manual_seed(170 + rank)

    method = object.__new__(Mxfp4MoEMethod)
    method.quant_type = QuantType.per_1x32
    method.use_triton = method.use_triton_ep = method.is_guinterleave = False
    method.fused_experts = None
    method.hidden_pad = method.intermediate_pad = 0
    layer = torch.nn.Module()
    layer.use_ep = layer.apply_router_weight_on_input = False
    layer.custom_routing_function = layer.w13_input_scale = None
    layer.w13_bias = layer.w2_bias = None
    layer.activation = ActivationType.Swiglu
    layer.swiglu_limit = 10.0
    layer.hidden_size, intermediate, experts = 6144, 768, 129
    layer.top_k, layer.num_fused_shared_experts = 4, 1
    hidden, topk = layer.hidden_size, layer.top_k + layer.num_fused_shared_experts
    layer.w13_weight = torch.randint(
        256, (experts, 2 * intermediate, hidden // 2), device="cuda", dtype=torch.uint8
    ).view(dtypes.fp4x2)
    layer.w2_weight = torch.randint(
        256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8
    ).view(dtypes.fp4x2)
    layer.w13_weight.is_shuffled = layer.w2_weight.is_shuffled = True
    s1 = torch.full(
        (experts, 2 * intermediate, hidden // 32), 120, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    s2 = torch.full(
        (experts, hidden, intermediate // 32), 120, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    quant = get_hip_quant(QuantType.per_1x32)
    for tokens in (16, 8192, 32768):
        x = torch.randn(tokens // world, hidden, dtype=torch.bfloat16, device="cuda")
        gathered = sp.sp_moe_gather(x)
        actual, scale = method.gather_sp_input(layer, x)
        if tokens == 16:
            assert scale is None
            same_bytes(actual, gathered)
            continue
        reference, reference_scale = quant(gathered, quant_dtype=dtypes.fp4x2)
        same_bytes(actual, reference)
        same_bytes(scale, reference_scale)
        ids = (
            torch.rand(tokens, experts - 1, device="cuda").topk(topk - 1).indices.int()
        )
        ids = torch.cat((ids, torch.full_like(ids[:, :1], experts - 1)), dim=1)
        weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), dim=-1)
        kwargs = dict(
            activation=layer.activation,
            quant_type=method.quant_type,
            w1_scale=s1,
            w2_scale=s2,
            swiglu_limit=layer.swiglu_limit,
            dtype=torch.bfloat16,
        )
        expected = fused_moe(
            gathered, layer.w13_weight, layer.w2_weight, weights, ids, **kwargs
        )
        result = fused_moe(
            actual,
            layer.w13_weight,
            layer.w2_weight,
            weights,
            ids,
            a1_scale=scale,
            **kwargs,
        )
        assert torch.isfinite(result).all()
        same_bytes(result, expected)
        same_bytes(sp.sp_moe_reduce_scatter(result), sp.sp_moe_reduce_scatter(expected))
        torch.cuda.synchronize()
        with ps.graph_capture() as capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture.stream):
                graph_q, graph_scale = method.gather_sp_input(layer, x)
        for _ in range(2):
            x.normal_()
            graph.replay()
            reference, reference_scale = quant(
                sp.sp_moe_gather(x), quant_dtype=dtypes.fp4x2
            )
            same_bytes(graph_q, reference)
            same_bytes(graph_scale, reference_scale)
        if rank == 0:
            print(
                f"PASS SP{world} {tokens}: quant bytes, scales, MoE, reduce-scatter, graph replay",
                flush=True,
            )
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


if __name__ == "__main__":
    main()
