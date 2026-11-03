#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 F5 validation for MiniMax-M3 prefill registered AR input buffers.

Run only on an idle four-GPU host:

  torchrun --standalone --nproc-per-node=4 \
    tools/validate_minimax_m3_ar_input_buffer_tp4.py

For M=2517 and 4536 this compares the ordinary producer allocation with a
producer writing directly into custom AR's leased input arena. It requires
bitwise-identical fused AR+Gemma-RMSNorm outputs in eager and graph replay and
fails if the optimized profiler region contains a device-to-device memcpy.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from aiter import (
    dtypes,
    gemm_a8w8_bpreshuffle,
    gemm_a8w8_bpreshuffle_out,
    pertoken_quant,
)
from aiter.ops.shuffle import shuffle_weight
from torch.profiler import ProfilerActivity, profile

from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_lease_registered_input_buffer,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)

_MS = (2517, 4536)
_HIDDEN = 6144
_PROJECTION_K = 2048
_WORLD = 4


def _dtod_events(prof) -> list[str]:
    names = []
    for event in prof.events():
        name = event.name
        normalized = name.lower().replace(" ", "")
        if "memcpy" in normalized and (
            "dtod" in normalized or "devicetodevice" in normalized
        ):
            names.append(name)
    return sorted(set(names))


def _profile(step):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = step()
        torch.cuda.synchronize()
    copied = _dtod_events(prof)
    if copied:
        raise AssertionError(f"optimized region still contains DtoD memcpy: {copied}")
    return result


def _assert_bitwise(label: str, expected, actual):
    for name, lhs, rhs in zip(("norm", "residual"), expected, actual):
        if not torch.equal(lhs, rhs):
            mismatches = torch.count_nonzero(
                lhs.view(torch.int16) != rhs.view(torch.int16)
            )
            raise AssertionError(
                f"{label} {name} is not bitwise equal: "
                f"{int(mismatches)} elements differ"
            )


def _case(rank: int, m: int, lease, graphs: list):
    device = torch.device("cuda", rank)
    local_gen = torch.Generator(device=device).manual_seed(0xF500 + 17 * m + rank)
    common_gen = torch.Generator(device=device).manual_seed(0x393 + m)
    activation = torch.randn(
        (m, _PROJECTION_K),
        dtype=torch.bfloat16,
        device=device,
        generator=local_gen,
    )
    projection_weight = torch.randn(
        (_HIDDEN, _PROJECTION_K),
        dtype=torch.bfloat16,
        device=device,
        generator=local_gen,
    )
    xq, x_scale = pertoken_quant(activation, quant_dtype=dtypes.fp8)
    wq, w_scale = pertoken_quant(projection_weight, quant_dtype=dtypes.fp8)
    wq = shuffle_weight(wq, layout=(16, 16))
    residual = torch.randn(
        (m, _HIDDEN), dtype=torch.bfloat16, device=device, generator=common_gen
    )
    norm_weight = torch.randn(
        (_HIDDEN,), dtype=torch.bfloat16, device=device, generator=common_gen
    )
    registered_input = lease.view_like(residual)

    def fused(inp):
        return tensor_model_parallel_fused_allreduce_rmsnorm(
            inp,
            residual,
            norm_weight,
            1e-6,
            gemma_norm=True,
        )

    ordinary_input = gemm_a8w8_bpreshuffle(
        xq, wq, x_scale, w_scale, dtype=dtypes.bf16
    )
    expected = fused(ordinary_input)
    torch.cuda.synchronize()

    def eager_step():
        gemm_a8w8_bpreshuffle_out(
            registered_input, xq, wq, x_scale, w_scale
        )
        return fused(registered_input)

    eager = _profile(eager_step)
    _assert_bitwise(f"M={m} eager", expected, eager)

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture, torch.cuda.graph(graph, stream=capture.stream):
        gemm_a8w8_bpreshuffle_out(
            registered_input, xq, wq, x_scale, w_scale
        )
        graph_result = fused(registered_input)
    graphs.append(
        (
            graph,
            graph_result,
            xq,
            wq,
            x_scale,
            w_scale,
            residual,
            norm_weight,
            registered_input,
        )
    )

    graph.replay()
    torch.cuda.synchronize()
    _assert_bitwise(f"M={m} graph", expected, graph_result)
    _profile(graph.replay)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != _WORLD:
        raise RuntimeError(f"F5 validation requires TP4, got WORLD_SIZE={world}")

    torch.cuda.set_device(local_rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    lease = None
    graphs = []
    try:
        lease = tensor_model_parallel_lease_registered_input_buffer()
        if lease is None:
            raise RuntimeError("custom AR registered input-buffer lease is unavailable")
        for m in _MS:
            get_tp_group().barrier()
            _case(local_rank, m, lease, graphs)
        get_tp_group().barrier()
        if rank == 0:
            print(
                "PASS: M=2517,4536 eager+graph bitwise; "
                "optimized profiler regions contain no Memcpy DtoD"
            )
    finally:
        torch.cuda.synchronize()
        if lease is not None:
            lease.release()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
