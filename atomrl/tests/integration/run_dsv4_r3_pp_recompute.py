# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Two-rank PP replay/recompute smoke for the patched Megatron RouterReplay."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.distributed as dist
from atomrl.engine.training.megatron_engine import MegatronEngine
from megatron.core.tensor_parallel.random import checkpoint
from megatron.core.transformer.moe.router_replay import RouterReplay


def _routes(offset: int, num_layers: int, topk: int, num_experts: int) -> torch.Tensor:
    routes = torch.empty(1, 3, num_layers, topk, dtype=torch.int16)
    for layer in range(num_layers):
        for choice in range(topk):
            routes[0, :, layer, choice] = (layer + offset + choice) % num_experts
    return routes


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size == 2:
        layers_per_rank = [2, 2]
    elif world_size == 4:
        layers_per_rank = [11, 11, 11, 10]
    else:
        raise ValueError(f"expected PP world size 2 or 4, got {world_size}")
    num_layers = sum(layers_per_rank)
    num_experts = 256
    topk = 6
    torch.cuda.set_device(rank)

    RouterReplay.clear_global_router_replay_instances()
    instances = [RouterReplay() for _ in range(layers_per_rank[rank])]

    engine = MegatronEngine.__new__(MegatronEngine)
    engine._pp_rank = rank
    engine._pp_size = world_size
    engine._layers_per_pp_rank = layers_per_rank
    engine._tp_rank = 0
    engine._tp_size = 1
    engine._tfcfg = SimpleNamespace(sequence_parallel=False)
    engine._dims = SimpleNamespace(
        num_layers=num_layers,
        num_experts=num_experts,
    )

    observed: list[list[torch.Tensor]] = [[] for _ in instances]

    def routed_forward(scores: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer, replay in enumerate(instances):
            values, indices = replay.get_replay_topk(
                scores,
                topk,
                default_compute_topk=lambda value, topk, *_args, **_kwargs: torch.topk(
                    value, topk, dim=1
                ),
            )
            observed[layer].append(indices.detach().cpu())
            outputs.append(values)
        return torch.stack(outputs).sum(dim=0)

    scores0 = torch.arange(
        4 * num_experts,
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    ).view(4, num_experts)
    scores1 = (scores0.detach() + 100).requires_grad_(True)

    engine._r3_clear()
    engine._r3_set_microbatch_routes(
        _routes(0, num_layers, topk, num_experts),
        row=0,
        start=0,
        length=4,
        padded_length=4,
    )
    output0 = checkpoint(routed_forward, False, scores0)
    engine._r3_set_microbatch_routes(
        _routes(7, num_layers, topk, num_experts),
        row=0,
        start=0,
        length=4,
        padded_length=4,
    )
    output1 = checkpoint(routed_forward, False, scores1)

    # Megatron 1F1B retires the oldest outstanding microbatch first.
    output0.sum().backward()
    output1.sum().backward()

    layer_offset = sum(layers_per_rank[:rank])
    for local_layer, layer_events in enumerate(observed):
        global_layer = layer_offset + local_layer
        filler = (
            torch.div(
                torch.arange(topk, dtype=torch.int64) * num_experts,
                topk,
                rounding_mode="floor",
            )
            + global_layer * topk
        ).remainder(num_experts)
        expected0 = torch.tensor(
            [[global_layer + choice for choice in range(topk)]] * 3,
            dtype=torch.int64,
        ).remainder(num_experts)
        expected0 = torch.cat((expected0, filler.unsqueeze(0)), dim=0)
        expected1 = torch.tensor(
            [[global_layer + 7 + choice for choice in range(topk)]] * 3,
            dtype=torch.int64,
        ).remainder(num_experts)
        expected1 = torch.cat((expected1, filler.unsqueeze(0)), dim=0)
        assert len(layer_events) == 4
        torch.testing.assert_close(layer_events[0], expected0)
        torch.testing.assert_close(layer_events[1], expected1)
        torch.testing.assert_close(layer_events[2], expected0)
        torch.testing.assert_close(layer_events[3], expected1)

    engine._r3_clear()
    assert all(not replay.replay_backward_list for replay in instances)
    coverage = torch.tensor(
        [layer_offset, layer_offset + len(instances)],
        dtype=torch.int64,
        device="cuda",
    )
    gathered = [torch.empty_like(coverage) for _ in range(world_size)]
    dist.all_gather(gathered, coverage)
    assert [item.tolist() for item in gathered] == [
        [sum(layers_per_rank[:stage]), sum(layers_per_rank[: stage + 1])]
        for stage in range(world_size)
    ]
    dist.barrier()
    if rank == 0:
        print(f"PP{world_size}_R3_RECOMPUTE_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
