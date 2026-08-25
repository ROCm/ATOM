# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Synthetic inter-node DP+EP validation runner.

This entrypoint is intentionally not an OpenAI server. Launch it on every node
with matching global DP parameters to validate that ATOM/AITER can form one
global DP/EP process group while each node owns only its local DP slice.
"""

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkerSpec:
    global_dp_rank: int
    local_dp_rank: int
    model_rank: int
    local_rank: int


def build_worker_specs(
    *,
    data_parallel_size: int,
    data_parallel_size_local: int,
    data_parallel_rank: int,
    tensor_parallel_size: int,
    prefill_context_parallel_size: int,
) -> list[WorkerSpec]:
    """Build local worker specs for this node.

    `data_parallel_rank` is the first global DP rank owned by this node.
    `model_rank` is the local TP/PCP rank passed to AITER init_dist_env.
    """

    model_world = tensor_parallel_size * prefill_context_parallel_size
    specs: list[WorkerSpec] = []
    for local_dp_rank in range(data_parallel_size_local):
        global_dp_rank = data_parallel_rank + local_dp_rank
        for model_rank in range(model_world):
            local_rank = local_dp_rank * model_world + model_rank
            specs.append(
                WorkerSpec(
                    global_dp_rank=global_dp_rank,
                    local_dp_rank=local_dp_rank,
                    model_rank=model_rank,
                    local_rank=local_rank,
                )
            )
    return specs


def validate_topology(
    *,
    data_parallel_size: int,
    data_parallel_size_local: int,
    data_parallel_rank: int,
    tensor_parallel_size: int,
    prefill_context_parallel_size: int,
    visible_gpu_count: int,
) -> None:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be > 0")
    if data_parallel_size_local <= 0:
        raise ValueError("data_parallel_size_local must be > 0")
    if data_parallel_rank < 0:
        raise ValueError("data_parallel_rank must be >= 0")
    if data_parallel_rank + data_parallel_size_local > data_parallel_size:
        raise ValueError(
            "local DP rank range exceeds global data_parallel_size: "
            f"rank_start={data_parallel_rank}, local_size={data_parallel_size_local}, "
            f"global_size={data_parallel_size}"
        )
    required_workers = (
        data_parallel_size_local * tensor_parallel_size * prefill_context_parallel_size
    )
    if visible_gpu_count < required_workers:
        raise ValueError(
            f"topology requires {required_workers} local GPU workers but only "
            f"{visible_gpu_count} visible GPUs are available"
        )


def _worker(spec_dict: dict, args_dict: dict) -> None:
    import torch
    import torch.distributed as dist

    from aiter import init_dist_env, destroy_dist_env
    from aiter.dist.parallel_state import get_dp_group, get_ep_group, get_tp_group
    from atom.utils import get_distributed_init_method

    spec = WorkerSpec(**spec_dict)
    args = argparse.Namespace(**args_dict)

    torch.cuda.set_device(spec.local_rank)
    init_method = get_distributed_init_method(
        args.data_parallel_master_ip, args.data_parallel_master_port
    )

    init_dist_env(
        args.tensor_parallel_size,
        rankID=spec.model_rank,
        backend=args.backend,
        distributed_init_method=init_method,
        local_rank=spec.local_rank,
        data_parallel_size=args.data_parallel_size,
        data_parallel_rank=spec.global_dp_rank,
        prefill_context_model_parallel_size=args.prefill_context_parallel_size,
    )

    try:
        device = torch.device(f"cuda:{spec.local_rank}")
        ep_group = get_ep_group()
        dp_group = get_dp_group()
        tp_group = get_tp_group()

        value = torch.tensor(
            [float(dist.get_rank())], dtype=torch.float32, device=device
        )
        dist.all_reduce(value, group=ep_group.device_group)

        info = {
            "global_rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "global_dp_rank": spec.global_dp_rank,
            "local_dp_rank": spec.local_dp_rank,
            "model_rank": spec.model_rank,
            "local_rank": spec.local_rank,
            "device": str(device),
            "tp_group": tp_group.ranks,
            "dp_group": dp_group.ranks,
            "ep_group": ep_group.ranks,
            "ep_all_reduce_sum": float(value.item()),
        }
        print(json.dumps(info, sort_keys=True), flush=True)
    finally:
        destroy_dist_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-parallel-size", "-dp", type=int, required=True)
    parser.add_argument("--data-parallel-size-local", type=int, required=True)
    parser.add_argument("--data-parallel-rank", type=int, required=True)
    parser.add_argument("--data-parallel-master-ip", type=str, required=True)
    parser.add_argument("--data-parallel-master-port", type=int, default=29500)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--prefill-context-parallel-size", "-pcp", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    validate_topology(
        data_parallel_size=args.data_parallel_size,
        data_parallel_size_local=args.data_parallel_size_local,
        data_parallel_rank=args.data_parallel_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        prefill_context_parallel_size=args.prefill_context_parallel_size,
        visible_gpu_count=torch.cuda.device_count(),
    )
    specs = build_worker_specs(
        data_parallel_size=args.data_parallel_size,
        data_parallel_size_local=args.data_parallel_size_local,
        data_parallel_rank=args.data_parallel_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        prefill_context_parallel_size=args.prefill_context_parallel_size,
    )
    args_dict = vars(args)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker, args=(asdict(spec), args_dict), daemon=False)
        for spec in specs
    ]
    for proc in procs:
        proc.start()
    failures = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures.append((proc.pid, proc.exitcode))
    if failures:
        raise SystemExit(f"worker failures: {failures}")


if __name__ == "__main__":
    main()
