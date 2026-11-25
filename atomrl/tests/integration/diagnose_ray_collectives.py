# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import datetime
import os
import socket
import time

import ray
import torch
import torch.distributed as dist
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def ray_head_ip() -> str:
    head_ip = os.environ.get("RAY_HEAD_IP")
    if not head_ip:
        raise RuntimeError("set RAY_HEAD_IP to the reachable Ray head-node address")
    return head_ip


@ray.remote(num_gpus=1, num_cpus=1)
class CollectiveProbe:
    def run(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        port: int,
        elements: int,
        iterations: int,
    ) -> dict[str, object]:
        device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{ray_head_ip()}:{port}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120),
            device_id=device if device.type == "cuda" else None,
        )
        value = torch.full(
            (elements,),
            float(rank + 1),
            dtype=torch.float32,
            device=device,
        )
        dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
        started = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(value)
            value.div_(world_size * (world_size + 1) / 2)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        result = {
            "backend": backend,
            "rank": rank,
            "host": socket.gethostname(),
            "node_ip": ray.util.get_node_ip_address(),
            "device": str(device),
            "result": float(value[0].cpu()),
            "elapsed_s": elapsed,
            "bytes_per_rank": value.numel() * value.element_size() * iterations,
            "interfaces": sorted(name for _, name in socket.if_nameindex()),
            "nccl_socket_ifname": os.getenv("NCCL_SOCKET_IFNAME"),
            "nccl_ib_hca": os.getenv("NCCL_IB_HCA"),
            "nccl_net": os.getenv("NCCL_NET"),
        }
        dist.destroy_process_group()
        return result


def main() -> None:
    head_ip = ray_head_ip()
    ray.init(address="auto")
    nodes = [node for node in ray.nodes() if node["Alive"]]
    nodes.sort(key=lambda node: node["NodeManagerAddress"] != head_ip)
    if len(nodes) != 3:
        raise RuntimeError(f"expected 3 alive Ray nodes, got {len(nodes)}")

    env_vars = {
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "ens14np0",
        "NCCL_IB_HCA": "mlx5_0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET": "IB",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_DEBUG": "WARN",
    }
    probes = [
        CollectiveProbe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"],
                soft=False,
            ),
            runtime_env={"env_vars": env_vars},
        ).remote()
        for node in nodes
    ]

    for backend, port, elements, iterations in (
        ("gloo", 29627, 1, 1),
        ("nccl", 29628, 16 * 1024 * 1024, 4),
    ):
        results = ray.get(
            [
                probe.run.remote(
                    backend=backend,
                    rank=rank,
                    world_size=len(probes),
                    port=port,
                    elements=elements,
                    iterations=iterations,
                )
                for rank, probe in enumerate(probes)
            ]
        )
        print({"backend": backend, "results": results}, flush=True)


if __name__ == "__main__":
    main()
