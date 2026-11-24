# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Stress the TP=8 RCCL AllGather observed immediately before the DSV4 fault."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

ELEMENTS_PER_RANK = 517_120


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"expected TP=8, got {world_size}"

    send = torch.full(
        (ELEMENTS_PER_RANK,),
        rank + 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    recv = torch.empty(
        world_size * ELEMENTS_PER_RANK,
        dtype=torch.bfloat16,
        device="cuda",
    )

    for iteration in range(args.iterations):
        dist.all_gather_into_tensor(recv, send)
        if iteration == 0 or (iteration + 1) % 50 == 0:
            torch.cuda.synchronize()
            rows = recv.view(world_size, ELEMENTS_PER_RANK)
            expected = torch.arange(
                1,
                world_size + 1,
                dtype=torch.bfloat16,
                device="cuda",
            )
            torch.testing.assert_close(rows[:, 0], expected, rtol=0, atol=0)
            torch.testing.assert_close(rows[:, -1], expected, rtol=0, atol=0)
            if rank == 0:
                print(f"iteration={iteration + 1}/{args.iterations} ok", flush=True)

    torch.cuda.synchronize()
    if rank == 0:
        print(
            "PASS "
            f"world_size={world_size} elements_per_rank={ELEMENTS_PER_RANK} "
            f"recv_bytes={recv.numel() * recv.element_size()} "
            f"iterations={args.iterations}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
