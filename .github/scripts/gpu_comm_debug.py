"""Lightweight GPU communication smoke test for CI runners."""

from __future__ import annotations

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl")
    parser.add_argument(
        "--device-mib", type=int, default=_env_int("GPU_COMM_DEBUG_DEVICE_MIB", 16)
    )
    parser.add_argument(
        "--iters", type=int, default=_env_int("GPU_COMM_DEBUG_ITERS", 3)
    )
    args = parser.parse_args()

    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)
    world_size = _env_int("WORLD_SIZE", 1)
    host = socket.gethostname()

    print(
        f"[comm-debug] host={host} rank={rank} local_rank={local_rank} "
        f"world_size={world_size} backend={args.backend}",
        flush=True,
    )
    print(
        f"[comm-debug] torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} "
        f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}",
        flush=True,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("No ROCm/CUDA device is visible to torch")

    device_index = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    print(
        f"[comm-debug] rank={rank} device={device} name={torch.cuda.get_device_name(device)}",
        flush=True,
    )

    dist.init_process_group(backend=args.backend)
    try:
        torch.cuda.synchronize(device)
        init_started = time.monotonic()
        dist.barrier()
        torch.cuda.synchronize(device)
        print(
            f"[comm-debug] rank={rank} barrier_seconds={time.monotonic() - init_started:.3f}",
            flush=True,
        )

        elements = max(
            1,
            args.device_mib
            * 1024
            * 1024
            // torch.tensor([], dtype=torch.float32).element_size(),
        )
        expected = world_size * (world_size + 1) / 2
        tensor = torch.empty(elements, dtype=torch.float32, device=device)

        for iteration in range(args.iters):
            tensor.fill_(rank + 1)
            started = time.monotonic()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            observed = float(tensor[0].item())
            if observed != expected:
                raise RuntimeError(
                    f"all_reduce mismatch on rank {rank}: observed={observed}, expected={expected}"
                )
            if rank == 0:
                mib = args.device_mib
                print(
                    f"[comm-debug] all_reduce iteration={iteration} size_mib={mib} seconds={elapsed:.3f}",
                    flush=True,
                )

        dist.barrier()
        if rank == 0:
            print("[comm-debug] PASS", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
