"""Two-node RCCL compatibility check for LumenRL sender -> ATOM receiver.

Run under Spur with five tasks (rank 0 trainer, ranks 1..4 ATOM workers).
"""

from __future__ import annotations

import json
import os
import socket
from typing import ClassVar

import torch
import torch.distributed as dist
from lumenrl.engine.inference.rdma_weight_transfer import send_weight_stream

from atom.rollout.rdma_weight_receiver import receive_weight_stream
from atom.rollout.weight_updater import WeightUpdaterMixin


class PackedProjection(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        self.weight_scale = torch.nn.Parameter(torch.ones(1, device=device))

    @staticmethod
    def weight_loader(param, tensor, shard_id):
        param.data[{"q": 0, "k": 1}[shard_id]].copy_(tensor.reshape(-1))


class PackedModel(torch.nn.Module):
    packed_modules_mapping: ClassVar[dict] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
    }

    def __init__(self, device: torch.device):
        super().__init__()
        self.qkv_proj = PackedProjection(device)


class CompatRunner(WeightUpdaterMixin):
    def __init__(self, device: torch.device, rank: int):
        self.model = PackedModel(device)
        self.device = device
        self.rank = rank - 1
        self.world_size = 4
        self.label = f"compat-rank-{rank}"
        self.clear_count = 0

    def clear_kv_cache(self):
        self.clear_count += 1

    @staticmethod
    def _is_fp8_param(module, param):
        return isinstance(module, PackedProjection)

    def _requantize_fp8_weight(self, module, param_name, param, tensor):
        param.data.copy_(tensor)
        module.weight_scale.data.fill_(2.0)
        return True

    def _invalidate_cudagraphs_after_weight_update(self):
        pass


def weights_for(version: int, device: torch.device, *, complete: bool = True):
    q = torch.tensor([version, version + 0.25], dtype=torch.bfloat16, device=device)
    k = torch.tensor([version + 0.5, version + 0.75], dtype=torch.bfloat16, device=device)
    values = [("q_proj.weight", q)]
    if complete:
        values.append(("k_proj.weight", k))
    return values


def expected_weight(version: int, device: torch.device):
    return torch.stack([tensor for _, tensor in weights_for(version, device)]).float()


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def main() -> None:
    rank_base = int(os.getenv("RDMA_TEST_RANK_BASE", "0"))
    rank = rank_base + int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(
        os.getenv("RDMA_TEST_WORLD_SIZE", os.environ["SLURM_NTASKS"])
    )
    if world_size != 5:
        raise RuntimeError(f"expected trainer + DP2xTP2 world of 5, got {world_size}")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    master_port = int(os.getenv("RDMA_TEST_MASTER_PORT", "29612"))
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{os.environ['RDMA_TEST_MASTER_ADDR']}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    runner = CompatRunner(device, rank) if rank else None
    completed = []

    for version in (1, 2, 3):
        if rank == 0:
            stats = send_weight_stream(
                dist.group.WORLD,
                weights_for(version, device),
                bucket_size_bytes=4,
                version=version,
            )
            local = {"role": "sender", **stats}
        else:
            stats = receive_weight_stream(
                dist.group.WORLD,
                runner,
                device=device,
                expected_version=version,
                verify_full_load=True,
            )
            torch.testing.assert_close(
                runner.model.qkv_proj.weight,
                expected_weight(version, device),
            )
            local = {
                "role": "receiver",
                "rank": rank,
                "clear_count": runner.clear_count,
                **stats,
            }
        round_stats = gather(local)
        if rank == 0:
            receivers = round_stats[1:]
            invariant = {
                (item["version"], item["buckets"], item["weights"], item["bytes"])
                for item in receivers
            }
            if invariant != {(float(version), 2.0, 2.0, 8.0)}:
                raise RuntimeError(f"receiver statistics diverged: {receivers}")
            completed.append(round_stats)

    failure_version = 4
    if rank == 0:
        send_weight_stream(
            dist.group.WORLD,
            weights_for(failure_version, device, complete=False),
            bucket_size_bytes=4,
            version=failure_version,
        )
        failure = {"role": "sender", "sent_incomplete": True}
    else:
        try:
            receive_weight_stream(
                dist.group.WORLD,
                runner,
                device=device,
                expected_version=failure_version,
                verify_full_load=True,
            )
        except RuntimeError as exc:
            fenced = False
            try:
                runner.assert_weight_update_ready()
            except RuntimeError:
                fenced = True
            failure = {
                "role": "receiver",
                "rank": rank,
                "failed": True,
                "fenced": fenced,
                "error": str(exc),
            }
        else:
            raise RuntimeError("incomplete stream unexpectedly committed")
    failure_stats = gather(failure)
    if rank == 0 and not all(
        item.get("failed") and item.get("fenced") for item in failure_stats[1:]
    ):
        raise RuntimeError(f"failure did not fence every receiver: {failure_stats}")

    recovery_version = 5
    if rank == 0:
        recovery = send_weight_stream(
            dist.group.WORLD,
            weights_for(recovery_version, device),
            bucket_size_bytes=4,
            version=recovery_version,
        )
        recovery_local = {"role": "sender", **recovery}
    else:
        recovery = receive_weight_stream(
            dist.group.WORLD,
            runner,
            device=device,
            expected_version=recovery_version,
            verify_full_load=True,
        )
        runner.assert_weight_update_ready()
        torch.testing.assert_close(
            runner.model.qkv_proj.weight,
            expected_weight(recovery_version, device),
        )
        recovery_local = {"role": "receiver", "rank": rank, **recovery}
    recovery_stats = gather(recovery_local)

    if rank == 0:
        print(
            "RDMA_COMPAT_RESULT "
            + json.dumps(
                {
                    "status": "PASS",
                    "nodes": sorted(
                        {item["host"] for item in gather({"host": socket.gethostname()})}
                    ),
                    "world_size": world_size,
                    "versions": [1, 2, 3],
                    "rounds": completed,
                    "failure": failure_stats,
                    "recovery": recovery_stats,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        gather({"host": socket.gethostname()})

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
