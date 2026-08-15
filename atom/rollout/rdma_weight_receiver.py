# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-compatible bucketed weight receive path for LumenRL.

The wire format intentionally mirrors
``lumenrl.engine.inference.rdma_weight_transfer`` v1.  ATOM's transaction
state is local to the receiver and does not add acknowledgements or commands.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger("atom")

_CMD_END = 0
_CMD_BUCKET = 1
_HEADER_WORDS = 4  # command, metadata bytes, payload bytes, version


def _broadcast_header(group, *, device: torch.device) -> torch.Tensor:
    header = torch.empty(_HEADER_WORDS, dtype=torch.int64, device=device)
    dist.broadcast(header, src=0, group=group)
    return header


def _decode_bucket(
    metadata_tensor: torch.Tensor,
    payload: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    metadata = json.loads(bytes(metadata_tensor.cpu().tolist()).decode("utf-8"))
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError("invalid RDMA weight metadata: expected a non-empty list")

    weights: list[tuple[str, torch.Tensor]] = []
    for entry in metadata:
        try:
            name = str(entry["name"])
            shape = [int(dim) for dim in entry["shape"]]
            dtype = getattr(torch, entry["dtype"])
            start = int(entry["offset"])
            nbytes = int(entry["nbytes"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError(f"invalid RDMA weight metadata entry: {entry}") from exc
        end = start + nbytes
        if start < 0 or nbytes <= 0 or end > payload.numel():
            raise RuntimeError(
                f"invalid RDMA payload range for {name}: "
                f"offset={start}, nbytes={nbytes}, payload={payload.numel()}"
            )
        value = payload[start:end].view(dtype).view(shape)
        if value.numel() * value.element_size() != nbytes:
            raise RuntimeError(
                f"RDMA metadata size mismatch for {name}: "
                f"shape={shape}, dtype={dtype}, nbytes={nbytes}"
            )
        weights.append((name, value))
    return weights


@torch.no_grad()
def receive_weight_stream(
    group,
    runner,
    *,
    device: torch.device,
    expected_version: int,
    verify_full_load: bool = True,
) -> dict[str, float]:
    """Receive the frozen LumenRL v1 stream and transactionally load ATOM."""
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    started = time.perf_counter()
    runner.begin_weight_update(expected_version)

    try:
        while True:
            header = _broadcast_header(group, device=device)
            command, metadata_bytes, payload_bytes, version = [
                int(value) for value in header.cpu().tolist()
            ]
            if version != expected_version:
                raise RuntimeError(
                    "RDMA weight version mismatch: "
                    f"expected {expected_version}, got {version}"
                )
            if command == _CMD_END:
                break
            if command != _CMD_BUCKET or metadata_bytes <= 0 or payload_bytes <= 0:
                raise RuntimeError(f"invalid RDMA weight header: {header.tolist()}")

            metadata_tensor = torch.empty(
                metadata_bytes, dtype=torch.uint8, device=device
            )
            payload = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
            dist.broadcast(metadata_tensor, src=0, group=group)
            dist.broadcast(payload, src=0, group=group)
            weights = _decode_bucket(metadata_tensor, payload)
            runner.apply_weight_bucket(weights, payload_bytes=payload_bytes)
            total_bytes += payload_bytes
            total_weights += len(weights)
            total_buckets += 1

        manifest = runner.commit_weight_update(
            expected_version, verify_full_load=verify_full_load
        )
    except Exception as exc:
        if getattr(runner, "_weight_update_transaction", None) is not None:
            runner.abort_weight_update(expected_version, exc)
        raise

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "version": float(expected_version),
        "buckets": float(total_buckets),
        "weights": float(total_weights),
        "bytes": float(total_bytes),
        "seconds": elapsed,
        "gbps": (total_bytes * 8 / 1e9 / elapsed) if elapsed > 0 else 0.0,
        "loaded_internal": float(manifest["loaded_internal"]),
    }


class RDMAWeightReceiverMixin:
    """Worker methods matching LumenRL's vLLM collective-RPC contract."""

    def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        from lumenrl.utils.independent_process_group import (
            init_independent_process_group,
        )

        groups = getattr(self, "_rdma_weight_groups", None)
        if groups is None:
            groups = {}
            self._rdma_weight_groups = groups
        if group_name in groups:
            return True

        dp_rank_local = int(
            getattr(self.config.parallel_config, "data_parallel_rank_local", 0) or 0
        )
        tp_size = int(self.world_size)
        rank = int(base_rank) + dp_rank_local * tp_size + int(self.rank)
        if rank <= 0 or rank >= int(world_size):
            raise ValueError(
                f"invalid ATOM RDMA rank={rank}, world_size={world_size}, "
                f"base_rank={base_rank}, dp_rank_local={dp_rank_local}, "
                f"tp_rank={self.rank}, tp_size={tp_size}"
            )

        groups[group_name] = init_independent_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=int(timeout_s)),
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        logger.warning(
            "%s joined RDMA weight group %s as rank=%d/%d "
            "(dp_local=%d, tp_rank=%d)",
            self.label,
            group_name,
            rank,
            world_size,
            dp_rank_local,
            self.rank,
        )
        return True

    def receive_weights_rdma(
        self,
        group_name: str,
        version: int,
        verify_full_load: bool = True,
    ) -> dict[str, float]:
        groups = getattr(self, "_rdma_weight_groups", {})
        if group_name not in groups:
            raise RuntimeError(f"RDMA weight group is not initialized: {group_name}")
        stats = receive_weight_stream(
            groups[group_name],
            self,
            device=self.device,
            expected_version=int(version),
            verify_full_load=bool(verify_full_load),
        )
        logger.warning("%s RDMA weight reload verified: %s", self.label, stats)
        return stats

    def destroy_rdma_weight_group(self, group_name: str) -> bool:
        groups = getattr(self, "_rdma_weight_groups", {})
        group = groups.pop(group_name, None)
        if group is not None:
            dist.destroy_process_group(group)
        return True
