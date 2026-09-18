# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Receive trained weights straight into resident GPU memory over RCCL.

Every TP worker joins an independent process group alongside the trainer, so the
trainer broadcasts BF16 buckets directly into this engine's parameters. No
safetensors round-trip, no shared folder, no host bounce.

The wire format is frozen by the sending side
(``lumenrl.engine.inference.rdma_weight_transfer``) and must match byte for byte:

    header   4x int64, broadcast from rank 0:
             [command, metadata_bytes, payload_bytes, version]
             command 0 = end of stream, 1 = a bucket follows
    metadata ``metadata_bytes`` of uint8, a JSON list of
             {"name", "shape", "dtype", "offset", "nbytes"}
             where dtype is a torch name with the "torch." prefix stripped
    payload  ``payload_bytes`` of uint8; each entry is the slice
             [offset, offset+nbytes) viewed as its dtype and shape

Two things make this more than a memcpy. ATOM's parameters are *fused* --
``qkv_proj`` and ``gate_up_proj`` combine several checkpoint tensors -- and the
shards of one fused parameter can land in different buckets, so state has to
survive across buckets. And applying weights in place means a stream that fails
halfway leaves the model as a mix of old and new, which is worse than not having
started: inference would keep serving, quietly wrong. Hence the transaction
below, and the fence that refuses to serve after a partial write.
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


def _recv_header(group, *, device: torch.device) -> tuple[int, int, int, int]:
    header = torch.empty(_HEADER_WORDS, dtype=torch.int64, device=device)
    dist.broadcast(header, src=0, group=group)
    command, metadata_bytes, payload_bytes, version = (
        int(v) for v in header.cpu().tolist()
    )
    return command, metadata_bytes, payload_bytes, version


def _decode_bucket(
    metadata_tensor: torch.Tensor, payload: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Turn one received bucket into named tensor views over the payload.

    Views, not copies: the payload stays alive for the duration of the bucket's
    application, and copying here would double the peak footprint of a transfer
    already sized in tens of GB.
    """
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
        # Bounds-check before slicing: a corrupt offset would otherwise either
        # silently truncate or read a neighbouring tensor's bytes.
        if start < 0 or nbytes <= 0 or end > payload.numel():
            raise RuntimeError(
                f"RDMA payload range out of bounds for {name}: "
                f"offset={start} nbytes={nbytes} payload={payload.numel()}"
            )
        value = payload[start:end].view(dtype).view(shape)
        if value.numel() * value.element_size() != nbytes:
            raise RuntimeError(
                f"RDMA metadata size mismatch for {name}: "
                f"shape={shape} dtype={dtype} nbytes={nbytes}"
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
    """Consume one weight stream and apply it as a single transaction.

    Returns throughput and coverage statistics. On any failure the runner is
    fenced rather than left half-updated -- see ``abort_weight_update``.
    """
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    started = time.perf_counter()

    runner.begin_weight_update(expected_version)
    try:
        while True:
            command, metadata_bytes, payload_bytes, version = _recv_header(
                group, device=device
            )
            # Checked on every header, not just the first: a version change
            # mid-stream means two senders are interleaving on one group, and
            # the halves would silently mix.
            if version != expected_version:
                raise RuntimeError(
                    f"RDMA weight version mismatch: expected {expected_version}, "
                    f"got {version}"
                )
            if command == _CMD_END:
                break
            if command != _CMD_BUCKET or metadata_bytes <= 0 or payload_bytes <= 0:
                raise RuntimeError(
                    f"invalid RDMA weight header: command={command} "
                    f"metadata_bytes={metadata_bytes} payload_bytes={payload_bytes}"
                )

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
        # Fence before re-raising: the parameters are now a mix of versions, so
        # serving must stop until a later full reload succeeds.
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
    """The worker methods a trainer-side orchestrator drives over collective_rpc.

    Named to match the vLLM worker extension, so the same caller drives either
    backend.
    """

    def init_rdma_weight_group(
        self,
        master_addr: str,
        master_port: int,
        base_rank: int,
        world_size: int,
        group_name: str,
        timeout_s: int = 600,
    ) -> bool:
        """Join the trainer's weight-broadcast group as this rank.

        Rank 0 is the trainer, so every worker sits at ``base_rank`` or above.
        The caller assigns one ``base_rank`` per replica; this rank's offset
        within the replica accounts for both TP and any local DP, because a
        replica running DP internally contributes more than ``tp_size`` ranks.
        """
        # Vendored, so a container with ATOM but no RL framework still works.
        from atom.utils.independent_process_group import (
            init_independent_process_group,
        )

        groups = getattr(self, "_rdma_weight_groups", None)
        if groups is None:
            groups = {}
            self._rdma_weight_groups = groups
        if group_name in groups:
            # Idempotent: the orchestrator may retry init after a partial
            # failure elsewhere, and rejoining would deadlock the rendezvous.
            logger.warning(
                "%s: RDMA weight group %s already joined, reusing",
                getattr(self, "label", "runner"),
                group_name,
            )
            return True

        parallel = getattr(getattr(self, "config", None), "parallel_config", None)
        dp_rank_local = int(getattr(parallel, "data_parallel_rank_local", 0) or 0)
        tp_size = int(getattr(self, "world_size", 1) or 1)
        rank = int(base_rank) + dp_rank_local * tp_size + int(self.rank)

        if rank <= 0 or rank >= int(world_size):
            raise ValueError(
                f"invalid RDMA rank {rank} for world_size={world_size} "
                f"(base_rank={base_rank}, dp_local={dp_rank_local}, "
                f"tp_rank={self.rank}, tp_size={tp_size}); rank 0 is the trainer"
            )

        groups[group_name] = init_independent_process_group(
            backend="nccl",  # RCCL on ROCm
            init_method=f"tcp://{master_addr}:{master_port}",
            timeout=timedelta(seconds=int(timeout_s)),
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            "%s: joined RDMA weight group %s as rank %d/%d " "(dp_local=%d tp_rank=%d)",
            getattr(self, "label", "runner"),
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
        """Receive one version of the weights into resident memory."""
        groups = getattr(self, "_rdma_weight_groups", {})
        if group_name not in groups:
            raise RuntimeError(
                f"RDMA weight group is not initialized: {group_name!r}; "
                "call init_rdma_weight_group first"
            )
        stats = receive_weight_stream(
            groups[group_name],
            self,
            device=self.device,
            expected_version=int(version),
            verify_full_load=bool(verify_full_load),
        )
        logger.info(
            "%s: RDMA weight reload v%d verified: %s",
            getattr(self, "label", "runner"),
            int(version),
            stats,
        )
        return stats

    def destroy_rdma_weight_group(self, group_name: str) -> bool:
        groups = getattr(self, "_rdma_weight_groups", {})
        group = groups.pop(group_name, None)
        if group is None:
            return True
        try:
            dist.destroy_process_group(group)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            # A failure here is not worth propagating: the caller is usually
            # already unwinding from something else, and the group is gone from
            # our map either way.
            logger.warning(
                "%s: tearing down RDMA weight group %s failed: %s",
                getattr(self, "label", "runner"),
                group_name,
                exc,
            )
        return True
