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
import traceback
from collections import Counter
from datetime import timedelta
from itertools import pairwise

import torch
import torch.distributed as dist

logger = logging.getLogger("atom")

_CMD_END = 0
_CMD_BUCKET = 1
_HEADER_WORDS = 4  # command, metadata bytes, payload bytes, version

# What a header is held to before anything is allocated for it. Metadata is a
# JSON entry per tensor, so 64 MiB is hundreds of thousands of them, and no
# bucket carries a terabyte: past either, the header is corrupt or another
# protocol's.
_MAX_METADATA_BYTES = 64 << 20
_MAX_PAYLOAD_BYTES = 1 << 40


class RDMAStreamOutOfStep(RuntimeError):
    """This rank can no longer follow the stream.

    A broadcast is received whole, into a buffer of the size it was sent with,
    so a frame this rank cannot allocate for is one it cannot take part in --
    and the sender, already broadcasting it, cannot be told. Nothing more can
    ride the group: ``receive_weights_rdma`` tears down this rank's end, and
    the sender and any peer still in that broadcast are released by the
    group's timeout.
    """


def _unreceivable(
    device: torch.device, metadata_bytes: int, payload_bytes: int
) -> str | None:
    """Why a frame of these sizes cannot be allocated for here, if it cannot."""
    if metadata_bytes < 0 or payload_bytes < 0:
        return "a negative size"
    if metadata_bytes > _MAX_METADATA_BYTES:
        return f"metadata over the {_MAX_METADATA_BYTES}-byte ceiling"
    if payload_bytes > _MAX_PAYLOAD_BYTES:
        return f"a payload over the {_MAX_PAYLOAD_BYTES}-byte ceiling"
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        # Blocks the caching allocator holds unused are allocatable too.
        cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
            device
        )
        needed = metadata_bytes + payload_bytes
        if needed > free + cached:
            return f"{needed} bytes needed, {free + cached} free on {device}"
    return None


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
    spans: list[tuple[int, int, str]] = []
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
        spans.append((start, end, name))

    # Each entry can be in bounds and still be wrong as a set: a repeated name
    # writes one parameter twice, and overlapping ranges hand the same bytes to
    # two parameters. Either way coverage still passes. The sender packs one
    # tensor per name, back to back, so neither is ever legitimate.
    names = [name for _, _, name in spans]
    if len(set(names)) != len(names):
        repeated = sorted(n for n, seen in Counter(names).items() if seen > 1)
        raise RuntimeError(f"RDMA bucket repeats weight names: {repeated[:20]}")
    spans.sort()
    for (_, prev_end, prev_name), (start, _, name) in pairwise(spans):
        if start < prev_end:
            raise RuntimeError(
                f"RDMA payload ranges overlap: {prev_name} ends at byte "
                f"{prev_end}, {name} starts at {start}"
            )
    return weights


@torch.no_grad()
def receive_weight_stream(
    group,
    runner,
    *,
    device: torch.device,
    expected_version: int,
) -> dict[str, float]:
    """Consume one weight stream and apply it as a single transaction.

    Returns throughput and coverage statistics. Once this stream's reload has
    begun, any failure fences the runner rather than leaving it half-updated --
    see ``abort_weight_update``. A stream refused at begin wrote nothing, and
    leaves the runner as it was, a reload another caller has open included.

    A failure on this rank does not end its part in the stream. The trainer and
    every other rank are still in the broadcasts, and one rank leaving early
    hangs all of them in the next one -- the orchestrator then sees a timeout
    naming nobody instead of this rank's error. So the rest of the stream is
    received and discarded, the header's sizes being all that is needed to stay
    in step, and the failure is raised once the end marker arrives. The one
    exception is a frame this rank cannot allocate for, which raises
    ``RDMAStreamOutOfStep`` at its header.
    """
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    started = time.perf_counter()
    failure: Exception | None = None
    drained = 0
    began = False

    try:
        try:
            runner.begin_weight_update(expected_version)
            began = True
        except Exception as exc:  # noqa: BLE001 - raised at the end marker
            # Refused before anything was written -- a replayed or older
            # version, or a reload already open. Received all the same, or the
            # trainer and every other rank wait in the next broadcast. Not
            # aborted on the way out: a reload already open is its caller's to
            # finish, and the weights are as the refusal found them.
            failure = exc
            logger.error(
                "RDMA weight stream v%d refused on this rank, receiving it "
                "before reporting that: %s",
                expected_version,
                exc,
            )
        while True:
            # The last bucket goes before anything else is allocated. Held
            # until its names were reassigned, it was still resident while the
            # next one was allocated -- two buckets at the peak -- and the
            # final one stayed resident through commit.
            metadata_tensor = payload = weights = None
            command, metadata_bytes, payload_bytes, version = _recv_header(
                group, device=device
            )
            # Checked on every header, not just the first: a version change
            # mid-stream means two senders are interleaving on one group, and
            # the halves would silently mix.
            if failure is None and version != expected_version:
                failure = RuntimeError(
                    f"RDMA weight version mismatch: expected {expected_version}, "
                    f"got {version}"
                )
            if command == _CMD_END:
                # The contract is [END, 0, 0, version]. Anything else is a
                # corrupt stream or a mismatched sender, not a clean finish.
                if failure is None and (metadata_bytes != 0 or payload_bytes != 0):
                    failure = RuntimeError(
                        f"invalid RDMA end marker: metadata_bytes={metadata_bytes} "
                        f"payload_bytes={payload_bytes}, both must be 0"
                    )
                break
            frame = (
                f"command={command} metadata_bytes={metadata_bytes} "
                f"payload_bytes={payload_bytes}"
            )
            # Checked before anything is allocated: an allocation that fails
            # here takes this rank out of the broadcast the sender has begun.
            reason = _unreceivable(device, metadata_bytes, payload_bytes)
            if reason is not None:
                raise RDMAStreamOutOfStep(
                    f"RDMA weight frame cannot be received: {frame} ({reason})"
                )

            # Every frame but the end marker is followed by these two
            # broadcasts, so they are received before the frame is judged: one
            # this rank cannot use still has to keep it in step.
            try:
                metadata_tensor = torch.empty(
                    metadata_bytes, dtype=torch.uint8, device=device
                )
                payload = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
            except RuntimeError as exc:  # out of memory, within the bounds
                raise RDMAStreamOutOfStep(
                    f"RDMA weight frame cannot be received: {frame} ({exc})"
                ) from exc
            dist.broadcast(metadata_tensor, src=0, group=group)
            dist.broadcast(payload, src=0, group=group)
            if failure is None and (
                command != _CMD_BUCKET or metadata_bytes == 0 or payload_bytes == 0
            ):
                failure = RuntimeError(f"invalid RDMA weight header: {frame}")
            if failure is not None:
                drained += 1
                continue

            try:
                weights = _decode_bucket(metadata_tensor, payload)
                runner.apply_weight_bucket(weights, payload_bytes=payload_bytes)
            except Exception as exc:  # noqa: BLE001 - raised at the end marker
                # Kept until the end marker, so the frames it was raised
                # through must not keep this bucket alive while the rest of
                # the stream is drained.
                traceback.clear_frames(exc.__traceback__)
                failure = exc
                logger.error(
                    "RDMA weight bucket %d failed on this rank, receiving the "
                    "rest of the stream before reporting it: %s",
                    total_buckets + 1,
                    exc,
                )
                continue

            total_bytes += payload_bytes
            total_weights += len(weights)
            total_buckets += 1

        if failure is not None:
            if drained:
                logger.error(
                    "RDMA weight stream v%d: discarded %d bucket(s) after the "
                    "failure",
                    expected_version,
                    drained,
                )
            raise failure

        manifest = runner.commit_weight_update(expected_version)
        # Inside the try: the writes are asynchronous, so a device fault in
        # them, or in commit's own finalisation, surfaces only here, after
        # commit has declared the version good.
        torch.cuda.synchronize(device)
    except Exception as exc:
        if began:
            # Fence before re-raising: the parameters are now a mix of
            # versions, so serving must stop until a later full reload succeeds.
            runner.abort_weight_update(expected_version, exc)
        raise

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
        within the replica accounts for any local DP, because a replica running
        DP internally contributes one engine's worth of ranks per DP rank.
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

        config = getattr(self, "config", None)
        parallel = getattr(config, "parallel_config", None)
        dp_rank_local = int(getattr(parallel, "data_parallel_rank_local", 0) or 0)
        # The engine's own stride, which ModelRunner places devices by: one
        # worker per TP shard that has a process, per prefill-context rank. The
        # logical TP width over-counts under simulated TP and under-counts under
        # PCP, where one DP replica then takes another's ranks.
        workers = int(getattr(config, "tp_world_size", 1) or 1) * int(
            getattr(config, "prefill_context_parallel_size", 1) or 1
        )
        rank = int(base_rank) + dp_rank_local * workers + int(self.rank)

        if rank <= 0 or rank >= int(world_size):
            raise ValueError(
                f"invalid RDMA rank {rank} for world_size={world_size} "
                f"(base_rank={base_rank}, dp_local={dp_rank_local}, "
                f"worker={self.rank}, workers_per_engine={workers}); "
                f"rank 0 is the trainer"
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
        """Receive one version of the weights into resident memory.

        Commits only a stream that rewrote every parameter. ``verify_full_load``
        is accepted because the caller drives the vLLM worker with the same
        arguments, but it cannot turn that off: a commit of part of the model
        would serve the rest from another version.
        """
        label = getattr(self, "label", "runner")
        groups = getattr(self, "_rdma_weight_groups", {})
        if group_name not in groups:
            raise RuntimeError(
                f"RDMA weight group is not initialized: {group_name!r}; "
                "call init_rdma_weight_group first"
            )
        if not verify_full_load:
            logger.warning(
                "%s: verify_full_load=False ignored; an RDMA reload commits only "
                "once it has rewritten every parameter",
                label,
            )
        try:
            stats = receive_weight_stream(
                groups[group_name],
                self,
                device=self.device,
                expected_version=int(version),
            )
        except RDMAStreamOutOfStep as exc:
            logger.error(
                "%s: left RDMA weight group %s, which has to be initialized "
                "again: %s",
                label,
                group_name,
                exc,
            )
            self.destroy_rdma_weight_group(group_name)
            raise
        logger.info(
            "%s: RDMA weight reload v%d committed: %s", label, int(version), stats
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
