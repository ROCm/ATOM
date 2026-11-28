# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Bucketed BF16 weight broadcast over an isolated RCCL process group."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist

from atom.rollout import rdma_protocol

STAGING_MODES = ("none", "cpu", "gpu")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _missing_reloadable_names(
    expected_names: set[str],
    loaded_names: set[str],
    *,
    streamed_scales: bool = False,
) -> list[str]:
    """Return missing checkpoint weights, excluding generated/static parameters."""
    from atomrl.engine.inference.vllm_fp8_utils import (
        TensorNameClass,
        classify_tensor_name,
    )
    from atomrl.engine.inference.vllm_moe_weight_sync import _coverage_aliases

    missing = {
        name
        for name in expected_names - loaded_names
        if loaded_names.isdisjoint(_coverage_aliases(name))
    }
    return sorted(
        name
        for name in missing
        if classify_tensor_name(
            name,
            streamed_scales=streamed_scales,
        )
        is TensorNameClass.RELOADABLE
    )


def _encode_bucket(
    weights: list[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not weights:
        raise ValueError("cannot encode an empty weight bucket")
    prepared: list[tuple[str, torch.Tensor]] = []
    entries: list[dict[str, Any]] = []
    offset = 0
    for name, tensor in weights:
        value = tensor.detach()
        if not value.is_cuda:
            value = value.to("cuda", non_blocking=True)
        value = value.contiguous()
        nbytes = _tensor_nbytes(value)
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "offset": offset,
                "nbytes": nbytes,
            }
        )
        prepared.append((name, value))
        offset += nbytes

    device = prepared[0][1].device
    payload = torch.empty(offset, dtype=torch.uint8, device=device)
    for entry, (_, value) in zip(entries, prepared):
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        payload[start:end].copy_(value.view(torch.uint8).reshape(-1))

    raw_meta = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    metadata = torch.tensor(list(raw_meta), dtype=torch.uint8, device=device)
    return metadata, payload


def _broadcast_header(
    group,
    *,
    device: torch.device,
    command: int = 0,
    metadata_bytes: int = 0,
    payload_bytes: int = 0,
    version: int = 0,
) -> torch.Tensor:
    header = torch.tensor(
        [command, metadata_bytes, payload_bytes, version],
        dtype=torch.int64,
        device=device,
    )
    dist.broadcast(header, src=0, group=group)
    return header


@torch.no_grad()
def send_weight_stream(
    group,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    bucket_size_bytes: int,
    version: int,
) -> dict[str, float]:
    """Pack HF-named tensors into bounded GPU buckets and RCCL-broadcast them.

    Timings: ``gather_s`` and ``pack_s`` are host time spent producing tensors
    and packing buckets. ``bcast_s`` is GPU time across the broadcasts, from
    CUDA events (a host sync would serialize packing with sending), and
    includes waiting for receivers to post theirs. ``final_sync_s`` is the
    drain after the end marker.
    """
    if bucket_size_bytes <= 0:
        raise ValueError("bucket_size_bytes must be positive")
    device = torch.device("cuda", torch.cuda.current_device())
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_bytes = 0
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    gather_s = 0.0
    pack_s = 0.0
    bcast_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    started = time.perf_counter()

    def flush() -> None:
        nonlocal bucket, bucket_bytes, total_bytes, total_weights, total_buckets, pack_s
        if not bucket:
            return
        t0 = time.perf_counter()
        metadata, payload = _encode_bucket(bucket)
        pack_s += time.perf_counter() - t0
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        _broadcast_header(
            group,
            device=device,
            command=rdma_protocol.CMD_BUCKET,
            metadata_bytes=metadata.numel(),
            payload_bytes=payload.numel(),
            version=version,
        )
        dist.broadcast(metadata, src=0, group=group)
        dist.broadcast(payload, src=0, group=group)
        end.record()
        bcast_events.append((begin, end))
        total_bytes += payload.numel()
        total_weights += len(bucket)
        total_buckets += 1
        bucket = []
        bucket_bytes = 0

    it = iter(weights)
    while True:
        t0 = time.perf_counter()
        try:
            name, tensor = next(it)
        except StopIteration:
            break
        finally:
            gather_s += time.perf_counter() - t0
        nbytes = _tensor_nbytes(tensor)
        if bucket and bucket_bytes + nbytes > bucket_size_bytes:
            flush()
        bucket.append((name, tensor))
        bucket_bytes += nbytes
        if bucket_bytes >= bucket_size_bytes:
            flush()
    flush()
    _broadcast_header(
        group, device=device, command=rdma_protocol.CMD_END, version=version
    )
    t0 = time.perf_counter()
    torch.cuda.synchronize(device)
    final_sync_s = time.perf_counter() - t0
    elapsed = time.perf_counter() - started
    bcast_s = sum(b.elapsed_time(e) for b, e in bcast_events) / 1e3
    return {
        "version": float(version),
        "buckets": float(total_buckets),
        "weights": float(total_weights),
        "bytes": float(total_bytes),
        "seconds": elapsed,
        "gbps": (total_bytes * 8 / 1e9 / elapsed) if elapsed > 0 else 0.0,
        "gather_s": gather_s,
        "pack_s": pack_s,
        "bcast_s": bcast_s,
        "final_sync_s": final_sync_s,
    }


@torch.no_grad()
def receive_weight_stream(
    group,
    model,
    *,
    device: torch.device,
    expected_version: int,
    verify_full_load: bool = True,
    streamed_scales: bool = False,
    fingerprint_tracker=None,
    finalize_fingerprints: bool = True,
    staging: str = "cpu",
    moe_router=None,
) -> dict[str, Any]:
    """Receive RCCL buckets and load them directly into a resident vLLM model.

    ``staging`` decides what ``load_weights`` gets for each tensor: ``none``
    passes views into the bucket payload (safe only when the loaders copy
    before returning, as on BF16 and prequantized FP8), ``cpu`` and ``gpu``
    pass copies. Online-FP8 reload holds tensors until a module is complete,
    and a held view would keep its whole payload alive.

    ``moe_router`` (a ``FusedMoEWeightRouter``) loads the fused expert tensors
    an FSDP2 trainer sends, which ``model.load_weights`` would drop.
    """
    from atomrl.engine.inference.vllm_fp8_utils import ReloadFingerprintTracker

    if staging not in STAGING_MODES:
        raise ValueError(
            f"invalid RDMA staging mode {staging!r}; expected one of {STAGING_MODES}"
        )
    if not finalize_fingerprints and fingerprint_tracker is None:
        raise ValueError(
            "finalize_fingerprints=False requires an external fingerprint_tracker"
        )
    fingerprints = fingerprint_tracker
    if fingerprints is None:
        fingerprints = ReloadFingerprintTracker(
            model,
            streamed_scales=streamed_scales,
        )
    elif fingerprints.streamed_scales != bool(streamed_scales):
        raise ValueError(
            "fingerprint_tracker streamed_scales does not match RDMA stream mode"
        )
    loaded_names: set[str] = set()
    total_bytes = 0
    total_weights = 0
    total_buckets = 0
    # Host time inside this generator, by phase; whatever load_weights spends
    # outside it is its own loading. Each phase ends in a host sync (header,
    # metadata, .cpu()), so wall time is the transfer time.
    timing = {
        "wait_s": 0.0,
        "recv_s": 0.0,
        "host_copy_s": 0.0,
        "observe_s": 0.0,
        "route_s": 0.0,
    }
    started = time.perf_counter()
    last_source: dict[str, Any] | None = None

    def received_weights():
        nonlocal total_bytes, total_weights, total_buckets, last_source
        while True:
            t0 = time.perf_counter()
            header = _broadcast_header(group, device=device)
            command, metadata_bytes, payload_bytes, version = [
                int(x) for x in header.cpu().tolist()
            ]
            timing["wait_s"] += time.perf_counter() - t0
            if version != expected_version:
                raise RuntimeError(
                    f"RDMA weight version mismatch: expected {expected_version}, got {version}"
                )
            if command == rdma_protocol.CMD_END:
                return
            if (
                command != rdma_protocol.CMD_BUCKET
                or metadata_bytes <= 0
                or payload_bytes <= 0
            ):
                raise RuntimeError(f"invalid RDMA weight header: {header.tolist()}")

            t0 = time.perf_counter()
            metadata_tensor = torch.empty(
                metadata_bytes,
                dtype=torch.uint8,
                device=device,
            )
            payload = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
            dist.broadcast(metadata_tensor, src=0, group=group)
            dist.broadcast(payload, src=0, group=group)
            metadata = json.loads(bytes(metadata_tensor.cpu().tolist()).decode("utf-8"))
            timing["recv_s"] += time.perf_counter() - t0
            total_bytes += payload_bytes
            total_weights += len(metadata)
            total_buckets += 1
            for entry in metadata:
                last_source = entry
                dtype = getattr(torch, entry["dtype"])
                start = int(entry["offset"])
                end = start + int(entry["nbytes"])
                t0 = time.perf_counter()
                value = payload[start:end].view(dtype).view(entry["shape"])
                if staging == "cpu":
                    value = value.cpu()
                elif staging == "gpu":
                    value = value.clone()
                timing["host_copy_s"] += time.perf_counter() - t0
                if os.environ.get("ATOMRL_WEIGHT_SYNC_INTEGRITY", "0") == "1":
                    from atomrl.engine.inference.weight_integrity import (
                        sampled_tensor_integrity,
                    )

                    integrity = sampled_tensor_integrity(entry["name"], value)
                    if not integrity["all_finite"]:
                        raise FloatingPointError(
                            "non-finite weight at stage=rdma_recv "
                            f"bucket={total_buckets - 1} "
                            f"tensor={entry['name']}: {integrity}"
                        )
                t0 = time.perf_counter()
                fingerprints.observe_source([(entry["name"], value)])
                timing["observe_s"] += time.perf_counter() - t0
                to_load = [(entry["name"], value)]
                if moe_router is not None:
                    t0 = time.perf_counter()
                    to_load, moe_loaded = moe_router.route(to_load)
                    loaded_names.update(moe_loaded)
                    timing["route_s"] += time.perf_counter() - t0
                yield from to_load

    t_load = time.perf_counter()
    try:
        loaded = model.load_weights(received_weights())
    except Exception as exc:
        if last_source is None:
            raise
        raise RuntimeError(
            "vLLM model.load_weights failed while loading RDMA tensor: "
            f"source={last_source['name']}; "
            f"shape={tuple(last_source['shape'])}; "
            f"dtype=torch.{last_source['dtype']}"
        ) from exc
    load_total_s = time.perf_counter() - t_load
    if loaded is None:
        raise RuntimeError(
            "vLLM model.load_weights returned no manifest during RDMA reload"
        )
    loaded_names.update(loaded)
    t_verify = time.perf_counter()

    # The transfer protocol mode is authoritative. Source observation may
    # classify individual FP8 tensors, but it must not change whether scale
    # tensors are required for the stream as a whole.
    fingerprints.streamed_scales = bool(streamed_scales)
    if verify_full_load:
        expected_names = {name for name, _ in model.named_parameters()}
        if streamed_scales:
            expected_names.update(
                name
                for name, _ in model.named_buffers()
                if name.endswith("weight_scale_inv")
            )
        missing = _missing_reloadable_names(
            expected_names,
            loaded_names,
            streamed_scales=streamed_scales,
        )
        if missing:
            raise RuntimeError(
                "Incomplete vLLM RDMA weight reload: "
                f"streamed_scales={bool(streamed_scales)}; "
                f"loaded {len(loaded_names)}/{len(expected_names)} internal "
                f"parameters from {total_weights} HF tensors; missing={missing[:20]}"
            )
    torch.cuda.synchronize(device)
    verify_s = time.perf_counter() - t_verify
    elapsed = time.perf_counter() - started
    stats = {
        "version": float(expected_version),
        "buckets": float(total_buckets),
        "weights": float(total_weights),
        "bytes": float(total_bytes),
        "seconds": elapsed,
        "gbps": (total_bytes * 8 / 1e9 / elapsed) if elapsed > 0 else 0.0,
        "loaded_internal": float(len(loaded_names)),
        **timing,
        "load_weights_s": max(0.0, load_total_s - sum(timing.values())),
        "verify_s": verify_s,
    }
    if finalize_fingerprints:
        t0 = time.perf_counter()
        stats["verification"] = fingerprints.finalize()
        stats["finalize_s"] = time.perf_counter() - t0
    return stats
