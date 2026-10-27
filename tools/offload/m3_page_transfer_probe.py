#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Manual single-GPU M3 PAGE roundtrip of the default BlockGPUConnector path.

Uses the actual DSV4PageSlotCodec used by M3, with 297 synthetic opaque PAGE
regions matching 60 K/V/FP32-scale sets and 57 BF16 index-K regions. It loads
no model. CPU fixture copies/checks and setup/cleanup fences are outside the
reported wall times. Run only when the operator has reserved the GPU.

This checks the copy/codec/staging contract, not scheduler lifetime races,
LMCache put/eviction, TP8 behavior, model outputs or serving throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--chunk-blocks", type=int, default=2)
    parser.add_argument("--staging-chunks", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--host-allocator", choices=("lmcache", "torch"), default="lmcache"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "blocks",
        "block_size",
        "chunk_blocks",
        "staging_chunks",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 1:
        parser.error("--warmup must be at least 1 (exclude Triton compilation)")
    if args.blocks <= args.chunk_blocks * args.staging_chunks:
        parser.error("--blocks must exercise reuse of the staging buffer")
    return args


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def run(args, report):
    # Imports are intentionally lazy: --help/compilation require no GPU stack.
    import torch

    import atom.kv_transfer.offload._block_gpu_connector as connector_module
    from atom.kv_transfer.disaggregation.types import KVTransferRegion
    from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
    from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec

    if not torch.cuda.is_available():
        raise RuntimeError("a reserved CUDA/HIP GPU is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    os.environ["OFFLOAD_GPU_STAGING_CHUNKS"] = str(args.staging_chunks)
    os.environ.pop("OFFLOAD_GPU_STAGING_MAX_BYTES", None)
    os.environ["OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER"] = "0"
    # Opaque per-block byte geometry is exact; tensor contents are synthetic.
    # FP8 K/V:128 bytes/token each; FP32 scales:4 each; index-K:128 BF16.
    roles = []
    for layer in range(60):
        for role, width in (("k", 128), ("v", 128), ("k_scale", 4), ("v_scale", 4)):
            roles.append((f"layer{layer}.{role}", width))
        if layer >= 3:
            roles.append((f"layer{layer}.index_k", 256))
    assert len(roles) == 297 and sum(width for _, width in roles) == 30432
    total_blocks = args.blocks + 2  # unselected first/last blocks are guards
    cpu_regions = []
    gpu_regions = []
    regions = []
    generator = torch.Generator(device="cpu").manual_seed(731)
    for role, width in roles:
        unit_bytes = width * args.block_size
        reference = torch.randint(
            0, 256, (total_blocks, unit_bytes), dtype=torch.uint8, generator=generator
        )
        tensor = reference.to(device)
        cpu_regions.append(reference)
        gpu_regions.append(tensor)
        regions.append(
            KVTransferRegion(
                base_addr=tensor.data_ptr(),
                total_bytes=tensor.numel(),
                unit_bytes=unit_bytes,
                semantic_role=role,
            )
        )
    codec = DSV4PageSlotCodec(
        page_regions=regions,
        slot_regions=(),
        num_blocks=total_blocks,
        num_slots=0,
        device=device,
    )
    if not codec.has_fused_chunk_major_staging:
        raise RuntimeError("the actual M3 fused PAGE codec is unavailable")
    assert codec.bytes_per_block == 30432 * args.block_size
    block_ids = list(range(1, args.blocks + 1))
    random.Random(731).shuffle(block_ids)
    selected = torch.tensor(block_ids, dtype=torch.long, device=device)
    chunk_size = args.block_size * args.chunk_blocks
    # Exercise a partial final physical block whenever the block has >1 token.
    token_count = args.blocks * args.block_size - int(args.block_size > 1)
    starts = list(range(0, token_count, chunk_size))
    ends = [min(start + chunk_size, token_count) for start in starts]
    sizes = [
        ((end + args.block_size - 1) // args.block_size - start // args.block_size)
        * codec.bytes_per_block
        for start, end in zip(starts, ends, strict=True)
    ]
    # Independent CPU oracle for the codec's item-major layout: every block
    # stores the ordered region bytes before advancing to the next block.
    # This matches the documented item_pos * bytes_per_item + region offset,
    # without using GPU gather/scatter to construct the expected host payload.
    references = [
        torch.cat(
            [
                reference[block_id]
                for block_id in block_ids[
                    start
                    // args.block_size : (end + args.block_size - 1)
                    // args.block_size
                ]
                for reference in cpu_regions
            ]
        )
        for start, end in zip(starts, ends, strict=True)
    ]
    guard_size = 64
    strides = [((size + guard_size + 4095) // 4096) * 4096 for size in sizes]
    pool_size = sum(strides)
    pool = None
    free_pool = None
    connectors = {}
    rows = []
    report.update(
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(device),
        device=str(device),
        region_count=len(regions),
        bytes_per_token=30432,
        block_size=args.block_size,
        bytes_per_block=codec.bytes_per_block,
        chunks=len(starts),
        payload_bytes=sum(sizes),
        host_pool_bytes=pool_size,
        host_allocator=args.host_allocator,
        block_ids=block_ids,
        token_count=token_count,
        connector_sha256=hashlib.sha256(
            Path(connector_module.__file__).read_bytes()
        ).hexdigest(),
    )
    try:
        if args.host_allocator == "lmcache":
            from lmcache.v1.memory_management import (
                _allocate_cpu_memory,
                _free_cpu_memory,
            )

            # Same raw pinned allocation + torch.frombuffer path as LMCache.
            # Keep the pool/view owners alive through the final stream fence.
            pool = _allocate_cpu_memory(pool_size)

            def free_pool():
                _free_cpu_memory(pool, pool_size)

        else:
            pool = torch.empty(pool_size, dtype=torch.uint8, pin_memory=True)
        memory_objs = []
        guards = []
        offset = 0
        for size, stride in zip(sizes, strides, strict=True):
            memory_objs.append(SimpleNamespace(tensor=pool[offset : offset + size]))
            guards.append(pool[offset + size : offset + size + guard_size])
            offset += stride
        physical_pins = [bool(obj.tensor.is_pinned()) for obj in memory_objs]
        report["tensor_physical_pins"] = physical_pins
        connector = BlockGPUConnector(codec, args.block_size, chunk_size=chunk_size)
        connectors["default"] = connector

        def assert_fast_path(direction, stats):
            if stats["stats_available"] != 1 or stats["transfer_succeeded"] != 1:
                raise AssertionError(f"{direction} transfer evidence unavailable")
            if stats["completed_bytes"] != sum(sizes):
                raise AssertionError(f"{direction} transfer byte count is incomplete")
            if stats["batch_block_ids_enabled"] != 1:
                raise AssertionError(f"{direction} did not use prepared block IDs")
            if stats["batch_id_uploads"] != 1:
                raise AssertionError(
                    f"{direction} expected exactly one block-ID upload"
                )
            if stats["batch_id_groups"] != stats["groups"]:
                raise AssertionError(f"{direction} did not prepare every staging group")
            expected_async = sum(physical_pins)
            if stats["async_host_copy_chunks"] != expected_async:
                raise AssertionError(
                    f"{direction} async-copy count differs from physical pin state"
                )
            expected_blocking = len(physical_pins) - expected_async
            if stats["blocking_host_copy_chunks"] != expected_blocking:
                raise AssertionError(f"{direction} blocking-copy count is incorrect")
            if stats["async_host_copy_enabled"] != 1:
                raise AssertionError(f"{direction} combined async path was not enabled")

        def roundtrip(iteration, warmup):
            # Fixture preparation and its fence are deliberately outside both
            # timed calls; do not rely on implicit default-stream ordering.
            for tensor, reference in zip(gpu_regions, cpu_regions, strict=True):
                tensor.copy_(reference)
            pool.fill_(0xA5)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            connector.batched_from_gpu(
                memory_objs,
                starts,
                ends,
                block_ids=block_ids,
                req_id=f"probe-{iteration}-d2h",
            )
            d2h_ms = (time.perf_counter() - started) * 1000
            d2h = connector.last_transfer_stats()
            assert_fast_path("d2h", d2h)
            # Host views are inspected only after the connector's final fence.
            for index, (obj, reference) in enumerate(
                zip(memory_objs, references, strict=True)
            ):
                if not torch.equal(obj.tensor, reference):
                    raise AssertionError(
                        f"D2H host chunk {index} differs from CPU oracle"
                    )
            if not all(bool(torch.all(guard == 0xA5)) for guard in guards):
                raise AssertionError("D2H overwrote a host guard")
            for tensor in gpu_regions:
                tensor.index_fill_(0, selected, 0)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            connector.batched_to_gpu(
                memory_objs,
                starts,
                ends,
                block_ids=block_ids,
                req_id=f"probe-{iteration}-h2d",
            )
            h2d_ms = (time.perf_counter() - started) * 1000
            h2d = connector.last_transfer_stats()
            assert_fast_path("h2d", h2d)
            for role, tensor, reference in zip(
                roles, gpu_regions, cpu_regions, strict=True
            ):
                if not torch.equal(tensor.cpu(), reference):
                    raise AssertionError(
                        f"roundtrip/guard mismatch in region {role[0]}"
                    )
            if not all(bool(torch.all(guard == 0xA5)) for guard in guards):
                raise AssertionError("H2D overwrote a host guard")
            row = {
                "iteration": iteration,
                "warmup": warmup,
                "byte_exact": True,
                "d2h_ms": d2h_ms,
                "h2d_ms": h2d_ms,
                "d2h": d2h,
                "h2d": h2d,
            }
            rows.append(row)
            report["rows"] = rows

        for iteration in range(args.warmup):
            roundtrip(iteration, True)
        for iteration in range(args.iterations):
            roundtrip(iteration, False)
        measured = [row for row in rows if not row["warmup"]]
        report["median_ms"] = {
            "d2h_ms": statistics.median(row["d2h_ms"] for row in measured),
            "h2d_ms": statistics.median(row["h2d_ms"] for row in measured),
        }
        report["byte_exact"] = True
        report["async_exercised"] = all(
            row[direction]["async_host_copy_chunks"] == len(physical_pins)
            for row in measured
            for direction in ("d2h", "h2d")
        )
        report["batch_ids_exercised"] = all(
            row[direction]["batch_id_uploads"] == 1
            and row[direction]["batch_id_groups"] == row[direction]["groups"]
            for row in measured
            for direction in ("d2h", "h2d")
        )
        report["status"] = (
            "pass"
            if report["async_exercised"] and report["batch_ids_exercised"]
            else "fast_path_not_exercised"
        )
    finally:
        # Free externally pinned host memory only after confirmed GPU quiescence.
        # This cleanup fence is outside the per-transfer measurements.
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:  # noqa: BLE001 - report and retain host memory
            report["cleanup_error"] = repr(exc)
            report["host_pool_free_skipped"] = True
        else:
            for connector in connectors.values():
                connector.close()
            if free_pool is not None:
                free_pool()


def main():
    args = arguments()
    report = {
        "status": "running",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Single-GPU synthetic opaque M3 PAGE geometry using the real fused codec; no model/TP/LMCache store/eviction test.",
    }
    try:
        run(args, report)
    except Exception:  # noqa: BLE001 - serialize probe failures and exit nonzero
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    rendered = json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "pass" and "cleanup_error" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
