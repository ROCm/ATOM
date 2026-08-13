# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip(
    "torch",
    reason="real PyTorch with ROCm or CUDA is required for GPU+disk integration",
)
_HAS_SUPPORTED_GPU = not (
    not hasattr(torch, "version")
    or (
        getattr(torch.version, "hip", None) is None
        and getattr(torch.version, "cuda", None) is None
    )
    or not hasattr(torch, "cuda")
    or not torch.cuda.is_available()
)
_LMCACHE_AVAILABLE = importlib.util.find_spec("lmcache") is not None
_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
if _LMCACHE_AVAILABLE:
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.metadata import LMCacheMetadata
if _TRITON_AVAILABLE:
    from atom.kv_transfer.offload import triton_kv_staging

    assert callable(triton_kv_staging.fused_pack_chunk_major)
    assert callable(triton_kv_staging.gather_copy_plan)

_SKIP_REASON = (
    "a ROCm or CUDA GPU is required"
    if not _HAS_SUPPORTED_GPU
    else (
        "the external lmcache package is required for GPU+disk integration"
        if not _LMCACHE_AVAILABLE
        else "the external triton package is required for GPU+disk integration"
    )
)
pytestmark = pytest.mark.skipif(
    not _HAS_SUPPORTED_GPU or not _LMCACHE_AVAILABLE or not _TRITON_AVAILABLE,
    reason=_SKIP_REASON,
)

from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload.atom_kv_byte_codec import ATOMKVByteCodec
from atom.kv_transfer.offload.atom_lmcache_gpu_connector import (
    ATOMLMCacheGPUConnector,
)
from atom.kv_transfer.offload.atom_page_region_codec import ATOMPageRegionCodec
from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata
from atom.kv_transfer.offload.slot_sidecar_codec import ATOMSlotSidecarCodec
from atom.kv_transfer.offload.slot_sidecar_format import (
    SlotSidecarHeader,
    SlotSidecarKey,
    decode_sidecar,
    encode_sidecar,
)
from atom.kv_transfer.offload.slot_sidecar_store import SlotSidecarStore

_FINGERPRINT = bytes.fromhex("00112233445566778899aabbccddeeff")


def _uint8_payload(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    count = math.prod(shape)
    return (
        torch.arange(count, dtype=torch.int64, device="cuda")
        .add(offset)
        .remainder(251)
        .to(torch.uint8)
        .reshape(shape)
    )


def _make_aiter_kv_caches(
    *,
    num_blocks: int,
    block_size: int,
) -> dict[str, SimpleNamespace]:
    num_heads = 2
    head_dim = 16
    pack = 16
    caches: dict[str, SimpleNamespace] = {}
    for layer_id in range(2):
        base = layer_id * 47
        caches[f"layer-{layer_id}"] = SimpleNamespace(
            # AITER x-packed K: (blocks, heads, head_dim/x, block_size, x).
            k_cache=_uint8_payload(
                (num_blocks, num_heads, head_dim // pack, block_size, pack),
                base,
            ),
            # AITER head-major V: (blocks, heads, head_dim, block_size).
            v_cache=_uint8_payload(
                (num_blocks, num_heads, head_dim, block_size),
                base + 13,
            ),
            k_scale=torch.linspace(
                0.25 + layer_id,
                1.75 + layer_id,
                steps=num_blocks * num_heads,
                dtype=torch.float32,
                device="cuda",
            ).reshape(num_blocks, num_heads),
            v_scale=torch.linspace(
                2.25 + layer_id,
                3.75 + layer_id,
                steps=num_blocks * num_heads,
                dtype=torch.float32,
                device="cuda",
            ).reshape(num_blocks, num_heads),
        )
    return caches


def _clone_segments(
    kv_caches: dict[str, SimpleNamespace],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        layer_name: {
            field: getattr(cache, field).clone()
            for field in ("k_cache", "v_cache", "k_scale", "v_scale")
        }
        for layer_name, cache in kv_caches.items()
    }


def _wait_for_disk_hit(
    engine,
    tokens: list[int],
    disk_path: Path,
    *,
    timeout_s: float = 20.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        files = list(disk_path.rglob("*.pt"))
        if len(files) == 2 and engine.lookup(
            tokens, search_range=["LocalDiskBackend"]
        ) == len(tokens):
            return files
        time.sleep(0.01)
    return list(disk_path.rglob("*.pt"))


def _wait_for_sidecar_hit(
    store: SlotSidecarStore,
    key: SlotSidecarKey,
    disk_path: Path,
    *,
    min_files: int,
    timeout_s: float = 20.0,
) -> list[Path]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        files = list(disk_path.rglob("*.pt"))
        if len(files) >= min_files and store.contains(key):
            return files
        time.sleep(0.01)
    return list(disk_path.rglob("*.pt"))


def _synchronize_producer_stream() -> None:
    """Mirror production's producer-event fence before background packing."""

    producer_event = torch.cuda.Event()
    producer_event.record(torch.cuda.current_stream())
    producer_event.synchronize()


def test_gpu_kv_round_trip_through_local_disk_backend(tmp_path: Path):
    """Round-trip AITER-layout GPU KV bytes through LMCache local disk."""
    num_blocks = 4
    block_size = 4
    chunk_size = 8
    tokens = list(range(num_blocks * block_size))
    block_ids = list(range(num_blocks))
    instance_id = f"atom-gpu-disk-e2e-{uuid.uuid4().hex}"

    kv_caches = _make_aiter_kv_caches(
        num_blocks=num_blocks,
        block_size=block_size,
    )
    expected = _clone_segments(kv_caches)
    codec = ATOMKVByteCodec(kv_caches, num_blocks=num_blocks)
    assert codec.has_fused_chunk_major_staging
    gpu_connector = ATOMLMCacheGPUConnector(
        codec,
        block_size=block_size,
        chunk_size=chunk_size,
    )

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=False,
        max_local_cpu_size=0.01,
        local_disk=str(tmp_path),
        max_local_disk_size=0.01,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
        use_gds=False,
    )
    base_metadata = LMCacheMetadata(
        model_name="atom-gpu-disk-e2e",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(2, 2, chunk_size, 2, 16),
        chunk_size=chunk_size,
        engine_id=instance_id,
    )
    metadata = ATOMRawBytesLMCacheMetadata(
        base_metadata,
        atom_block_size=block_size,
        bytes_per_block=codec.bytes_per_block,
    )

    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        gpu_connector,
        lambda tensor, source: None,
        lambda obj, source: obj,
    )
    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        assert engine.storage_manager is not None
        assert "LocalDiskBackend" in engine.storage_manager.list_backends()

        _synchronize_producer_stream()
        engine.store(tokens, block_ids=block_ids)
        disk_files = _wait_for_disk_hit(engine, tokens, tmp_path)

        assert len(disk_files) == 2
        assert all(path.stat().st_size > 0 for path in disk_files)
        engine.set_hot_cache(False)
        assert engine.lookup(tokens, search_range=["LocalCPUBackend"]) == 0

        for cache in kv_caches.values():
            cache.k_cache.zero_()
            cache.v_cache.zero_()
            cache.k_scale.zero_()
            cache.v_scale.zero_()
        torch.cuda.synchronize()

        ret_mask = engine.retrieve(tokens, block_ids=block_ids)
        torch.cuda.synchronize()

        assert bool(ret_mask.all())
        for layer_name, cache in kv_caches.items():
            for field, expected_tensor in expected[layer_name].items():
                assert torch.equal(getattr(cache, field), expected_tensor), (
                    "GPU KV mismatch after LocalDiskBackend round trip: "
                    f"{layer_name}.{field}"
                )
    finally:
        LMCacheEngineBuilder.destroy(instance_id)


def test_page_major_page_and_full_slot_round_trip_through_local_disk_backend(
    tmp_path: Path,
):
    """Round-trip DSV4-style page regions plus one complete AOS1 SLOT."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_blocks = 4
    block_size = 4
    chunk_size = 8
    tokens = list(range(num_blocks * block_size))
    block_ids = list(range(num_blocks))
    instance_id = f"atom-page-slot-gpu-disk-e2e-{uuid.uuid4().hex}"
    model_name = "atom-page-slot-gpu-disk-e2e"

    page_tensors = [
        _uint8_payload((num_blocks, 193), 11),
        _uint8_payload((num_blocks, 127), 79),
    ]
    expected_pages = [tensor.clone() for tensor in page_tensors]
    page_regions = [
        KVTransferRegion(
            base_addr=tensor.data_ptr(),
            total_bytes=tensor.numel(),
            unit_bytes=tensor.shape[1],
        )
        for tensor in page_tensors
    ]
    page_codec = ATOMPageRegionCodec(
        page_regions,
        num_blocks=num_blocks,
        device=device,
    )

    num_slots = 3
    slot_tensors = [
        _uint8_payload((num_slots, 181), 37),
        _uint8_payload((num_slots, 109), 151),
    ]
    expected_slots = [tensor.clone() for tensor in slot_tensors]
    slot_regions = [
        KVTransferRegion(
            base_addr=tensor.data_ptr(),
            total_bytes=tensor.numel(),
            unit_bytes=tensor.shape[1],
            reverse_indexed=True,
        )
        for tensor in slot_tensors
    ]
    slot_codec = ATOMSlotSidecarCodec(
        slot_regions,
        num_slots=num_slots,
        staging_slots=1,
        device=device,
    )
    assert page_codec.has_fused_chunk_major_staging
    assert slot_codec.has_fused_copy_plan_staging

    gpu_connector = ATOMLMCacheGPUConnector(
        page_codec,
        block_size=block_size,
        chunk_size=chunk_size,
    )
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=False,
        max_local_cpu_size=0.01,
        local_disk=str(tmp_path),
        max_local_disk_size=0.01,
        store_location="LocalDiskBackend",
        retrieve_locations=["LocalDiskBackend"],
        use_gds=False,
    )
    base_metadata = LMCacheMetadata(
        model_name=model_name,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 2, chunk_size, 1, page_codec.bytes_per_block),
        chunk_size=chunk_size,
        engine_id=instance_id,
    )
    metadata = ATOMRawBytesLMCacheMetadata(
        base_metadata,
        atom_block_size=block_size,
        bytes_per_block=page_codec.bytes_per_block,
    )
    engine = LMCacheEngineBuilder.get_or_create(
        instance_id,
        config,
        metadata,
        gpu_connector,
        lambda tensor, source: None,
        lambda obj, source: obj,
    )

    try:
        engine.fmt = MemoryFormat.KV_2LTD
        engine.post_init()
        assert engine.storage_manager is not None
        assert "LocalDiskBackend" in engine.storage_manager.list_backends()

        _synchronize_producer_stream()
        engine.store(tokens, block_ids=block_ids)
        page_files = _wait_for_disk_hit(engine, tokens, tmp_path)
        assert len(page_files) == 2
        assert engine.lookup(tokens, search_range=["LocalDiskBackend"]) == len(tokens)

        source_group = 0
        destination_group = 1
        slot_codec.snapshot_to_staging(source_group, 0)
        torch.cuda.synchronize()
        payload = bytes(slot_codec.staging_view(0).cpu().tolist())
        expected_payload = bytes(
            torch.cat([tensor[-1] for tensor in expected_slots]).cpu().tolist()
        )
        assert payload == expected_payload

        boundary_hash = 0x0123456789ABCDEF
        sidecar_key = SlotSidecarKey(
            boundary_block_hash=boundary_hash,
            fingerprint=_FINGERPRINT,
            tp_size=1,
            tp_rank=0,
        )
        sidecar_blob = encode_sidecar(
            SlotSidecarHeader(
                boundary_tokens=len(tokens),
                boundary_block_hash=boundary_hash,
                payload_bytes=None,
                payload_crc32=None,
                fingerprint=_FINGERPRINT,
                tp_size=1,
                tp_rank=0,
            ),
            payload,
        )
        sidecar_store = SlotSidecarStore(
            engine,
            model_name=model_name,
            world_size=1,
            worker_id=0,
        )
        assert sidecar_store.put(sidecar_key, sidecar_blob)
        all_files = _wait_for_sidecar_hit(
            sidecar_store,
            sidecar_key,
            tmp_path,
            min_files=3,
        )
        assert len(all_files) >= 3

        engine.set_hot_cache(False)
        for tensor in page_tensors:
            tensor.zero_()
        destination_row = num_slots - destination_group - 1
        for tensor in slot_tensors:
            tensor[destination_row].zero_()
        torch.cuda.synchronize()

        ret_mask = engine.retrieve(tokens, block_ids=block_ids)
        loaded_sidecar = sidecar_store.get(sidecar_key)
        assert loaded_sidecar is not None
        _, restored_payload = decode_sidecar(
            memoryview(loaded_sidecar.numpy()),
            expected_fingerprint=_FINGERPRINT,
            expected_tp_size=1,
            expected_tp_rank=0,
            expected_boundary_tokens=len(tokens),
            expected_boundary_block_hash=boundary_hash,
            expected_payload_bytes=slot_codec.payload_bytes,
        )
        slot_codec.staging_view(0).copy_(
            torch.tensor(list(restored_payload), dtype=torch.uint8, device=device)
        )
        slot_codec.restore_from_staging(destination_group, 0)
        torch.cuda.synchronize()

        assert bool(ret_mask.all())
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(page_tensors, expected_pages, strict=True)
        )
        source_row = num_slots - source_group - 1
        assert all(
            torch.equal(actual[destination_row], expected[source_row])
            for actual, expected in zip(slot_tensors, expected_slots, strict=True)
        )
    finally:
        LMCacheEngineBuilder.destroy(instance_id)
