# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end GPU test for the offload-unit GPU connector (save/load)."""

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.offload.hybrid.kv_bundle import BundleGeometry
from atom.kv_transfer.offload.hybrid.kv_bundle_codec import (
    BundleCodec,
    BundleError,
)
from atom.kv_transfer.offload.hybrid.gpu_connector import (
    BundleGPUConnector,
    SourceRegion,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="connector data path requires GPU"
)

NAMES = ["compressed", "swa", "csa_state"]


def _geometry() -> BundleGeometry:
    return BundleGeometry.build(
        model_tag="deepseek-v4",
        tp_size=1,
        tp_rank=0,
        fields=dict(num_layers=4, head_dim=128, kv_dtype="bf16", block_size=256),
    )


def _blocks(num_blocks: int, unit_bytes: int) -> torch.Tensor:
    return torch.randint(
        0, 256, (num_blocks * unit_bytes,), dtype=torch.uint8, device="cuda"
    )


def _setup():
    codec = BundleCodec(_geometry())
    conn = BundleGPUConnector(codec, device="cuda")
    comp = _blocks(4, 512)
    swa = _blocks(2, 384)
    state = _blocks(1, 1024)
    comps = [
        ("compressed", [SourceRegion(comp.data_ptr(), 512, [0, 1, 2, 3])]),
        ("swa", [SourceRegion(swa.data_ptr(), 384, [0, 1])]),
        ("csa_state", [SourceRegion(state.data_ptr(), 1024, [0])]),
    ]
    layout = codec.build_layout_n([("compressed", 4 * 512), ("swa", 2 * 384), ("csa_state", 1024)])
    return codec, conn, comp, swa, state, comps, layout


def test_connector_save_produces_valid_unit():
    codec, conn, comp, swa, state, comps, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)  # host MemoryObj
    conn.save_bundle(memory_obj=mem, components=comps, boundary_B=1024)
    got = codec.disassemble_n(mem, NAMES, expect_boundary_B=1024)
    assert torch.equal(got["compressed"], comp.cpu())
    assert torch.equal(got["swa"], swa.cpu())
    assert torch.equal(got["csa_state"], state.cpu())


def test_connector_load_round_trips_to_fresh_kv():
    codec, conn, comp, swa, state, comps, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)
    conn.save_bundle(memory_obj=mem, components=comps, boundary_B=1024)
    comp2 = torch.zeros_like(comp)
    swa2 = torch.zeros_like(swa)
    state2 = torch.zeros_like(state)
    dst = [
        ("compressed", [SourceRegion(comp2.data_ptr(), 512, [0, 1, 2, 3])]),
        ("swa", [SourceRegion(swa2.data_ptr(), 384, [0, 1])]),
        ("csa_state", [SourceRegion(state2.data_ptr(), 1024, [0])]),
    ]
    conn.load_bundle(memory_obj=mem, components=dst, expect_boundary_B=1024)
    torch.cuda.synchronize()
    assert torch.equal(comp2, comp)
    assert torch.equal(swa2, swa)
    assert torch.equal(state2, state)


def test_connector_load_corrupt_fails_closed():
    codec, conn, comp, swa, state, comps, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)
    conn.save_bundle(memory_obj=mem, components=comps, boundary_B=1024)
    mem[layout.components[0].off + 5] ^= 0xFF
    with pytest.raises(BundleError, match="CRC"):
        conn.load_bundle(memory_obj=mem, components=comps, expect_boundary_B=1024)


def test_connector_size_mismatch_fails():
    codec, conn, comp, swa, state, comps, layout = _setup()
    mem = torch.empty(layout.total_bytes + 256, dtype=torch.uint8)
    with pytest.raises(BundleError, match="size"):
        conn.save_bundle(memory_obj=mem, components=comps, boundary_B=1024)
