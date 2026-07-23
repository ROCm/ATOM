# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end GPU test for the DSV4 offload-unit GPU connector (save/load)."""

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.offload.dsv4.unit import (
    LAYOUT_VERSION,
    DSV4OffloadUnitGeometry,
)
from atom.kv_transfer.offload.dsv4.unit_codec import (
    DSV4OffloadUnitCodec,
    DSV4OffloadUnitError,
)
from atom.kv_transfer.offload.dsv4.gpu_connector import (
    DSV4OffloadUnitGPUConnector,
    SourceRegion,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="connector data path requires GPU"
)


def _geometry() -> DSV4OffloadUnitGeometry:
    return DSV4OffloadUnitGeometry(
        model_name="deepseek-v4",
        layout_version=LAYOUT_VERSION,
        num_layers=4,
        head_dim=128,
        window_size=128,
        block_size=256,
        swa_block_size=256,
        k1_csa=64,
        k2_hca=2,
        kv_dtype="bf16",
        compress_ratios=(4, 128, 4, 128),
        tp_size=1,
        tp_rank=0,
    )


def _blocks(num_blocks: int, unit_bytes: int) -> torch.Tensor:
    return torch.randint(
        0, 256, (num_blocks * unit_bytes,), dtype=torch.uint8, device="cuda"
    )


def _setup():
    codec = DSV4OffloadUnitCodec(_geometry())
    conn = DSV4OffloadUnitGPUConnector(codec, device="cuda")
    comp = _blocks(4, 512)
    swa = _blocks(2, 384)
    state = _blocks(1, 1024)
    compressed = [SourceRegion(comp.data_ptr(), 512, [0, 1, 2, 3])]
    swa_r = [SourceRegion(swa.data_ptr(), 384, [0, 1])]
    state_r = [SourceRegion(state.data_ptr(), 1024, [0])]
    layout = codec.build_layout(
        compressed_bytes=4 * 512, swa_bytes=2 * 384, csa_state_bytes=1024
    )
    return codec, conn, comp, swa, state, compressed, swa_r, state_r, layout


def test_connector_save_produces_valid_unit():
    codec, conn, comp, swa, state, compressed, swa_r, state_r, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)  # host MemoryObj
    conn.save_unit(
        memory_obj=mem,
        compressed=compressed,
        swa=swa_r,
        csa_state=state_r,
        boundary_B=1024,
    )
    got_c, got_s, got_state = codec.disassemble(mem, expect_boundary_B=1024)
    assert torch.equal(got_c, comp.cpu())
    assert torch.equal(got_s, swa.cpu())
    assert torch.equal(got_state, state.cpu())


def test_connector_load_round_trips_to_fresh_kv():
    codec, conn, comp, swa, state, compressed, swa_r, state_r, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)
    conn.save_unit(
        memory_obj=mem,
        compressed=compressed,
        swa=swa_r,
        csa_state=state_r,
        boundary_B=1024,
    )
    comp2 = torch.zeros_like(comp)
    swa2 = torch.zeros_like(swa)
    state2 = torch.zeros_like(state)
    conn.load_unit(
        memory_obj=mem,
        compressed=[SourceRegion(comp2.data_ptr(), 512, [0, 1, 2, 3])],
        swa=[SourceRegion(swa2.data_ptr(), 384, [0, 1])],
        csa_state=[SourceRegion(state2.data_ptr(), 1024, [0])],
        expect_boundary_B=1024,
    )
    torch.cuda.synchronize()
    assert torch.equal(comp2, comp)
    assert torch.equal(swa2, swa)
    assert torch.equal(state2, state)


def test_connector_load_corrupt_fails_closed():
    codec, conn, comp, swa, state, compressed, swa_r, state_r, layout = _setup()
    mem = torch.empty(layout.total_bytes, dtype=torch.uint8)
    conn.save_unit(
        memory_obj=mem,
        compressed=compressed,
        swa=swa_r,
        csa_state=state_r,
        boundary_B=1024,
    )
    mem[layout.compressed_off + 5] ^= 0xFF
    with pytest.raises(DSV4OffloadUnitError, match="CRC"):
        conn.load_unit(
            memory_obj=mem,
            compressed=[SourceRegion(comp.data_ptr(), 512, [0, 1, 2, 3])],
            swa=[SourceRegion(swa.data_ptr(), 384, [0, 1])],
            csa_state=[SourceRegion(state.data_ptr(), 1024, [0])],
            expect_boundary_B=1024,
        )


def test_connector_size_mismatch_fails():
    codec, conn, comp, swa, state, compressed, swa_r, state_r, layout = _setup()
    mem = torch.empty(layout.total_bytes + 256, dtype=torch.uint8)
    with pytest.raises(DSV4OffloadUnitError, match="size"):
        conn.save_unit(
            memory_obj=mem,
            compressed=compressed,
            swa=swa_r,
            csa_state=state_r,
            boundary_B=1024,
        )
