# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the DSV4 offload-unit codec (assemble/disassemble/fail-closed)."""

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


def _geometry(**overrides) -> DSV4OffloadUnitGeometry:
    base = dict(
        model_name="deepseek-v4",
        layout_version=LAYOUT_VERSION,
        num_layers=61,
        head_dim=128,
        window_size=128,
        block_size=256,
        swa_block_size=256,
        k1_csa=64,
        k2_hca=2,
        kv_dtype="bf16",
        compress_ratios=(0, 4, 128, 4, 128),
        tp_size=1,
        tp_rank=0,
    )
    base.update(overrides)
    return DSV4OffloadUnitGeometry(**base)


def _rand(n: int) -> torch.Tensor:
    return torch.randint(0, 256, (n,), dtype=torch.uint8)


def test_round_trip_recovers_components_byte_identical():
    codec = DSV4OffloadUnitCodec(_geometry())
    compressed = _rand(4096)
    swa = _rand(777)
    csa_state = _rand(129)
    unit = codec.assemble(
        compressed=compressed, swa=swa, csa_state=csa_state, boundary_B=16384
    )
    got_c, got_s, got_state = codec.disassemble(unit, expect_boundary_B=16384)
    assert torch.equal(got_c, compressed)
    assert torch.equal(got_s, swa)
    assert torch.equal(got_state, csa_state)


def test_round_trip_zero_sized_swa_and_state():
    codec = DSV4OffloadUnitCodec(_geometry())
    compressed = _rand(512)
    swa = torch.empty(0, dtype=torch.uint8)
    csa_state = torch.empty(0, dtype=torch.uint8)
    unit = codec.assemble(
        compressed=compressed, swa=swa, csa_state=csa_state, boundary_B=256
    )
    got_c, got_s, got_state = codec.disassemble(unit)
    assert torch.equal(got_c, compressed)
    assert got_s.numel() == 0
    assert got_state.numel() == 0


def test_assemble_into_preallocated_out():
    codec = DSV4OffloadUnitCodec(_geometry())
    compressed = _rand(1024)
    swa = _rand(256)
    csa_state = _rand(64)
    layout = codec.build_layout(
        compressed_bytes=1024, swa_bytes=256, csa_state_bytes=64
    )
    out = torch.empty(layout.total_bytes, dtype=torch.uint8)
    unit = codec.assemble(
        compressed=compressed,
        swa=swa,
        csa_state=csa_state,
        boundary_B=1024,
        out=out,
    )
    assert unit.data_ptr() == out.data_ptr()
    got_c, got_s, got_state = codec.disassemble(unit)
    assert torch.equal(got_c, compressed)
    assert torch.equal(got_s, swa)
    assert torch.equal(got_state, csa_state)


def test_corrupt_payload_fails_crc():
    codec = DSV4OffloadUnitCodec(_geometry())
    unit = codec.assemble(
        compressed=_rand(2048), swa=_rand(300), csa_state=_rand(64), boundary_B=4096
    )
    layout = codec.build_layout(
        compressed_bytes=2048, swa_bytes=300, csa_state_bytes=64
    )
    idx = layout.compressed_off + 10
    unit[idx] ^= 0xFF
    with pytest.raises(DSV4OffloadUnitError, match="CRC"):
        codec.disassemble(unit)


def test_fingerprint_mismatch_fails_closed():
    saver = DSV4OffloadUnitCodec(_geometry(tp_rank=0))
    unit = saver.assemble(
        compressed=_rand(512), swa=_rand(64), csa_state=_rand(64), boundary_B=512
    )
    # A different shard must refuse to load it.
    loader = DSV4OffloadUnitCodec(_geometry(tp_rank=1))
    with pytest.raises(DSV4OffloadUnitError, match="fingerprint"):
        loader.disassemble(unit)


def test_boundary_mismatch_fails():
    codec = DSV4OffloadUnitCodec(_geometry())
    unit = codec.assemble(
        compressed=_rand(512), swa=_rand(64), csa_state=_rand(64), boundary_B=512
    )
    with pytest.raises(DSV4OffloadUnitError, match="boundary_B"):
        codec.disassemble(unit, expect_boundary_B=1024)


def test_truncated_unit_fails_size_check():
    codec = DSV4OffloadUnitCodec(_geometry())
    unit = codec.assemble(
        compressed=_rand(512), swa=_rand(64), csa_state=_rand(64), boundary_B=512
    )
    with pytest.raises(DSV4OffloadUnitError):
        codec.disassemble(unit[:-1])


def test_wrong_out_size_rejected():
    codec = DSV4OffloadUnitCodec(_geometry())
    with pytest.raises(DSV4OffloadUnitError, match="out size"):
        codec.assemble(
            compressed=_rand(512),
            swa=_rand(64),
            csa_state=_rand(64),
            boundary_B=512,
            out=torch.empty(16, dtype=torch.uint8),
        )
