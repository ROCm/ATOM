# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the offload-unit codec (assemble/disassemble/fail-closed)."""

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.offload.hybrid.kv_bundle import BundleGeometry
from atom.kv_transfer.offload.hybrid.kv_bundle_codec import (
    BundleCodec,
    BundleError,
)

NAMES = ["compressed", "swa", "csa_state"]


def _geometry(**overrides) -> BundleGeometry:
    fields = dict(num_layers=61, head_dim=128, kv_dtype="bf16", block_size=256)
    kw = dict(model_tag="deepseek-v4", tp_size=1, tp_rank=0)
    kw.update(overrides)
    return BundleGeometry.build(fields=fields, **kw)


def _rand(n: int) -> torch.Tensor:
    return torch.randint(0, 256, (n,), dtype=torch.uint8)


def _comps(*sizes):
    return [(NAMES[i], _rand(n)) for i, n in enumerate(sizes)]


def test_round_trip_recovers_components_byte_identical():
    codec = BundleCodec(_geometry())
    comps = _comps(4096, 777, 129)
    unit = codec.assemble_n(comps, boundary_B=16384)
    got = codec.disassemble_n(unit, NAMES, expect_boundary_B=16384)
    for name, t in comps:
        assert torch.equal(got[name], t)


def test_round_trip_zero_sized_components():
    codec = BundleCodec(_geometry())
    comps = [("compressed", _rand(512)),
             ("swa", torch.empty(0, dtype=torch.uint8)),
             ("csa_state", torch.empty(0, dtype=torch.uint8))]
    unit = codec.assemble_n(comps, boundary_B=256)
    got = codec.disassemble_n(unit, NAMES)
    assert torch.equal(got["compressed"], comps[0][1])
    assert got["swa"].numel() == 0 and got["csa_state"].numel() == 0


def test_assemble_into_preallocated_out():
    codec = BundleCodec(_geometry())
    comps = _comps(1024, 256, 64)
    layout = codec.build_layout_n([(n, int(t.numel())) for n, t in comps])
    out = torch.empty(layout.total_bytes, dtype=torch.uint8)
    unit = codec.assemble_n(comps, boundary_B=1024, out=out)
    assert unit.data_ptr() == out.data_ptr()
    got = codec.disassemble_n(unit, NAMES)
    for name, t in comps:
        assert torch.equal(got[name], t)


def test_corrupt_payload_fails_crc():
    codec = BundleCodec(_geometry())
    comps = _comps(2048, 300, 64)
    unit = codec.assemble_n(comps, boundary_B=4096)
    layout = codec.build_layout_n([(n, int(t.numel())) for n, t in comps])
    unit[layout.components[0].off + 10] ^= 0xFF
    with pytest.raises(BundleError, match="CRC"):
        codec.disassemble_n(unit, NAMES)


def test_fingerprint_mismatch_fails_closed():
    saver = BundleCodec(_geometry(tp_rank=0))
    unit = saver.assemble_n(_comps(512, 64, 64), boundary_B=512)
    loader = BundleCodec(_geometry(tp_rank=1))
    with pytest.raises(BundleError, match="fingerprint"):
        loader.disassemble_n(unit, NAMES)


def test_boundary_mismatch_fails():
    codec = BundleCodec(_geometry())
    unit = codec.assemble_n(_comps(512, 64, 64), boundary_B=512)
    with pytest.raises(BundleError, match="boundary_B"):
        codec.disassemble_n(unit, NAMES, expect_boundary_B=1024)


def test_truncated_unit_fails_size_check():
    codec = BundleCodec(_geometry())
    unit = codec.assemble_n(_comps(512, 64, 64), boundary_B=512)
    with pytest.raises(BundleError):
        codec.disassemble_n(unit[:-1], NAMES)


def test_component_count_mismatch_fails():
    codec = BundleCodec(_geometry())
    unit = codec.assemble_n(_comps(512, 64, 64), boundary_B=512)
    with pytest.raises(BundleError):
        codec.disassemble_n(unit, ["compressed", "swa"])  # expected 3


def test_wrong_out_size_rejected():
    codec = BundleCodec(_geometry())
    with pytest.raises(BundleError, match="out size"):
        codec.assemble_n(
            _comps(512, 64, 64), boundary_B=512,
            out=torch.empty(16, dtype=torch.uint8),
        )
