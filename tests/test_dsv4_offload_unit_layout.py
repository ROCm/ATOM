# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the DSV4 offload-unit container layout, header, fingerprint."""

from __future__ import annotations

import pytest

from atom.kv_transfer.offload.dsv4.unit import (
    ALIGN,
    HEADER_BYTES,
    LAYOUT_VERSION,
    MAGIC,
    DSV4OffloadUnitGeometry,
    DSV4OffloadUnitLayout,
    pack_header,
    unpack_header,
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


def test_layout_offsets_are_aligned_and_ordered():
    layout = DSV4OffloadUnitLayout.build(
        compressed_bytes=1000,
        swa_bytes=300,
        csa_state_bytes=50,
    )
    assert layout.header_bytes % ALIGN == 0
    assert layout.compressed_off % ALIGN == 0
    assert layout.swa_off % ALIGN == 0
    assert layout.csa_state_off % ALIGN == 0
    assert layout.total_bytes % ALIGN == 0
    # Ordered and non-overlapping.
    assert layout.compressed_off == layout.header_bytes
    assert layout.swa_off >= layout.compressed_off + layout.compressed_bytes
    assert layout.csa_state_off >= layout.swa_off + layout.swa_bytes
    assert layout.total_bytes >= layout.csa_state_off + layout.csa_state_bytes


def test_layout_zero_sized_components():
    layout = DSV4OffloadUnitLayout.build(
        compressed_bytes=0, swa_bytes=0, csa_state_bytes=0
    )
    assert layout.total_bytes == layout.header_bytes
    assert layout.compressed_off == layout.header_bytes


def test_layout_rejects_negative():
    with pytest.raises(ValueError):
        DSV4OffloadUnitLayout.build(
            compressed_bytes=-1, swa_bytes=0, csa_state_bytes=0
        )


def test_header_round_trip():
    geo = _geometry()
    layout = DSV4OffloadUnitLayout.build(
        compressed_bytes=4096, swa_bytes=512, csa_state_bytes=128
    )
    raw = pack_header(
        layout=layout,
        geometry=geo,
        boundary_B=16384,
        payload_crc32=0xDEADBEEF,
    )
    assert len(raw) == layout.header_bytes == HEADER_BYTES
    assert raw[:4] == MAGIC
    header = unpack_header(raw)
    assert header.layout_version == LAYOUT_VERSION
    assert header.boundary_B == 16384
    assert header.compressed_bytes == 4096
    assert header.swa_bytes == 512
    assert header.csa_state_bytes == 128
    assert header.total_bytes == layout.total_bytes
    assert header.payload_crc32 == 0xDEADBEEF
    assert header.fingerprint == geo.fingerprint()


def test_header_bad_magic_fails():
    geo = _geometry()
    layout = DSV4OffloadUnitLayout.build(
        compressed_bytes=16, swa_bytes=16, csa_state_bytes=16
    )
    raw = bytearray(
        pack_header(
            layout=layout, geometry=geo, boundary_B=256, payload_crc32=0
        )
    )
    raw[0] = ord("X")
    with pytest.raises(ValueError):
        unpack_header(bytes(raw))


def test_header_version_mismatch_fails():
    geo = _geometry(layout_version=LAYOUT_VERSION + 99)
    layout = DSV4OffloadUnitLayout.build(
        compressed_bytes=16, swa_bytes=16, csa_state_bytes=16
    )
    raw = pack_header(layout=layout, geometry=geo, boundary_B=256, payload_crc32=0)
    with pytest.raises(ValueError):
        unpack_header(raw)


def test_fingerprint_is_deterministic_and_sensitive():
    a = _geometry().fingerprint()
    b = _geometry().fingerprint()
    assert a == b
    assert len(a) == 16
    # Any geometry change flips the fingerprint.
    for change in (
        dict(tp_rank=1),
        dict(tp_size=2),
        dict(kv_dtype="fp8"),
        dict(head_dim=64),
        dict(window_size=256),
        dict(compress_ratios=(0, 4, 128)),
        dict(model_name="other"),
    ):
        assert _geometry(**change).fingerprint() != a, change


def test_header_too_short_fails():
    with pytest.raises(ValueError):
        unpack_header(b"\x00" * 4)
