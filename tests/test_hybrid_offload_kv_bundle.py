# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the offload-unit container layout, header, and fingerprint."""

from __future__ import annotations

import pytest

from atom.kv_transfer.offload.hybrid.kv_bundle import (
    ALIGN,
    HEADER_BYTES,
    LAYOUT_VERSION,
    MAGIC,
    MAX_COMPONENTS,
    BundleGeometry,
    BundleLayout,
    pack_bundle_header,
    unpack_bundle_header,
)


def _geometry(**field_overrides) -> BundleGeometry:
    fields = dict(num_layers=61, head_dim=128, kv_dtype="bf16", block_size=128)
    fields.update(field_overrides.pop("fields", {}))
    kw = dict(model_tag="deepseek-v4", tp_size=1, tp_rank=0)
    kw.update(field_overrides)
    return BundleGeometry.build(fields=fields, **kw)


def _layout(*sizes) -> BundleLayout:
    return BundleLayout.build([(f"c{i}", n) for i, n in enumerate(sizes)])


def test_layout_offsets_are_aligned_and_ordered():
    layout = _layout(1000, 300, 50)
    assert layout.header_bytes % ALIGN == 0
    prev_end = layout.header_bytes
    for c in layout.components:
        assert c.off % ALIGN == 0
        assert c.off >= prev_end
        prev_end = c.off + c.nbytes
    assert layout.total_bytes % ALIGN == 0
    assert layout.total_bytes >= prev_end
    assert layout.components[0].off == layout.header_bytes


def test_layout_zero_sized_components():
    layout = _layout(0, 0, 0)
    assert layout.total_bytes == layout.header_bytes
    assert layout.components[0].off == layout.header_bytes


def test_layout_rejects_negative():
    with pytest.raises(ValueError):
        _layout(-1, 0, 0)


def test_layout_rejects_too_many_components():
    with pytest.raises(ValueError):
        _layout(*([16] * (MAX_COMPONENTS + 1)))


def test_header_round_trip():
    geo = _geometry()
    layout = _layout(4096, 512, 128)
    raw = pack_bundle_header(
        layout=layout, fingerprint=geo.fingerprint(), boundary_B=16384,
        payload_crc32=0xDEADBEEF, layout_version=LAYOUT_VERSION,
    )
    assert len(raw) == layout.header_bytes == HEADER_BYTES
    assert raw[:4] == MAGIC
    h = unpack_bundle_header(raw)
    assert h.layout_version == LAYOUT_VERSION
    assert h.boundary_B == 16384
    assert [nb for nb, _ in h.components] == [4096, 512, 128]
    assert h.total_bytes == layout.total_bytes
    assert h.payload_crc32 == 0xDEADBEEF
    assert h.fingerprint == geo.fingerprint()


def test_header_bad_magic_fails():
    layout = _layout(16, 16, 16)
    raw = bytearray(
        pack_bundle_header(
            layout=layout, fingerprint=b"\x00" * 16, boundary_B=256, payload_crc32=0
        )
    )
    raw[0] = ord("X")
    with pytest.raises(ValueError):
        unpack_bundle_header(bytes(raw))


def test_header_version_mismatch_fails():
    layout = _layout(16, 16, 16)
    raw = pack_bundle_header(
        layout=layout, fingerprint=b"\x00" * 16, boundary_B=256, payload_crc32=0,
        layout_version=LAYOUT_VERSION + 99,
    )
    with pytest.raises(ValueError):
        unpack_bundle_header(raw)


def test_header_too_short_fails():
    with pytest.raises(ValueError):
        unpack_bundle_header(b"\x00" * 4)


def test_fingerprint_is_deterministic_and_sensitive():
    a = _geometry().fingerprint()
    assert a == _geometry().fingerprint()
    assert len(a) == 16
    assert _geometry(tp_rank=1).fingerprint() != a
    assert _geometry(tp_size=2).fingerprint() != a
    assert _geometry(model_tag="other").fingerprint() != a
    assert _geometry(fields={"head_dim": 64}).fingerprint() != a
    assert _geometry(fields={"kv_dtype": "fp8"}).fingerprint() != a
