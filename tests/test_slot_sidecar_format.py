# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError
import struct
import zlib

import pytest
import torch

from atom.kv_transfer.offload.hybrid import sidecar_format as slot_sidecar_format
from atom.kv_transfer.offload.hybrid.sidecar_format import (
    HEADER_BYTES,
    LAYOUT_VERSION,
    MAGIC,
    SidecarFormatError,
    SlotSidecarHeader,
    SlotSidecarKey,
    decode_sidecar_tensor,
    decode_sidecar,
    encode_sidecar,
    finalize_sidecar_tensor_,
)

FINGERPRINT = bytes.fromhex("00112233445566778899aabbccddeeff")
OTHER_FINGERPRINT = bytes.fromhex("ffeeddccbbaa99887766554433221100")


def _header(
    *,
    boundary_tokens: int = 256,
    boundary_block_hash: int = 0x0123456789ABCDEF,
    payload_bytes: int | None = None,
    payload_crc32: int | None = None,
    fingerprint: bytes = FINGERPRINT,
    tp_size: int = 4,
    tp_rank: int = 1,
) -> SlotSidecarHeader:
    return SlotSidecarHeader(
        boundary_tokens=boundary_tokens,
        boundary_block_hash=boundary_block_hash,
        payload_bytes=payload_bytes,
        payload_crc32=payload_crc32,
        fingerprint=fingerprint,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )


def _blob(payload: bytes = b"slot-payload") -> bytes:
    return encode_sidecar(_header(), payload)


def _readonly_memoryview(value: bytearray) -> memoryview:
    return memoryview(value).toreadonly()


def _decode(
    blob: bytes | bytearray | memoryview,
    *,
    expected_fingerprint: bytes = FINGERPRINT,
    expected_tp_size: int = 4,
    expected_tp_rank: int = 1,
    expected_boundary_tokens: int = 256,
    expected_boundary_block_hash: int = 0x0123456789ABCDEF,
    expected_payload_bytes: int = len(b"slot-payload"),
):
    return decode_sidecar(
        blob,
        expected_fingerprint=expected_fingerprint,
        expected_tp_size=expected_tp_size,
        expected_tp_rank=expected_tp_rank,
        expected_boundary_tokens=expected_boundary_tokens,
        expected_boundary_block_hash=expected_boundary_block_hash,
        expected_payload_bytes=expected_payload_bytes,
    )


def test_format_constants_are_fixed():
    assert MAGIC == b"AOS1"
    assert LAYOUT_VERSION == 1
    assert HEADER_BYTES == 128


def test_key_has_deterministic_canonical_string_and_pr1683_storage_hash():
    key = SlotSidecarKey(
        boundary_block_hash=0x123,
        fingerprint=FINGERPRINT,
        tp_size=4,
        tp_rank=1,
    )
    canonical = "atom-slot-v1:4:1:0000000000000123:00112233445566778899aabbccddeeff"

    assert key.canonical_string() == canonical
    assert str(key) == canonical
    assert key.storage_hash() == 10062454250606645
    assert 0 <= key.storage_hash() < (1 << 63)
    assert SlotSidecarKey(0x123, FINGERPRINT, 4, 1).storage_hash() == 10062454250606645


@pytest.mark.parametrize(
    "other",
    [
        SlotSidecarKey(0x124, FINGERPRINT, 4, 1),
        SlotSidecarKey(0x123, OTHER_FINGERPRINT, 4, 1),
        SlotSidecarKey(0x123, FINGERPRINT, 8, 1),
        SlotSidecarKey(0x123, FINGERPRINT, 4, 2),
    ],
)
def test_key_changes_across_hash_fingerprint_and_tp_geometry(other):
    base = SlotSidecarKey(0x123, FINGERPRINT, 4, 1)

    assert other.canonical_string() != base.canonical_string()
    assert other.storage_hash() != base.storage_hash()


def test_key_is_immutable():
    key = SlotSidecarKey(0x123, FINGERPRINT, 4, 1)

    with pytest.raises(FrozenInstanceError):
        key.tp_rank = 2


@pytest.mark.parametrize("boundary_block_hash", [-1, 1 << 64])
def test_key_rejects_out_of_range_boundary_hash(boundary_block_hash: int):
    with pytest.raises(SidecarFormatError, match="boundary_block_hash"):
        SlotSidecarKey(boundary_block_hash, FINGERPRINT, 4, 1)


@pytest.mark.parametrize("fingerprint", [b"x" * 15, b"x" * 17])
def test_key_rejects_non_16_byte_fingerprint(fingerprint: bytes):
    with pytest.raises(SidecarFormatError, match="fingerprint.*16"):
        SlotSidecarKey(1, fingerprint, 4, 1)


@pytest.mark.parametrize("tp_size", [0, -1, 1 << 32])
def test_key_rejects_invalid_tp_size(tp_size: int):
    with pytest.raises(SidecarFormatError, match="tp_size"):
        SlotSidecarKey(1, FINGERPRINT, tp_size, 0)


@pytest.mark.parametrize("tp_rank", [-1, 4, 1 << 32])
def test_key_rejects_invalid_tp_rank(tp_rank: int):
    with pytest.raises(SidecarFormatError, match="tp_rank"):
        SlotSidecarKey(1, FINGERPRINT, 4, tp_rank)


def test_header_round_trip_is_fixed_size_little_endian_and_zero_padded():
    payload = b"\x00whole-slot\xff"

    blob = encode_sidecar(_header(), payload)
    header, decoded_payload = _decode(
        blob,
        expected_payload_bytes=len(payload),
    )

    assert len(blob) == HEADER_BYTES + len(payload)
    assert blob[HEADER_BYTES:] == payload
    assert blob[:4] == MAGIC
    assert struct.unpack_from("<I", blob, 4)[0] == LAYOUT_VERSION
    assert struct.unpack_from("<I", blob, 8)[0] == 0
    assert struct.unpack_from("<Q", blob, 12)[0] == 256
    assert struct.unpack_from("<Q", blob, 20)[0] == 0x0123456789ABCDEF
    assert struct.unpack_from("<Q", blob, 28)[0] == len(payload)
    assert struct.unpack_from("<I", blob, 36)[0] == zlib.crc32(payload) & 0xFFFFFFFF
    assert blob[40:56] == FINGERPRINT
    assert struct.unpack_from("<I", blob, 56)[0] == 4
    assert struct.unpack_from("<I", blob, 60)[0] == 1
    assert blob[64:HEADER_BYTES] == b"\x00" * (HEADER_BYTES - 64)
    assert header == SlotSidecarHeader(
        boundary_tokens=256,
        boundary_block_hash=0x0123456789ABCDEF,
        payload_bytes=len(payload),
        payload_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
        fingerprint=FINGERPRINT,
        tp_size=4,
        tp_rank=1,
    )
    assert decoded_payload == payload


def test_tensor_framing_writes_header_in_place_and_decode_returns_payload_view():
    payload = torch.tensor(list(b"tensor-slot-payload"), dtype=torch.uint8)
    framed = torch.empty(HEADER_BYTES + payload.numel(), dtype=torch.uint8)
    framed[HEADER_BYTES:].copy_(payload)
    payload_ptr = framed[HEADER_BYTES:].data_ptr()

    result = finalize_sidecar_tensor_(framed, _header(payload_bytes=payload.numel()))
    header, decoded = decode_sidecar_tensor(
        framed,
        expected_fingerprint=FINGERPRINT,
        expected_tp_size=4,
        expected_tp_rank=1,
        expected_boundary_tokens=256,
        expected_boundary_block_hash=0x0123456789ABCDEF,
        expected_payload_bytes=payload.numel(),
    )

    assert result is framed
    assert header.payload_crc32 == zlib.crc32(payload.numpy()) & 0xFFFFFFFF
    assert decoded.data_ptr() == payload_ptr
    assert decoded.untyped_storage().data_ptr() == framed.untyped_storage().data_ptr()
    decoded[0] ^= 1
    assert framed[HEADER_BYTES].item() == decoded[0].item()


def test_tensor_framing_rejects_crc_corruption_without_payload_clone():
    payload = torch.tensor(list(b"tensor-slot-payload"), dtype=torch.uint8)
    framed = torch.empty(HEADER_BYTES + payload.numel(), dtype=torch.uint8)
    framed[HEADER_BYTES:].copy_(payload)
    finalize_sidecar_tensor_(framed, _header(payload_bytes=payload.numel()))
    framed[-1] ^= 1

    with pytest.raises(SidecarFormatError, match="CRC"):
        decode_sidecar_tensor(
            framed,
            expected_fingerprint=FINGERPRINT,
            expected_tp_size=4,
            expected_tp_rank=1,
            expected_boundary_tokens=256,
            expected_boundary_block_hash=0x0123456789ABCDEF,
            expected_payload_bytes=payload.numel(),
        )


def test_header_is_immutable():
    header, _ = _decode(_blob())

    with pytest.raises(FrozenInstanceError):
        header.boundary_tokens = 512


def test_encode_rejects_stale_payload_size():
    payload = b"payload"

    with pytest.raises(SidecarFormatError, match="payload_bytes"):
        encode_sidecar(_header(payload_bytes=len(payload) + 1), payload)


def test_encode_rejects_stale_payload_crc():
    payload = b"payload"
    stale_crc = (zlib.crc32(payload) ^ 1) & 0xFFFFFFFF

    with pytest.raises(SidecarFormatError, match="CRC"):
        encode_sidecar(_header(payload_crc32=stale_crc), payload)


def test_header_rejects_crc_corruption():
    blob = _blob()
    corrupted = blob[:-1] + bytes([blob[-1] ^ 1])

    with pytest.raises(SidecarFormatError, match="CRC"):
        _decode(corrupted)


@pytest.mark.parametrize("malformed", [_blob()[:-1], _blob() + b"\x00"])
def test_header_rejects_truncated_and_extra_blobs(malformed: bytes):
    with pytest.raises(SidecarFormatError, match="size"):
        _decode(malformed)


def test_header_rejects_blob_shorter_than_fixed_header():
    with pytest.raises(SidecarFormatError, match="smaller than header"):
        _decode(b"\x00" * (HEADER_BYTES - 1))


def test_header_rejects_bad_magic():
    malformed = bytearray(_blob())
    malformed[:4] = b"BAD!"

    with pytest.raises(SidecarFormatError, match="magic"):
        _decode(malformed)


def test_header_rejects_bad_version():
    malformed = bytearray(_blob())
    struct.pack_into("<I", malformed, 4, LAYOUT_VERSION + 1)

    with pytest.raises(SidecarFormatError, match="version"):
        _decode(malformed)


def test_header_rejects_nonzero_flags():
    malformed = bytearray(_blob())
    struct.pack_into("<I", malformed, 8, 1)

    with pytest.raises(SidecarFormatError, match="flags"):
        _decode(malformed)


def test_header_rejects_nonzero_reserved_bytes():
    malformed = bytearray(_blob())
    malformed[HEADER_BYTES - 1] = 1

    with pytest.raises(SidecarFormatError, match="reserved"):
        _decode(malformed)


def test_header_rejects_declared_payload_size_mismatch():
    malformed = bytearray(_blob())
    struct.pack_into("<Q", malformed, 28, len(b"slot-payload") + 1)

    with pytest.raises(SidecarFormatError, match="payload_bytes"):
        _decode(malformed)


def test_header_rejects_compatibility_fingerprint_mismatch():
    with pytest.raises(SidecarFormatError, match="fingerprint"):
        _decode(
            _blob(),
            expected_fingerprint=OTHER_FINGERPRINT,
        )


@pytest.mark.parametrize("expected_fingerprint", [b"x" * 15, b"x" * 17])
def test_decode_rejects_invalid_expected_fingerprint(expected_fingerprint: bytes):
    with pytest.raises(SidecarFormatError, match="fingerprint.*16"):
        _decode(
            _blob(),
            expected_fingerprint=expected_fingerprint,
        )


@pytest.mark.parametrize(
    ("expected_tp_size", "expected_tp_rank"),
    [(8, 1), (4, 2)],
)
def test_header_rejects_expected_tp_geometry_mismatch(
    expected_tp_size: int,
    expected_tp_rank: int,
):
    with pytest.raises(SidecarFormatError, match="TP"):
        _decode(
            _blob(),
            expected_tp_size=expected_tp_size,
            expected_tp_rank=expected_tp_rank,
        )


@pytest.mark.parametrize(
    ("expected_boundary_tokens", "expected_boundary_block_hash"),
    [(512, 0x0123456789ABCDEF), (256, 0xFEDCBA9876543210)],
)
def test_header_rejects_expected_boundary_mismatch(
    expected_boundary_tokens: int,
    expected_boundary_block_hash: int,
):
    with pytest.raises(SidecarFormatError, match="boundary"):
        _decode(
            _blob(),
            expected_boundary_tokens=expected_boundary_tokens,
            expected_boundary_block_hash=expected_boundary_block_hash,
        )


@pytest.mark.parametrize(
    "missing",
    [
        "expected_boundary_tokens",
        "expected_boundary_block_hash",
        "expected_payload_bytes",
    ],
)
def test_decode_requires_boundary_and_payload_expectations(missing: str):
    kwargs = {
        "expected_fingerprint": FINGERPRINT,
        "expected_tp_size": 4,
        "expected_tp_rank": 1,
        "expected_boundary_tokens": 256,
        "expected_boundary_block_hash": 0x0123456789ABCDEF,
        "expected_payload_bytes": len(b"slot-payload"),
    }
    del kwargs[missing]

    with pytest.raises(SidecarFormatError, match=f"{missing}.*required"):
        decode_sidecar(_blob(), **kwargs)


def test_boundary_identity_is_checked_before_payload_crc():
    malformed = bytearray(_blob())
    malformed[-1] ^= 1

    with pytest.raises(SidecarFormatError, match="boundary_tokens"):
        _decode(malformed, expected_boundary_tokens=512)


def test_expected_payload_size_is_checked_before_payload_crc():
    malformed = bytearray(_blob())
    malformed[-1] ^= 1

    with pytest.raises(SidecarFormatError, match="payload_bytes"):
        _decode(malformed, expected_payload_bytes=1 << 30)


@pytest.mark.parametrize("wrapper", [bytes, bytearray, memoryview])
def test_encode_and_decode_accept_contiguous_bytes_like_inputs(wrapper):
    payload = b"slot-payload"
    encoded = encode_sidecar(_header(), wrapper(payload))

    header, decoded = _decode(wrapper(encoded))

    assert header.payload_bytes == len(payload)
    assert decoded == payload
    assert isinstance(encoded, bytes)
    assert isinstance(decoded, bytes)


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param(lambda value: value, id="bytearray"),
        pytest.param(memoryview, id="writable-memoryview"),
        pytest.param(_readonly_memoryview, id="readonly-memoryview"),
    ],
)
def test_encode_snapshots_mutable_backing_before_checksum(monkeypatch, wrapper):
    payload = bytearray(b"slot-payload")
    original_payload = bytes(payload)
    real_crc32 = zlib.crc32

    def crc32_then_mutate_input(data):
        checksum = real_crc32(data)
        payload[:] = b"x" * len(payload)
        return checksum

    monkeypatch.setattr(zlib, "crc32", crc32_then_mutate_input)

    encoded = encode_sidecar(_header(), wrapper(payload))

    assert bytes(payload) != original_payload
    assert encoded[HEADER_BYTES:] == original_payload
    payload[:] = b"y" * len(payload)
    assert encoded[HEADER_BYTES:] == original_payload


@pytest.mark.parametrize(
    "wrapper",
    [
        pytest.param(lambda value: value, id="bytearray"),
        pytest.param(memoryview, id="writable-memoryview"),
        pytest.param(_readonly_memoryview, id="readonly-memoryview"),
    ],
)
def test_decode_snapshots_mutable_backing_before_checksum(monkeypatch, wrapper):
    blob = bytearray(_blob())
    original_payload = bytes(blob[HEADER_BYTES:])
    real_crc32 = zlib.crc32

    def crc32_then_mutate_input(data):
        checksum = real_crc32(data)
        blob[:4] = b"BAD!"
        blob[HEADER_BYTES:] = b"x" * len(original_payload)
        return checksum

    monkeypatch.setattr(zlib, "crc32", crc32_then_mutate_input)

    header, payload = _decode(wrapper(blob))

    assert bytes(blob[:4]) == b"BAD!"
    assert header.boundary_tokens == 256
    assert header.boundary_block_hash == 0x0123456789ABCDEF
    assert payload == original_payload
    blob[HEADER_BYTES:] = b"y" * len(original_payload)
    assert payload == original_payload


def test_decode_identity_rejection_snapshots_only_fixed_header(monkeypatch):
    payload = b"x" * (256 * 1024)
    blob = bytearray(_blob(payload))
    snapshot_sizes = []

    def record_snapshot(view):
        snapshot_sizes.append(len(view))
        return bytes(view)

    monkeypatch.setattr(
        slot_sidecar_format,
        "_snapshot_bytes",
        record_snapshot,
        raising=False,
    )

    with pytest.raises(SidecarFormatError, match="boundary_tokens"):
        _decode(
            blob,
            expected_boundary_tokens=512,
            expected_payload_bytes=len(payload),
        )

    assert snapshot_sizes == [HEADER_BYTES]


def test_decode_total_size_rejection_snapshots_only_fixed_header(monkeypatch):
    payload = b"x" * (256 * 1024)
    blob = bytearray(_blob(payload) + b"extra")
    snapshot_sizes = []

    def record_snapshot(view):
        snapshot_sizes.append(len(view))
        return bytes(view)

    monkeypatch.setattr(
        slot_sidecar_format,
        "_snapshot_bytes",
        record_snapshot,
        raising=False,
    )

    with pytest.raises(SidecarFormatError, match="framed size"):
        _decode(blob, expected_payload_bytes=len(payload))

    assert snapshot_sizes == [HEADER_BYTES]


def test_decode_snapshots_header_then_payload_once(monkeypatch):
    blob = bytearray(_blob())
    snapshot_sizes = []

    def record_snapshot(view):
        snapshot_sizes.append(len(view))
        return bytes(view)

    monkeypatch.setattr(
        slot_sidecar_format,
        "_snapshot_bytes",
        record_snapshot,
        raising=False,
    )

    _, payload = _decode(blob)

    assert snapshot_sizes == [HEADER_BYTES, len(b"slot-payload")]
    assert payload == b"slot-payload"


def test_decode_mutation_between_snapshots_fails_crc(monkeypatch):
    blob = bytearray(_blob())
    payload_bytes = len(b"slot-payload")
    snapshot_sizes = []

    def snapshot_then_mutate(view):
        snapshot = bytes(view)
        snapshot_sizes.append(len(view))
        if len(snapshot_sizes) == 1:
            blob[HEADER_BYTES:] = b"x" * payload_bytes
        return snapshot

    monkeypatch.setattr(
        slot_sidecar_format,
        "_snapshot_bytes",
        snapshot_then_mutate,
        raising=False,
    )

    with pytest.raises(SidecarFormatError, match="CRC"):
        _decode(blob)

    assert snapshot_sizes == [HEADER_BYTES, payload_bytes]


def test_decode_rejects_noncontiguous_memoryview():
    malformed = memoryview(_blob() * 2)[::2]

    with pytest.raises(SidecarFormatError, match="contiguous"):
        _decode(malformed)


@pytest.mark.parametrize("boundary_tokens", [0, -1, 1 << 64])
def test_header_rejects_invalid_boundary_tokens(boundary_tokens: int):
    with pytest.raises(SidecarFormatError, match="boundary_tokens"):
        _header(boundary_tokens=boundary_tokens)


@pytest.mark.parametrize("boundary_block_hash", [-1, 1 << 64])
def test_header_rejects_invalid_boundary_hash(boundary_block_hash: int):
    with pytest.raises(SidecarFormatError, match="boundary_block_hash"):
        _header(boundary_block_hash=boundary_block_hash)


@pytest.mark.parametrize("payload_bytes", [0, -1, 1 << 64])
def test_header_rejects_invalid_concrete_payload_size(payload_bytes: int):
    with pytest.raises(SidecarFormatError, match="payload_bytes"):
        _header(payload_bytes=payload_bytes)


@pytest.mark.parametrize("payload_crc32", [-1, 1 << 32])
def test_header_rejects_invalid_payload_crc(payload_crc32: int):
    with pytest.raises(SidecarFormatError, match="payload_crc32"):
        _header(payload_crc32=payload_crc32)


@pytest.mark.parametrize("fingerprint", [b"x" * 15, b"x" * 17])
def test_header_rejects_non_16_byte_fingerprint(fingerprint: bytes):
    with pytest.raises(SidecarFormatError, match="fingerprint.*16"):
        _header(fingerprint=fingerprint)


@pytest.mark.parametrize("tp_size", [0, -1, 1 << 32])
def test_header_rejects_invalid_tp_size(tp_size: int):
    with pytest.raises(SidecarFormatError, match="tp_size"):
        _header(tp_size=tp_size, tp_rank=0)


@pytest.mark.parametrize("tp_rank", [-1, 4, 1 << 32])
def test_header_rejects_invalid_tp_rank(tp_rank: int):
    with pytest.raises(SidecarFormatError, match="tp_rank"):
        _header(tp_size=4, tp_rank=tp_rank)


def test_encode_rejects_empty_payload():
    with pytest.raises(SidecarFormatError, match="payload"):
        encode_sidecar(_header(), b"")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SlotSidecarKey(True, FINGERPRINT, 4, 1),
        lambda: SlotSidecarKey(1, FINGERPRINT, True, 0),
        lambda: SlotSidecarKey(1, FINGERPRINT, 4, False),
        lambda: _header(boundary_tokens=True),
        lambda: _header(payload_bytes=True),
    ],
)
def test_integer_fields_reject_builtin_boolean_scalars(factory):
    with pytest.raises(SidecarFormatError, match="boolean"):
        factory()


def test_integer_fields_reject_numpy_boolean_scalars():
    np = pytest.importorskip("numpy")

    with pytest.raises(SidecarFormatError, match="boolean"):
        SlotSidecarKey(np.bool_(True), FINGERPRINT, 4, 1)
    with pytest.raises(SidecarFormatError, match="boolean"):
        _header(tp_rank=np.bool_(False))


def test_integer_fields_accept_numpy_integral_scalars():
    np = pytest.importorskip("numpy")
    payload = b"slot-payload"
    key = SlotSidecarKey(
        np.uint64(0x123),
        FINGERPRINT,
        np.int64(4),
        np.int32(1),
    )
    header = SlotSidecarHeader(
        boundary_tokens=np.int64(256),
        boundary_block_hash=np.uint64(0x0123456789ABCDEF),
        payload_bytes=None,
        payload_crc32=None,
        fingerprint=FINGERPRINT,
        tp_size=np.int32(4),
        tp_rank=np.int16(1),
    )

    decoded_header, decoded_payload = decode_sidecar(
        encode_sidecar(header, memoryview(payload)),
        expected_fingerprint=FINGERPRINT,
        expected_tp_size=np.int64(4),
        expected_tp_rank=np.int32(1),
        expected_boundary_tokens=np.int64(256),
        expected_boundary_block_hash=np.uint64(0x0123456789ABCDEF),
        expected_payload_bytes=np.int64(len(payload)),
    )

    assert key.boundary_block_hash == 0x123
    assert type(key.boundary_block_hash) is int
    assert decoded_header.boundary_tokens == 256
    assert decoded_payload == payload
