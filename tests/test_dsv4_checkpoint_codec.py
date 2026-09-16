# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import zlib

import pytest
import torch

from atom.kv_transfer.offload.hybrid.dsv4.codec import (
    HEADER_BYTES,
    DSV4CheckpointCodec,
    DSV4CheckpointError,
    DSV4CheckpointHeader,
    DSV4CheckpointKey,
)

FINGERPRINT = bytes.fromhex("00112233445566778899aabbccddeeff")


def _codec(*, tp_rank: int = 1) -> DSV4CheckpointCodec:
    return DSV4CheckpointCodec(
        fingerprint=FINGERPRINT,
        tp_size=4,
        tp_rank=tp_rank,
    )


def test_checkpoint_codec_encode_decode_round_trip():
    codec = _codec()
    payload = b"page-and-slot-checkpoint"

    encoded = codec.encode(
        payload,
        boundary_tokens=8_192,
        boundary_block_hash=0x0123456789ABCDEF,
    )
    framed = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
    header, decoded = codec.decode_tensor(
        framed,
        expected_boundary_tokens=8_192,
        expected_boundary_block_hash=0x0123456789ABCDEF,
        expected_payload_bytes=len(payload),
    )

    assert isinstance(header, DSV4CheckpointHeader)
    assert header.boundary_tokens == 8_192
    assert header.boundary_block_hash == 0x0123456789ABCDEF
    assert header.payload_bytes == len(payload)
    assert header.payload_crc32 == zlib.crc32(payload) & 0xFFFFFFFF
    assert header.fingerprint == FINGERPRINT
    assert (header.tp_size, header.tp_rank) == (4, 1)
    assert bytes(decoded.tolist()) == payload
    assert decoded.data_ptr() == framed.data_ptr() + HEADER_BYTES


def test_checkpoint_codec_finalizes_caller_owned_tensor_in_place():
    codec = _codec(tp_rank=0)
    payload = torch.tensor(list(b"slot-state"), dtype=torch.uint8)
    framed = torch.empty(HEADER_BYTES + payload.numel(), dtype=torch.uint8)
    framed[HEADER_BYTES:].copy_(payload)

    result = codec.finalize_tensor_(
        framed,
        boundary_tokens=16_384,
        boundary_block_hash=0xABC,
    )
    _, decoded = codec.decode_tensor(
        framed,
        expected_boundary_tokens=16_384,
        expected_boundary_block_hash=0xABC,
        expected_payload_bytes=payload.numel(),
    )

    assert result is framed
    assert torch.equal(decoded, payload)
    assert decoded.untyped_storage().data_ptr() == framed.untyped_storage().data_ptr()


def test_checkpoint_key_is_rank_local_and_deterministic():
    key = _codec().make_key(boundary_block_hash=0x123)

    assert isinstance(key, DSV4CheckpointKey)
    assert key.canonical_string() == (
        "atom-slot-v1:4:1:0000000000000123:00112233445566778899aabbccddeeff"
    )
    assert key.storage_hash() == 10062454250606645
    assert key == _codec().make_key(boundary_block_hash=0x123)
    assert key != _codec(tp_rank=2).make_key(boundary_block_hash=0x123)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fingerprint", b"too-short", "fingerprint.*16"),
        ("tp_size", 0, "tp_size"),
        ("tp_rank", 4, "tp_rank"),
    ],
)
def test_checkpoint_codec_rejects_invalid_identity(field, value, message):
    kwargs = {"fingerprint": FINGERPRINT, "tp_size": 4, "tp_rank": 1}
    kwargs[field] = value

    with pytest.raises(DSV4CheckpointError, match=message):
        DSV4CheckpointCodec(**kwargs)


def test_checkpoint_decode_fails_closed_on_identity_and_crc_mismatch():
    payload = b"checkpoint"
    framed = torch.frombuffer(
        bytearray(
            _codec().encode(
                payload,
                boundary_tokens=8_192,
                boundary_block_hash=0x123,
            )
        ),
        dtype=torch.uint8,
    )

    with pytest.raises(DSV4CheckpointError, match="boundary_tokens mismatch"):
        _codec().decode_tensor(
            framed,
            expected_boundary_tokens=16_384,
            expected_boundary_block_hash=0x123,
            expected_payload_bytes=len(payload),
        )

    framed[-1] ^= 1
    with pytest.raises(DSV4CheckpointError, match="CRC"):
        _codec().decode_tensor(
            framed,
            expected_boundary_tokens=8_192,
            expected_boundary_block_hash=0x123,
            expected_payload_bytes=len(payload),
        )


def test_checkpoint_decode_rejects_another_tp_shard():
    payload = b"checkpoint"
    framed = torch.frombuffer(
        bytearray(
            _codec(tp_rank=1).encode(
                payload,
                boundary_tokens=8_192,
                boundary_block_hash=0x123,
            )
        ),
        dtype=torch.uint8,
    )

    with pytest.raises(DSV4CheckpointError, match="TP geometry mismatch"):
        _codec(tp_rank=2).decode_tensor(
            framed,
            expected_boundary_tokens=8_192,
            expected_boundary_block_hash=0x123,
            expected_payload_bytes=len(payload),
        )


@pytest.mark.parametrize("payload_bytes", [0, -1, True])
def test_checkpoint_frame_size_rejects_invalid_payload_size(payload_bytes):
    with pytest.raises(DSV4CheckpointError, match="payload_bytes"):
        DSV4CheckpointCodec.frame_size(payload_bytes=payload_bytes)


def test_checkpoint_frame_size_includes_fixed_header():
    assert DSV4CheckpointCodec.frame_size(payload_bytes=17) == HEADER_BYTES + 17
