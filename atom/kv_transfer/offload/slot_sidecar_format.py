# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic keying and fixed AOS1 framing for SLOT sidecars."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import struct
import zlib

import torch

MAGIC = b"AOS1"
LAYOUT_VERSION = 1
HEADER_BYTES = 128

_FLAGS_NONE = 0
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_STORAGE_HASH_MASK = (1 << 63) - 1
_FINGERPRINT_BYTES = 16
_REQUIRED = object()

# Wire offsets are intentionally stable:
#   magic[0:4], version[4:8], flags[8:12], boundary_tokens[12:20],
#   boundary_block_hash[20:28], payload_bytes[28:36], payload_crc32[36:40],
#   fingerprint[40:56], tp_size[56:60], tp_rank[60:64], reserved[64:128].
_HEADER_PREFIX = struct.Struct("<4sIIQQQI16sII")
_RESERVED_OFFSET = _HEADER_PREFIX.size
assert _RESERVED_OFFSET == 64
assert _RESERVED_OFFSET <= HEADER_BYTES


class SidecarFormatError(ValueError):
    """Raised when a sidecar key, header, or payload fails validation."""


def _is_boolean_scalar(value: object) -> bool:
    value_type = type(value)
    return isinstance(value, bool) or (
        value_type.__module__.split(".", 1)[0] == "numpy"
        and value_type.__name__ in {"bool", "bool_"}
    )


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if _is_boolean_scalar(value):
        raise SidecarFormatError(f"{name} must be an integer, not a boolean scalar")
    if not isinstance(value, Integral):
        raise SidecarFormatError(f"{name} must be an integer")
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise SidecarFormatError(
            f"{name} must be in [{minimum}, {maximum}], got {normalized}"
        )
    return normalized


def _fingerprint_bytes(value: object, *, name: str = "fingerprint") -> bytes:
    view = _contiguous_byte_view(value, name=name)
    if len(view) != _FINGERPRINT_BYTES:
        raise SidecarFormatError(
            f"{name} must be exactly {_FINGERPRINT_BYTES} bytes, got {len(view)}"
        )
    return bytes(view)


def _contiguous_byte_view(value: object, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except (TypeError, ValueError) as exc:
        raise SidecarFormatError(f"{name} must be bytes-like") from exc
    if not view.c_contiguous:
        raise SidecarFormatError(f"{name} must be a contiguous bytes-like object")
    try:
        return view if view.format == "B" and view.ndim == 1 else view.cast("B")
    except TypeError as exc:
        raise SidecarFormatError(
            f"{name} must expose a contiguous byte representation"
        ) from exc


def _snapshot_bytes(view: memoryview) -> bytes:
    return bytes(view)


def _stable_entry_byte_view(value: object, *, name: str) -> memoryview:
    view = _contiguous_byte_view(value, name=name)
    if not isinstance(value, bytes):
        view = memoryview(_snapshot_bytes(view))
    return view


def _tp_geometry(
    tp_size: object, tp_rank: object, *, prefix: str = ""
) -> tuple[int, int]:
    size_name = f"{prefix}tp_size"
    rank_name = f"{prefix}tp_rank"
    normalized_size = _require_int(
        size_name,
        tp_size,
        minimum=1,
        maximum=_UINT32_MAX,
    )
    normalized_rank = _require_int(
        rank_name,
        tp_rank,
        minimum=0,
        maximum=_UINT32_MAX,
    )
    if normalized_rank >= normalized_size:
        raise SidecarFormatError(
            f"{rank_name} must be smaller than {size_name}; "
            f"got rank={normalized_rank}, size={normalized_size}"
        )
    return normalized_size, normalized_rank


@dataclass(frozen=True)
class SlotSidecarKey:
    """Content-addressed identity for one rank's SLOT sidecar."""

    boundary_block_hash: int
    fingerprint: bytes
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        boundary_block_hash = _require_int(
            "boundary_block_hash",
            self.boundary_block_hash,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        fingerprint = _fingerprint_bytes(self.fingerprint)
        tp_size, tp_rank = _tp_geometry(self.tp_size, self.tp_rank)
        object.__setattr__(self, "boundary_block_hash", boundary_block_hash)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "tp_rank", tp_rank)

    def canonical_string(self) -> str:
        """Return the stable text hashed for LMCache's ``chunk_hash``."""
        return (
            f"atom-slot-v1:{self.tp_size}:{self.tp_rank}:"
            f"{self.boundary_block_hash:016x}:{self.fingerprint.hex()}"
        )

    def storage_hash(self) -> int:
        """Return PR #1683's BLAKE2b-8 digest masked to nonnegative 63 bits."""
        digest = hashlib.blake2b(
            self.canonical_string().encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") & _STORAGE_HASH_MASK

    def __str__(self) -> str:
        return self.canonical_string()


@dataclass(frozen=True)
class SlotSidecarHeader:
    """Logical AOS1 header.

    ``payload_bytes`` and ``payload_crc32`` may be ``None`` when passed to
    :func:`encode_sidecar`; encoding always derives both from the payload.
    Decoded headers always contain concrete integer values.

    ``fingerprint`` is an opaque compatibility identifier supplied by the
    caller. This format validates equality but does not infer what it covers.
    """

    boundary_tokens: int
    boundary_block_hash: int
    payload_bytes: int | None
    payload_crc32: int | None
    fingerprint: bytes
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        boundary_tokens = _require_int(
            "boundary_tokens",
            self.boundary_tokens,
            minimum=1,
            maximum=_UINT64_MAX,
        )
        boundary_block_hash = _require_int(
            "boundary_block_hash",
            self.boundary_block_hash,
            minimum=0,
            maximum=_UINT64_MAX,
        )
        payload_bytes = self.payload_bytes
        if payload_bytes is not None:
            payload_bytes = _require_int(
                "payload_bytes",
                payload_bytes,
                minimum=1,
                maximum=_UINT64_MAX,
            )
        payload_crc32 = self.payload_crc32
        if payload_crc32 is not None:
            payload_crc32 = _require_int(
                "payload_crc32",
                payload_crc32,
                minimum=0,
                maximum=_UINT32_MAX,
            )
        fingerprint = _fingerprint_bytes(self.fingerprint)
        tp_size, tp_rank = _tp_geometry(self.tp_size, self.tp_rank)

        object.__setattr__(self, "boundary_tokens", boundary_tokens)
        object.__setattr__(self, "boundary_block_hash", boundary_block_hash)
        object.__setattr__(self, "payload_bytes", payload_bytes)
        object.__setattr__(self, "payload_crc32", payload_crc32)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "tp_size", tp_size)
        object.__setattr__(self, "tp_rank", tp_rank)


def _encoded_header(
    header: SlotSidecarHeader,
    payload_view: memoryview,
) -> bytes:
    if not isinstance(header, SlotSidecarHeader):
        raise SidecarFormatError("header must be a SlotSidecarHeader")
    if not payload_view:
        raise SidecarFormatError("payload must contain at least one byte")

    payload_bytes = len(payload_view)
    payload_crc32 = zlib.crc32(payload_view) & _UINT32_MAX
    if header.payload_bytes is not None and header.payload_bytes != payload_bytes:
        raise SidecarFormatError(
            f"header payload_bytes={header.payload_bytes} does not match "
            f"payload size={payload_bytes}"
        )
    if header.payload_crc32 is not None and header.payload_crc32 != payload_crc32:
        raise SidecarFormatError(
            f"header payload CRC={header.payload_crc32:#010x} does not match "
            f"computed CRC={payload_crc32:#010x}"
        )

    prefix = _HEADER_PREFIX.pack(
        MAGIC,
        LAYOUT_VERSION,
        _FLAGS_NONE,
        header.boundary_tokens,
        header.boundary_block_hash,
        payload_bytes,
        payload_crc32,
        header.fingerprint,
        header.tp_size,
        header.tp_rank,
    )
    encoded_header = prefix + b"\x00" * (HEADER_BYTES - len(prefix))
    if len(encoded_header) != HEADER_BYTES:
        raise AssertionError("internal AOS1 header size mismatch")
    return encoded_header


def encode_sidecar(
    header: SlotSidecarHeader,
    payload: bytes | bytearray | memoryview,
) -> bytes:
    """Encode one immutable AOS1 byte string, snapshotting mutable input."""

    payload_view = _stable_entry_byte_view(payload, name="payload")
    encoded_header = _encoded_header(header, payload_view)
    return b"".join((encoded_header, payload_view))


def _cpu_uint8_tensor_view(value: object, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SidecarFormatError(f"{name} must be a torch.Tensor")
    if value.dtype is not torch.uint8:
        raise SidecarFormatError(f"{name} tensor must have dtype torch.uint8")
    if value.device.type != "cpu":
        raise SidecarFormatError(f"{name} tensor must be on the CPU")
    if not value.is_contiguous():
        raise SidecarFormatError(f"{name} tensor must be contiguous")
    return value.reshape(-1)


def finalize_sidecar_tensor_(
    framed: torch.Tensor,
    header: SlotSidecarHeader,
) -> torch.Tensor:
    """Compute payload CRC and write only the AOS1 header into ``framed``."""

    flat = _cpu_uint8_tensor_view(framed, name="framed")
    if flat.numel() <= HEADER_BYTES:
        raise SidecarFormatError("framed tensor must contain a nonempty payload")
    payload_view = memoryview(flat[HEADER_BYTES:].numpy())
    encoded_header = _encoded_header(header, payload_view)
    flat[:HEADER_BYTES].copy_(
        torch.frombuffer(bytearray(encoded_header), dtype=torch.uint8)
    )
    return framed


def _decode_sidecar_view(
    blob: bytes | bytearray | memoryview,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
    snapshot_payload: bool = False,
) -> tuple[SlotSidecarHeader, bytes | memoryview]:
    """Validate and decode one AOS1 sidecar, failing closed on any mismatch.

    Boundary identity and payload size expectations are runtime-mandatory. The
    exact expected size bounds work before checksum work or payload allocation,
    so a corrupt size field cannot amplify memory use.

    CRC32 detects accidental corruption in trusted LMCache storage. It is not
    authentication and does not protect against adversarial tampering.
    """
    blob_view = _contiguous_byte_view(blob, name="blob")
    if len(blob_view) < HEADER_BYTES:
        raise SidecarFormatError(
            f"sidecar size {len(blob_view)} is smaller than header size {HEADER_BYTES}"
        )
    header_snapshot = _snapshot_bytes(blob_view[:HEADER_BYTES])

    (
        magic,
        layout_version,
        flags,
        boundary_tokens,
        boundary_block_hash,
        payload_bytes,
        payload_crc32,
        fingerprint,
        tp_size,
        tp_rank,
    ) = _HEADER_PREFIX.unpack_from(header_snapshot)

    if magic != MAGIC:
        raise SidecarFormatError(f"bad sidecar magic {magic!r}; expected {MAGIC!r}")
    if layout_version != LAYOUT_VERSION:
        raise SidecarFormatError(
            f"bad sidecar layout version {layout_version}; expected {LAYOUT_VERSION}"
        )
    if flags != _FLAGS_NONE:
        raise SidecarFormatError(f"unsupported sidecar flags {flags:#x}")
    if any(header_snapshot[_RESERVED_OFFSET:HEADER_BYTES]):
        raise SidecarFormatError("sidecar reserved header bytes must all be zero")

    for name, value in (
        ("expected_boundary_tokens", expected_boundary_tokens),
        ("expected_boundary_block_hash", expected_boundary_block_hash),
        ("expected_payload_bytes", expected_payload_bytes),
    ):
        if value is _REQUIRED:
            raise SidecarFormatError(f"{name} is required")

    expected_fingerprint = _fingerprint_bytes(
        expected_fingerprint,
        name="expected_fingerprint",
    )
    expected_tp_size, expected_tp_rank = _tp_geometry(
        expected_tp_size,
        expected_tp_rank,
        prefix="expected_",
    )
    expected_boundary_tokens = _require_int(
        "expected_boundary_tokens",
        expected_boundary_tokens,
        minimum=1,
        maximum=_UINT64_MAX,
    )
    expected_boundary_block_hash = _require_int(
        "expected_boundary_block_hash",
        expected_boundary_block_hash,
        minimum=0,
        maximum=_UINT64_MAX,
    )
    expected_payload_bytes = _require_int(
        "expected_payload_bytes",
        expected_payload_bytes,
        minimum=1,
        maximum=_UINT64_MAX,
    )

    header = SlotSidecarHeader(
        boundary_tokens=boundary_tokens,
        boundary_block_hash=boundary_block_hash,
        payload_bytes=payload_bytes,
        payload_crc32=payload_crc32,
        fingerprint=fingerprint,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    if boundary_tokens != expected_boundary_tokens:
        raise SidecarFormatError(
            "sidecar boundary_tokens mismatch: "
            f"stored={boundary_tokens}, expected={expected_boundary_tokens}"
        )
    if boundary_block_hash != expected_boundary_block_hash:
        raise SidecarFormatError(
            "sidecar boundary_block_hash mismatch: "
            f"stored={boundary_block_hash:#018x}, "
            f"expected={expected_boundary_block_hash:#018x}"
        )
    if payload_bytes != expected_payload_bytes:
        raise SidecarFormatError(
            "sidecar payload_bytes mismatch: "
            f"stored={payload_bytes}, expected={expected_payload_bytes}"
        )
    if fingerprint != expected_fingerprint:
        raise SidecarFormatError("sidecar fingerprint mismatch")
    if tp_size != expected_tp_size or tp_rank != expected_tp_rank:
        raise SidecarFormatError(
            "sidecar TP geometry mismatch: "
            f"stored=({tp_size}, {tp_rank}), "
            f"expected=({expected_tp_size}, {expected_tp_rank})"
        )

    expected_total_bytes = HEADER_BYTES + expected_payload_bytes
    if len(blob_view) != expected_total_bytes:
        raise SidecarFormatError(
            f"sidecar size {len(blob_view)} does not match expected framed size "
            f"{expected_total_bytes}"
        )

    payload_view = blob_view[HEADER_BYTES:]
    payload = _snapshot_bytes(payload_view) if snapshot_payload else payload_view
    actual_crc32 = zlib.crc32(payload) & _UINT32_MAX
    if actual_crc32 != payload_crc32:
        raise SidecarFormatError(
            f"sidecar payload CRC mismatch: stored={payload_crc32:#010x}, "
            f"actual={actual_crc32:#010x}"
        )

    return header, payload


def decode_sidecar(
    blob: bytes | bytearray | memoryview,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
) -> tuple[SlotSidecarHeader, bytes]:
    """Validate AOS1 and return an ownership-independent payload snapshot."""

    header, payload = _decode_sidecar_view(
        blob,
        expected_fingerprint,
        expected_tp_size,
        expected_tp_rank,
        expected_boundary_tokens=expected_boundary_tokens,
        expected_boundary_block_hash=expected_boundary_block_hash,
        expected_payload_bytes=expected_payload_bytes,
        snapshot_payload=True,
    )
    if not isinstance(payload, bytes):
        raise AssertionError("legacy AOS1 decode did not snapshot payload")
    return header, payload


def decode_sidecar_tensor(
    framed: torch.Tensor,
    expected_fingerprint: bytes | bytearray | memoryview,
    expected_tp_size: int,
    expected_tp_rank: int,
    *,
    expected_boundary_tokens: object = _REQUIRED,
    expected_boundary_block_hash: object = _REQUIRED,
    expected_payload_bytes: object = _REQUIRED,
) -> tuple[SlotSidecarHeader, torch.Tensor]:
    """Validate a CPU uint8 frame and return its zero-copy payload view."""

    flat = _cpu_uint8_tensor_view(framed, name="framed")
    header, _ = _decode_sidecar_view(
        memoryview(flat.numpy()),
        expected_fingerprint,
        expected_tp_size,
        expected_tp_rank,
        expected_boundary_tokens=expected_boundary_tokens,
        expected_boundary_block_hash=expected_boundary_block_hash,
        expected_payload_bytes=expected_payload_bytes,
    )
    return header, flat[HEADER_BYTES:]
