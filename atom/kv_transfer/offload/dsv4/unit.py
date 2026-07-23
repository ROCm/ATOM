# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 terminal-checkpoint *offload unit* container format.

An "offload unit" is the single, self-contained thing that gets offloaded and
later reloaded to resume a DSV4 request at a 128-aligned boundary ``B``. It is
one opaque ``uint8`` MemoryObj (LMCache stays oblivious to its contents) holding
everything needed to restore the request: compressed KV, the raw SWA tail, and
the CSA overlap continuation state.

(Named "offload unit" rather than "block" on purpose: "block" is already the
paged-KV vocabulary — ``block_table`` / ``block_size`` / ``num_blocks`` — so an
"offload block" would be ambiguous.)

Offload-unit layout for a terminal boundary ``B`` (``B % 128 == 0``)::

    +----------------------------------------------------------------+
    | header (fixed HEADER_BYTES)                                    |
    |   magic | layout_version | flags | boundary_B                  |
    |   compressed_bytes | swa_bytes | csa_state_bytes | total_bytes |
    |   payload_crc32 | geometry_fingerprint(16)                     |
    +----------------------------------------------------------------+
    | compressed component  (CSA main KV | HCA main KV | CSA idx KV) |
    +----------------------------------------------------------------+
    | raw SWA terminal tail [B-W, B)                                 |
    +----------------------------------------------------------------+
    | CSA main / CSA indexer overlap continuation state @B          |
    |   (reuses the v4_state_pool slot; HCA ring rides along inert)  |
    +----------------------------------------------------------------+
    | padding                                                        |
    +----------------------------------------------------------------+

HCA (``ratio=128``, ``overlap=False``) is offloaded as *compressed KV only* —
no ring state is saved, reset, or restored. Correctness holds by construction
because ``B`` is 128-aligned: the resumed first HCA block starts fresh and never
reads ring history. See ``dsv4-lmcache-bundle-plan.md``.

The container is deliberately geometry-agnostic: it lays out three opaque
component byte-blobs given their sizes. DSV4-specific sourcing (which GPU regions
feed each component) lives in the connector / gather kernel, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import zlib

# Bytes identifying a DSV4 offload unit payload; bump LAYOUT_VERSION on any
# format or section-ordering change so stale objects fail closed on load.
MAGIC = b"DV4U"
LAYOUT_VERSION = 1

# Section alignment. 256 keeps every component start GPU-copy friendly and leaves
# room for the header. HEADER_BYTES is fixed and aligned so the first component
# begins at a known offset.
ALIGN = 256
HEADER_BYTES = 256

# struct layout of the meaningful header prefix (little-endian). The remaining
# HEADER_BYTES - _HEADER_STRUCT.size bytes are zero padding / reserved.
#   magic(4s) version(I) flags(I) boundary_B(Q)
#   compressed_bytes(Q) swa_bytes(Q) csa_state_bytes(Q) total_bytes(Q)
#   payload_crc32(I) fingerprint(16s)
_HEADER_STRUCT = struct.Struct("<4sIIQQQQQI16s")
assert _HEADER_STRUCT.size <= HEADER_BYTES

# Header flag bits.
FLAG_NONE = 0


def _align_up(n: int, align: int = ALIGN) -> int:
    return ((int(n) + align - 1) // align) * align


@dataclass(frozen=True)
class DSV4OffloadUnitGeometry:
    """Everything that must match between save and load for a unit to be valid.

    A mismatch on ANY field means the stored bytes cannot be interpreted by the
    loading engine (different model, shard, dtype, or paging), so the fingerprint
    is the load-compatibility gate: a differing fingerprint => fail closed =>
    recompute.
    """

    model_name: str
    layout_version: int
    num_layers: int
    head_dim: int
    window_size: int  # SWA W (= 128 for DSV4)
    block_size: int
    swa_block_size: int
    k1_csa: int  # compressed slots per block for CSA (block_size // 4)
    k2_hca: int  # compressed slots per block for HCA (block_size // 128)
    kv_dtype: str
    compress_ratios: tuple[int, ...]
    tp_size: int
    tp_rank: int

    def _canonical_bytes(self) -> bytes:
        # Deterministic, cross-process-stable encoding (no repr()/hash()).
        parts = [
            self.model_name,
            str(self.layout_version),
            str(self.num_layers),
            str(self.head_dim),
            str(self.window_size),
            str(self.block_size),
            str(self.swa_block_size),
            str(self.k1_csa),
            str(self.k2_hca),
            self.kv_dtype,
            ",".join(str(r) for r in self.compress_ratios),
            str(self.tp_size),
            str(self.tp_rank),
        ]
        return "|".join(parts).encode("utf-8")

    def fingerprint(self) -> bytes:
        """16-byte geometry digest embedded in the header."""
        return hashlib.blake2b(self._canonical_bytes(), digest_size=16).digest()


@dataclass(frozen=True)
class DSV4OffloadUnitLayout:
    """Section offsets/sizes for one offload unit, derived from component sizes."""

    compressed_bytes: int
    swa_bytes: int
    csa_state_bytes: int

    header_bytes: int
    compressed_off: int
    swa_off: int
    csa_state_off: int
    total_bytes: int
    alignment: int

    @classmethod
    def build(
        cls,
        *,
        compressed_bytes: int,
        swa_bytes: int,
        csa_state_bytes: int,
        alignment: int = ALIGN,
    ) -> "DSV4OffloadUnitLayout":
        for name, val in (
            ("compressed_bytes", compressed_bytes),
            ("swa_bytes", swa_bytes),
            ("csa_state_bytes", csa_state_bytes),
        ):
            if int(val) < 0:
                raise ValueError(
                    f"DSV4OffloadUnitLayout: {name} must be >= 0, got {val}"
                )
        header_bytes = _align_up(HEADER_BYTES, alignment)
        compressed_off = header_bytes
        swa_off = _align_up(compressed_off + int(compressed_bytes), alignment)
        csa_state_off = _align_up(swa_off + int(swa_bytes), alignment)
        total_bytes = _align_up(csa_state_off + int(csa_state_bytes), alignment)
        return cls(
            compressed_bytes=int(compressed_bytes),
            swa_bytes=int(swa_bytes),
            csa_state_bytes=int(csa_state_bytes),
            header_bytes=header_bytes,
            compressed_off=compressed_off,
            swa_off=swa_off,
            csa_state_off=csa_state_off,
            total_bytes=total_bytes,
            alignment=int(alignment),
        )

    def payload_slice(self) -> tuple[int, int]:
        """[start, end) byte range covered by the CRC (everything after header)."""
        return self.header_bytes, self.total_bytes


@dataclass(frozen=True)
class DSV4OffloadUnitHeader:
    """Parsed header fields."""

    layout_version: int
    flags: int
    boundary_B: int
    compressed_bytes: int
    swa_bytes: int
    csa_state_bytes: int
    total_bytes: int
    payload_crc32: int
    fingerprint: bytes


def pack_header(
    *,
    layout: DSV4OffloadUnitLayout,
    geometry: DSV4OffloadUnitGeometry,
    boundary_B: int,
    payload_crc32: int,
    flags: int = FLAG_NONE,
) -> bytes:
    """Serialize a fixed-size (HEADER_BYTES) header block."""
    prefix = _HEADER_STRUCT.pack(
        MAGIC,
        int(geometry.layout_version),
        int(flags),
        int(boundary_B),
        int(layout.compressed_bytes),
        int(layout.swa_bytes),
        int(layout.csa_state_bytes),
        int(layout.total_bytes),
        int(payload_crc32) & 0xFFFFFFFF,
        geometry.fingerprint(),
    )
    return prefix + b"\x00" * (layout.header_bytes - len(prefix))


def unpack_header(raw: bytes | memoryview) -> DSV4OffloadUnitHeader:
    """Parse + magic/version-check a header block. Raises on mismatch."""
    if len(raw) < _HEADER_STRUCT.size:
        raise ValueError(
            f"DSV4 offload unit header too short: {len(raw)} < {_HEADER_STRUCT.size}"
        )
    (
        magic,
        layout_version,
        flags,
        boundary_B,
        compressed_bytes,
        swa_bytes,
        csa_state_bytes,
        total_bytes,
        payload_crc32,
        fingerprint,
    ) = _HEADER_STRUCT.unpack_from(bytes(raw[: _HEADER_STRUCT.size]))
    if magic != MAGIC:
        raise ValueError(
            f"DSV4 offload unit: bad magic {magic!r} (expected {MAGIC!r})"
        )
    if int(layout_version) != LAYOUT_VERSION:
        raise ValueError(
            f"DSV4 offload unit: layout_version {layout_version} != {LAYOUT_VERSION}"
        )
    return DSV4OffloadUnitHeader(
        layout_version=int(layout_version),
        flags=int(flags),
        boundary_B=int(boundary_B),
        compressed_bytes=int(compressed_bytes),
        swa_bytes=int(swa_bytes),
        csa_state_bytes=int(csa_state_bytes),
        total_bytes=int(total_bytes),
        payload_crc32=int(payload_crc32),
        fingerprint=bytes(fingerprint),
    )


def payload_crc(buf: bytes | memoryview) -> int:
    """CRC32 over the payload region (header excluded by the caller's slice)."""
    return zlib.crc32(bytes(buf)) & 0xFFFFFFFF
