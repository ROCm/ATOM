# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Opaque *offload bundle* container format for the hybrid layout family.

An "offload bundle" is the single, self-contained thing that gets offloaded and
later reloaded to resume a request at an aligned boundary ``B``. It is one opaque
``uint8`` MemoryObj (LMCache stays oblivious to its contents) holding an ordered
list of component byte-blobs — whatever a model's profile declares (e.g. DSV4:
compressed KV + SWA tail + CSA overlap state; Qwen3-Next: full-attn KV + GDN
recurrent state).

(Named "offload bundle" rather than "block" on purpose: "block" is already the
paged-KV vocabulary — ``block_table`` / ``block_size`` / ``num_blocks`` — so an
"offload block" would be ambiguous.)

Offload-unit layout::

    +----------------------------------------------------------------+
    | header (fixed HEADER_BYTES)                                    |
    |   magic | layout_version | flags | boundary_B                  |
    |   n_components | total_bytes | payload_crc32                    |
    |   geometry_fingerprint(16) | per-component (nbytes, off) table  |
    +----------------------------------------------------------------+
    | component[0] bytes  (aligned)                                  |
    | component[1] bytes  (aligned)                                  |
    | ...                                                            |
    +----------------------------------------------------------------+
    | padding                                                        |
    +----------------------------------------------------------------+

The container is geometry-agnostic: it lays out N opaque component byte-blobs
given their sizes. Which GPU regions feed each component (block / swa / slot /
staging) is decided by the profile + ``BundleSources`` in the connector, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import zlib

# Bytes identifying an offload bundle payload; bump LAYOUT_VERSION on any format or
# section-ordering change so stale objects fail closed on load. v2 generalized
# the header from three fixed size fields (compressed/swa/csa_state) to a
# variable component table so one container serves DSV4, hybrid KV (Qwen3-Next /
# Kimi K3), and future profiles.
MAGIC = b"AOB1"  # ATOM Offload Unit v1
LAYOUT_VERSION = 2

# Section alignment. 256 keeps every component start GPU-copy friendly and leaves
# room for the header. HEADER_BYTES is fixed and aligned so the first component
# begins at a known offset.
ALIGN = 256
HEADER_BYTES = 256

# Generic (N-component) header. Fixed-size prefix + a per-component table:
#   prefix: magic(4s) version(I) flags(I) boundary_B(Q)
#           n_components(I) total_bytes(Q) payload_crc32(I) fingerprint(16s)
#   table:  n_components x (nbytes(Q) off(Q))
# Component *names* are not stored — the codec knows them from the profile /
# fixed order; the header carries only sizes + offsets so load can byte-slice
# and cross-check against the expected component sizes (fail closed on mismatch).
_BUNDLE_PREFIX = struct.Struct("<4sIIQIQI16s")
_BUNDLE_COMPONENT = struct.Struct("<QQ")
# How many components fit in the fixed header after the prefix.
MAX_COMPONENTS = (HEADER_BYTES - _BUNDLE_PREFIX.size) // _BUNDLE_COMPONENT.size
assert _BUNDLE_PREFIX.size <= HEADER_BYTES and MAX_COMPONENTS >= 3

# Header flag bits.
FLAG_NONE = 0


def _align_up(n: int, align: int = ALIGN) -> int:
    return ((int(n) + align - 1) // align) * align


# =====================================================================
# Generic geometry fingerprint (model-agnostic; profiles supply the fields)
# =====================================================================
@dataclass(frozen=True)
class BundleGeometry:
    """Model-agnostic load-compatibility fingerprint for a unit profile.

    ``fields`` is an ordered tuple of ``(key, str-value)`` pairs describing
    everything that must match between save and load for a unit's bytes to be
    interpretable (num layers, head dims, dtype, per-model dims, ...). A change
    in any field flips the fingerprint => load fails closed => recompute.

    One codec/connector drives every profile (DSV4, Qwen3-Next, ...) — the
    profile supplies the fields.
    """

    model_tag: str
    layout_version: int
    tp_size: int
    tp_rank: int
    fields: tuple[tuple[str, str], ...]

    @classmethod
    def build(
        cls,
        *,
        model_tag: str,
        tp_size: int,
        tp_rank: int,
        fields: dict,
        layout_version: int = LAYOUT_VERSION,
    ) -> "BundleGeometry":
        # Sorted for a deterministic, order-independent fingerprint.
        items = tuple(sorted((str(k), str(v)) for k, v in fields.items()))
        return cls(
            model_tag=str(model_tag),
            layout_version=int(layout_version),
            tp_size=int(tp_size),
            tp_rank=int(tp_rank),
            fields=items,
        )

    def _canonical_bytes(self) -> bytes:
        parts = [
            self.model_tag,
            str(self.layout_version),
            str(self.tp_size),
            str(self.tp_rank),
        ]
        parts += [f"{k}={v}" for k, v in self.fields]
        return "|".join(parts).encode("utf-8")

    def fingerprint(self) -> bytes:
        return hashlib.blake2b(self._canonical_bytes(), digest_size=16).digest()


# =====================================================================
# Generic N-component container: BundleComponent / BundleLayout / header codec
# =====================================================================
@dataclass(frozen=True)
class BundleComponent:
    """One opaque byte-blob inside a unit: its name, size, and aligned offset."""

    name: str
    nbytes: int
    off: int


@dataclass(frozen=True)
class BundleLayout:
    """Ordered component sections of one offload bundle, derived from sizes.

    The component *order* is significant: it fixes the byte layout and the
    header table order. Two profiles with the same ordered (name, nbytes) list
    produce byte-identical payload regions.
    """

    components: tuple[BundleComponent, ...]
    header_bytes: int
    total_bytes: int
    alignment: int

    @classmethod
    def build(
        cls,
        sizes: "list[tuple[str, int]]",
        *,
        alignment: int = ALIGN,
    ) -> "BundleLayout":
        header_bytes = _align_up(HEADER_BYTES, alignment)
        if len(sizes) > MAX_COMPONENTS:
            raise ValueError(
                f"offload bundle: {len(sizes)} components exceeds MAX_COMPONENTS="
                f"{MAX_COMPONENTS}"
            )
        comps: list[BundleComponent] = []
        off = header_bytes
        for name, nbytes in sizes:
            if int(nbytes) < 0:
                raise ValueError(
                    f"offload bundle: component {name!r} nbytes must be >= 0, got {nbytes}"
                )
            comps.append(BundleComponent(str(name), int(nbytes), off))
            off = _align_up(off + int(nbytes), alignment)
        return cls(
            components=tuple(comps),
            header_bytes=header_bytes,
            total_bytes=off,
            alignment=int(alignment),
        )

    def payload_slice(self) -> tuple[int, int]:
        """[start, end) byte range covered by the CRC (everything after header)."""
        return self.header_bytes, self.total_bytes

    def component(self, name: str) -> BundleComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"offload bundle: no component named {name!r}")


@dataclass(frozen=True)
class BundleHeader:
    """Parsed generic header. ``components`` is an ordered list of (nbytes, off)."""

    layout_version: int
    flags: int
    boundary_B: int
    total_bytes: int
    payload_crc32: int
    fingerprint: bytes
    components: tuple[tuple[int, int], ...]


def pack_bundle_header(
    *,
    layout: BundleLayout,
    fingerprint: bytes,
    boundary_B: int,
    payload_crc32: int,
    layout_version: int = LAYOUT_VERSION,
    flags: int = FLAG_NONE,
) -> bytes:
    """Serialize a fixed-size (header_bytes) generic unit header."""
    n = len(layout.components)
    if n > MAX_COMPONENTS:
        raise ValueError(f"offload bundle: {n} components exceeds {MAX_COMPONENTS}")
    prefix = _BUNDLE_PREFIX.pack(
        MAGIC,
        int(layout_version),
        int(flags),
        int(boundary_B),
        int(n),
        int(layout.total_bytes),
        int(payload_crc32) & 0xFFFFFFFF,
        bytes(fingerprint),
    )
    table = b"".join(_BUNDLE_COMPONENT.pack(int(c.nbytes), int(c.off)) for c in layout.components)
    raw = prefix + table
    return raw + b"\x00" * (layout.header_bytes - len(raw))


def unpack_bundle_header(raw: bytes | memoryview) -> BundleHeader:
    """Parse + magic/version-check a generic unit header. Raises on mismatch."""
    if len(raw) < _BUNDLE_PREFIX.size:
        raise ValueError(
            f"offload bundle header too short: {len(raw)} < {_BUNDLE_PREFIX.size}"
        )
    raw = bytes(raw)
    (magic, version, flags, boundary_B, n, total_bytes, crc, fp) = _BUNDLE_PREFIX.unpack_from(
        raw
    )
    if magic != MAGIC:
        raise ValueError(f"offload bundle: bad magic {magic!r} (expected {MAGIC!r})")
    if int(version) != LAYOUT_VERSION:
        raise ValueError(
            f"offload bundle: layout_version {version} != {LAYOUT_VERSION}"
        )
    if int(n) > MAX_COMPONENTS:
        raise ValueError(f"offload bundle: n_components {n} exceeds {MAX_COMPONENTS}")
    need = _BUNDLE_PREFIX.size + int(n) * _BUNDLE_COMPONENT.size
    if len(raw) < need:
        raise ValueError(f"offload bundle header truncated: {len(raw)} < {need}")
    comps: list[tuple[int, int]] = []
    for i in range(int(n)):
        nbytes, off = _BUNDLE_COMPONENT.unpack_from(
            raw, _BUNDLE_PREFIX.size + i * _BUNDLE_COMPONENT.size
        )
        comps.append((int(nbytes), int(off)))
    return BundleHeader(
        layout_version=int(version),
        flags=int(flags),
        boundary_B=int(boundary_B),
        total_bytes=int(total_bytes),
        payload_crc32=int(crc),
        fingerprint=bytes(fp),
        components=tuple(comps),
    )


def payload_crc(buf: bytes | memoryview) -> int:
    """CRC32 over the payload region (header excluded by the caller's slice)."""
    return zlib.crc32(bytes(buf)) & 0xFFFFFFFF
