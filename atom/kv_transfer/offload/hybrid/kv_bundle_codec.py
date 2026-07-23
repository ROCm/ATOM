# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Assemble / disassemble an opaque *offload bundle* (N ordered components).

This codec owns the *container* only: it places the ordered component byte-blobs
into their aligned :class:`BundleLayout` sections, writes/validates the header +
payload CRC, and gates load on the geometry fingerprint. It does NOT know how
those bytes are sourced from the GPU KV cache — that is the connector's /
``BundleSources`` / gather-kernel job.

The GPU save path gathers components directly into the offload bundle buffer at the
layout offsets (avoiding an extra copy) and finalizes the header + CRC on the
host MemoryObj after D2H. The CPU-tensor :meth:`assemble_n` / :meth:`disassemble_n`
here are the reference/round-trip path used by unit tests and the fail-closed
validation on load.
"""

from __future__ import annotations

import torch

from atom.kv_transfer.offload.hybrid.kv_bundle import (
    BundleGeometry,
    BundleHeader,
    BundleLayout,
    pack_bundle_header,
    payload_crc,
    unpack_bundle_header,
)


class BundleError(RuntimeError):
    """Raised on any offload bundle validation failure (=> caller fails closed)."""


def _as_uint8_1d(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise BundleError(f"offload bundle: {name} must be a torch.Tensor")
    if t.dtype != torch.uint8:
        raise BundleError(f"offload bundle: {name} must be uint8, got {t.dtype}")
    flat = t.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat


def _crc_over(buf: torch.Tensor) -> int:
    """CRC32 over a uint8 tensor region (moves to host if needed)."""
    host = buf if buf.device.type == "cpu" else buf.cpu()
    return payload_crc(memoryview(host.contiguous().numpy()))


class BundleCodec:
    """Container codec for one geometry (one model / shard / layout version)."""

    def __init__(self, geometry: BundleGeometry) -> None:
        self.geometry = geometry
        self._fingerprint = geometry.fingerprint()

    # -- N-component container API ----------------------------------------
    def build_layout_n(self, sizes: "list[tuple[str, int]]") -> BundleLayout:
        """Ordered component layout for any profile (block/swa/slot/staging)."""
        return BundleLayout.build(sizes)

    def read_header_generic(self, unit: torch.Tensor) -> BundleHeader:
        """Parse + fully validate a generic unit header. Raises on failure."""
        buf = _as_uint8_1d(unit, "unit")
        head_host = buf[: _header_read_len(buf)]
        head_host = head_host if head_host.device.type == "cpu" else head_host.cpu()
        try:
            header = unpack_bundle_header(memoryview(head_host.contiguous().numpy()))
        except ValueError as exc:
            raise BundleError(str(exc)) from exc
        if header.fingerprint != self._fingerprint:
            raise BundleError(
                "offload bundle: geometry fingerprint mismatch "
                "(model/shard/dtype/layout changed) => recompute"
            )
        if header.total_bytes != int(buf.numel()):
            raise BundleError(
                f"offload bundle: total_bytes {header.total_bytes} != actual "
                f"{int(buf.numel())}"
            )
        return header

    def assemble_n(
        self,
        components: "list[tuple[str, torch.Tensor]]",
        *,
        boundary_B: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack ordered named components + header into one uint8 offload bundle."""
        comps = [(str(name), _as_uint8_1d(t, str(name))) for name, t in components]
        layout = BundleLayout.build([(name, int(t.numel())) for name, t in comps])
        if out is None:
            out = torch.zeros(layout.total_bytes, dtype=torch.uint8)
        else:
            out = _as_uint8_1d(out, "out")
            if int(out.numel()) != layout.total_bytes:
                raise BundleError(
                    f"offload bundle: out size {int(out.numel())} != layout total "
                    f"{layout.total_bytes}"
                )
            out.zero_()
        for (_name, t), c in zip(comps, layout.components):
            out[c.off : c.off + t.numel()] = t.to(out.device)
        start, end = layout.payload_slice()
        crc = _crc_over(out[start:end])
        header = pack_bundle_header(
            layout=layout,
            fingerprint=self._fingerprint,
            boundary_B=int(boundary_B),
            payload_crc32=crc,
            layout_version=self.geometry.layout_version,
        )
        out[: layout.header_bytes] = torch.frombuffer(
            bytearray(header), dtype=torch.uint8
        ).to(out.device)
        return out

    def disassemble_n(
        self,
        unit: torch.Tensor,
        names: "list[str]",
        *,
        expect_boundary_B: int | None = None,
        verify_crc: bool = True,
    ) -> "dict[str, torch.Tensor]":
        """Validate a unit and return {name: view} for the given ordered names."""
        buf = _as_uint8_1d(unit, "unit")
        header = self.read_header_generic(buf)
        if expect_boundary_B is not None and header.boundary_B != int(expect_boundary_B):
            raise BundleError(
                f"offload bundle: boundary_B {header.boundary_B} != expected "
                f"{int(expect_boundary_B)}"
            )
        if len(header.components) != len(names):
            raise BundleError(
                f"offload bundle: header has {len(header.components)} components, "
                f"expected {len(names)}"
            )
        layout = BundleLayout.build(
            [(names[i], header.components[i][0]) for i in range(len(names))]
        )
        if layout.total_bytes != int(buf.numel()):
            raise BundleError(
                f"offload bundle: header-derived size {layout.total_bytes} != actual "
                f"{int(buf.numel())}"
            )
        for i, c in enumerate(layout.components):
            if c.off != header.components[i][1]:
                raise BundleError(
                    f"offload bundle: component {names[i]!r} offset {c.off} disagrees "
                    f"with header {header.components[i][1]}"
                )
        if verify_crc:
            start, end = layout.payload_slice()
            actual = _crc_over(buf[start:end])
            if actual != header.payload_crc32:
                raise BundleError(
                    "offload bundle: payload CRC mismatch "
                    f"(stored={header.payload_crc32:#010x} actual={actual:#010x}) "
                    "=> corrupt, recompute"
                )
        return {c.name: buf[c.off : c.off + c.nbytes] for c in layout.components}


def _header_read_len(buf: torch.Tensor) -> int:
    from atom.kv_transfer.offload.hybrid.kv_bundle import HEADER_BYTES

    return min(int(buf.numel()), HEADER_BYTES)
