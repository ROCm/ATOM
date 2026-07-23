# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Assemble / disassemble a DSV4 terminal-checkpoint *offload unit*.

This codec owns the *container* only: it places three opaque component
byte-blobs (compressed KV, raw SWA tail, CSA overlap state) into the
:class:`DSV4OffloadUnitLayout` sections, writes/validates the header + payload
CRC, and gates load on the geometry fingerprint. It does NOT know how those
bytes are sourced from the GPU KV cache — that is the connector's / gather
kernel's job.

The GPU save path gathers components directly into the offload-unit buffer at
the layout offsets (avoiding an extra copy) and finalizes the header + CRC on
the host MemoryObj after D2H. The CPU-tensor :meth:`assemble` / :meth:`disassemble`
here are the reference/round-trip path used by unit tests and the fail-closed
validation on load.
"""

from __future__ import annotations

import torch

from atom.kv_transfer.offload.dsv4.unit import (
    DSV4OffloadUnitGeometry,
    DSV4OffloadUnitHeader,
    DSV4OffloadUnitLayout,
    pack_header,
    payload_crc,
    unpack_header,
)


class DSV4OffloadUnitError(RuntimeError):
    """Raised on any offload-unit validation failure (=> caller fails closed)."""


def _as_uint8_1d(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise DSV4OffloadUnitError(f"DSV4 offload unit: {name} must be a torch.Tensor")
    if t.dtype != torch.uint8:
        raise DSV4OffloadUnitError(
            f"DSV4 offload unit: {name} must be uint8, got {t.dtype}"
        )
    flat = t.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat


def _crc_over(buf: torch.Tensor) -> int:
    """CRC32 over a uint8 tensor region (moves to host if needed)."""
    host = buf if buf.device.type == "cpu" else buf.cpu()
    return payload_crc(memoryview(host.contiguous().numpy()))


class DSV4OffloadUnitCodec:
    """Container codec for one geometry (one model / shard / layout version)."""

    def __init__(self, geometry: DSV4OffloadUnitGeometry) -> None:
        self.geometry = geometry
        self._fingerprint = geometry.fingerprint()

    # -- layout -----------------------------------------------------------
    def build_layout(
        self,
        *,
        compressed_bytes: int,
        swa_bytes: int,
        csa_state_bytes: int,
    ) -> DSV4OffloadUnitLayout:
        return DSV4OffloadUnitLayout.build(
            compressed_bytes=compressed_bytes,
            swa_bytes=swa_bytes,
            csa_state_bytes=csa_state_bytes,
        )

    # -- save -------------------------------------------------------------
    def assemble(
        self,
        *,
        compressed: torch.Tensor,
        swa: torch.Tensor,
        csa_state: torch.Tensor,
        boundary_B: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack the three components + header into a single uint8 offload unit."""
        comp = _as_uint8_1d(compressed, "compressed")
        swa_b = _as_uint8_1d(swa, "swa")
        state = _as_uint8_1d(csa_state, "csa_state")
        layout = self.build_layout(
            compressed_bytes=int(comp.numel()),
            swa_bytes=int(swa_b.numel()),
            csa_state_bytes=int(state.numel()),
        )
        if out is None:
            out = torch.zeros(layout.total_bytes, dtype=torch.uint8)
        else:
            out = _as_uint8_1d(out, "out")
            if int(out.numel()) != layout.total_bytes:
                raise DSV4OffloadUnitError(
                    "DSV4 offload unit: out size "
                    f"{int(out.numel())} != layout total {layout.total_bytes}"
                )
            out.zero_()

        # Place components at their aligned offsets.
        out[layout.compressed_off : layout.compressed_off + comp.numel()] = comp.to(
            out.device
        )
        out[layout.swa_off : layout.swa_off + swa_b.numel()] = swa_b.to(out.device)
        out[layout.csa_state_off : layout.csa_state_off + state.numel()] = state.to(
            out.device
        )

        # CRC over the whole payload region (post-header, including padding gaps,
        # which are zero and thus deterministic).
        start, end = layout.payload_slice()
        crc = _crc_over(out[start:end])
        header = pack_header(
            layout=layout,
            geometry=self.geometry,
            boundary_B=int(boundary_B),
            payload_crc32=crc,
        )
        header_t = torch.frombuffer(bytearray(header), dtype=torch.uint8)
        out[: layout.header_bytes] = header_t.to(out.device)
        return out

    # -- load -------------------------------------------------------------
    def read_header(self, unit: torch.Tensor) -> DSV4OffloadUnitHeader:
        """Parse + fully validate an offload-unit header. Raises on failure."""
        buf = _as_uint8_1d(unit, "unit")
        head_host = buf[: _header_read_len(buf)]
        head_host = head_host if head_host.device.type == "cpu" else head_host.cpu()
        try:
            header = unpack_header(memoryview(head_host.contiguous().numpy()))
        except ValueError as exc:
            raise DSV4OffloadUnitError(str(exc)) from exc
        if header.fingerprint != self._fingerprint:
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: geometry fingerprint mismatch "
                "(model/shard/dtype/layout changed) => recompute"
            )
        if header.total_bytes != int(buf.numel()):
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: total_bytes "
                f"{header.total_bytes} != actual {int(buf.numel())}"
            )
        return header

    def layout_from_header(
        self, header: DSV4OffloadUnitHeader
    ) -> DSV4OffloadUnitLayout:
        return self.build_layout(
            compressed_bytes=header.compressed_bytes,
            swa_bytes=header.swa_bytes,
            csa_state_bytes=header.csa_state_bytes,
        )

    def disassemble(
        self,
        unit: torch.Tensor,
        *,
        expect_boundary_B: int | None = None,
        verify_crc: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate an offload unit and return (compressed, swa, csa_state) views."""
        buf = _as_uint8_1d(unit, "unit")
        header = self.read_header(buf)
        if expect_boundary_B is not None and header.boundary_B != int(
            expect_boundary_B
        ):
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: boundary_B "
                f"{header.boundary_B} != expected {int(expect_boundary_B)}"
            )
        layout = self.layout_from_header(header)
        if layout.total_bytes != int(buf.numel()):
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: header-derived layout size "
                f"{layout.total_bytes} != actual {int(buf.numel())}"
            )
        if verify_crc:
            start, end = layout.payload_slice()
            actual = _crc_over(buf[start:end])
            if actual != header.payload_crc32:
                raise DSV4OffloadUnitError(
                    "DSV4 offload unit: payload CRC mismatch "
                    f"(stored={header.payload_crc32:#010x} actual={actual:#010x}) "
                    "=> corrupt, recompute"
                )
        compressed = buf[
            layout.compressed_off : layout.compressed_off + header.compressed_bytes
        ]
        swa = buf[layout.swa_off : layout.swa_off + header.swa_bytes]
        csa_state = buf[
            layout.csa_state_off : layout.csa_state_off + header.csa_state_bytes
        ]
        return compressed, swa, csa_state


def _header_read_len(buf: torch.Tensor) -> int:
    from atom.kv_transfer.offload.dsv4.unit import HEADER_BYTES

    return min(int(buf.numel()), HEADER_BYTES)
