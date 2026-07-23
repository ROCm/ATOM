# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache-facing GPU connector for the DSV4 offload unit.

LMCache sees exactly one opaque, fixed-size ``uint8`` MemoryObj per DSV4
terminal checkpoint. This connector owns the save/load data path between that
MemoryObj and the three GPU sources:

    save (from_gpu):  KV sources --gather--> GPU staging --D2H--> MemoryObj
                      then finalize header + payload CRC on the host object.
    load (to_gpu):    validate header/CRC/fingerprint on the host object,
                      MemoryObj --H2D--> GPU staging --scatter--> KV sources.

Phase 1 uses a single full-unit staging buffer (correct + testable). The
bounded two-stream ping-pong that the MHA connector uses
(``ATOMLMCacheGPUConnector`` / ``OFFLOAD_GPU_STAGING_CHUNKS``) is wired in
Phase 2; see ``dsv4-lmcache-bundle-plan.md``.

The connector is geometry-agnostic: the worker-side connector (Phase 2) supplies
the per-component source regions (compressed ``block_regions`` keyed by
``seq.block_table``, ``swa_block_regions`` keyed by ``seq.swa_block_table``, and
the ``v4_state_pool`` CSA-state slot) and the terminal boundary ``B``.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Sequence

import torch

from atom.kv_transfer.offload.atom_lmcache_staging import (
    _PipelineStage,
    _StagingBuffer,
    _ThreadTransferState,
    run_staged_pipeline,
)
from atom.kv_transfer.offload.dsv4.unit import pack_header, payload_crc
from atom.kv_transfer.offload.dsv4.unit_codec import (
    DSV4OffloadUnitCodec,
    DSV4OffloadUnitError,
)
from atom.kv_transfer.offload.triton_offload_gather import (
    GatherRegion,
    gather_regions,
    scatter_regions,
)


@dataclass(frozen=True)
class SourceRegion:
    """A KV source region for one component (before offset assignment).

    ``base_addr`` is a raw device address; ``physical_ids`` are the block/slot
    ids to move, in the order they should appear in the offload unit.
    """

    base_addr: int
    unit_bytes: int
    physical_ids: Sequence[int]

    def nbytes(self) -> int:
        return int(self.unit_bytes) * sum(1 for i in self.physical_ids if int(i) >= 0)


def _component_bytes(regions: Sequence[SourceRegion]) -> int:
    return sum(r.nbytes() for r in regions)


@dataclass(frozen=True)
class _UnitGroup:
    """A staged-pipeline group. V1 uses one group = the whole offload unit."""

    nbytes: int


class DSV4OffloadUnitGPUConnector:
    """Move one DSV4 offload unit between a host MemoryObj and GPU KV sources."""

    def __init__(
        self,
        codec: DSV4OffloadUnitCodec,
        *,
        device: torch.device | str,
    ) -> None:
        self.codec = codec
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DSV4 offload connector requires a CUDA/HIP device")
        # Per-thread pack/copy streams + staging buffer, shared with the chunked
        # connector's two-stream pipeline. V1 uses one group (the whole unit);
        # bounded tiling for ping-pong overlap is a follow-up.
        self._tls = threading.local()

    # -- staging ----------------------------------------------------------
    def _thread_state(self) -> _ThreadTransferState:
        state = getattr(self._tls, "state", None)
        if state is None:
            state = _ThreadTransferState(self.device, self.device.type == "cuda")
            self._tls.state = state
        return state

    def _ensure_staging_buffer(
        self, staging_buffer: _StagingBuffer, nbytes: int
    ) -> torch.Tensor:
        nbytes = int(nbytes)
        if staging_buffer.tensor is None or int(staging_buffer.tensor.numel()) < nbytes:
            staging_buffer.tensor = torch.empty(
                nbytes, dtype=torch.uint8, device=self.device
            )
            staging_buffer.free_event_valid = False
        return staging_buffer.tensor[:nbytes]

    # -- layout -----------------------------------------------------------
    def _layout_and_regions(
        self,
        *,
        compressed: Sequence[SourceRegion],
        swa: Sequence[SourceRegion],
        csa_state: Sequence[SourceRegion],
    ):
        """Build the unit layout and place each source region at its unit offset."""
        layout = self.codec.build_layout(
            compressed_bytes=_component_bytes(compressed),
            swa_bytes=_component_bytes(swa),
            csa_state_bytes=_component_bytes(csa_state),
        )
        gathers: list[GatherRegion] = []
        for base_off, comp_regions in (
            (layout.compressed_off, compressed),
            (layout.swa_off, swa),
            (layout.csa_state_off, csa_state),
        ):
            off = base_off
            for r in comp_regions:
                gathers.append(
                    GatherRegion(r.base_addr, int(r.unit_bytes), r.physical_ids, off)
                )
                off += r.nbytes()
        return layout, gathers

    # -- save -------------------------------------------------------------
    def save_unit(
        self,
        *,
        memory_obj: torch.Tensor,
        compressed: Sequence[SourceRegion],
        swa: Sequence[SourceRegion],
        csa_state: Sequence[SourceRegion],
        boundary_B: int,
    ) -> None:
        """Gather KV sources into ``memory_obj`` as a valid offload unit.

        Runs the shared two-stream pipeline: stage A zeros the staging buffer and
        gathers the source regions (pack_stream); stage B copies the payload
        D2H into the host MemoryObj (copy_stream). The header + payload CRC are
        finalized on the host after the pipeline synchronizes.
        """
        if memory_obj.dtype != torch.uint8:
            raise DSV4OffloadUnitError("DSV4 offload connector: MemoryObj must be uint8")
        host = memory_obj.reshape(-1)
        layout, gathers = self._layout_and_regions(
            compressed=compressed, swa=swa, csa_state=csa_state
        )
        if int(host.numel()) != layout.total_bytes:
            raise DSV4OffloadUnitError(
                "DSV4 offload connector: MemoryObj size "
                f"{int(host.numel())} != unit total {layout.total_bytes}"
            )
        start, end = layout.payload_slice()
        state = self._thread_state()

        def _pack(_group, device_buf: torch.Tensor) -> None:
            device_buf.zero_()  # deterministic (CRC-stable) padding gaps
            gather_regions(device_buf, gathers, stream=state.pack_stream)

        def _d2h(_group, device_buf: torch.Tensor) -> None:
            host[start:end].copy_(device_buf[start:end], non_blocking=True)

        run_staged_pipeline(
            state,
            [_UnitGroup(nbytes=layout.total_bytes)],
            stage_a=_PipelineStage(state.pack_stream, _pack),
            stage_b=_PipelineStage(state.copy_stream, _d2h),
            ensure_buffer=self._ensure_staging_buffer,
            group_nbytes=lambda g: g.nbytes,
        )
        # Finalize header + CRC on the host object (payload now landed).
        crc = payload_crc(memoryview(host[start:end].contiguous().numpy()))
        header = pack_header(
            layout=layout,
            geometry=self.codec.geometry,
            boundary_B=int(boundary_B),
            payload_crc32=crc,
        )
        host[: layout.header_bytes] = torch.frombuffer(
            bytearray(header), dtype=torch.uint8
        )

    # -- load -------------------------------------------------------------
    def load_unit(
        self,
        *,
        memory_obj: torch.Tensor,
        compressed: Sequence[SourceRegion],
        swa: Sequence[SourceRegion],
        csa_state: Sequence[SourceRegion],
        expect_boundary_B: int | None = None,
    ) -> None:
        """Validate ``memory_obj`` and scatter its components to KV sources.

        Raises :class:`DSV4OffloadUnitError` on any validation failure so the
        caller can fail closed and recompute. Runs the shared two-stream
        pipeline: stage A copies the payload H2D (copy_stream); stage B scatters
        it into the KV regions (pack_stream).
        """
        host = memory_obj.reshape(-1)
        # Full fail-closed validation (magic/version/fingerprint/size/CRC/boundary).
        header = self.codec.read_header(host)
        if expect_boundary_B is not None and header.boundary_B != int(
            expect_boundary_B
        ):
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: boundary_B "
                f"{header.boundary_B} != expected {int(expect_boundary_B)}"
            )
        layout, gathers = self._layout_and_regions(
            compressed=compressed, swa=swa, csa_state=csa_state
        )
        if layout.total_bytes != int(host.numel()):
            raise DSV4OffloadUnitError(
                "DSV4 offload connector: region-derived size "
                f"{layout.total_bytes} != MemoryObj {int(host.numel())}"
            )
        if (
            layout.compressed_bytes != header.compressed_bytes
            or layout.swa_bytes != header.swa_bytes
            or layout.csa_state_bytes != header.csa_state_bytes
        ):
            raise DSV4OffloadUnitError(
                "DSV4 offload connector: component sizes disagree with header "
                f"(header c/s/state={header.compressed_bytes}/{header.swa_bytes}/"
                f"{header.csa_state_bytes}, regions="
                f"{layout.compressed_bytes}/{layout.swa_bytes}/{layout.csa_state_bytes})"
            )
        start, end = layout.payload_slice()
        actual_crc = payload_crc(memoryview(host[start:end].contiguous().numpy()))
        if actual_crc != header.payload_crc32:
            raise DSV4OffloadUnitError(
                "DSV4 offload unit: payload CRC mismatch "
                f"(stored={header.payload_crc32:#010x} actual={actual_crc:#010x}) "
                "=> corrupt, recompute"
            )
        # H2D payload into staging, then scatter to KV sources, via the pipeline.
        state = self._thread_state()

        def _h2d(_group, device_buf: torch.Tensor) -> None:
            device_buf[start:end].copy_(host[start:end], non_blocking=True)

        def _scatter(_group, device_buf: torch.Tensor) -> None:
            scatter_regions(device_buf, gathers, stream=state.pack_stream)

        run_staged_pipeline(
            state,
            [_UnitGroup(nbytes=layout.total_bytes)],
            stage_a=_PipelineStage(state.copy_stream, _h2d),
            stage_b=_PipelineStage(state.pack_stream, _scatter),
            ensure_buffer=self._ensure_staging_buffer,
            group_nbytes=lambda g: g.nbytes,
        )


class DSV4OffloadUnitLMCacheMetadata:
    """LMCache engine metadata proxy: one fixed-size opaque uint8 MemoryObj.

    ``get_shapes`` returns the offload-unit byte size (a function of the terminal
    boundary via ``unit_bytes_for``), ``get_dtypes`` is ``[uint8]``, and
    ``get_num_groups`` is 1. As in the MHA path, ``engine.fmt`` is set to a
    tensor-accepting ``MemoryFormat`` (e.g. ``KV_2LTD``) purely to satisfy the
    LocalCPU allocator; the real shape is forced here.
    """

    def __init__(self, base_metadata, *, unit_bytes: int) -> None:
        self._base = base_metadata
        self.__dict__.update(vars(base_metadata))
        self.dsv4_unit_bytes = int(unit_bytes)
        if self.dsv4_unit_bytes <= 0:
            raise ValueError("DSV4 offload metadata: unit_bytes must be > 0")

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def is_first_rank(self) -> bool:
        return self._base.is_first_rank()

    def get_dtypes(self) -> list[torch.dtype]:
        return [torch.uint8]

    def get_shapes(self, num_tokens: int | None = None) -> list[torch.Size]:
        return [torch.Size((self.dsv4_unit_bytes,))]

    def get_num_groups(self) -> int:
        return 1
