# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache-facing GPU connector for the offload bundle.

LMCache sees exactly one opaque, fixed-size ``uint8`` MemoryObj per terminal
checkpoint. This connector owns the save/load data path between that MemoryObj
and the GPU sources:

    save (from_gpu):  KV sources --gather--> GPU staging --D2H--> MemoryObj
                      then finalize header + payload CRC on the host object.
    load (to_gpu):    validate header/CRC/fingerprint on the host object,
                      MemoryObj --H2D--> GPU staging --scatter--> KV sources.

The connector is geometry-agnostic: the worker-side connector supplies the
ordered per-component source regions (via a profile + ``BundleSources`` —
block / swa / slot / staging kinds) and the terminal boundary ``B``.
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
from atom.kv_transfer.offload.hybrid.kv_bundle import (
    pack_bundle_header,
    payload_crc,
)
from atom.kv_transfer.offload.hybrid.kv_bundle_codec import (
    BundleCodec,
    BundleError,
)
from atom.kv_transfer.offload.triton_kv_staging import (
    GatherRegion,
    gather_regions,
    scatter_regions,
)


@dataclass(frozen=True)
class SourceRegion:
    """A KV source region for one component (before offset assignment).

    ``base_addr`` is a raw device address; ``physical_ids`` are the block/slot
    ids to move, in the order they should appear in the offload bundle.
    """

    base_addr: int
    unit_bytes: int
    physical_ids: Sequence[int]

    def nbytes(self) -> int:
        return int(self.unit_bytes) * sum(1 for i in self.physical_ids if int(i) >= 0)


def _component_bytes(regions: Sequence[SourceRegion]) -> int:
    return sum(r.nbytes() for r in regions)


@dataclass(frozen=True)
class _BundleGroup:
    """A staged-pipeline group. V1 uses one group = the whole offload bundle."""

    nbytes: int


class BundleGPUConnector:
    """Move one offload bundle between a host MemoryObj and GPU KV sources."""

    def __init__(
        self,
        codec: BundleCodec,
        *,
        device: torch.device | str,
    ) -> None:
        self.codec = codec
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("offload connector requires a CUDA/HIP device")
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
    def _layout_and_regions_n(
        self, components: "Sequence[tuple[str, Sequence[SourceRegion]]]"
    ):
        """Build the unit layout and place each component's regions at its offset.

        ``components`` is an ordered list of ``(name, source_regions)``; the order
        fixes the byte layout. Generic over profile (block / swa / slot / staging).
        """
        layout = self.codec.build_layout_n(
            [(name, _component_bytes(regs)) for name, regs in components]
        )
        gathers: list[GatherRegion] = []
        for c, (_name, comp_regions) in zip(layout.components, components):
            off = c.off
            for r in comp_regions:
                gathers.append(
                    GatherRegion(r.base_addr, int(r.unit_bytes), r.physical_ids, off)
                )
                off += r.nbytes()
        return layout, gathers

    # -- save (generic) ---------------------------------------------------
    def save_bundle(
        self,
        *,
        memory_obj: torch.Tensor,
        components: "Sequence[tuple[str, Sequence[SourceRegion]]]",
        boundary_B: int,
    ) -> None:
        """Gather KV sources for an ordered component list into ``memory_obj``.

        Two-stream pipeline: stage A zeros staging + gathers (pack_stream); stage
        B copies payload D2H (copy_stream). Header + CRC finalized on the host.
        """
        if memory_obj.dtype != torch.uint8:
            raise BundleError("offload connector: MemoryObj must be uint8")
        host = memory_obj.reshape(-1)
        layout, gathers = self._layout_and_regions_n(components)
        if int(host.numel()) != layout.total_bytes:
            raise BundleError(
                f"offload connector: MemoryObj size {int(host.numel())} != unit total "
                f"{layout.total_bytes}"
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
            [_BundleGroup(nbytes=layout.total_bytes)],
            stage_a=_PipelineStage(state.pack_stream, _pack),
            stage_b=_PipelineStage(state.copy_stream, _d2h),
            ensure_buffer=self._ensure_staging_buffer,
            group_nbytes=lambda g: g.nbytes,
        )
        crc = payload_crc(memoryview(host[start:end].contiguous().numpy()))
        header = pack_bundle_header(
            layout=layout,
            fingerprint=self.codec.geometry.fingerprint(),
            boundary_B=int(boundary_B),
            payload_crc32=crc,
            layout_version=self.codec.geometry.layout_version,
        )
        host[: layout.header_bytes] = torch.frombuffer(
            bytearray(header), dtype=torch.uint8
        )

    # -- load (generic) ---------------------------------------------------
    def load_bundle(
        self,
        *,
        memory_obj: torch.Tensor,
        components: "Sequence[tuple[str, Sequence[SourceRegion]]]",
        expect_boundary_B: int | None = None,
    ) -> None:
        """Validate ``memory_obj`` and scatter its components to KV sources.

        Raises :class:`BundleError` on any validation failure so the
        caller can fail closed and recompute.
        """
        host = memory_obj.reshape(-1)
        header = self.codec.read_header_generic(host)
        if expect_boundary_B is not None and header.boundary_B != int(expect_boundary_B):
            raise BundleError(
                f"offload bundle: boundary_B {header.boundary_B} != expected "
                f"{int(expect_boundary_B)}"
            )
        layout, gathers = self._layout_and_regions_n(components)
        if layout.total_bytes != int(host.numel()):
            raise BundleError(
                f"offload connector: region-derived size {layout.total_bytes} != "
                f"MemoryObj {int(host.numel())}"
            )
        if len(header.components) != len(layout.components):
            raise BundleError(
                f"offload bundle: header has {len(header.components)} components, "
                f"regions have {len(layout.components)}"
            )
        for i, c in enumerate(layout.components):
            if header.components[i][0] != c.nbytes:
                raise BundleError(
                    f"offload bundle: component {c.name!r} size disagrees with header "
                    f"(header={header.components[i][0]} regions={c.nbytes})"
                )
        start, end = layout.payload_slice()
        actual_crc = payload_crc(memoryview(host[start:end].contiguous().numpy()))
        if actual_crc != header.payload_crc32:
            raise BundleError(
                "offload bundle: payload CRC mismatch "
                f"(stored={header.payload_crc32:#010x} actual={actual_crc:#010x}) "
                "=> corrupt, recompute"
            )
        state = self._thread_state()

        def _h2d(_group, device_buf: torch.Tensor) -> None:
            device_buf[start:end].copy_(host[start:end], non_blocking=True)

        def _scatter(_group, device_buf: torch.Tensor) -> None:
            scatter_regions(device_buf, gathers, stream=state.pack_stream)

        run_staged_pipeline(
            state,
            [_BundleGroup(nbytes=layout.total_bytes)],
            stage_a=_PipelineStage(state.copy_stream, _h2d),
            stage_b=_PipelineStage(state.pack_stream, _scatter),
            ensure_buffer=self._ensure_staging_buffer,
            group_nbytes=lambda g: g.nbytes,
        )


class BundleLMCacheMetadata:
    """LMCache engine metadata proxy: one fixed-size opaque uint8 MemoryObj.

    ``get_shapes`` returns the offload bundle byte size (a function of the terminal
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
            raise ValueError("offload metadata: unit_bytes must be > 0")

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
