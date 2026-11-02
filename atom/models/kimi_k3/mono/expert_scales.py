# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Architecture-explicit Kimi expert E8M0 scale materialization.

AITER's loaded expert payload is byte-compatible between gfx950 and gfx1250,
but its scale plane is not.  gfx950 uses ``shuffle_scale``'s padded 256-row by
8-group permutation.  gfx1250's ``moe_shuffle_scale`` uses n32k4: one
contiguous dword holds four K32 scales for one output row.  The source layout
must come from the quant method's architecture metadata; shape is validation,
never a layout discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    ExpertScaleLayout,
    ExpertScaleSpec,
)
from atom.mono.plan.architecture import (
    GFX950,
    GFX1250,
    Architecture,
    architecture_for,
)


@dataclass(frozen=True)
class _SourceToken:
    identity: int
    data_ptr: int
    version: int
    dtype: str
    device: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _source_token(source: torch.Tensor) -> _SourceToken:
    return _SourceToken(
        id(source),
        source.data_ptr(),
        source._version,
        str(source.dtype),
        str(source.device),
        tuple(source.shape),
        tuple(source.stride()),
    )


@dataclass(frozen=True)
class ExpertScaleKey:
    source: _SourceToken
    source_layout: ExpertScaleLayout
    target_architecture: str
    target_layout: ExpertScaleLayout
    spec: ExpertScaleSpec


@dataclass(frozen=True)
class ExpertScaleHandle:
    """A materialized tensor plus enough metadata to reject stale reuse."""

    tensor: torch.Tensor
    key: ExpertScaleKey
    _source: torch.Tensor

    @property
    def spec(self) -> ExpertScaleSpec:
        return self.key.spec

    @property
    def layout(self) -> ExpertScaleLayout:
        return self.key.target_layout

    def validate(
        self,
        architecture: Architecture,
        spec: ExpertScaleSpec | None = None,
    ) -> None:
        if architecture.gfx != self.key.target_architecture:
            raise ValueError(
                f"{self.spec.name}: materialized for "
                f"{self.key.target_architecture}, not {architecture.gfx}"
            )
        if _source_token(self._source) != self.key.source:
            raise ValueError(f"{self.spec.name}: stale expert scale materialization")
        if spec is not None and spec != self.spec:
            raise ValueError(f"{self.spec.name}: expert scale spec mismatch")
        expected = self.spec.storage_shape(self.layout)
        if tuple(self.tensor.shape) != expected:
            raise ValueError(
                f"{self.spec.name}: materialized shape "
                f"{tuple(self.tensor.shape)}, expected {expected}"
            )


class ExpertScaleCache:
    """Per-expert-module cache isolated by source, TP, architecture and layout."""

    def __init__(self) -> None:
        self._entries: dict[ExpertScaleKey, ExpertScaleHandle] = {}

    def get(self, key: ExpertScaleKey) -> ExpertScaleHandle | None:
        handle = self._entries.get(key)
        if handle is not None:
            handle.validate(architecture_for(key.target_architecture), key.spec)
        return handle

    def put(self, handle: ExpertScaleHandle) -> ExpertScaleHandle:
        stale = [
            key
            for key in self._entries
            if key.spec == handle.key.spec
            and key.source_layout is handle.key.source_layout
            and key.target_architecture == handle.key.target_architecture
            and key.target_layout is handle.key.target_layout
            and key.source != handle.key.source
        ]
        for key in stale:
            del self._entries[key]
        self._entries[handle.key] = handle
        return handle


def _validate_source(
    source: torch.Tensor,
    spec: ExpertScaleSpec,
    layout: ExpertScaleLayout,
) -> torch.Tensor:
    if source.element_size() != 1:
        raise ValueError(f"{spec.name}: expert scales must use one-byte E8M0")
    if not source.is_contiguous():
        raise ValueError(f"{spec.name}: expert scale storage must be contiguous")
    expected = spec.storage_shape(layout)
    if tuple(source.shape) != expected:
        raise ValueError(
            f"{spec.name}: {layout.value} shape {tuple(source.shape)}, "
            f"expected {expected}"
        )
    return source.view(torch.uint8)


def materialize_expert_scale(
    source: torch.Tensor,
    *,
    spec: ExpertScaleSpec,
    source_layout: ExpertScaleLayout,
    architecture: Architecture,
    cache: ExpertScaleCache,
) -> ExpertScaleHandle:
    """Return the architecture's scale layout, reusing an exact cache hit."""

    if architecture is GFX950:
        target_layout = ExpertScaleLayout.GFX950_SHUFFLE
        if source_layout is not target_layout:
            raise ValueError("gfx950 Kimi requires its existing shuffle_scale layout")
    elif architecture is GFX1250:
        target_layout = ExpertScaleLayout.GFX1250_N32K4
    else:
        raise ValueError(f"unsupported Kimi expert architecture {architecture.gfx}")

    source_bytes = _validate_source(source, spec, source_layout)
    key = ExpertScaleKey(
        _source_token(source),
        source_layout,
        architecture.gfx,
        target_layout,
        spec,
    )
    found = cache.get(key)
    if found is not None:
        return found

    if source_layout is target_layout:
        tensor = source_bytes.reshape(spec.storage_shape(target_layout))
    else:
        packing = spec.packing
        source_for_destination = [0] * packing.gfx1250_scale_bytes
        for old, destination in packing.scale_repack_map():
            source_for_destination[destination] = old
        indices = torch.tensor(
            source_for_destination,
            dtype=torch.int64,
            device=source.device,
        )
        tensor = source_bytes.reshape(-1).index_select(0, indices)
        tensor = tensor.reshape(spec.storage_shape(target_layout))

    handle = ExpertScaleHandle(tensor, key, source)
    handle.validate(architecture, spec)
    return cache.put(handle)


@dataclass(frozen=True)
class KimiExpertScaleHandles:
    architecture: Architecture
    w13: ExpertScaleHandle
    w2: ExpertScaleHandle

    def validate(self) -> None:
        if self.architecture is not GFX1250:
            raise ValueError("the K2 wave32 declaration requires gfx1250 scales")
        self.w13.validate(self.architecture)
        self.w2.validate(self.architecture)
        if self.w13.spec.tp_rank != self.w2.spec.tp_rank:
            raise ValueError("expert scale handles disagree on TP rank")
        if self.w13.spec.tp_size != self.w2.spec.tp_size:
            raise ValueError("expert scale handles disagree on TP size")

    def kernel_arguments(self) -> dict[str, torch.Tensor]:
        self.validate()
        return {"w13s": self.w13.tensor, "w2s": self.w2.tensor}
