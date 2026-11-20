# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Zero-copy binding for Kimi expert E8M0 scales in native gfx1250 layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    ExpertScaleLayout,
    ExpertScaleSpec,
)
from atom.mono.plan.architecture import GFX1250, Architecture


@dataclass(frozen=True)
class ExpertScaleHandle:
    """A validated zero-copy view of one native gfx1250 scale plane."""

    tensor: torch.Tensor
    spec: ExpertScaleSpec
    layout: ExpertScaleLayout
    architecture: Architecture

    def validate(
        self,
        architecture: Architecture,
        spec: ExpertScaleSpec | None = None,
    ) -> None:
        if architecture != self.architecture:
            raise ValueError(
                f"{self.spec.name}: bound for {self.architecture.gfx}, "
                f"not {architecture.gfx}"
            )
        if spec is not None and spec != self.spec:
            raise ValueError(f"{self.spec.name}: expert scale spec mismatch")
        expected = self.spec.storage_shape(self.layout)
        if tuple(self.tensor.shape) != expected:
            raise ValueError(
                f"{self.spec.name}: bound shape "
                f"{tuple(self.tensor.shape)}, expected {expected}"
            )


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


def bind_expert_scale(
    source: torch.Tensor,
    *,
    spec: ExpertScaleSpec,
    source_layout: ExpertScaleLayout,
    architecture: Architecture,
) -> ExpertScaleHandle:
    """Validate and return the native gfx1250 scale tensor without conversion."""

    if architecture is not GFX1250:
        raise ValueError(f"unsupported Kimi expert architecture {architecture.gfx}")
    target_layout = ExpertScaleLayout.GFX1250_N32K4
    if source_layout is not target_layout:
        raise ValueError(
            f"{spec.name}: gfx1250 mono requires native {target_layout.value} scales"
        )
    tensor = _validate_source(source, spec, source_layout).reshape(
        spec.storage_shape(target_layout)
    )
    handle = ExpertScaleHandle(tensor, spec, target_layout, architecture)
    handle.validate(architecture, spec)
    return handle


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
