# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host runner for gfx1250 local-compute stages and external TP collectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from atom.models.kimi_k3.mono.external_collectives import (
    CollectiveBoundary,
    CollectiveKind,
    ExternalCollective,
    ExternalCollectivePlan,
)

AllReduce = Callable[[torch.Tensor], torch.Tensor | None]
AllGather = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MoeLocalPartials:
    """Rank-local outputs retained across the routed-latent collective."""

    routed: torch.Tensor
    shared: torch.Tensor


@dataclass(frozen=True)
class PackedShardedFrontPartials:
    payload: torch.Tensor
    retained: object


class ExternalCollectiveK2Runner:
    """Execute packed rank-local K2 regions around regular TP collectives.

    Stage callables are deliberately passed in rather than imported here.  The
    device builder owns their weights, ABI and launchers; this class owns the
    stream order and validates every tensor before a collective can be issued.
    """

    def __init__(
        self,
        plan: ExternalCollectivePlan,
        all_gather: AllGather,
        all_reduce: AllReduce,
    ) -> None:
        self.plan = plan
        self._all_reduce = all_reduce
        self._all_gather = all_gather

    @staticmethod
    def _validate(
        tensor: torch.Tensor,
        boundary: CollectiveBoundary,
        *,
        stage: str,
    ) -> None:
        expected = (
            boundary.local_shape if stage.endswith("partial") else boundary.global_shape
        )
        if tensor.shape != expected:
            raise ValueError(
                f"{stage}: expected {expected}, " f"got {tuple(tensor.shape)}"
            )
        expected_dtype = {
            "f32": torch.float32,
            "bf16": torch.bfloat16,
            "u8": torch.uint8,
        }[boundary.dtype]
        if tensor.dtype != expected_dtype:
            raise ValueError(f"{stage}: expected {expected_dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{stage}: collective partial must be contiguous")

    def _reduce(
        self,
        name: ExternalCollective,
        partial: torch.Tensor,
    ) -> torch.Tensor:
        boundary = self.plan.boundary(name)
        self._validate(partial, boundary, stage=f"{name.value} partial")
        reduced = self._all_reduce(partial)
        if reduced is None:
            reduced = partial
        self._validate(reduced, boundary, stage=f"{name.value} reduced")
        return reduced

    def _gather(
        self,
        name: ExternalCollective,
        partial: torch.Tensor,
    ) -> torch.Tensor:
        boundary = self.plan.boundary(name)
        if boundary.kind is not CollectiveKind.ALL_GATHER:
            raise ValueError(f"{name.value}: boundary is not all-gather")
        self._validate(partial, boundary, stage=f"{name.value} partial")
        gathered = self._all_gather(partial)
        self._validate(gathered, boundary, stage=f"{name.value} gathered")
        return gathered

    def run(
        self,
        oproj_local: Callable[[], torch.Tensor],
        front_local: Callable[[torch.Tensor], PackedShardedFrontPartials],
        finish_local: Callable[[torch.Tensor, object], MoeLocalPartials],
        finalize_local: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run the single-gather peer-major front payload contract."""

        oproj = self._reduce(ExternalCollective.OPROJ, oproj_local())
        front = front_local(oproj)
        gathered = self._gather(
            ExternalCollective.FRONT_PAYLOAD,
            front.payload,
        )
        partials = finish_local(gathered, front.retained)
        routed = self._reduce(
            ExternalCollective.ROUTED_LATENT,
            partials.routed,
        )
        final_partial = finalize_local(routed, partials.shared)
        return self._reduce(ExternalCollective.FINAL_HIDDEN, final_partial)
