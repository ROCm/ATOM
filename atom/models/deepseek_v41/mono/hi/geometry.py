# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Geometry of the shared layer pieces: what blocks take as arguments.

The recipe builds these from the model's geometry (``v41.config``); blocks (``blocks``) never read a model's
constants. They are frozen (hashable): blocks put them in their build keys.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NormGeometry:
    """An RMSNorm over rows of ``width`` elements."""

    width: int
    eps: float


@dataclass(frozen=True)
class HCGeometry:
    """Hyper-connections: ``streams`` residual streams of ``hidden`` (the flat row is
    stream-major, ``streams * hidden``); each sub-layer's HC mixes them through a low-rank
    path of ``lowrank`` columns and normalizes each stream with ``eps``."""

    hidden: int
    streams: int
    lowrank: int
    eps: float

    @property
    def width(self) -> int:
        """The flat stream row: streams * hidden."""
        return self.streams * self.hidden
