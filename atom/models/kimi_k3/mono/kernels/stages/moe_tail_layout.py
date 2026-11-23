# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""LDS allocation for the packed gfx1250 K2 tail."""

from dataclasses import dataclass

from atom.models.kimi_k3.mono.config import LATENT
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    LDS_PAD,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX1250

ALIGNMENT = 16


@dataclass(frozen=True)
class TailLdsRegion:
    name: str
    offset: int
    bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class TailLdsPlan:
    tokens: int
    gemv_regions: tuple[TailLdsRegion, ...]
    norm_regions: tuple[TailLdsRegion, ...]

    def __post_init__(self) -> None:
        if not 1 <= self.tokens <= GFX1250_MAX_TOKENS:
            raise ValueError("tail LDS tokens outside supported range")
        for regions in (self.gemv_regions, self.norm_regions):
            previous = 0
            for region in regions:
                if region.offset % ALIGNMENT:
                    raise ValueError(f"{region.name}: unaligned LDS offset")
                if region.offset < previous:
                    raise ValueError(f"{region.name}: overlapping LDS region")
                previous = region.end

    @property
    def required_bytes(self) -> int:
        return max(
            max((region.end for region in self.gemv_regions), default=0),
            max((region.end for region in self.norm_regions), default=0),
        )

    @property
    def token_tiles(self):
        return Gfx1250TokenTiling(self.tokens).tiles

    @property
    def lds_rows(self) -> int:
        return Gfx1250TokenTiling(self.tokens).lds_rows


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def tail_lds_plan(tokens: int, waves: int) -> TailLdsPlan:
    if waves <= 0:
        raise ValueError("tail LDS plan needs waves")
    tile_rows = Gfx1250TokenTiling(tokens).lds_rows
    activation = tile_rows * (LATENT + LDS_PAD) * 2
    accumulator = (
        waves * GFX1250.wave_size * GFX1250_GEMV.accumulator_elements_per_lane * 4
    )
    accumulator_offset = _align(activation)
    gemv = (
        TailLdsRegion("tail_gemv_rows", 0, activation),
        TailLdsRegion("tail_gemv_accumulator", accumulator_offset, accumulator),
    )
    norm = (TailLdsRegion("tail_norm_tree", 0, tokens * 64 * 4),)
    return TailLdsPlan(tokens, gemv, norm)
