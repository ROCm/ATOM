# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only architecture contracts for Kimi K2's latent-MoE front half."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX950, GFX1250, Architecture

TP = 8
HIDDEN = 7168
LATENT = 3584
EXPERTS = 896
TOPK = 16
ROWS = 16
UG_GROUPS = 12
MX_GROUP = 32
MX_WORDS = MX_GROUP // 2
MX_PACKED_WORDS = MX_GROUP // 8


def f32_round(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def ordered_f32_key(value: float) -> int:
    """Mirror ``moe._okey`` including signed-zero ordering."""

    bits = struct.unpack("<I", struct.pack("<f", f32_round(value)))[0]
    if bits & 0x80000000:
        return (~bits) & 0xFFFFFFFF
    return bits ^ 0x80000000


@dataclass(frozen=True)
class RouterContract:
    architecture: Architecture

    @property
    def experts_per_lane(self) -> int:
        if EXPERTS % self.architecture.wave_size:
            raise ValueError("experts must divide the router wave")
        return EXPERTS // self.architecture.wave_size

    def experts_for_lane(self, lane: int) -> tuple[int, ...]:
        if not 0 <= lane < self.architecture.wave_size:
            raise ValueError(f"lane {lane} outside router wave")
        return tuple(
            lane + self.architecture.wave_size * index
            for index in range(self.experts_per_lane)
        )

    def select(
        self, scores: list[float], sigmoid: list[float]
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Reference the device key/tie order and serial sigmoid renormalization."""

        if len(scores) != EXPERTS or len(sigmoid) != EXPERTS:
            raise ValueError(f"router needs exactly {EXPERTS} values")
        clean_scores = [
            f32_round(-3.0e38) if math.isnan(value) else f32_round(value)
            for value in scores
        ]
        clean_sigmoid = [
            f32_round(0.0) if math.isnan(value) else f32_round(value)
            for value in sigmoid
        ]
        selected = tuple(
            sorted(
                range(EXPERTS),
                key=lambda expert: (-ordered_f32_key(clean_scores[expert]), expert),
            )[:TOPK]
        )
        total = f32_round(0.0)
        for expert in selected:
            total = f32_round(total + clean_sigmoid[expert])
        inverse = f32_round(1.0 / total)
        weights = tuple(
            f32_round(clean_sigmoid[expert] * inverse) for expert in selected
        )
        return selected, weights


@dataclass(frozen=True)
class RouteUnion:
    experts: tuple[int, ...]
    slot_of: tuple[int, ...]
    kof: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RouteUnionContract:
    architecture: Architecture
    threads: int
    tokens: int

    def __post_init__(self) -> None:
        if self.threads <= 0 or self.threads % self.architecture.wave_size:
            raise ValueError("route threads must be a positive wave multiple")
        maximum = GFX1250_MAX_TOKENS if self.architecture is GFX1250 else ROWS
        if not 1 <= self.tokens <= maximum:
            raise ValueError(f"route tokens must be in [1, {maximum}]")

    @property
    def waves(self) -> int:
        return self.threads // self.architecture.wave_size

    @property
    def experts_per_thread(self) -> int:
        return (EXPERTS + self.threads - 1) // self.threads

    def experts_for_thread(self, tid: int) -> tuple[int, ...]:
        """Contiguous ownership makes scan order equal ascending expert order."""

        if not 0 <= tid < self.threads:
            raise ValueError(f"thread {tid} outside route CTA")
        first = tid * self.experts_per_thread
        return tuple(range(first, min(first + self.experts_per_thread, EXPERTS)))

    @property
    def scan_words(self) -> int:
        return self.waves

    def build(self, routes: tuple[tuple[int, ...], ...]) -> RouteUnion:
        if len(routes) != self.tokens:
            raise ValueError("route rows do not match token count")
        for row in routes:
            if len(row) != TOPK or len(set(row)) != TOPK:
                raise ValueError("each route row must contain 16 unique experts")
            if any(not 0 <= expert < EXPERTS for expert in row):
                raise ValueError("route expert outside model range")
        experts = tuple(sorted({expert for row in routes for expert in row}))
        slots = [-1] * EXPERTS
        for slot, expert in enumerate(experts):
            slots[expert] = slot
        kof = [[-1] * self.tokens for _ in experts]
        for token, row in enumerate(routes):
            for topk, expert in enumerate(row):
                kof[slots[expert]][token] = topk
        return RouteUnion(
            experts,
            tuple(slots),
            tuple(tuple(row) for row in kof),
        )

    @property
    def token_tiles(self):
        if self.architecture is not GFX1250:
            raise ValueError("route token tiling is gfx1250-only")
        return Gfx1250TokenTiling(self.tokens).tiles

    def kof_tile(
        self, union: RouteUnion, tile_index: int
    ) -> tuple[tuple[int, ...], ...]:
        """Return the compact tile-local KOF matrix planned for LDS."""

        tile = Gfx1250TokenTiling(self.tokens).tile(tile_index)
        return tuple(row[tile.row_start : tile.row_stop] for row in union.kof)


@dataclass(frozen=True)
class LatentQuantContract:
    architecture: Architecture
    threads: int
    tokens: int

    def __post_init__(self) -> None:
        if self.threads <= 0 or self.threads % self.architecture.wave_size:
            raise ValueError("quant threads must be a positive wave multiple")
        maximum = GFX1250_MAX_TOKENS if self.architecture is GFX1250 else ROWS
        if not 1 <= self.tokens <= maximum:
            raise ValueError(f"quant tokens must be in [1, {maximum}]")

    @property
    def blocks_per_token(self) -> int:
        return LATENT // MX_GROUP

    @property
    def blocks(self) -> int:
        return self.tokens * self.blocks_per_token

    def blocks_for_thread(self, tid: int) -> tuple[int, ...]:
        if not 0 <= tid < self.threads:
            raise ValueError(f"thread {tid} outside quant CTA")
        return tuple(range(tid, self.blocks, self.threads))

    def coordinates(self, block: int) -> tuple[int, int]:
        if not 0 <= block < self.blocks:
            raise ValueError(f"quant block {block} outside latent tensor")
        return block // self.blocks_per_token, block % self.blocks_per_token

    def output_offsets(self, block: int) -> tuple[tuple[int, ...], int]:
        token, group = self.coordinates(block)
        payload = tuple(
            token * (LATENT // 8) + group * MX_PACKED_WORDS + word
            for word in range(MX_PACKED_WORDS)
        )
        scale = token * self.blocks_per_token + group
        return payload, scale

    @property
    def token_tiles(self):
        if self.architecture is not GFX1250:
            raise ValueError("quant token tiling is gfx1250-only")
        return Gfx1250TokenTiling(self.tokens).tiles

    @staticmethod
    def e8m0_roundup(amax: float) -> int:
        """CPU mirror of Kimi's capped ceil exponent for ``amax / 6``."""

        working = f32_round(f32_round(amax) * f32_round(1.0 / 6.0))
        bits = struct.unpack("<I", struct.pack("<f", working))[0]
        return min(((bits + 0x7FFFFF) >> 23) & 0xFF, 254)


ROUTER_CONTRACTS = {
    GFX950.gfx: RouterContract(GFX950),
    GFX1250.gfx: RouterContract(GFX1250),
}


def router_contract_for(architecture: Architecture) -> RouterContract:
    try:
        return ROUTER_CONTRACTS[architecture.gfx]
    except KeyError as error:
        raise ValueError(f"no K2 router contract for {architecture.gfx}") from error
