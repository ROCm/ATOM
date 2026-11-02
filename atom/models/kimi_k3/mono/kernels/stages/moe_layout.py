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
ROUTER_GROUPS = EXPERTS // TP // ROWS
ROUTER_PARTS = 4
LATENT_GROUPS = LATENT // TP // ROWS
LATENT_PARTS = 2
GATE_UP_TASKS = 96
UG_GROUPS = 12
MX_GROUP = 32
MX_WORDS = MX_GROUP // 2
MX_PACKED_WORDS = MX_GROUP // 8


@dataclass(frozen=True)
class K2Dependency:
    path: str
    stage: str
    assumptions: tuple[str, ...]
    gfx1250_status: str


K2_FRONT_HALF_DEPENDENCIES = (
    K2Dependency(
        "kernels/layer.py",
        "fused placement",
        (
            "BLOCKS persistent round-robin",
            "THREADS launch geometry",
            "WAVES accumulator LDS",
            "lane=tid%64 and wave=tid//64",
            "fixed CTA ranges O0=56 and E0=160",
        ),
        "declaration only; launch and residency remain refused",
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "early router/latent/gate-up GEMV",
        (
            "wave64 MFMA accumulator layout",
            "R_TASKS=28, L_TASKS=56, G_TASKS=96 model schedule",
            "THREADS-strided publication",
        ),
        "GEMV backend exists; pure task schedule is architecture-independent",
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "router and top-16",
        (
            "wave 0 only",
            "14 experts per wave64 lane",
            "64-bit ballot and lane-prefix counts",
            "serial top-16 renormalization order",
        ),
        "wave32 ownership and separate device backend implemented",
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "route union",
        (
            "two experts per thread assumes 512-thread CTA",
            "wave64 block exclusive scan",
            "permlane16/32 scan exchange through shared device helpers",
            "WAVES-word scan LDS",
            "ascending expert slot order",
        ),
        "wave32 contiguous-chunk scan contract/backend implemented",
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "latent MXFP4 quantization",
        (
            "one owner per token/32-element block",
            "16 packed BF16 words to four E2M1 dwords and one E8M0 code",
            "THREADS-strided block ownership",
            "native scaled FP4 conversion",
        ),
        "layout/ownership and portable AITER E2M1 backend implemented",
    ),
    K2Dependency(
        "kernels/layer.py and kernels/stages/moe.py",
        "gate/up task placement",
        (
            "96 early GEMV tasks",
            "reversed persistent UG task ownership",
            "UG expert compute uses eight-wave scaled MFMA",
        ),
        "pre-compute schedule proven; scaled-WMMA boundary remains refused",
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "expert gate/up and down",
        (
            "mfma_scaled FP4 gate/up and down",
            "wave64 accumulator v4 layout",
            "lane%16 and lane//16 matrix ownership",
            "WAVES expert batching",
            "gfx950 shuffle_scale weight-scale layout",
        ),
        (
            "gfx1250 v16/v8/v16 WMMA backend and architecture-keyed "
            "runtime scale materialization implemented"
        ),
    ),
    K2Dependency(
        "kernels/stages/moe.py",
        "post-expert tail stages",
        (
            "wave64 lnorm/reduce ownership",
            "shared-down and up GEMV task placement",
        ),
        (
            "gfx1250 local shared-down/latent-reduce/lnorm/up/final-reduce "
            "backend implemented; XGMI staging readiness remains refused"
        ),
    ),
)


@dataclass(frozen=True)
class K2DeviceEvidence:
    primitive: str
    source: str
    lines: str
    contract: str


K2_GFX1250_DEVICE_EVIDENCE = (
    K2DeviceEvidence(
        "wave32 ballot",
        "aiter/ops/flydsl/kernels/mega_moe_gfx1250/dispatch_tdm.py",
        "397-404, 780",
        "gfx1250 route predicates use rocdl ballot with an i32 result",
    ),
    K2DeviceEvidence(
        "wave32 shuffle_xor",
        "aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py",
        "1114-1123, 1146",
        "wave32 FP4 group reductions and packing use shuffle_xor",
    ),
    K2DeviceEvidence(
        "portable E2M1 quantization",
        "aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py",
        "1125-1147",
        "gfx1250 emits E8M0 and RNE E2M1 nibbles with quant_utils",
    ),
    K2DeviceEvidence(
        "BF16 WMMA early GEMV",
        "aiter/ops/flydsl/kernels/pa_decode/op_gemm.py",
        "Gfx1250Gemm.mma",
        "v16 BF16 operands and v8 FP32 accumulator",
    ),
)


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


@dataclass(frozen=True)
class EarlyTask:
    kind: str
    group: int
    part: int


@dataclass(frozen=True)
class K2TaskSchedule:
    blocks: int
    fused: bool

    def __post_init__(self) -> None:
        if self.blocks <= 0:
            raise ValueError("K2 schedule needs a positive block count")

    @property
    def early_tasks(self) -> tuple[EarlyTask, ...]:
        return (
            tuple(
                EarlyTask("router", group, part)
                for group in range(ROUTER_GROUPS)
                for part in range(ROUTER_PARTS)
            )
            + tuple(
                EarlyTask("latent", group, part)
                for group in range(LATENT_GROUPS)
                for part in range(LATENT_PARTS)
            )
            + tuple(EarlyTask("gate_up", group, 0) for group in range(GATE_UP_TASKS))
        )

    def early_owner(self, task: EarlyTask) -> int:
        if task.kind == "router":
            ordinal = task.group * ROUTER_PARTS + task.part
            return 160 + ordinal if self.fused else ordinal
        if task.kind == "latent":
            ordinal = task.group * LATENT_PARTS + task.part
            return (
                (160 + ROUTER_GROUPS * ROUTER_PARTS + ordinal)
                if self.fused
                else (ROUTER_GROUPS * ROUTER_PARTS + ordinal)
            )
        if task.kind == "gate_up":
            return (
                56 + task.group
                if self.fused
                else (
                    ROUTER_GROUPS * ROUTER_PARTS
                    + LATENT_GROUPS * LATENT_PARTS
                    + task.group
                )
            )
        raise ValueError(f"unknown early task kind {task.kind!r}")

    def validate_early_placement(self) -> None:
        owners = [self.early_owner(task) for task in self.early_tasks]
        if any(owner >= self.blocks for owner in owners):
            raise ValueError(
                f"{self.blocks} blocks do not cover the K2 early task placement"
            )
        keyed = [(task.kind, task.group, task.part) for task in self.early_tasks]
        if len(keyed) != len(set(keyed)):
            raise ValueError("duplicate K2 early task")

    def route_owner(self, token: int, tokens: int) -> int:
        if not 0 <= token < tokens <= GFX1250_MAX_TOKENS:
            raise ValueError("invalid K2 route token")
        owner = token if self.fused else len(self.early_tasks) + token
        if owner >= self.blocks:
            raise ValueError(f"{self.blocks} blocks do not cover route token {token}")
        return owner

    def ug_tasks_for_block(self, total: int, block: int) -> tuple[int, ...]:
        """Match the production reverse-placement persistent schedule."""

        if total < 0 or not 0 <= block < self.blocks:
            raise ValueError("invalid UG task ownership query")
        first = self.blocks - 1 - block
        return tuple(range(first, total, self.blocks))


ROUTER_CONTRACTS = {
    GFX950.gfx: RouterContract(GFX950),
    GFX1250.gfx: RouterContract(GFX1250),
}


def router_contract_for(architecture: Architecture) -> RouterContract:
    try:
        return ROUTER_CONTRACTS[architecture.gfx]
    except KeyError as error:
        raise ValueError(f"no K2 router contract for {architecture.gfx}") from error


def front_contracts_for(
    architecture: Architecture, *, threads: int, tokens: int
) -> tuple[RouterContract, RouteUnionContract, LatentQuantContract]:
    """Build the complete CPU-only front-half contract for one CTA topology."""

    return (
        router_contract_for(architecture),
        RouteUnionContract(architecture, threads, tokens),
        LatentQuantContract(architecture, threads, tokens),
    )


def front_backend_for(architecture: Architecture, *, threads: int, tokens: int):
    """Select only the separately implemented gfx1250 device backend."""

    if architecture is not GFX1250:
        raise ValueError(
            "the production gfx950 path remains the explicit moe.py module"
        )
    from atom.models.kimi_k3.mono.kernels.stages.moe_gfx1250 import (
        StageBackend,
    )

    return StageBackend(
        *front_contracts_for(architecture, threads=threads, tokens=tokens)
    )
