# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Architecture-separated FP4 expert matrix and packing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from atom.models.kimi_k3.mono.kernels.token_tiling import Gfx1250TokenTiling
from atom.mono.plan.architecture import GFX950, GFX1250, Architecture

FP4_PER_DWORD = 8
K_STEP = 128
MX_GROUP = 32
SCALES_PER_STEP = K_STEP // MX_GROUP
TOKENS = 16
TOPK = 16
ROUTED_INTERMEDIATE = 384
LATENT = 3584
UG_GROUPS = ROUTED_INTERMEDIATE // 32
DOWN_GROUPS_GFX1250 = LATENT // 32


class ExpertScaleLayout(str, Enum):
    GFX950_SHUFFLE = "gfx950-shuffle-scale"
    GFX1250_N32K4 = "gfx1250-n32k4"


@dataclass(frozen=True)
class ExpertScaleSpec:
    """One rank-local expert scale plane's logical and TP identity."""

    name: str
    experts: int
    rows_per_expert: int
    k: int
    tp_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("expert scale plane needs a name")
        if self.experts <= 0 or self.rows_per_expert <= 0:
            raise ValueError("expert scale dimensions must be positive")
        if self.rows_per_expert % 32 or self.k % 128:
            raise ValueError("n32k4 needs N%32==0 and K%128==0")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("expert scale TP rank is outside its TP group")

    @property
    def rows(self) -> int:
        return self.experts * self.rows_per_expert

    @property
    def groups(self) -> int:
        return self.k // MX_GROUP

    @property
    def packing(self) -> ExpertWeightPacking:
        return ExpertWeightPacking(self.rows, self.k)

    def storage_shape(self, layout: ExpertScaleLayout) -> tuple[int, ...]:
        if layout is ExpertScaleLayout.GFX950_SHUFFLE:
            padded_rows = (self.rows + 255) // 256 * 256
            padded_groups = (self.groups + 7) // 8 * 8
            return padded_rows, padded_groups
        if layout is ExpertScaleLayout.GFX1250_N32K4:
            return self.experts, self.rows_per_expert // 32, self.groups * 32
        raise ValueError(f"unsupported expert scale layout {layout!r}")


@dataclass(frozen=True)
class ExpertDeviceEvidence:
    primitive: str
    source: str
    lines: str
    contract: str


GFX1250_EXPERT_EVIDENCE = (
    ExpertDeviceEvidence(
        "scaled FP4 WMMA",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py",
        "238-243, 1173-1187, 1199-1221",
        (
            "wmma_scale_f32_32x16x128_f4 takes v16-i32 weight, v8-i32 "
            "activation, v16-f32 accumulator, two packed E8M0 scale words, "
            "scaleAType=0, and scaleBType selecting the M16 half"
        ),
    ),
    ExpertDeviceEvidence(
        "operand lane loads",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py",
        "1040-1062, 1095-1129",
        (
            "lane%16 owns an M row, lane//16 owns alternating K32 groups, "
            "and two K64 halves concatenate into the WMMA fragments"
        ),
    ),
    ExpertDeviceEvidence(
        "accumulator output mapping",
        "aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py",
        "1763-1773, 2030-2089, 2120-2168",
        (
            "v16 accumulator splits into two v8 N16 fragments; lane%16 is "
            "the output row and lane//16 selects eight columns in each half"
        ),
    ),
    ExpertDeviceEvidence(
        "weight payload shuffle",
        "aiter/ops/shuffle.py",
        "10-25, 39-51, 124-147, 192-208",
        (
            "non-interleaved gfx1250 and gfx950 payloads share the 16-row by "
            "32-packed-byte shuffle_weight layout"
        ),
    ),
    ExpertDeviceEvidence(
        "weight n32k4 scales",
        "aiter/ops/shuffle.py",
        "243-294, 423-442",
        (
            "gfx1250 MoE packs four consecutive K32 E8M0 bytes per output "
            "row in n32k4 order; gfx950 uses a distinct shuffle_scale layout"
        ),
    ),
)


@dataclass(frozen=True)
class ExpertMatrixContract:
    architecture: Architecture
    wave_size: int
    output_rows: int
    weight_dwords: int
    activation_dwords: int
    accumulator_elements: int
    weight_scale_bytes: int
    activation_scale_bytes: int
    instruction: str

    def __post_init__(self) -> None:
        if self.wave_size != self.architecture.wave_size:
            raise ValueError("expert wave size disagrees with architecture")
        outputs = self.wave_size * self.accumulator_elements
        if outputs != TOKENS * self.output_rows:
            raise ValueError(
                f"accumulator covers {outputs}, expected "
                f"{TOKENS}x{self.output_rows}"
            )


GFX950_EXPERT = ExpertMatrixContract(
    GFX950,
    wave_size=64,
    output_rows=16,
    weight_dwords=4,
    activation_dwords=4,
    accumulator_elements=4,
    weight_scale_bytes=1,
    activation_scale_bytes=1,
    instruction="mfma_scale_f32_16x16x128_f8f6f4",
)

GFX1250_EXPERT = ExpertMatrixContract(
    GFX1250,
    wave_size=32,
    output_rows=32,
    weight_dwords=16,
    activation_dwords=8,
    accumulator_elements=16,
    weight_scale_bytes=4,
    activation_scale_bytes=4,
    instruction="wmma_scale_f32_32x16x128_f4",
)


def expert_contract_for(
    architecture: Architecture,
) -> ExpertMatrixContract:
    if architecture is GFX950:
        return GFX950_EXPERT
    if architecture is GFX1250:
        return GFX1250_EXPERT
    raise ValueError(f"no expert matrix contract for {architecture.gfx}")


@dataclass(frozen=True)
class Gfx1250FragmentLayout:
    """Coordinates of one N32xM16xK128 scaled-WMMA operation."""

    def _lane(self, lane: int) -> tuple[int, int]:
        if not 0 <= lane < 32:
            raise ValueError(f"lane {lane} outside wave32")
        return lane % 16, lane // 16

    def weight_coordinate(self, lane: int, dword: int, nibble: int) -> tuple[int, int]:
        """Return ``(output_row, k)`` for one v16-i32 weight nibble."""

        lane16, kgroup_parity = self._lane(lane)
        if not 0 <= dword < 16 or not 0 <= nibble < FP4_PER_DWORD:
            raise ValueError("weight fragment coordinate outside v16 FP4")
        output_half = dword // 8
        within_half = dword % 8
        k64_half = within_half // 4
        word = within_half % 4
        group = 2 * k64_half + kgroup_parity
        return (
            output_half * 16 + lane16,
            group * MX_GROUP + word * FP4_PER_DWORD + nibble,
        )

    def activation_coordinate(
        self, lane: int, dword: int, nibble: int
    ) -> tuple[int, int]:
        """Return ``(token_row, k)`` for one v8-i32 activation nibble."""

        lane16, kgroup_parity = self._lane(lane)
        if not 0 <= dword < 8 or not 0 <= nibble < FP4_PER_DWORD:
            raise ValueError("activation fragment coordinate outside v8 FP4")
        k64_half = dword // 4
        word = dword % 4
        group = 2 * k64_half + kgroup_parity
        return (
            lane16,
            group * MX_GROUP + word * FP4_PER_DWORD + nibble,
        )

    def accumulator_coordinate(self, lane: int, element: int) -> tuple[int, int]:
        """Return ``(token_row, output_column)`` for one v16-f32 value."""

        lane16, column_half = self._lane(lane)
        if not 0 <= element < 16:
            raise ValueError("accumulator element outside v16")
        output_half = element // 8
        return lane16, output_half * 16 + column_half * 8 + element % 8

    def weight_scale_coordinates(
        self, lane: int, step: int
    ) -> tuple[tuple[int, int], ...]:
        if step < 0:
            raise ValueError("negative K128 step")
        self._lane(lane)
        return tuple(
            (lane, step * SCALES_PER_STEP + byte) for byte in range(SCALES_PER_STEP)
        )

    def activation_scale_coordinates(
        self, lane: int, step: int
    ) -> tuple[tuple[int, int], ...]:
        lane16, _ = self._lane(lane)
        if step < 0:
            raise ValueError("negative K128 step")
        return tuple(
            (lane16, step * SCALES_PER_STEP + byte) for byte in range(SCALES_PER_STEP)
        )

    def output_columns_for_lane(self, lane: int) -> tuple[int, ...]:
        return tuple(
            self.accumulator_coordinate(lane, element)[1] for element in range(16)
        )


GFX1250_FRAGMENT = Gfx1250FragmentLayout()


@dataclass(frozen=True)
class ExpertWeightPacking:
    """Index maps between raw FP4/E8M0 planes and architecture layouts."""

    rows: int
    k: int

    def __post_init__(self) -> None:
        if self.rows % 32 or self.k % K_STEP:
            raise ValueError("expert weights need N%32==0 and K%128==0")

    @property
    def packed_columns(self) -> int:
        return self.k // 2

    @property
    def scale_groups(self) -> int:
        return self.k // MX_GROUP

    @property
    def gfx950_scale_bytes(self) -> int:
        padded_rows = ((self.rows + 255) // 256) * 256
        padded_groups = ((self.scale_groups + 7) // 8) * 8
        return padded_rows * padded_groups

    @property
    def gfx1250_scale_bytes(self) -> int:
        return self.rows * self.scale_groups

    def payload_index(self, row: int, k: int) -> tuple[int, int]:
        """Shared shuffle_weight byte index and nibble for ``(row, k)``."""

        if not 0 <= row < self.rows or not 0 <= k < self.k:
            raise ValueError("weight payload coordinate outside plane")
        byte_column, nibble = divmod(k, 2)
        k_tiles = self.packed_columns // 32
        index = (
            (((row // 16) * k_tiles + byte_column // 32) * 2 + (byte_column % 32) // 16)
            * 16
            + row % 16
        ) * 16 + byte_column % 16
        return index, nibble

    def payload_coordinate(self, index: int, nibble: int) -> tuple[int, int]:
        if not 0 <= index < self.rows * self.packed_columns or not 0 <= nibble < 2:
            raise ValueError("weight payload index outside plane")
        k_tiles = self.packed_columns // 32
        byte = index % 16
        value = index // 16
        row16 = value % 16
        value //= 16
        inner = value % 2
        tile = value // 2
        k_tile = tile % k_tiles
        row_tile = tile // k_tiles
        row = row_tile * 16 + row16
        byte_column = k_tile * 32 + inner * 16 + byte
        return row, 2 * byte_column + nibble

    def gfx950_scale_index(self, row: int, group: int) -> int:
        """Existing mono ``shuffle_scale`` byte index."""

        if not 0 <= row < self.rows or not 0 <= group < self.scale_groups:
            raise ValueError("weight scale coordinate outside plane")
        padded_groups = ((self.scale_groups + 7) // 8) * 8
        row32, row_half, row16 = row // 32, (row // 16) % 2, row % 16
        group8 = group // 8
        group_half = (group // 4) % 2
        group4 = group % 4
        return (
            (((row32 * (padded_groups // 8) + group8) * 4 + group4) * 16 + row16) * 2
            + group_half
        ) * 2 + row_half

    def gfx1250_scale_index(self, row: int, group: int) -> int:
        """gfx1250 ``shuffle_scale_n32k4`` byte index."""

        if not 0 <= row < self.rows or not 0 <= group < self.scale_groups:
            raise ValueError("weight scale coordinate outside plane")
        return (
            (row // 32 * (self.scale_groups // 4) + group // 4) * 32 + row % 32
        ) * 4 + group % 4

    def gfx1250_scale_coordinate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.rows * self.scale_groups:
            raise ValueError("gfx1250 scale index outside plane")
        byte = index % 4
        value = index // 4
        row32 = value % 32
        tile = value // 32
        group4 = tile % (self.scale_groups // 4)
        row_tile = tile // (self.scale_groups // 4)
        return row_tile * 32 + row32, group4 * 4 + byte

    def scale_repack_map(self) -> tuple[tuple[int, int], ...]:
        """``(gfx950 source, gfx1250 destination)`` for every real scale."""

        return tuple(
            (
                self.gfx950_scale_index(row, group),
                self.gfx1250_scale_index(row, group),
            )
            for row in range(self.rows)
            for group in range(self.scale_groups)
        )

    def repack_scales(self, gfx950: bytes) -> bytes:
        """Materialize n32k4 bytes from the existing gfx950 shuffled plane."""

        if len(gfx950) != self.gfx950_scale_bytes:
            raise ValueError(
                f"gfx950 scale plane has {len(gfx950)} bytes, expected "
                f"{self.gfx950_scale_bytes}"
            )
        result = bytearray(self.gfx1250_scale_bytes)
        for source, destination in self.scale_repack_map():
            result[destination] = gfx950[source]
        return bytes(result)


@dataclass(frozen=True)
class ExpertTaskContract:
    waves: int

    def __post_init__(self) -> None:
        if self.waves <= 0:
            raise ValueError("expert task contract needs waves")

    def ug_k_steps(self, wave: int) -> tuple[int, ...]:
        if not 0 <= wave < self.waves:
            raise ValueError("wave outside expert CTA")
        if self.waves != 8:
            raise ValueError("order-preserving gfx1250 gate/up requires eight waves")
        if wave % 2:
            return ()
        half = wave // 4
        first = half * (LATENT // K_STEP // 2)
        return tuple(range(first, first + LATENT // K_STEP // 2))

    @staticmethod
    def ug_role(wave: int) -> str | None:
        if not 0 <= wave < 8:
            raise ValueError("wave outside eight-wave expert CTA")
        if wave % 2:
            return None
        return "gate" if wave % 4 == 0 else "up"

    @staticmethod
    def ug_reduction_waves(role: str) -> tuple[int, int]:
        if role == "gate":
            return 0, 4
        if role == "up":
            return 2, 6
        raise ValueError(f"unknown expert role {role!r}")

    @property
    def down_tasks(self) -> int:
        return DOWN_GROUPS_GFX1250

    def down_slots(self, union_size: int, wave: int) -> tuple[int, ...]:
        if not 0 <= union_size <= TOKENS * TOPK:
            raise ValueError("invalid expert union size")
        if not 0 <= wave < self.waves:
            raise ValueError("wave outside expert CTA")
        return tuple(range(wave, union_size, self.waves))

    @property
    def ug_accumulator_elements(self) -> int:
        if self.waves != 8:
            raise ValueError("order-preserving gfx1250 gate/up requires eight waves")
        return 4 * 32 * GFX1250_EXPERT.accumulator_elements

    @property
    def down_contribution_elements(self) -> int:
        return TOKENS * TOPK * GFX1250_EXPERT.output_rows


def expert_backend_for(architecture: Architecture, *, waves: int, tokens: int = TOKENS):
    if architecture is not GFX1250:
        raise ValueError("the production gfx950 expert path remains stages/moe.py")
    from atom.models.kimi_k3.mono.kernels.stages.moe_expert_gfx1250 import (
        ExpertBackend,
    )

    return ExpertBackend(
        ExpertTaskContract(waves),
        Gfx1250TokenTiling(tokens),
    )
