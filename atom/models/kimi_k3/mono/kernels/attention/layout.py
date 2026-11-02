# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only matrix and schedule contracts for Kimi K1 attention.

The o-projection and KDA in-projection use the evidenced row-major GEMV
fragments from ``stages.gemv_layout``.  KDA's f_b x f_a gate uses the same
per-wave 16x16 result ABI, but owns a distinct set of 16 output rows per wave,
so its ownership is described separately here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX950_GEMV,
    GFX1250_GEMV,
    K_PER_CHUNK,
    GemvLayout,
    layout_for,
    stage_backend_for,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX950, GFX1250, Architecture

ROWS = 16
HIDDEN = 7168
GATE_K = 128
GATE_ROW_GROUPS = 128 // ROWS
LDS_PAD = 8
STATE_ROWS = 16
STATE_COLUMNS = 128
RECURRENCE_LANES = 32
RECURRENCE_COLUMNS_PER_LANE = STATE_COLUMNS // RECURRENCE_LANES
SEAM_COLUMNS = 128
SEAM_COLUMN_GROUPS = SEAM_COLUMNS // 4
SEAM_SLICES = 56
SEAM_SLICE_GROUPS = 8
SEAM_SLICES_PER_GROUP = SEAM_SLICES // SEAM_SLICE_GROUPS


@dataclass(frozen=True)
class KdaMatrixContract:
    """Architecture-explicit KDA projection/gate ownership."""

    architecture: Architecture
    waves: int
    gemv: GemvLayout

    def __post_init__(self) -> None:
        if self.gemv.architecture is not self.architecture:
            raise ValueError("KDA GEMV layout does not match its architecture")
        if self.waves <= 0:
            raise ValueError(f"waves must be positive, got {self.waves}")

    @property
    def group_columns(self) -> int:
        """Columns consumed by one chunk per wave."""

        return self.waves * K_PER_CHUNK

    @property
    def projection_groups(self) -> int:
        if HIDDEN % self.group_columns:
            raise ValueError(
                f"KDA hidden {HIDDEN} is not divisible by {self.waves} waves x "
                f"{K_PER_CHUNK} columns"
            )
        return HIDDEN // self.group_columns

    def projection_chunk_ids(self, wave: int) -> tuple[int, ...]:
        return self.gemv.chunk_ids(HIDDEN, wave, self.waves)

    def slab_plan(
        self, rows: int, *, lds_capacity_bytes: int, fixed_lds_bytes: int
    ) -> tuple[tuple[int, int], ...]:
        """Balanced ``(first group, groups)`` slabs matching KDA's loop order."""

        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        if not 0 <= fixed_lds_bytes < lds_capacity_bytes:
            raise ValueError("fixed LDS must be smaller than the capacity")
        groups = self.projection_groups
        for slab_count in range(1, groups + 1):
            widest = -(-groups // slab_count)
            needed = (
                rows * (widest * self.group_columns + LDS_PAD) * 2 + fixed_lds_bytes
            )
            if needed <= lds_capacity_bytes:
                result = []
                first = 0
                for slab in range(slab_count):
                    take = -(-(groups - first) // (slab_count - slab))
                    result.append((first, take))
                    first += take
                return tuple(result)
        one_group = rows * (self.group_columns + LDS_PAD) * 2 + fixed_lds_bytes
        raise ValueError(
            f"{rows} rows: one KDA wave group needs {one_group} LDS bytes, "
            f"capacity is {lds_capacity_bytes}"
        )

    def gate_output_coordinates(
        self, wave: int, lane: int
    ) -> tuple[tuple[int, int], ...]:
        """``(head row, token)`` outputs owned by one gate accumulator."""

        if not 0 <= wave < self.waves:
            raise ValueError(f"wave {wave} outside [0, {self.waves})")
        local = self.gemv.accumulator_coordinates(lane)
        return tuple((wave * ROWS + row, token) for row, token in local)

    def gate_row_groups(self, wave: int) -> tuple[int, ...]:
        """16-row f_b groups assigned to a wave by round robin."""

        if not 0 <= wave < self.waves:
            raise ValueError(f"wave {wave} outside [0, {self.waves})")
        return tuple(range(wave, GATE_ROW_GROUPS, self.waves))

    @property
    def recurrence_subgroups_per_wave(self) -> int:
        """Independent 32-lane state-row reductions resident in one wave."""

        if self.architecture.wave_size % RECURRENCE_LANES:
            raise ValueError(
                f"{self.architecture.gfx}: wave size is not divisible by "
                f"{RECURRENCE_LANES} recurrence lanes"
            )
        return self.architecture.wave_size // RECURRENCE_LANES

    def recurrence_rows(self, wave: int, subgroup: int) -> tuple[int, ...]:
        """State rows serially owned by one wave/subgroup."""

        if not 0 <= wave < self.waves:
            raise ValueError(f"wave {wave} outside [0, {self.waves})")
        subgroups = self.recurrence_subgroups_per_wave
        if not 0 <= subgroup < subgroups:
            raise ValueError(f"subgroup {subgroup} outside [0, {subgroups})")
        first = wave * subgroups + subgroup
        stride = self.waves * subgroups
        return tuple(range(first, STATE_ROWS, stride))

    def recurrence_coordinates(
        self, wave: int, lane: int
    ) -> tuple[tuple[int, int], ...]:
        """``(state row, K column)`` scalar groups owned by one lane."""

        if not 0 <= lane < self.architecture.wave_size:
            raise ValueError(f"lane {lane} outside [0, {self.architecture.wave_size})")
        subgroup, lane32 = divmod(lane, RECURRENCE_LANES)
        return tuple(
            (row, lane32 * RECURRENCE_COLUMNS_PER_LANE + element)
            for row in self.recurrence_rows(wave, subgroup)
            for element in range(RECURRENCE_COLUMNS_PER_LANE)
        )

    @property
    def recurrence_reduction_offsets(self) -> tuple[int, ...]:
        """The per-state-row reduction order, unchanged on wave32."""

        return (16, 8, 4, 2, 1)

    def qk_columns(self, lane: int) -> tuple[int, ...]:
        """Q/K row columns loaded by one architecture-specific lane."""

        if not 0 <= lane < self.architecture.wave_size:
            raise ValueError(f"lane {lane} outside [0, {self.architecture.wave_size})")
        if self.architecture is GFX950:
            return lane, lane + 64
        if self.architecture is GFX1250:
            return lane, lane + 32, lane + 64, lane + 96
        raise ValueError(f"no Q/K layout for {self.architecture.gfx}")

    @property
    def qk_reduction_offsets(self) -> tuple[int, ...]:
        """XOR steps after each lane's initial partial is formed."""

        if self.architecture is GFX950:
            return (32, 16, 8, 4, 2, 1)
        if self.architecture is GFX1250:
            # The local merge of (lane, lane+64) with (lane+32, lane+96)
            # exactly replaces gfx950's first offset-32 exchange.
            return (16, 8, 4, 2, 1)
        raise ValueError(f"no Q/K reduction for {self.architecture.gfx}")

    def qk_reduction_tree(self, lane: int = 0):
        """Symbolic add tree proving wave32 preserves wave64 norm order."""

        columns = {
            current_lane: (
                ("add", current_lane, current_lane + 64)
                if self.architecture is GFX950
                else (
                    "add",
                    ("add", current_lane, current_lane + 64),
                    ("add", current_lane + 32, current_lane + 96),
                )
            )
            for current_lane in range(self.architecture.wave_size)
        }
        for offset in self.qk_reduction_offsets:
            columns = {
                current_lane: (
                    "add",
                    columns[current_lane],
                    columns[current_lane ^ offset],
                )
                for current_lane in range(self.architecture.wave_size)
            }
        return columns[lane]

    @property
    def accumulator_lds_elements(self) -> int:
        """f32 elements for one partial 16x16 tile per wave."""

        return (
            self.waves
            * self.architecture.wave_size
            * self.gemv.accumulator_elements_per_lane
        )

    def _check_seam(self, tokens: int, sources: int) -> None:
        maximum = GFX1250_MAX_TOKENS if self.architecture is GFX1250 else ROWS
        if not 1 <= tokens <= maximum:
            raise ValueError(
                f"{self.architecture.gfx} attention tokens must be in "
                f"[1, {maximum}], got {tokens}"
            )
        if not 1 <= sources <= 32:
            raise ValueError(
                f"wave32-compatible attention sources must be in [1, 32], "
                f"got {sources}"
            )

    def slice_work(
        self, tokens: int, sources: int, thread: int
    ) -> tuple[tuple[int, int, int], ...]:
        """``(token, source, four-column group)`` work owned by one thread."""

        self._check_seam(tokens, sources)
        threads = self.waves * self.architecture.wave_size
        if not 0 <= thread < threads:
            raise ValueError(f"thread {thread} outside [0, {threads})")
        work = tokens * sources * SEAM_COLUMN_GROUPS
        return tuple(
            (
                item // (sources * SEAM_COLUMN_GROUPS),
                (item // SEAM_COLUMN_GROUPS) % sources,
                item % SEAM_COLUMN_GROUPS,
            )
            for item in range(thread, work, threads)
        )

    def mix_tokens(self, tokens: int, wave: int, subgroup: int = 0) -> tuple[int, ...]:
        """Token rows serially owned by one 32-lane mix subgroup."""

        self._check_seam(tokens, 1)
        if not 0 <= wave < self.waves:
            raise ValueError(f"wave {wave} outside [0, {self.waves})")
        subgroups = self.recurrence_subgroups_per_wave
        if not 0 <= subgroup < subgroups:
            raise ValueError(f"subgroup {subgroup} outside [0, {subgroups})")
        first = wave * subgroups + subgroup
        stride = self.waves * subgroups
        return tuple(range(first, tokens, stride))

    def gate_source(self, lane: int, sources: int) -> int | None:
        """Softmax source owned by a lane, or None for an identity lane."""

        self._check_seam(1, sources)
        if not 0 <= lane < self.architecture.wave_size:
            raise ValueError(f"lane {lane} outside [0, {self.architecture.wave_size})")
        return lane if lane < sources else None

    @staticmethod
    def mix_indices(
        token: int, source: int, sources: int, column_group: int
    ) -> tuple[int, ...]:
        """Four ``vals[token, source, column]`` coordinates mixed by a lane."""

        if not 0 <= column_group < SEAM_COLUMN_GROUPS:
            raise ValueError(f"column group {column_group} outside [0, 32)")
        if not 0 <= source < sources:
            raise ValueError(f"source {source} outside [0, {sources})")
        return tuple(
            ((token * sources + source) * SEAM_COLUMNS + column_group * 4 + element)
            for element in range(4)
        )

    @staticmethod
    def _tree(leaves, offsets):
        values = dict(enumerate(leaves))
        for offset in offsets:
            values = {
                lane: ("op", values[lane], values[lane ^ offset]) for lane in values
            }
        return values[0]

    @staticmethod
    def _without_identities(tree, identity):
        if not isinstance(tree, tuple):
            return tree
        _, left, right = tree
        left = KdaMatrixContract._without_identities(left, identity)
        right = KdaMatrixContract._without_identities(right, identity)
        if left == identity:
            return right
        if right == identity:
            return left
        return "op", left, right

    def softmax_reduction_tree(self, sources: int, identity):
        """Max/sum tree after removing inactive-lane identity operations.

        This equivalence is for the kernel's finite live logits and nonnegative
        exponentials: gfx950's inactive upper half contributes its max sentinel
        or +0, so its offset-32 operation is an exact identity.
        """

        self._check_seam(1, sources)
        leaves = [
            lane if lane < sources else identity
            for lane in range(self.architecture.wave_size)
        ]
        tree = self._tree(leaves, self.architecture.reduction_offsets)
        return self._without_identities(tree, identity)

    def seam_lds_elements(self, tokens: int, sources: int) -> tuple[int, int]:
        """Tile-local ``(red f32, vals f32)`` elements required by gate/mix."""

        self._check_seam(tokens, sources)
        lds_tokens = (
            Gfx1250TokenTiling(tokens).lds_rows
            if self.architecture is GFX1250
            else tokens
        )
        red = max(
            SEAM_SLICE_GROUPS * sources * 2,
            lds_tokens * (sources + 1),
            128 + SEAM_SLICE_GROUPS * lds_tokens,
        )
        vals = lds_tokens * sources * SEAM_COLUMNS
        return red, vals

    def token_tiles(self, tokens: int):
        """M16 schedule required around gfx1250 seam device callables."""

        if self.architecture is not GFX1250:
            raise ValueError("attention token tiling is gfx1250-only")
        self._check_seam(tokens, 1)
        return Gfx1250TokenTiling(tokens).tiles


def kda_contract_for(architecture: Architecture, *, waves: int) -> KdaMatrixContract:
    return KdaMatrixContract(architecture, waves, layout_for(architecture))


def back_backend_for(architecture: Architecture, *, waves: int):
    """Select the o-projection GEMV backend explicitly."""

    return stage_backend_for(architecture, waves=waves)


_KDA_MODULES = {
    GFX950.gfx: "atom.models.kimi_k3.mono.kernels.attention.kda",
    GFX1250.gfx: "atom.models.kimi_k3.mono.kernels.attention.kda_gfx1250",
}


def kda_backend_for(architecture: Architecture, *, waves: int):
    """Select KDA's projection/gate backend.

    Both currently require eight waves because one wave owns each of the eight
    16-row f_b groups.  This is an explicit candidate constraint, not a claimed
    gfx1250 hardware or deployment constant.
    """

    contract = kda_contract_for(architecture, waves=waves)
    if waves != GATE_ROW_GROUPS:
        raise ValueError(
            f"KDA gate backend requires {GATE_ROW_GROUPS} explicit waves, got {waves}"
        )
    module = import_module(_KDA_MODULES[architecture.gfx])
    if architecture is GFX950:
        return module
    return module.StageBackend(contract)


assert GFX950_GEMV.architecture is GFX950
assert GFX1250_GEMV.architecture is GFX1250
