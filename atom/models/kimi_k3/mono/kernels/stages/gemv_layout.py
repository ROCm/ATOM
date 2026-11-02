# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-Python contracts for Kimi mono's architecture-specific BF16 GEMV.

The device implementations are deliberately separate modules.  This file
describes their distributed fragments and K-chunk ownership without importing
FlyDSL, so every coordinate can be exhaustively checked on a CPU-only host.

The gfx1250 mapping is taken from AITER's device code, not inferred from vector
sizes:

* ``pa_decode/op_lds.py:Gfx1250Memory.operand`` loads both row-major operands
  from ``lane16 * stride + half * 16 + j * 32`` and
  ``pipeline.py`` passes those values as both ``a`` and ``b`` to
  ``Gfx1250Gemm.mma``.
* ``gemm_a16w16_kernel_gfx1250.py`` constructs row-major ``A_SIDE`` and
  ``B_SIDE`` through the same ``_lane_bases`` and ``_frag`` functions:
  ``lane16`` selects the row, ``lane_kgrp * 8`` selects the first K pack, and
  the two fragment halves are 16 K elements apart.
* ``fmha_fwd_prefill_a16w16_m32x8.py`` documents K-as-A and Q-as-B as matching
  v16 fragments and the v8 result mapping used below.  Its V-as-A fragment is
  transpose-loaded and intentionally outside this row-major GEMV contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from atom.models.kimi_k3.mono.kernels.token_tiling import (
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX950, GFX1250, Architecture
from atom.mono.plan.execution import WAVES as GFX950_WAVES

ROWS = 16
K_PER_MMA = 32
K_PER_CHUNK = 128
PACK_ELEMENTS = 8
LDS_PAD = 8


@dataclass(frozen=True)
class GemvLayout:
    """One architecture's wave-distributed 16-row GEMV fragments."""

    architecture: Architecture
    operand_packs_per_lane: int
    operand_pack_stride: int

    def __post_init__(self) -> None:
        wave = self.architecture.wave_size
        operand_elements = self.operand_elements_per_lane
        if wave * operand_elements != ROWS * K_PER_MMA:
            raise ValueError(
                f"{self.architecture.gfx}: {wave} lanes x {operand_elements} "
                f"operand elements != {ROWS}x{K_PER_MMA}"
            )
        accumulator_elements = (
            wave * self.architecture.bf16_mma.accumulator_elements_per_lane
        )
        if accumulator_elements != ROWS * ROWS:
            raise ValueError(
                f"{self.architecture.gfx}: distributed accumulator does not "
                f"cover {ROWS}x{ROWS}"
            )
        if K_PER_CHUNK % K_PER_MMA:
            raise ValueError("K chunk must contain whole matrix operations")

    @property
    def operand_elements_per_lane(self) -> int:
        return self.operand_packs_per_lane * PACK_ELEMENTS

    @property
    def accumulator_elements_per_lane(self) -> int:
        return self.architecture.bf16_mma.accumulator_elements_per_lane

    @property
    def mma_steps_per_chunk(self) -> int:
        return K_PER_CHUNK // K_PER_MMA

    def _lane(self, lane: int) -> tuple[int, int]:
        if not 0 <= lane < self.architecture.wave_size:
            raise ValueError(
                f"{self.architecture.gfx}: lane {lane} outside "
                f"[0, {self.architecture.wave_size})"
            )
        return lane % ROWS, lane // ROWS

    def _row_major_operand_coordinates(
        self, lane: int, mma_step: int
    ) -> tuple[tuple[int, int], ...]:
        if not 0 <= mma_step < self.mma_steps_per_chunk:
            raise ValueError(f"matrix step {mma_step} outside one K128 chunk")
        row, lane_group = self._lane(lane)
        base = mma_step * K_PER_MMA + lane_group * PACK_ELEMENTS
        return tuple(
            (row, base + pack * self.operand_pack_stride + element)
            for pack in range(self.operand_packs_per_lane)
            for element in range(PACK_ELEMENTS)
        )

    def a_operand_coordinates(
        self, lane: int, mma_step: int
    ) -> tuple[tuple[int, int], ...]:
        """Row-major WMMA A ``(row, k)`` elements held by ``lane``."""

        return self._row_major_operand_coordinates(lane, mma_step)

    def b_operand_coordinates(
        self, lane: int, mma_step: int
    ) -> tuple[tuple[int, int], ...]:
        """Row-major WMMA B ``(row, k)`` elements held by ``lane``."""

        return self._row_major_operand_coordinates(lane, mma_step)

    def accumulator_coordinates(self, lane: int) -> tuple[tuple[int, int], ...]:
        """``(output row, token column)`` held by one accumulator lane."""

        token, lane_group = self._lane(lane)
        row0 = lane_group * self.accumulator_elements_per_lane
        return tuple(
            (row0 + element, token)
            for element in range(self.accumulator_elements_per_lane)
        )

    def accumulator_offset(self, wave: int, lane: int) -> int:
        """First f32 of a lane's accumulator in wave-major scratch."""

        self._lane(lane)
        if wave < 0:
            raise ValueError(f"wave must be nonnegative, got {wave}")
        return (
            wave * self.architecture.wave_size + lane
        ) * self.accumulator_elements_per_lane

    def accumulator_location(self, row: int, token: int) -> tuple[int, int]:
        """``(lane, element)`` that owns output ``[row, token]``."""

        if not 0 <= row < ROWS or not 0 <= token < ROWS:
            raise ValueError(f"output coordinate {(row, token)} outside {ROWS}x{ROWS}")
        lane_group, element = divmod(row, self.accumulator_elements_per_lane)
        return token + ROWS * lane_group, element

    def token_tiling(self, rows: int) -> Gfx1250TokenTiling:
        """Return the reusable M16 row schedule for gfx1250 GEMVs."""

        if self.architecture is not GFX1250:
            raise ValueError("token tiling is a gfx1250 WMMA contract")
        return Gfx1250TokenTiling(rows)

    def chunk_ids(self, k: int, wave: int, waves: int) -> tuple[int, ...]:
        """K128 chunks owned by ``wave`` when a block has ``waves`` waves."""

        if k <= 0 or k % K_PER_CHUNK:
            raise ValueError(f"K must be a positive multiple of {K_PER_CHUNK}, got {k}")
        if waves <= 0 or not 0 <= wave < waves:
            raise ValueError(f"wave {wave} outside a {waves}-wave block")
        return tuple(range(wave, k // K_PER_CHUNK, waves))

    def chunks_per_wave(self, k: int, waves: int) -> tuple[int, int]:
        """``(maximum chunks owned by a wave, total chunks)``."""

        if k <= 0 or k % K_PER_CHUNK:
            raise ValueError(f"K must be a positive multiple of {K_PER_CHUNK}, got {k}")
        if waves <= 0:
            raise ValueError(f"waves must be positive, got {waves}")
        chunks = k // K_PER_CHUNK
        return (chunks + waves - 1) // waves, chunks

    def slab_plan(
        self, k: int, rows: int, waves: int, lds_budget_bytes: int
    ) -> tuple[tuple[int, int], ...]:
        """Plan K slabs that fit an explicit activation-LDS budget.

        Splits occur only between complete wave groups, preserving each wave's
        accumulation order.  The budget is the caller's allocation for the
        padded BF16 activation rows, not an inferred device-wide capacity.
        """

        if k <= 0 or k % K_PER_CHUNK:
            raise ValueError(f"K must be a positive multiple of {K_PER_CHUNK}, got {k}")
        if rows <= 0:
            raise ValueError(f"rows must be positive, got {rows}")
        if waves <= 0:
            raise ValueError(f"waves must be positive, got {waves}")
        if lds_budget_bytes <= 0:
            raise ValueError(
                f"LDS slab budget must be positive, got {lds_budget_bytes}"
            )

        def bytes_for(columns: int) -> int:
            return rows * (columns + LDS_PAD) * 2

        if bytes_for(k) <= lds_budget_bytes:
            return ((0, k),)

        group_columns = waves * K_PER_CHUNK
        if k % group_columns:
            raise ValueError(
                f"K={k} cannot be split on {waves}-wave groups of "
                f"{group_columns} columns"
            )
        if bytes_for(group_columns) > lds_budget_bytes:
            raise ValueError(
                f"one {group_columns}-column wave group needs "
                f"{bytes_for(group_columns)} LDS bytes, budget is "
                f"{lds_budget_bytes}"
            )

        groups = k // group_columns
        groups_per_slab = (lds_budget_bytes // (2 * rows) - LDS_PAD) // group_columns
        groups_per_slab = max(1, groups_per_slab)
        slabs = []
        start_group = 0
        while start_group < groups:
            take = min(groups_per_slab, groups - start_group)
            slabs.append((start_group * group_columns, take * group_columns))
            start_group += take
        return tuple(slabs)

    def slab_width(self, k: int, rows: int, waves: int, lds_budget_bytes: int) -> int:
        """Widest K slab produced by :meth:`slab_plan`."""

        return max(
            width for _, width in self.slab_plan(k, rows, waves, lds_budget_bytes)
        )


GFX950_GEMV = GemvLayout(
    architecture=GFX950,
    operand_packs_per_lane=1,
    operand_pack_stride=0,
)


GFX1250_GEMV = GemvLayout(
    architecture=GFX1250,
    # gfx1250 WMMA's v16 BF16 operand is two 8-element packs.  Lane half
    # ``lane // 16`` selects K[0:8] or K[8:16], and the second pack advances
    # by 16 to K[16:24] or K[24:32].
    operand_packs_per_lane=2,
    operand_pack_stride=16,
)


_BACKEND_MODULES = {
    GFX950.gfx: "atom.models.kimi_k3.mono.kernels.stages.gemv",
    GFX1250.gfx: "atom.models.kimi_k3.mono.kernels.stages.gemv_gfx1250",
}


def layout_for(architecture: Architecture) -> GemvLayout:
    """Return the exact GEMV layout for ``architecture``."""

    if architecture is GFX950:
        return GFX950_GEMV
    if architecture is GFX1250:
        return GFX1250_GEMV
    raise ValueError(f"no Kimi GEMV layout for {architecture.gfx}")


def backend_for(architecture: Architecture):
    """Lazily import the device backend selected by an explicit architecture."""

    try:
        selected = layout_for(architecture)
    except ValueError as error:
        raise ValueError(f"no Kimi GEMV backend for {architecture.gfx}") from error
    module = _BACKEND_MODULES[selected.architecture.gfx]
    return import_module(module)


def stage_backend_for(architecture: Architecture, *, waves: int | None = None):
    """Return the uniform stage API for an explicitly selected architecture.

    gfx950 returns its historical module directly.  gfx1250 requires the
    caller to provide a wave count because no Kimi gfx1250 CTA geometry has
    been established; this selector must not invent one.
    """

    module = backend_for(architecture)
    if architecture is GFX950:
        if waves not in (None, GFX950_WAVES):
            raise ValueError(
                f"gfx950 Kimi GEMV is fixed to {GFX950_WAVES} waves, got {waves}"
            )
        return module
    if waves is None:
        raise ValueError("gfx1250 Kimi GEMV needs an explicit waves count")
    return module.StageBackend(waves)
