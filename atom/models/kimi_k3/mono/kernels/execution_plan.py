# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Execution and LDS contracts for gfx1250 Kimi builders.

gfx1250's wave32 and 320-KiB maximum workgroup LDS are evidenced architecture
facts. Callers still provide thread count, block count, and per-CTA LDS.
"""

from __future__ import annotations

from dataclasses import dataclass

from atom.models.kimi_k3.mono.kernels.attention import layout as attention_layout
from atom.models.kimi_k3.mono.kernels.stages import moe_layout, moe_tail_layout
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import (
    GFX1250_GEMV,
    LDS_PAD,
    GemvLayout,
    stage_backend_for,
)
from atom.models.kimi_k3.mono.kernels.stages.moe_expert_layout import (
    GFX1250_EXPERT,
    ExpertTaskContract,
    ExpertWeightPacking,
    expert_backend_for,
)
from atom.models.kimi_k3.mono.kernels.token_tiling import (
    GFX1250_MAX_TOKENS,
    Gfx1250TokenTiling,
)
from atom.mono.plan.architecture import GFX1250, Architecture

# Model constants from stages/moe.py and attention/kda.py.  They describe Kimi
# shapes, not gfx1250 hardware topology.
HIDDEN = 7168
LATENT = 3584
EXPERTS = 896
TOPK = 16
ROUTED_INTERMEDIATE = 384
ATTENTION_COLUMNS = 128
UG_GROUPS = ROUTED_INTERMEDIATE // 32
EARLY_K = (HIDDEN // 4, HIDDEN // 2, HIDDEN)

ALIGNMENT = 16
STANDALONE_TIMELINE_POINTS = 8
FUSED_TIMELINE_POINTS = 17


def _align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _struct_bytes(*field_bytes: int) -> int:
    offset = 0
    for size in field_bytes:
        if size < 0:
            raise ValueError(f"field size must be nonnegative, got {size}")
        offset = _align(offset) + size
    return _align(offset)


@dataclass(frozen=True)
class LdsRegion:
    """One 16-byte-aligned allocation in the candidate CTA's LDS."""

    name: str
    offset: int
    bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class KimiLdsPlan:
    """Conservative byte plan matching Kimi's sequential structs and stage union."""

    capacity_bytes: int
    regions: tuple[LdsRegion, ...]
    gemv_slabs: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError("LDS capacity must be positive")
        previous_end = 0
        for region in self.regions:
            if region.offset % ALIGNMENT:
                raise ValueError(f"{region.name}: unaligned offset {region.offset}")
            if region.offset < previous_end:
                raise ValueError(f"{region.name}: overlaps preceding LDS region")
            if region.end > self.capacity_bytes:
                raise ValueError(
                    f"{region.name}: LDS end {region.end} exceeds "
                    f"{self.capacity_bytes}"
                )
            previous_end = region.end

    @property
    def used_bytes(self) -> int:
        return max((region.end for region in self.regions), default=0)

    @property
    def stage_budget_bytes(self) -> int:
        return next(region.bytes for region in self.regions if region.name == "stage")


@dataclass(frozen=True)
class Gfx1250KimiBuildPlan:
    """Explicit candidate geometry for the architecture-separated GEMV substrate."""

    threads: int
    blocks: int
    lds_bytes: int
    tokens: int
    nblocks: int | None = None
    architecture: Architecture = GFX1250
    gemv_layout: GemvLayout = GFX1250_GEMV

    def __post_init__(self) -> None:
        if self.architecture is not GFX1250:
            raise ValueError("Gfx1250KimiBuildPlan only accepts GFX1250")
        if self.gemv_layout is not GFX1250_GEMV:
            raise ValueError("gfx1250 Kimi plan needs the gfx1250 GEMV layout")
        if self.threads <= 0 or self.threads % self.architecture.wave_size:
            raise ValueError(
                f"threads must be a positive multiple of wave32, got {self.threads}"
            )
        if self.blocks <= 0:
            raise ValueError(f"blocks must be explicit and positive, got {self.blocks}")
        if not 1 <= self.tokens <= GFX1250_MAX_TOKENS:
            raise ValueError(
                f"gfx1250 Kimi token rows must be in "
                f"[1, {GFX1250_MAX_TOKENS}], got {self.tokens}"
            )
        if self.nblocks is not None and self.nblocks < 0:
            raise ValueError(f"nblocks must be nonnegative, got {self.nblocks}")
        maximum = self.architecture.max_workgroup_lds_bytes
        if self.lds_bytes <= 0 or maximum is None or self.lds_bytes > maximum:
            raise ValueError(
                f"gfx1250 candidate LDS must be in [1, {maximum}], "
                f"got {self.lds_bytes}"
            )
        self.lds_plan()

    @property
    def waves(self) -> int:
        return self.threads // self.architecture.wave_size

    @property
    def token_tiling(self) -> Gfx1250TokenTiling:
        return Gfx1250TokenTiling(self.tokens)

    def lane_wave(self, tid: int) -> tuple[int, int]:
        """Return ``(lane, wave)`` under this candidate's wave32 topology."""

        if not 0 <= tid < self.threads:
            raise ValueError(f"thread {tid} outside [0, {self.threads})")
        return tid % self.architecture.wave_size, tid // self.architecture.wave_size

    def task_ids(self, total: int, block: int) -> tuple[int, ...]:
        """Persistent-grid round-robin tasks owned by one candidate block."""

        if total < 0:
            raise ValueError(f"task count must be nonnegative, got {total}")
        if not 0 <= block < self.blocks:
            raise ValueError(f"block {block} outside [0, {self.blocks})")
        return tuple(range(block, total, self.blocks))

    def _smem_bytes(self) -> int:
        accumulator = self.gemv_layout.accumulator_elements_per_lane
        red = self.waves * self.architecture.wave_size * accumulator
        red *= 4
        timeline_points = (
            FUSED_TIMELINE_POINTS
            if self.nblocks is not None
            else STANDALONE_TIMELINE_POINTS
        )
        return _struct_bytes(red, timeline_points * 8)

    def _route_bytes(self) -> int:
        union = min(self.tokens * TOPK, EXPERTS)
        tile_rows = self.token_tiling.lds_rows
        return _struct_bytes(
            self.tokens * 2 * TOPK * 4,
            EXPERTS * 4,
            union * 4,
            # KOF is rebuilt/consumed one M16 token tile at a time.  Keeping
            # the historical whole-S matrix would consume 144 KiB at S=48.
            union * tile_rows * 4,
            self.waves * 4,
        )

    def _non_early_stage_bytes(self) -> tuple[int, ...]:
        tile_rows = self.token_tiling.lds_rows
        ug = _struct_bytes(
            tile_rows * LATENT // 8 * 4,
            tile_rows * LATENT // 32 * 4,
            self.threads * 4,
            4 * self.architecture.wave_size * GFX1250_EXPERT.accumulator_elements * 4,
        )
        down = _struct_bytes(
            tile_rows * TOPK * 48 * 4,
            tile_rows * TOPK * UG_GROUPS * 4,
            tile_rows * TOPK * GFX1250_EXPERT.output_rows * 4,
        )
        stages = [ug, down]
        stages.append(
            moe_tail_layout.tail_lds_plan(self.tokens, self.waves).required_bytes
        )
        if self.nblocks is not None:
            sources = self.nblocks + 1
            stages.append(
                _struct_bytes(
                    tile_rows * sources * ATTENTION_COLUMNS * 4,
                    tile_rows * ATTENTION_COLUMNS * 4,
                )
            )
        return tuple(stages)

    def lds_plan(self) -> KimiLdsPlan:
        """Build and bounds-check the candidate's sequential LDS allocations."""

        smem = self._smem_bytes()
        route = self._route_bytes()
        stage_offset = _align(smem) + _align(route)
        stage_budget = self.lds_bytes - stage_offset
        if stage_budget <= 0:
            raise ValueError(
                f"fixed LDS regions need {stage_offset} bytes, candidate has "
                f"{self.lds_bytes}"
            )

        slabs = tuple(
            self.gemv_layout.slab_plan(
                k, self.token_tiling.lds_rows, self.waves, stage_budget
            )
            for k in EARLY_K
        )
        early = max(
            _struct_bytes(
                self.token_tiling.lds_rows
                * (max(width for _, width in plan) + LDS_PAD)
                * 2
            )
            for plan in slabs
        )
        stage_used = max((early, *self._non_early_stage_bytes()))
        if stage_used > stage_budget:
            raise ValueError(
                f"Kimi stage union needs {stage_used} LDS bytes after slabbing, "
                f"only {stage_budget} remain"
            )
        regions = (
            LdsRegion("smem", 0, smem),
            LdsRegion("route", _align(smem), route),
            LdsRegion("stage", stage_offset, stage_used),
        )
        return KimiLdsPlan(self.lds_bytes, regions, slabs)

    def stage_backend(self):
        """Select the wired gfx1250 GEMV API for this explicit wave count."""

        return stage_backend_for(self.architecture, waves=self.waves)

    def attention_back_backend(self):
        """Select K1/K2 attention back's o-projection matrix backend."""

        return attention_layout.back_backend_for(self.architecture, waves=self.waves)

    def kda_matrix_contract(self):
        """Return the CPU-only KDA projection/gate contract."""

        return attention_layout.kda_contract_for(self.architecture, waves=self.waves)

    def kda_slab_plan(
        self, rows: int, *, fixed_lds_bytes: int
    ) -> tuple[tuple[int, int], ...]:
        """Plan KDA in-projection slabs inside this candidate's LDS."""

        return self.kda_matrix_contract().slab_plan(
            rows,
            lds_capacity_bytes=self.lds_bytes,
            fixed_lds_bytes=fixed_lds_bytes,
        )

    @property
    def kda_accumulator_lds_elements(self) -> int:
        """Architecture-derived f32 accumulator scratch for KDA."""

        return self.kda_matrix_contract().accumulator_lds_elements

    def attention_seam_lds_bytes(self, sources: int) -> int:
        """Bounds-check KDA softmax/mix scratch for this candidate."""

        contract = self.kda_matrix_contract()
        red, vals = contract.seam_lds_elements(self.tokens, sources)
        red = max(red, contract.accumulator_lds_elements)
        smem = _struct_bytes(red * 4, 15 * 8)
        total = _align(smem) + _align(vals * 4)
        if total > self.lds_bytes:
            raise ValueError(
                f"attention seam needs {total} LDS bytes, candidate has "
                f"{self.lds_bytes}"
            )
        return total

    def kda_backend(self):
        """Select the implemented KDA matrix, norm, and recurrence slices."""

        if self.nblocks is None:
            raise ValueError(
                "KDA backend selection needs explicit nblocks for seam LDS"
            )
        self.attention_seam_lds_bytes(self.nblocks + 1)
        return attention_layout.kda_backend_for(self.architecture, waves=self.waves)

    def kda_matrix_backend(self):
        """Select matrix slices without claiming a complete seam contract."""

        return attention_layout.kda_backend_for(self.architecture, waves=self.waves)

    def moe_front_contracts(self):
        """Return CPU-only router, route-union, and latent-quant contracts."""

        return moe_layout.front_contracts_for(
            self.architecture,
            threads=self.threads,
            tokens=self.tokens,
        )

    def moe_front_backend(self):
        """Select the architecture-separated gfx1250 K2 front backend."""

        self.lds_plan()
        return moe_layout.front_backend_for(
            self.architecture,
            threads=self.threads,
            tokens=self.tokens,
        )

    def expert_task_contract(self) -> ExpertTaskContract:
        """Return gfx1250 expert K-step, slot, and LDS ownership."""

        return ExpertTaskContract(self.waves)

    def expert_weight_packings(
        self,
    ) -> tuple[ExpertWeightPacking, ExpertWeightPacking]:
        """Gate/up and down payload/scale packing contracts."""

        return (
            ExpertWeightPacking(2 * ROUTED_INTERMEDIATE, LATENT),
            ExpertWeightPacking(LATENT, ROUTED_INTERMEDIATE),
        )

    def expert_backend(self):
        """Select the separately implemented scaled-WMMA expert backend."""

        return expert_backend_for(
            self.architecture,
            waves=self.waves,
            tokens=self.tokens,
        )

    def moe_tail_contract(self):
        """Return wave32 post-expert ownership and staging contracts."""

        return moe_tail_layout.tail_contract_for(
            self.architecture,
            threads=self.threads,
            waves=self.waves,
            tokens=self.tokens,
        )

    def moe_tail_lds_plan(self):
        """Return and bounds-check the post-expert stage-union LDS."""

        layout = moe_tail_layout.tail_lds_plan(self.tokens, self.waves)
        if layout.required_bytes > self.lds_plan().stage_budget_bytes:
            raise ValueError(
                f"K2 tail needs {layout.required_bytes} stage LDS bytes, "
                f"budget is {self.lds_plan().stage_budget_bytes}"
            )
        return layout

    def moe_tail_backend(self):
        """Select the separately implemented local-compute tail backend."""

        self.moe_tail_lds_plan()
        return moe_tail_layout.tail_backend_for(
            self.architecture,
            threads=self.threads,
            waves=self.waves,
            tokens=self.tokens,
        )

    def build_k1_local(self, spec):
        """Build the typed, complete rank-local gfx1250 K1 callable."""

        from atom.models.kimi_k3.mono.kernels.attention import (
            kda_builder_gfx1250,
        )

        return kda_builder_gfx1250.build_local(self, spec=spec)

    def make_k1_local_runner(self, spec):
        """Build K1 and bind its product scratch-owning host adapter."""

        from atom.models.kimi_k3.mono.k1_local_runner_gfx1250 import (
            Gfx1250K1LocalRunner,
        )

        result = self.build_k1_local(spec)
        return Gfx1250K1LocalRunner(self, spec, result)
