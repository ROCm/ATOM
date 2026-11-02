# SPDX-License-Identifier: MIT
"""Feasibility gate for resident-CTA fused gfx1250 K2 intervals.

The staged implementation remains the correctness reference.  Emitting either
interval as one kernel requires a resident work-queue grid and grid-wide phase
barriers.  Source-level LDS legality is not residency evidence.
"""

from dataclasses import dataclass

# The staged projection's Smem holds an M16 x (K + pad) BF16 tile plus
# accumulators.  A fused candidate must either reserve this full footprint or
# replace it with the K-slab staging used by persistent K1; 40,960 bytes was
# only the smaller AttnRes/front-role estimate and was not a valid upper bound.
FRONT_REQUIRED_LDS_BYTES = 237_824
FINISH_REQUIRED_LDS_BYTES = 113_664
GATE_UP_TASKS_PER_M16_TILE = 256 * 12
DOWN_TASKS = 3584 // 32


@dataclass(frozen=True)
class FusedIntervalCandidate:
    name: str
    blocks: int
    threads: int
    required_lds_bytes: int
    maximum_tasks: int
    all_declared_ctas_resident: bool = True

    @property
    def grid(self) -> tuple[int]:
        return (self.blocks,)


@dataclass(frozen=True)
class FusedIntervalDecision:
    candidates: tuple[FusedIntervalCandidate, ...]
    runnable: bool
    refusal_code: str
    missing_evidence: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class FusedIntervalResourceEvidence:
    architecture: str
    cu_count: int
    kernel_symbol: str
    group_segment_fixed_size: int
    private_segment_fixed_size: int
    sgpr_count: int
    sgpr_spill_count: int
    vgpr_count: int
    vgpr_spill_count: int
    wavefront_size: int
    max_flat_workgroup_size: int
    max_active_blocks_per_cu: int
    measurement: str

    def validate(self, plan, candidate) -> None:
        if self.architecture != "gfx1250":
            raise ValueError("fused K2 evidence must name gfx1250")
        if self.cu_count != plan.blocks:
            raise ValueError("fused K2 grid needs exactly one participating CTA per CU")
        if candidate.grid != (self.cu_count,):
            raise ValueError("fused K2 grid does not cover measured CUs")
        if self.kernel_symbol != candidate.name:
            raise ValueError("fused K2 resource symbol mismatch")
        if not (0 < self.group_segment_fixed_size <= candidate.required_lds_bytes):
            raise ValueError("fused K2 compiled LDS exceeds its plan")
        if self.sgpr_count <= 0 or self.vgpr_count <= 0:
            raise ValueError("fused K2 register metadata is missing")
        if self.sgpr_spill_count < 0 or self.vgpr_spill_count < 0:
            raise ValueError("fused K2 spill metadata is invalid")
        if self.wavefront_size != 32:
            raise ValueError("fused K2 requires compiled wave32 metadata")
        if self.max_flat_workgroup_size < plan.threads:
            raise ValueError("fused K2 workgroup limit is too small")
        if self.max_active_blocks_per_cu < 1:
            raise ValueError("fused K2 has no resident CTA slot per CU")
        if not self.measurement:
            raise ValueError("fused K2 evidence needs a measurement source")


@dataclass(frozen=True)
class BoundFusedInterval:
    candidate: object
    evidence: FusedIntervalResourceEvidence
    runnable: bool = True
    launch_count: int = 1


def bind_fused_interval(plan, candidate, evidence) -> BoundFusedInterval:
    evidence.validate(plan, candidate)
    return BoundFusedInterval(candidate, evidence)


def assess_fused_intervals(plan) -> FusedIntervalDecision:
    """Return the precise pre-emission residency gate."""

    candidates = (
        FusedIntervalCandidate(
            f"k3_k2_gfx1250_fused_front_s{plan.tokens}",
            plan.blocks,
            plan.threads,
            FRONT_REQUIRED_LDS_BYTES,
            56,
        ),
        FusedIntervalCandidate(
            f"k3_k2_gfx1250_fused_finish_s{plan.tokens}",
            plan.blocks,
            plan.threads,
            FINISH_REQUIRED_LDS_BYTES,
            plan.token_tiling.tile_count * (GATE_UP_TASKS_PER_M16_TILE + DOWN_TASKS),
        ),
    )
    return FusedIntervalDecision(
        candidates,
        False,
        "gfx1250-k2-fused-interval-residency-unproven",
        (
            "target-compiled front and finish kernel symbols do not exist",
            "compiled LDS, SGPR, and VGPR metadata is unavailable",
            "one active 256-thread CTA per each declared CU is unmeasured",
            "fair progress for the resident work queue is unvalidated",
        ),
        (
            "emit and target-compile both 256-CTA candidates, then bind only "
            "if K1-style HSACO metadata and HIP occupancy admit one CTA/CU; "
            "retain staged external collectives as the fallback"
        ),
    )
