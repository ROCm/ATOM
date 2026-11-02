# SPDX-License-Identifier: MIT
"""Rank-local resident-grid protocol for fused gfx1250 K2 intervals.

This is a logical progress model.  It deliberately does not infer hardware
residency: a product launcher must separately bind compiled resources and HIP
occupancy for exactly ``blocks`` CTAs.
"""

from __future__ import annotations

from dataclasses import dataclass

FRONT_PHASES = ("slice", "gate", "mix", "normalize", "projection")
FRONT_SPLIT_PHASES = (
    "slice",
    "gate",
    "mix",
    "normalize",
    "projection_partial",
    "projection",
)
PROJECTION_TASKS = 7 + 28 + 96
ATTNRES_SLICES = 7168 // 128
FINISH_PHASES = (
    "latent_reorder",
    "route_shared_activation",
    "route_union_latent_quant",
    "shared_down_routed_gate_up",
    "routed_down",
)
SHARED_DOWN_TASKS = 7168 // 16
ROUTED_DOWN_TASKS = 3584 // 32
MAX_ROUTE_UNION_PER_TILE = 16 * 16
ROUTED_GATE_UP_GROUPS = 12


@dataclass(frozen=True)
class ProjectionWork:
    tile: int
    task: int
    k_partition: int


@dataclass(frozen=True)
class FinishWork:
    tile: int
    kind: str
    task: int


@dataclass(frozen=True)
class AttnResRoleTraffic:
    rows: int
    sources: int
    removed_values_bytes: int
    removed_mixed_bytes: int
    compact_partials_bytes: int
    compact_weights_bytes: int
    compact_mean_squares_bytes: int
    compact_rstd_bytes: int

    @classmethod
    def for_shape(cls, rows: int, sources: int) -> AttnResRoleTraffic:
        if not 1 <= rows <= 48:
            raise ValueError("AttnRes role rows must be in [1, 48]")
        if not 2 <= sources <= 32:
            raise ValueError("AttnRes role sources must be in [2, 32]")
        return cls(
            rows,
            sources,
            rows * sources * 7168 * 4,
            rows * 7168 * 4,
            ATTNRES_SLICES * rows * sources * 2 * 4,
            rows * (sources + 1) * 4,
            ATTNRES_SLICES * rows * 4,
            rows * 4,
        )

    @property
    def removed_global_bytes(self) -> int:
        return 2 * (self.removed_values_bytes + self.removed_mixed_bytes)

    @property
    def compact_global_bytes(self) -> int:
        return 2 * (
            self.compact_partials_bytes
            + self.compact_weights_bytes
            + self.compact_mean_squares_bytes
            + self.compact_rstd_bytes
        )

    @property
    def net_removed_global_bytes(self) -> int:
        return self.removed_global_bytes - self.compact_global_bytes


@dataclass(frozen=True)
class AttnResRoleSchedule:
    """CPU progress proof for resident role-LDS AttnRes."""

    rows: int
    sources: int
    slices: int = ATTNRES_SLICES

    @property
    def tiles(self) -> int:
        return (self.rows + 15) // 16

    @property
    def slice_roles(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (tile, slice_id)
            for tile in range(self.tiles)
            for slice_id in range(self.slices)
        )

    def prove_progress(self) -> None:
        traffic = AttnResRoleTraffic.for_shape(self.rows, self.sources)
        partials = {
            (slice_id, token, source, kind)
            for slice_id in range(self.slices)
            for token in range(self.rows)
            for source in range(self.sources)
            for kind in ("square", "dot")
        }
        gate_waits = set(partials)
        weights = {
            (token, item)
            for token in range(self.rows)
            for item in range(self.sources + 1)
        }
        slice_weight_waits = {
            (slice_id, token, item)
            for slice_id in range(self.slices)
            for token, item in weights
        }
        mean_squares = {
            (slice_id, token)
            for slice_id in range(self.slices)
            for token in range(self.rows)
        }
        norm_waits = set(mean_squares)
        ready = set(range(self.slices))
        tiled_ready = set(self.slice_roles)
        if gate_waits != partials or norm_waits != mean_squares:
            raise AssertionError("AttnRes role dependency coverage drifted")
        if len(slice_weight_waits) != self.slices * len(weights):
            raise AssertionError("AttnRes weight wait ownership overlaps")
        if len(ready) != self.slices:
            raise AssertionError("AttnRes ready publication overlaps")
        if len(tiled_ready) != self.tiles * self.slices:
            raise AssertionError("AttnRes tiled role ownership overlaps")
        if traffic.net_removed_global_bytes <= 0:
            raise AssertionError("AttnRes role fusion removes no traffic")


@dataclass(frozen=True)
class FrontProjectionSchedule:
    """Compile-time flattened token-tile/output-task/K-partition schedule."""

    tiles: int
    blocks: int
    k_partitions: int

    @classmethod
    def for_shape(cls, rows: int, blocks: int) -> FrontProjectionSchedule:
        if not 1 <= rows <= 48:
            raise ValueError("front schedule rows must be in [1, 48]")
        tiles = (rows + 15) // 16
        # One M16 tile has only 131 output tasks.  Split K in two only there;
        # two or three token tiles already expose 262/393 independent tasks.
        return cls(tiles, blocks, 2 if tiles == 1 else 1)

    @property
    def work(self) -> tuple[ProjectionWork, ...]:
        return tuple(
            ProjectionWork(tile, task, partition)
            for tile in range(self.tiles)
            for partition in range(self.k_partitions)
            for task in range(PROJECTION_TASKS)
        )

    def for_block(self, block: int) -> tuple[ProjectionWork, ...]:
        if not 0 <= block < self.blocks:
            raise ValueError("block outside projection schedule")
        return self.work[block :: self.blocks]

    def prove_coverage(self) -> None:
        assigned = tuple(
            item for block in range(self.blocks) for item in self.for_block(block)
        )
        if len(assigned) != len(set(assigned)) or set(assigned) != set(self.work):
            raise AssertionError("projection schedule coverage overlaps or drops work")


@dataclass(frozen=True)
class FinishIntervalSchedule:
    """Static work queues for one resident sharded-finish launch.

    Runtime route-union sizes mask routed gate/up tasks, but every possible
    task has exactly one compile-time owner.  Empty and sparse unions
    therefore change work, not grid geometry or barrier participation.
    """

    rows: int
    blocks: int

    def __post_init__(self) -> None:
        if not 1 <= self.rows <= 48:
            raise ValueError("finish schedule rows must be in [1, 48]")
        if self.blocks <= 0:
            raise ValueError("finish schedule needs a positive resident grid")

    @property
    def tiles(self) -> int:
        return (self.rows + 15) // 16

    @property
    def router_roles(self) -> tuple[int, ...]:
        return tuple(range(self.rows))

    @property
    def independent_work(self) -> tuple[FinishWork, ...]:
        return tuple(
            FinishWork(tile, "shared_down", task)
            for tile in range(self.tiles)
            for task in range(SHARED_DOWN_TASKS)
        ) + tuple(
            FinishWork(tile, "routed_gate_up", task)
            for tile in range(self.tiles)
            for task in range(MAX_ROUTE_UNION_PER_TILE * ROUTED_GATE_UP_GROUPS)
        )

    @property
    def routed_down_work(self) -> tuple[FinishWork, ...]:
        return tuple(
            FinishWork(tile, "routed_down", task)
            for tile in range(self.tiles)
            for task in range(ROUTED_DOWN_TASKS)
        )

    def for_block(
        self, block: int, *, routed_down: bool = False
    ) -> tuple[FinishWork, ...]:
        if not 0 <= block < self.blocks:
            raise ValueError("block outside finish schedule")
        work = self.routed_down_work if routed_down else self.independent_work
        return work[block :: self.blocks]

    def prove_coverage(self) -> None:
        if len(self.router_roles) != len(set(self.router_roles)):
            raise AssertionError("finish router ownership overlaps")
        for routed_down, expected in (
            (False, self.independent_work),
            (True, self.routed_down_work),
        ):
            assigned = tuple(
                item
                for block in range(self.blocks)
                for item in self.for_block(block, routed_down=routed_down)
            )
            if len(assigned) != len(set(assigned)):
                raise AssertionError("finish work ownership overlaps")
            if set(assigned) != set(expected):
                raise AssertionError("finish work ownership drops tasks")

    def active_gate_up_tasks(
        self, union_sizes: tuple[int, ...]
    ) -> tuple[FinishWork, ...]:
        if len(union_sizes) != self.tiles:
            raise ValueError("one route-union size is required per M16 tile")
        for size in union_sizes:
            if not 0 <= size <= MAX_ROUTE_UNION_PER_TILE:
                raise ValueError("route-union size is outside its tile bound")
        return tuple(
            item
            for item in self.independent_work
            if item.kind != "routed_gate_up"
            or item.task < union_sizes[item.tile] * ROUTED_GATE_UP_GROUPS
        )


@dataclass(frozen=True)
class ResidentGridAssumption:
    all_declared_ctas_resident: bool
    fair_cta_progress: bool = True
    hardware_validated: bool = False

    def require_logical(self) -> None:
        if not self.all_declared_ctas_resident or not self.fair_cta_progress:
            raise ValueError(
                "resident-grid progress needs all declared CTAs and fair progress"
            )
        if self.hardware_validated:
            raise ValueError("CPU proof cannot validate hardware residency")


@dataclass(frozen=True)
class PhaseArrival:
    phase: int
    block: int
    owner_thread: int = 0


@dataclass(frozen=True)
class ResidentGridProgress:
    phases: tuple[str, ...]
    arrivals: tuple[PhaseArrival, ...]
    reset_owners: tuple[int, ...]
    barriers: int
    assumption: ResidentGridAssumption


@dataclass(frozen=True)
class ResidentGridProtocol:
    blocks: int
    threads: int
    phases: tuple[str, ...] = FRONT_PHASES

    def __post_init__(self) -> None:
        if self.blocks <= 0 or self.threads <= 0:
            raise ValueError("resident-grid geometry must be positive")
        if len(self.phases) < 2 or len(set(self.phases)) != len(self.phases):
            raise ValueError("resident-grid phases must be unique and ordered")

    @property
    def synchronized_phases(self) -> int:
        return len(self.phases) - 1

    @property
    def mailbox_pairs(self) -> int:
        return self.blocks * self.synchronized_phases

    @property
    def scratch_bytes(self) -> int:
        # One monotonic i32 ticket counter, aligned to 16, followed by tagged
        # (payload, tag) i32 pairs.
        return 16 + self.mailbox_pairs * 8

    def mailbox_index(self, phase: int, block: int) -> int:
        if not 0 <= phase < self.synchronized_phases:
            raise ValueError("phase has no resident-grid barrier")
        if not 0 <= block < self.blocks:
            raise ValueError("block outside resident grid")
        return phase * self.blocks + block

    def reset_owner(self, pair: int) -> int:
        if not 0 <= pair < self.mailbox_pairs:
            raise ValueError("mailbox pair outside protocol scratch")
        return pair % self.blocks

    def prove_progress(
        self, assumption: ResidentGridAssumption
    ) -> ResidentGridProgress:
        assumption.require_logical()
        reset_owners = tuple(
            self.reset_owner(pair) for pair in range(self.mailbox_pairs)
        )
        arrivals = tuple(
            PhaseArrival(phase, block)
            for phase in range(self.synchronized_phases)
            for block in range(self.blocks)
        )
        if len(arrivals) != self.blocks * self.synchronized_phases:
            raise AssertionError("resident-grid arrival coverage drifted")
        return ResidentGridProgress(
            self.phases,
            arrivals,
            reset_owners,
            self.synchronized_phases,
            assumption,
        )
