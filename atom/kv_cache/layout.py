"""Architecture-neutral KV-pool layout results and default sizing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

ManagerKind = Literal["dense", "dsv4"]
LayoutKind = Literal["dense", "fixed", "full_retain", "arena"]


@dataclass(frozen=True, slots=True)
class KvLayoutOptions:
    """Runtime policy flags that select a physical pool layout."""

    unified_arena: bool = False
    full_retain: bool = False
    swa_tail_budget_frac: float = 0.2

    @classmethod
    def from_environment(cls) -> "KvLayoutOptions":
        from atom.utils import envs

        return cls(
            unified_arena=bool(envs.ATOM_V4_UNIFIED_KV_ARENA),
            full_retain=bool(envs.ATOM_SWA_FULL_RETAIN),
            swa_tail_budget_frac=float(envs.ATOM_SWA_TAIL_BUDGET_FRAC),
        )


@dataclass(frozen=True, slots=True)
class KvPoolLayout:
    """Complete output of KV memory-budget partitioning."""

    manager_kind: ManagerKind
    layout_kind: LayoutKind
    num_primary_blocks: int
    primary_block_bytes: int
    block_bytes: int
    available_for_pool: int
    num_swa_blocks: int = 0
    swa_block_bytes: int = 0
    swa_window_size: int = 0
    arena_specs: list[Any] | None = None
    per_req_cache_bytes: int = 0
    slots_per_req: int = 1
    max_per_req_cache_slots: int = 0
    per_req_cache_equiv_blocks: int = 0

    @property
    def num_kvcache_blocks(self) -> int:
        """Compatibility name used by ModelRunner and scheduler config."""
        return self.num_primary_blocks


def compute_total_kv_block_bytes(builders: Iterable[Any]) -> int:
    """Sum target and optional draft builders' per-block allocations."""
    total = sum(int(builder.compute_block_bytes()) for builder in builders)
    if total <= 0:
        raise ValueError(f"total KV block bytes must be positive, got {total}")
    return total


def compute_default_kv_pool_layout(
    *,
    available_for_pool: int,
    block_bytes: int,
    per_req_cache_bytes: int = 0,
    slots_per_req: int = 1,
    max_per_req_cache_slots: int = 0,
    per_req_cache_equiv_blocks: int = 0,
) -> KvPoolLayout:
    """Compute the default single-pool layout used by dense backends."""
    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be positive, got {block_bytes}")
    return KvPoolLayout(
        manager_kind="dense",
        layout_kind="dense",
        num_primary_blocks=max(0, available_for_pool // block_bytes),
        primary_block_bytes=block_bytes,
        block_bytes=block_bytes,
        available_for_pool=available_for_pool,
        per_req_cache_bytes=per_req_cache_bytes,
        slots_per_req=slots_per_req,
        max_per_req_cache_slots=max_per_req_cache_slots,
        per_req_cache_equiv_blocks=per_req_cache_equiv_blocks,
    )
