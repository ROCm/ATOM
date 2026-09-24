# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte-currency description of the cache classes an attention type needs.

There are exactly two pools, distinguished by what their size scales with:

  Pool.PAGE   scales with how much history is cached. The paged KV blocks.
  Pool.STATE  scales with in-flight requests, not with sequence length. The
              sliding window, the DeepSeek-V4 compressor ring, GDN/Mamba
              recurrent state. Classification follows the scaling, not the
              addressing: what puts a class in STATE is that its total is
              `max_num_seqs * per_req`, whatever the rows are indexed by.

A pool is a budget region, not a single entry size. Inside one pool there can
be several **entry classes**, each with its own index space, its own bytes
per entry, and — in the STATE pool — its own per-request multiplicity. The
sliding-window ring and the compressor ring live in the same pool and are
counted separately; collapsing them into one entry size would make the two
counts unrecoverable.

Entry classes are keyed by name. Two specs with the same name are the same
index space, so their `entry_bytes` add — that is how an Eagle3 draft KV pool
rides the target model's block ids, and how DeepSeek-V4's indexer cache rides
the same block table as its main compressed KV. Two specs with different names
are different index spaces even inside one pool and keep separate counts: the
sliding-window ring is `win_with_spec` rows per request while the compressor
ring is one entry per request, so they can never share a count.

This module deliberately defines no *architecture* vocabulary. A name is owned
by whatever consumes the count — `kv_block.py` names the per-request slot
class — and the backend that declares the spec imports it from there. The one name defined here, `PAGED_CLASS`, is
not an architecture's: it is the allocation regime itself, since PAGE is by
construction a single shared index space that every contributor adds bytes to.

Cross-request sharing is a property of the cache layer above, not of this
sizing. A STATE entry class is shared by copying it into the resuming
request's slot at a checkpoint, not by two requests pointing at one entry.
What is reserved here is the floor an in-flight request cannot run without;
retained-and-shared entries live in the same region and compete for the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Pool(Enum):
    """Which budget region an entry class draws from."""

    PAGE = "page"
    STATE = "state"


@dataclass(frozen=True)
class SubPoolSpec:
    """One entry class: a pool, an index space, and a per-entry byte cost."""

    pool: Pool
    name: str
    entry_bytes: int
    # STATE only. `entries_per_req` is what one in-flight request holds — 1
    # committed state, `1 + num_spec` when a rollback slot per speculated
    # token is kept, or the window block count for a sliding-window class.
    # `extra_entries` is a flat cushion on the class as a whole, on top of the
    # per-request term. What it covers is the declaring backend's business.
    entries_per_req: int = 0
    extra_entries: int = 0

    def __post_init__(self):
        if self.pool is Pool.STATE and self.entries_per_req < 1:
            raise ValueError(f"{self.name}: STATE entries need entries_per_req >= 1")
        if self.pool is Pool.PAGE and (self.entries_per_req or self.extra_entries):
            raise ValueError(f"{self.name}: PAGE entries are sized from the remainder")


# The paged class every backend contributes to. There is one block index
# space, so a draft KV pool or an indexer cache adds bytes to it rather than
# forming a pool of its own.
PAGED_CLASS = "kv"


def page_pool(entry_bytes: int) -> SubPoolSpec:
    """The paged KV entry class: sized from whatever the STATE pool leaves."""
    return SubPoolSpec(Pool.PAGE, PAGED_CLASS, entry_bytes)


def state_pool(
    name: str,
    entry_bytes: int,
    *,
    entries_per_req: int,
    extra_entries: int = 0,
) -> SubPoolSpec:
    """A per-request state entry class."""
    return SubPoolSpec(Pool.STATE, name, entry_bytes, entries_per_req, extra_entries)


@dataclass(frozen=True)
class PoolPlan:
    """Sizing result: per entry class, how many entries and at what cost.

    Consumers index `entries` by the class name they themselves declared —
    the runner that produced the plan never needs to know the names.
    """

    entries: dict[str, int]
    entry_bytes: dict[str, int]
    reserved_bytes: dict[str, int]
    entries_per_req: dict[str, int]
    paged_class: str | None = None

    @classmethod
    def empty(cls) -> PoolPlan:
        """The plan before sizing has run.

        Every count reads back as 0, which is what "no pool exists yet" means
        to a consumer. Model warmup happens before the budget is known, so a
        builder asking for its entry count then must get 0 rather than an
        AttributeError — the runner installs this at construction time and
        replaces it in `get_num_blocks`.
        """
        return cls(entries={}, entry_bytes={}, reserved_bytes={}, entries_per_req={})

    def with_paged_entries(self, count: int) -> PoolPlan:
        """Copy of the plan with the PAGE class resized to `count`.

        Pipeline-parallel stages reconcile their block counts to the global
        minimum after sizing; the plan has to follow so it stays the one place
        every consumer can read an entry count from.
        """
        if self.paged_class is None:
            raise ValueError("no PAGE entry class to resize")
        bytes_per = self.entry_bytes[self.paged_class]
        return replace(
            self,
            entries={**self.entries, self.paged_class: count},
            reserved_bytes={**self.reserved_bytes, self.paged_class: count * bytes_per},
        )

    @property
    def paged_entries(self) -> int:
        """Entries in the single PAGE class — the block count the scheduler
        and BlockManager are built around. Named-class-free so the runner can
        publish the plan without knowing any architecture's vocabulary."""
        return self.entries[self.paged_class] if self.paged_class else 0

    @property
    def total_reserved_bytes(self) -> int:
        return sum(self.reserved_bytes.values())


class InsufficientPoolBudget(RuntimeError):
    """The STATE floor alone exceeds the budget, leaving nothing to page with.

    Carries the numbers rather than a formatted message: the caller
    (ModelRunner) has the GPU-side context needed to suggest a fix.
    """

    def __init__(self, reserved_bytes: int, available_bytes: int, entries: int):
        super().__init__(f"state pool needs {reserved_bytes}B of {available_bytes}B")
        self.reserved_bytes = reserved_bytes
        self.available_bytes = available_bytes
        self.entries = entries


def merge_specs(specs: list[SubPoolSpec]) -> dict[str, SubPoolSpec]:
    """Collapse specs naming the same entry class, summing `entry_bytes`.

    Same name means same index space, so pool and multiplicity must agree.
    Different names stay separate even within one pool.
    """
    merged: dict[str, SubPoolSpec] = {}
    for spec in specs:
        prev = merged.get(spec.name)
        if prev is None:
            merged[spec.name] = spec
            continue
        if (
            prev.pool,
            prev.entries_per_req,
            prev.extra_entries,
        ) != (
            spec.pool,
            spec.entries_per_req,
            spec.extra_entries,
        ):
            raise ValueError(
                f"entry class {spec.name!r} declared twice with different "
                f"pool/multiplicity: {prev} vs {spec}"
            )
        merged[spec.name] = SubPoolSpec(
            spec.pool,
            spec.name,
            prev.entry_bytes + spec.entry_bytes,
            spec.entries_per_req,
            spec.extra_entries,
        )
    return merged


def plan_pools(
    specs: list[SubPoolSpec], available_bytes: int, max_num_seqs: int
) -> PoolPlan:
    """Turn entry-class declarations plus a byte budget into entry counts.

    Pure function — no CUDA, no config, no env. The STATE floor is reserved
    first because a request cannot run without it; the PAGE pool absorbs what
    is left.

    Raises `InsufficientPoolBudget` if the floor leaves nothing to page with.
    """
    merged = merge_specs(specs)
    entries: dict[str, int] = {}
    reserved: dict[str, int] = {}
    remaining = available_bytes

    state = {n: s for n, s in merged.items() if s.pool is Pool.STATE}
    for name, spec in state.items():
        count = max_num_seqs * spec.entries_per_req + spec.extra_entries
        cost = count * spec.entry_bytes
        entries[name], reserved[name] = count, cost
        remaining -= cost
    if state and remaining <= 0:
        raise InsufficientPoolBudget(
            reserved_bytes=sum(reserved.values()),
            available_bytes=available_bytes,
            entries=sum(entries.values()),
        )

    paged = [n for n, s in merged.items() if s.pool is Pool.PAGE]
    if len(paged) > 1:
        raise ValueError(f"more than one PAGE entry class: {paged}")
    for name in paged:
        spec = merged[name]
        count = max(0, remaining // spec.entry_bytes)
        entries[name], reserved[name] = count, count * spec.entry_bytes

    return PoolPlan(
        entries=entries,
        entry_bytes={n: s.entry_bytes for n, s in merged.items()},
        reserved_bytes=reserved,
        entries_per_req={n: s.entries_per_req for n, s in merged.items()},
        paged_class=paged[0] if paged else None,
    )
