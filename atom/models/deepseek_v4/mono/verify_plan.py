# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side live-row layout for V4 target verification.

Request segments include their anchor row. Padding is capacity, not a request
or a committed token. Building a layout does not admit a deployment or authorize
state forks: those need the session's separately qualified state transaction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from operator import index


@dataclass(frozen=True)
class VerifySegment:
    """A request's contiguous live rows and explicit state read/write slots."""

    start: int
    stop: int
    position: int
    slot_in: int
    slot_out: int

    @property
    def rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class VerifyPlan:
    """Packed request layout; inactive capacity must never update request state."""

    segments: tuple[VerifySegment, ...]
    padded_rows: int

    @property
    def live_rows(self) -> int:
        return self.segments[-1].stop if self.segments else 0

    @property
    def offsets(self) -> tuple[int, ...]:
        return (0, *(segment.stop for segment in self.segments))

    @property
    def max_seqlen_q(self) -> int:
        return max((segment.rows for segment in self.segments), default=0)

    @property
    def in_place(self) -> bool:
        return all(segment.slot_in == segment.slot_out for segment in self.segments)


@dataclass(frozen=True)
class VerifyLaunch:
    """One exact prepared shape and its lane within a packed publication."""

    offset: int
    rows: int
    tokens_per_seq: int
    lane: int


def plan_verify_launches(
    lengths: Sequence[int], prepared_shapes: Sequence[tuple[int, int]]
) -> tuple[VerifyLaunch, ...]:
    """Partition adjacent same-q requests without reordering or padding them.

    Shapes are (live rows, q), not graph capacities. A request must fit a
    single launch: splitting it into q=1 changes compressor/mailbox semantics.
    Repeated shapes use distinct lanes so captured publications never share
    their mutable metadata. The caller must own that many prepared lanes.
    """
    lens = tuple(index(value) for value in lengths)
    shapes = tuple((index(rows), index(q)) for rows, q in prepared_shapes)
    if any(length < 1 for length in lens):
        raise ValueError("verify requests require a positive anchor window")
    if any(rows < 1 or q < 1 or rows % q for rows, q in shapes):
        raise ValueError("prepared shapes must contain whole positive-q requests")
    widths: dict[int, set[int]] = {}
    for rows, q in shapes:
        widths.setdefault(q, set()).add(rows // q)
    launches = []
    lanes: Counter = Counter()
    request = offset = 0
    while request < len(lens):
        q = lens[request]
        stop = request + 1
        while stop < len(lens) and lens[stop] == q:
            stop += 1
        remaining = stop - request
        choices = sorted(widths.get(q, ()), reverse=True)
        # Require a singleton so a greedy largest-fit partition always has a
        # legal remainder. Serving prepares singleton shapes for every q.
        if 1 not in choices:
            raise ValueError(f"q={q} has no prepared single-request launch")
        while remaining:
            count = next(width for width in choices if width <= remaining)
            rows = count * q
            key = (rows, q)
            launches.append(VerifyLaunch(offset, rows, q, lanes[key]))
            lanes[key] += 1
            offset += rows
            remaining -= count
        request = stop
    return tuple(launches)


def build_verify_plan(
    lengths: Sequence[int],
    positions: Sequence[int],
    slots_in: Sequence[int],
    slots_out: Sequence[int],
    *,
    padded_rows: int,
    max_seqlen_q: int,
) -> VerifyPlan:
    """Validate CPU metadata without copying GPU tensors or inferring commits.

    ``positions`` contains only live rows, not graph padding. Shared read slots
    can represent prefix forks, but write slots must be distinct. The session
    must additionally check leases/generations, COW and commit/rollback readiness.
    ``max_seqlen_q`` is a capacity bound, not a fixed per-request verify length.
    """
    lens = tuple(index(value) for value in lengths)
    reads = tuple(index(value) for value in slots_in)
    writes = tuple(index(value) for value in slots_out)
    pos = tuple(index(value) for value in positions)
    capacity, max_q = index(padded_rows), index(max_seqlen_q)
    if capacity < 0 or max_q < 1:
        raise ValueError("verify capacity must be nonnegative and max q positive")
    if len(reads) != len(lens) or len(writes) != len(lens):
        raise ValueError("verify state slots must match request count")
    if any(not 1 <= length <= max_q for length in lens):
        raise ValueError("each request must contain 1..max q live rows")
    if len(pos) != sum(lens) or len(pos) > capacity:
        raise ValueError("live positions must match lengths and fit graph capacity")
    if any(slot < 0 for slot in (*reads, *writes)):
        raise ValueError("live requests cannot use padding state slots")
    if len(set(writes)) != len(writes):
        raise ValueError("verify requests cannot share a writable state slot")
    # A prefix can be read by several branches, but no branch may overwrite a
    # sibling's input while verification is in flight. Only its own in-place
    # update is legal; shared-prefix outputs must be disjoint COW slots.
    read_counts = Counter(reads)
    for slot_in, slot_out in zip(reads, writes):
        if read_counts[slot_out] > int(slot_in == slot_out):
            raise ValueError("verify output cannot overwrite another request's input")
    segments = []
    start = 0
    for length, slot_in, slot_out in zip(lens, reads, writes):
        stop = start + length
        anchor = pos[start]
        if anchor < 0 or any(
            pos[row] != anchor + row - start for row in range(start, stop)
        ):
            raise ValueError("request positions must be a contiguous anchor window")
        segments.append(VerifySegment(start, stop, anchor, slot_in, slot_out))
        start = stop
    return VerifyPlan(tuple(segments), capacity)


def build_verify_plan_from_offsets(
    offsets: Sequence[int],
    positions: Sequence[int],
    slots_in: Sequence[int],
    slots_out: Sequence[int],
    *,
    padded_rows: int,
    max_seqlen_q: int,
) -> VerifyPlan:
    """Read the native live-request cu_seqlens_q mirror, excluding its flat tail.

    Callers slice the mirror to scheduled_bs+1 before calling. Requiring its
    length to match live state slots prevents graph padding from becoming a
    zero-length request or hiding a missing live segment.
    """
    cu = tuple(index(value) for value in offsets)
    if not cu or cu[0] != 0 or len(cu) != len(slots_in) + 1:
        raise ValueError(
            "live verify offsets must start at zero and match request count"
        )
    return build_verify_plan(
        tuple(stop - start for start, stop in pairwise(cu)),
        positions,
        slots_in,
        slots_out,
        padded_rows=padded_rows,
        max_seqlen_q=max_seqlen_q,
    )
