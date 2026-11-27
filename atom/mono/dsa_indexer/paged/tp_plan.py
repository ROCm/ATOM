# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU/device arithmetic for replicated-cache TP indexer work and mailboxes.

Ranks own requests when there is enough independent work, split long requests
when peers would otherwise idle, and retain local short-step execution. These
are the K3 DSA planner's ownership decisions. Split ranges are contiguous and
chunk-aligned so rank/local ordering also preserves V4.1's global-index ties.
"""

from dataclasses import dataclass

from atom.mono.dsa_indexer.paged import plan as ip
from atom.mono.dsa_indexer.selection_plan import (
    DEFAULT_RADIX,
    ROW_COPIES,
    SHORT_VERIFY_CTX,
    SPLIT_MIN,
    split_request_keys,
)
from atom.mono.plan.arith import sel
from atom.mono.plan.layout import byte_layout, pair_layout

LOCAL, OWNED, SPLIT = 0, 1, 2
KINDS = ("sel", "blk")


def kernel_world(world, rows, capacity):
    """Use the measured local schedule for small bounded decode buckets.

    At <=8 rows and <=256K capacity, peer selection costs more than the
    replicated local scorer. Wider buckets and longer deployments retain TP
    ownership/K splitting. The decision uses TP-uniform build parameters.
    """
    return 1 if world > 1 and rows <= 8 and capacity <= SPLIT_MIN else world


def work_kind(request_bound, total, requests, world, local_step):
    """The same request-level decision on every rank, including idle ranks."""
    split = sel(
        requests < world,
        split_request_keys(request_bound, total, world, low_requests=True),
        split_request_keys(request_bound, total, world, low_requests=False),
    )
    return sel(local_step | (world == 1), LOCAL, sel(split, SPLIT, OWNED))


def key_window(bound, request_bound, request, world, rank, kind):
    """(first global column, local columns), shared by a request's query rows.

    The last query's bound determines a chunk-aligned partition; earlier
    causal rows clip it. Keeping a whole eight-row block on one rank lets the
    same partition serve the candidate-block maxima without cross-rank max.
    """
    span = ip.cdiv(ip.chunks(request_bound), world) * ip.CHUNK
    first = sel(kind == SPLIT, rank * span, 0)
    count = ip.imax(ip.imin(bound - first, span), 0)
    owns = (kind == LOCAL) | (request % world == rank)
    count = sel(kind == SPLIT, count, sel(owns, ip.imax(bound, 0), 0))
    return first, count


def needs_scores(global_bound, count):
    """A small local tail still needs scores when the global row needs TopK."""
    return ip.scored(global_bound) & (count > 0)


def row_rank_at(kind, row, rank, rows, world):
    return (kind * rows + row) * world + rank


def peer_layout(rows, world):
    return pair_layout(
        [(name, len(KINDS) * rows * world) for name in ("ix_count", "ix_done")]
        + [
            (
                "ix_hist",
                len(KINDS) * rows * world * DEFAULT_RADIX.passes * DEFAULT_RADIX.bins,
            )
        ]
    )


def plain_layout(rows):
    return byte_layout(
        [(f"ix_{kind}_pairs", rows * ip.KINDS[kind] * 8) for kind in KINDS]
    )


def layout_bytes(layout):
    return max(off + size for off, size in layout.values())


@dataclass(frozen=True)
class RowWork:
    kind: int
    owner: int
    first: int
    count: int
    scored: bool


def reference_work(bounds, batch_ids, world, rank, *, force_local=False):
    """CPU qualification of ragged request groups and empty padding rows.

    Device metadata derives the same request ordinal from contiguous groups.
    Refuse a repeated noncontiguous ID here rather than qualifying a layout
    that the native request-group walk does not describe.
    """
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError("unsupported TP group")
    if len(bounds) != len(batch_ids) or not 0 < len(bounds) <= 64:
        raise ValueError("unsupported query rows")
    requests, row_request, previous = [], [], None
    seen = set()
    for bound, batch in zip(bounds, batch_ids):
        if bound < 0:
            raise ValueError("negative bound")
        if batch < 0:
            row_request.append(-1)
            previous = None
            continue
        if batch != previous:
            if batch in seen:
                raise ValueError("request rows must be contiguous")
            seen.add(batch)
            requests.append([bound, bound])
        request = len(requests) - 1
        requests[request][1] = max(requests[request][1], bound)
        row_request.append(request)
        previous = batch
    total = sum(last for _, last in requests)
    local = force_local or all(first <= SHORT_VERIFY_CTX for first, _ in requests)
    work = []
    for bound, request in zip(bounds, row_request):
        if request < 0:
            work.append(RowWork(LOCAL, -1, 0, 0, False))
            continue
        last = requests[request][1]
        kind = work_kind(last, total, len(requests), world, local)
        first, count = key_window(bound, last, request, world, rank, kind)
        work.append(
            RowWork(kind, request % world, first, count, needs_scores(bound, count))
        )
    return work


def row_hist_words(rows):
    return len(KINDS) * rows * DEFAULT_RADIX.passes * ROW_COPIES * DEFAULT_RADIX.bins
