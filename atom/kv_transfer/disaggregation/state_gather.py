# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pack one KDA state slot's TP shard into a single contiguous image.

A stage sends every KDA layer of one request to one decode rank together.
DCP does not divide the state: each DCP rank of a TP rank receives the same
shard. The producer's conv state interleaves q, k and v, so the shard is many
short source ranges; the consumer stores that shard as one contiguous unit
per region. ``plan_state_shard`` lists those source ranges in destination
order and one segment per region, which is what a staged direct write sends
and what a landing READY asks the decode to scatter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# One PP stage of Kimi-K3.1 sends a decode rank 7–9 MiB. 12 MiB holds that
# with room for the cursor; a larger shard (decode TP == prefill TP) stays on
# the direct descriptor path.
STATE_SLOT_BYTES = 12 << 20
STATE_POOL_BYTES = 256 << 20
KDA_SLOT_LAYOUT = "kda_slot_v1"


@dataclass(frozen=True)
class StateShard:
    """Source pieces in destination order, and one segment per region.

    ``pieces`` are ``(producer_region, unit_offset, length, packed_offset)``:
    bytes of this request's source slot. ``segments`` are
    ``(consumer_region, packed_offset, dst_addr, length)``: one contiguous
    destination range per region.
    """

    pieces: tuple[tuple[int, int, int, int], ...]
    segments: tuple[tuple[int, int, int, int], ...]
    total: int


def plan_state_shard(
    *,
    src_units: Sequence[int],
    dst_bases: Sequence[int],
    dst_units: Sequence[int],
    tp_segments: Sequence[tuple[int, ...] | None],
    tp_replicated: Sequence[bool] | None,
    slot_map: Sequence[int],
    tp_ratio: int,
    tp_part: int,
    dst_slot: int,
) -> StateShard:
    """The bytes one decode rank owns, packed the way its slot stores them."""

    pieces: list[tuple[int, int, int, int]] = []
    segments: list[tuple[int, int, int, int]] = []
    packed = 0
    replicated = tp_replicated or ()
    for region, bps in enumerate(src_units):
        consumer = slot_map[region]
        dst_addr = dst_bases[consumer] + dst_slot * dst_units[consumer]
        start = packed
        whole = tp_ratio == 1 or (region < len(replicated) and replicated[region])
        if whole:
            pieces.append((region, 0, bps, packed))
            packed += bps
            segments.append((consumer, start, dst_addr, bps))
            continue
        segments_of = tp_segments[region]
        if not segments_of:
            raise ValueError(f"slot region {region} has no TP segments")
        src_off = 0
        for seg in segments_of:
            piece = seg // tp_ratio
            pieces.append((region, src_off + tp_part * piece, piece, packed))
            packed += piece
            src_off += seg
        segments.append((consumer, start, dst_addr, packed - start))
    return StateShard(tuple(pieces), tuple(segments), packed)


def copy_state_pieces(staging, views, src_slot: int, pieces) -> None:
    """Copy ``pieces`` of source slot ``src_slot`` into ``staging``.

    ``views[region][src_slot]`` is that slot's unit. A torch destination is
    copied on the current stream; a bytearray-like one is sliced in place.
    """

    for region, unit_off, length, packed, count, stride in _runs(pieces):
        unit = views[region][src_slot]
        if hasattr(staging, "copy_") and count > 1:
            # One strided copy for a run of equal pieces at a fixed pitch
            # (a conv row's q, k and v), instead of one launch per piece.
            src = unit.as_strided(
                (count, length), (stride, 1), unit.storage_offset() + unit_off
            )
            staging[packed : packed + count * length].view(count, length).copy_(src)
            continue
        for i in range(count):
            src = unit[unit_off + i * stride : unit_off + i * stride + length]
            dest = staging[packed + i * length : packed + (i + 1) * length]
            if hasattr(dest, "copy_"):
                dest.copy_(src)
            else:
                dest[:] = src


def _runs(pieces):
    """Group consecutive pieces of one region that share a length and pitch.

    Yields ``(region, unit_offset, length, packed_offset, count, pitch)``.
    """

    run = None
    for region, off, length, packed in pieces:
        if run is not None:
            r_region, r_off, r_len, r_packed, r_count, r_pitch = run
            if (
                r_region == region
                and r_len == length
                and packed == r_packed + r_count * r_len
            ):
                pitch = off - r_off if r_count == 1 else r_pitch
                if pitch > 0 and off == r_off + r_count * pitch:
                    run = (r_region, r_off, r_len, r_packed, r_count + 1, pitch)
                    continue
            yield run
        run = (region, off, length, packed, 1, 0)
    if run is not None:
        yield run
