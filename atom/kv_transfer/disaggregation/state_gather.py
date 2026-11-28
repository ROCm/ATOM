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

import numpy as np

# One PP stage of Kimi-K3.1 sends a decode rank 7–9 MiB. 12 MiB holds that
# with room for the cursor; a larger shard (decode TP == prefill TP) stays on
# the direct descriptor path.
STATE_SLOT_BYTES = 12 << 20
STATE_POOL_BYTES = 256 << 20
KDA_SLOT_LAYOUT = "kda_slot_v1"
# Bytes one GPU program of the one-launch gather copies: a 768 KiB SSM shard
# spreads over 12 programs instead of one warp walking it alone.
GATHER_SEGMENT_BYTES = 64 << 10


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

    ``views[region][src_slot]`` is that slot's unit. A CUDA destination is
    gathered by one kernel launch on the current stream over a segment table
    cached per layout (per-copy launches serialize across send workers);
    other torch tensors are copied, and a bytearray-like one is sliced in
    place.
    """
    if not _gather_on_gpu(staging, views, src_slot, pieces):
        _copy_runs(staging, views, src_slot, pieces)


def _copy_runs(staging, views, src_slot: int, pieces) -> None:
    """One copy per run of equal pieces (``_runs``); works on any backing."""
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


def gather_segments(
    src_addrs: Sequence[int], pieces, max_bytes: int = GATHER_SEGMENT_BYTES
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(region, src, dst, nbytes)`` byte segments that perform ``pieces``.

    ``src_addrs[region]`` is the address of the source slot's unit in that
    region; ``dst`` is the packed offset. Pieces longer than ``max_bytes`` are
    split, so one long piece does not keep a single GPU program busy.
    """
    p = np.asarray(pieces, dtype=np.int64).reshape(-1, 4)
    parts = -(-p[:, 2] // max_bytes)
    item = np.repeat(np.arange(len(p)), parts)
    off = (
        np.arange(item.size) - np.repeat(np.cumsum(parts) - parts, parts)
    ) * max_bytes
    region = p[item, 0]
    src = np.asarray(src_addrs, dtype=np.int64)[region] + p[item, 1] + off
    return region, src, p[item, 3] + off, np.minimum(max_bytes, p[item, 2] - off)


try:  # Triton is only needed on GPU; unit tests may import without it.
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["slot"])
    def _gather_slot_kernel(
        src, dst, seg_src, seg_dst, seg_len, seg_pitch, slot, BLOCK: tl.constexpr
    ):
        """Copy segment ``pid`` of source slot ``slot``, all in int32 words."""
        s = tl.program_id(0)
        so = tl.load(seg_src + s) + slot * tl.load(seg_pitch + s)
        do = tl.load(seg_dst + s)
        n = tl.load(seg_len + s)
        offs = tl.arange(0, BLOCK)
        for off in range(0, n, BLOCK):
            o = off + offs
            m = o < n
            tl.store(dst + do + o, tl.load(src + so + o, mask=m), mask=m)

except ImportError:  # pragma: no cover - exercised only without Triton
    _gather_slot_kernel = None

# Device segment tables by (pieces, region bases and pitches). A request's
# pieces depend only on the decode rank's TP part, so a stage keeps a few.
# Entries are never dropped: a worker may still be reading one on its stream.
_GATHER_TABLES: dict = {}
_GATHER_TABLES_MAX = 64


def _gather_table(views, pieces, device):
    """``(origin, table, count)`` for ``pieces`` of slot 0, or None if unaligned.

    ``table`` rows are int32-word source offset from ``origin``, destination
    offset, length and the region's slot pitch, so one table serves every
    source slot.
    """
    import torch

    from atom.kv_transfer.disaggregation.landing_scatter import (
        segment_table_capacity,
    )

    used = {piece[0] for piece in pieces}
    bases = [v.data_ptr() if r in used else 0 for r, v in enumerate(views)]
    pitches = [v.stride(0) * v.element_size() for v in views]
    key = (tuple(pieces), tuple(bases), tuple(pitches))
    hit = _GATHER_TABLES.get(key)
    if hit is not None:
        return hit
    origin = min(bases[r] for r in used)
    region, src, dst, nbytes = gather_segments(bases, pieces)
    pitch = np.asarray(pitches, dtype=np.int64)[region]
    src -= origin
    entry = None
    if not (origin % 4 or ((src | dst | nbytes | pitch) % 4).any()):
        table = np.zeros((4, segment_table_capacity(src.size)), dtype=np.int64)
        table[:, : src.size] = np.stack([src, dst, nbytes, pitch]) // 4
        side = torch.cuda.Stream(device)
        with torch.cuda.stream(side):
            dev = torch.from_numpy(table).to(device)
        side.synchronize()  # Workers read it from their own streams.
        entry = (origin, dev, src.size)
    if len(_GATHER_TABLES) < _GATHER_TABLES_MAX:
        _GATHER_TABLES[key] = entry
    return entry


def _gather_on_gpu(staging, views, src_slot: int, pieces) -> bool:
    """Gather ``pieces`` with one launch over a cached table; False if it cannot.

    The regions live in different tensors, so sources are addressed as int32
    offsets from the lowest region base, as the landing scatter does for its
    destinations. Unaligned bytes or no Triton keep the per-copy path.
    """
    # numpy arrays also have a ``device``, the string "cpu".
    device = getattr(staging, "device", None)
    if (
        getattr(device, "type", None) != "cuda"
        or not pieces
        or _gather_slot_kernel is None
        or (staging.data_ptr() | staging.numel()) % 4
    ):
        return False
    entry = _gather_table(views, pieces, device)
    if entry is None:
        return False
    import torch

    from atom.kv_transfer.disaggregation.landing_scatter import DevicePointer

    origin, table, count = entry
    _gather_slot_kernel[(count,)](
        DevicePointer(origin, device),
        staging.view(torch.int32),
        table[0],
        table[1],
        table[2],
        table[3],
        src_slot,
        BLOCK=256,
        num_warps=1,
    )
    return True


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
