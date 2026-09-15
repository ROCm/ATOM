# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton fused chunk-major staging for ATOM LMCache offload."""

from __future__ import annotations

import logging
import os
import threading
import time

import torch
import triton
import triton.language as tl

logger = logging.getLogger("atom")


def _probe_enabled() -> bool:
    raw = os.environ.get("OFFLOAD_STAGING_PROBE", "0").strip().lower()
    return bool(raw) and raw not in ("0", "false", "no", "off")


# Diagnostic only, and off unless the operator asks for it. py-spy puts ~46% of
# the save thread and ~56% of the load thread inside the event wait in
# ``_prepare``. Two mechanisms produce that sample count and they call for
# opposite fixes: the event is genuinely incomplete (the host outran the GPU by
# the slot count, so more slots help), or the event completed long ago and the
# cost is the GIL round trip a blocking runtime call pays to get the
# interpreter back (so removing the block helps and more slots do nothing).
# ``query()`` before the wait separates them: it reports completion without
# blocking, so a high ready fraction next to an expensive wait is the second
# mechanism and a low one is the first.
_STAGING_PROBE = _probe_enabled()
_PROBE_EVERY = 1000


class _ProbeAcc:
    __slots__ = ("copy_ms", "fill_ms", "n", "query_ms", "ready", "record_ms", "sync_ms")

    def __init__(self) -> None:
        self.n = 0
        self.ready = 0
        self.query_ms = 0.0
        self.sync_ms = 0.0
        self.fill_ms = 0.0
        self.copy_ms = 0.0
        self.record_ms = 0.0

    def reset(self) -> None:
        self.__init__()


_probe_state = threading.local()


def _probe_acc() -> _ProbeAcc:
    acc = getattr(_probe_state, "acc", None)
    if acc is None:
        acc = _ProbeAcc()
        _probe_state.acc = acc
    return acc


def _probe_report(acc: _ProbeAcc) -> None:
    n = acc.n
    logger.info(
        "[STAGING-PROBE] thread=%s n=%d ready=%.2f%% query=%.4f sync=%.4f "
        "fill=%.4f copy=%.4f record=%.4f (ms/call)",
        threading.current_thread().name,
        n,
        100.0 * acc.ready / n,
        acc.query_ms / n,
        acc.sync_ms / n,
        acc.fill_ms / n,
        acc.copy_ms / n,
        acc.record_ms / n,
    )
    acc.reset()


_BLOCK_BYTES = 1024
# 1024 bytes over two wavefronts is 8 bytes per lane. The tile used to be
# launched with eight warps, which the rectangular grid hid -- most programs
# had nothing to move, so what the ones that did cost per lane barely showed.
# Sized by the work, the shape is the whole cost: a sweep of tile x warps over
# the measured M3 geometry peaks flat along 8 bytes per lane (643/634/632 GB/s
# at 512x1, 1024x2, 2048x4) and falls off either side of it -- 448 GB/s at the
# eight warps this used to run, 45 GB/s at 8192x1. Two warps keeps the tile
# where every other part of this file already assumes it.
_NUM_WARPS = 2


@triton.jit
def _pack_chunk_major_kernel(
    device_buf,
    segment_ptrs,
    segment_block_bytes,
    segment_prefix_bytes,
    chunk_block_counts,
    chunk_block_offsets,
    chunk_output_bases,
    block_ids,
    tile_job,
    tile_pos,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.load(tile_job + pid)
    tile = tl.load(tile_pos + pid)
    chunk_id = job // NUM_SEGMENTS
    seg_id = job - chunk_id * NUM_SEGMENTS

    nblocks = tl.load(chunk_block_counts + chunk_id).to(tl.int64)
    seg_bytes = tl.load(segment_block_bytes + seg_id).to(tl.int64)
    nbytes = nblocks * seg_bytes
    offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
    mask = offsets < nbytes

    local_block = offsets // seg_bytes
    byte_in_block = offsets - local_block * seg_bytes
    block_offset = tl.load(chunk_block_offsets + chunk_id).to(tl.int64)
    physical_block = tl.load(
        block_ids + block_offset + local_block,
        mask=mask,
        other=0,
    ).to(tl.int64)

    seg_addr = tl.load(segment_ptrs + seg_id)
    src = (seg_addr + physical_block * seg_bytes + byte_in_block).to(
        tl.pointer_type(tl.uint8)
    )
    dst = (
        device_buf
        + tl.load(chunk_output_bases + chunk_id).to(tl.int64)
        + tl.load(segment_prefix_bytes + seg_id).to(tl.int64) * nblocks
        + offsets
    )
    data = tl.load(src, mask=mask)
    tl.store(dst, data, mask=mask)


@triton.jit
def _unpack_chunk_major_kernel(
    device_buf,
    segment_ptrs,
    segment_block_bytes,
    segment_prefix_bytes,
    chunk_block_counts,
    chunk_block_offsets,
    chunk_output_bases,
    block_ids,
    tile_job,
    tile_pos,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.load(tile_job + pid)
    tile = tl.load(tile_pos + pid)
    chunk_id = job // NUM_SEGMENTS
    seg_id = job - chunk_id * NUM_SEGMENTS

    nblocks = tl.load(chunk_block_counts + chunk_id).to(tl.int64)
    seg_bytes = tl.load(segment_block_bytes + seg_id).to(tl.int64)
    nbytes = nblocks * seg_bytes
    offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
    mask = offsets < nbytes

    local_block = offsets // seg_bytes
    byte_in_block = offsets - local_block * seg_bytes
    block_offset = tl.load(chunk_block_offsets + chunk_id).to(tl.int64)
    physical_block = tl.load(
        block_ids + block_offset + local_block,
        mask=mask,
        other=0,
    ).to(tl.int64)

    src = (
        device_buf
        + tl.load(chunk_output_bases + chunk_id).to(tl.int64)
        + tl.load(segment_prefix_bytes + seg_id).to(tl.int64) * nblocks
        + offsets
    )
    seg_addr = tl.load(segment_ptrs + seg_id)
    dst = (seg_addr + physical_block * seg_bytes + byte_in_block).to(
        tl.pointer_type(tl.uint8)
    )
    data = tl.load(src, mask=mask)
    tl.store(dst, data, mask=mask)


def _device_i64(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _validate_device_buf(device_buf: torch.Tensor) -> None:
    if not device_buf.is_cuda:
        raise ValueError("device_buf must be a CUDA/HIP tensor")
    if device_buf.dtype != torch.uint8:
        raise TypeError("device_buf must be uint8")
    if not device_buf.is_contiguous():
        raise ValueError("device_buf must be contiguous")


def _build_meta(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    _validate_device_buf(device_buf)
    if len(segment_tensors) != len(segment_block_bytes):
        raise ValueError("segment_tensors and segment_block_bytes size mismatch")
    if not segment_tensors:
        raise ValueError("at least one segment is required")

    device = device_buf.device
    segment_ptr_values: list[int] = []
    segment_prefix_values: list[int] = []
    bytes_per_block = 0
    for seg, nb in zip(segment_tensors, segment_block_bytes, strict=True):
        if not seg.is_cuda:
            raise ValueError("segment tensor must be CUDA/HIP")
        if seg.device != device:
            raise ValueError("segment/device mismatch")
        if not seg.is_contiguous():
            raise ValueError("segment tensor must be contiguous")
        nb = int(nb)
        if nb <= 0:
            raise ValueError("segment block bytes must be > 0")
        segment_ptr_values.append(int(seg.data_ptr()))
        segment_prefix_values.append(bytes_per_block)
        bytes_per_block += nb

    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    counts = [int(n) for n in chunk_block_counts]
    for nblocks in counts:
        if nblocks < 0:
            raise ValueError("chunk block count must be non-negative")
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += nblocks
        byte_offset += nblocks * bytes_per_block

    if len(block_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    if int(device_buf.numel()) < byte_offset:
        raise ValueError("device_buf is smaller than chunk-major staging output")

    tile_job, tile_pos = _tile_table(counts, segment_block_bytes, device)

    return (
        _device_i64(segment_ptr_values, device),
        _device_i64([int(x) for x in segment_block_bytes], device),
        _device_i64(segment_prefix_values, device),
        _device_i64(counts, device),
        _device_i64(chunk_block_offsets, device),
        _device_i64(chunk_output_bases, device),
        _device_i64([int(x) for x in block_ids], device),
        tile_job,
        tile_pos,
        torch.tensor([int(byte_offset)], dtype=torch.int64),
    )


def _tile_table(
    counts: list[int],
    segment_block_bytes,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One grid entry per tile that has bytes to move, and nothing else.

    The grid used to be rectangular: ``(chunk * segment, tiles)``, where the
    tile count came from the largest segment. Segments are not the same size --
    M3 stages 16 KiB of K and of V per block next to 512-byte MXFP8 scales and
    one much larger cache -- so every small segment was launched with the tile
    count of the biggest one and masked off almost all of it. On the measured
    geometry that is 1,847,024 programs to move 23 MiB, about 12 payload bytes
    each, and it costs what it sounds like: 15.8 GB/s against 276.6 GB/s for
    the same bytes in segments of one size.

    So size the grid by the work instead. Each job's tile count is its own,
    the table maps a flat program id back to (job, tile within job), and the
    result depends on the total bytes rather than on the widest segment. The
    table is a function of the geometry alone, so it is built once per plan and
    cached with it; it costs two int32 entries per tile, tens of KiB here.
    """
    nseg = len(segment_block_bytes)
    job_nbytes = torch.as_tensor(counts, dtype=torch.int64).repeat_interleave(
        nseg
    ) * torch.as_tensor(
        [int(nb) for nb in segment_block_bytes], dtype=torch.int64
    ).repeat(
        len(counts)
    )
    tiles = (job_nbytes + _BLOCK_BYTES - 1) // _BLOCK_BYTES
    jobs = torch.repeat_interleave(torch.arange(tiles.numel()), tiles)
    starts = torch.cumsum(tiles, 0) - tiles
    pos = torch.arange(int(tiles.sum())) - torch.repeat_interleave(starts, tiles)
    return (
        jobs.to(device=device, dtype=torch.int32),
        pos.to(device=device, dtype=torch.int32),
    )


class _ChunkMajorPlan:
    """Reusable pack/unpack metadata for one segment geometry and chunk shape.

    ``_build_meta`` walks every segment in Python and issues seven blocking H2D
    copies.  Six of the seven tensors it returns depend only on the segment
    geometry -- fixed for the life of the KV cache -- and on
    ``chunk_block_counts``, which repeats across staging groups; only
    ``block_ids`` actually varies.  Rebuilding all seven per group put that work
    on the transfer's critical path once per group, and with the model
    co-resident it dominated the group: 24.33 ms of a 24.43 ms "pack" against
    0.10 ms for the Triton kernel itself.

    Building the six once and pushing ``block_ids`` through a reused pinned
    tensor takes one 256-chunk save from 9.5 GB/s to 21.9 GB/s under a
    co-resident model, and to 36.7 GB/s once the device-to-host leg is
    asynchronous too.
    """

    __slots__ = (
        "_block_ids_d",
        "_chunk_block_counts",
        "_chunk_block_offsets",
        "_chunk_output_bases",
        "_grid",
        "_host_events",
        "_host_slots",
        "_host_views",
        "_nblocks",
        "_num_segments",
        "_output_nbytes",
        "_segment_block_bytes",
        "_segment_prefix",
        "_segment_ptrs",
        "_slot",
        "_tile_job",
        "_tile_pos",
    )

    # Host staging slots for block_ids. One would race: the pinned buffer is
    # handed to an asynchronous H2D, and the next group's host-side write could
    # land while that copy is still in flight. Waiting on the copy instead would
    # drain everything queued ahead of it on the stream -- the previous group's
    # pack kernel -- so rotate slots and only wait on wrap.
    #
    # Four slots did not get that wait down to never, and the wrap was not
    # cheap. py-spy on a live unprofiled server put 46% of the save thread and
    # 56% of the load thread inside this one wait. Probed in-server
    # (``OFFLOAD_STAGING_PROBE=1``), the event was already complete on 71% of
    # save groups and 84% of load groups; the rest wrapped into live work, and
    # a wrap that blocks costs ~22 ms (save) / ~35 ms (load) against 0.158 ms
    # of GPU work per group.
    #
    # The gap is not the transfer. A blocking runtime call drops the GIL to
    # sleep, the model's threads take it, and this one waits out a switch
    # interval to get it back. The non-blocking calls on the same path --
    # ``Event.record``, the H2D itself -- also drop the GIL and cost tens of
    # microseconds, so the expense is sleeping, not yielding.
    #
    # Depth therefore has to cover a burst, not a couple of groups: a host
    # issuing ~0.1 ms per group against a GPU draining ~0.158 ms cannot stay
    # ahead for long, but it only has to be ahead once per wrap to pay. Swept
    # end to end on GLM-5.2 MXFP4 (pool 64, 600 s, one seed, no instrument),
    # against a 1.23% cross-sweep spread:
    #
    #     slots   output tok/s   p50 TTFT   ITL
    #         4         353.77   1425.71   18.87
    #        32         383.15    742.31   18.00
    #       128         393.50    672.32   17.58
    #
    # Monotone on all three, so the knee is past 128 rather than at it; 128 is
    # where the marginal gain stopped being worth another arm, not a measured
    # optimum. It is cheap to sit here: a slot is nblocks*8 pinned bytes, so a
    # ring is single-digit KiB per cached plan.
    _NUM_HOST_SLOTS = 128

    def __init__(self, meta, num_segments: int, nblocks: int) -> None:
        (
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            self._tile_job,
            self._tile_pos,
            sizes,
        ) = meta
        self._num_segments = num_segments
        self._nblocks = nblocks
        self._output_nbytes = int(sizes[0].item())
        self._grid = (int(self._tile_job.numel()),)
        self._host_slots = [
            torch.empty(nblocks, dtype=torch.int64, pin_memory=True)
            for _ in range(self._NUM_HOST_SLOTS)
        ]
        # Numpy views over the same pinned storage, built once: ``.numpy()``
        # allocates a fresh wrapper per call and this is per staging group.
        self._host_views = [t.numpy() for t in self._host_slots]
        self._host_events = [torch.cuda.Event() for _ in range(self._NUM_HOST_SLOTS)]
        for event in self._host_events:
            event.record()
        self._slot = 0

    def _prepare(self, block_ids, device_buf: torch.Tensor) -> bool:
        # The cached metadata skips _build_meta, so device_buf still has to be
        # checked here; it is the one argument a caller can vary per call.
        _validate_device_buf(device_buf)
        if int(device_buf.numel()) < self._output_nbytes:
            raise ValueError("device_buf is smaller than chunk-major staging output")
        if len(block_ids) != self._nblocks:
            raise ValueError("block_ids length does not match chunk block counts")
        if self._output_nbytes == 0:
            return False
        slot = self._slot
        self._slot = (slot + 1) % self._NUM_HOST_SLOTS
        if _STAGING_PROBE:
            self._prepare_probed(slot, block_ids)
            return True
        self._host_events[slot].synchronize()
        # Assigning the sequence into the pinned buffer's numpy view converts it
        # in C, straight into the destination. ``torch.as_tensor(list_of_int)``
        # would unbox every element through the CPython API while holding the
        # GIL, and then copy the result -- the same cost tokens_to_tensor exists
        # to avoid, on a path that runs once per staging group.
        self._host_views[slot][:] = block_ids
        self._block_ids_d.copy_(self._host_slots[slot], non_blocking=True)
        self._host_events[slot].record()
        return True

    def _prepare_probed(self, slot: int, block_ids) -> None:
        """``_prepare``'s tail, with each runtime call timed separately.

        Same calls in the same order plus a non-blocking ``query()``, so the
        per-call numbers are comparable to each other; the total is not
        comparable to an unprobed run.
        """
        acc = _probe_acc()
        event = self._host_events[slot]
        t0 = time.perf_counter()
        ready = event.query()
        t1 = time.perf_counter()
        event.synchronize()
        t2 = time.perf_counter()
        self._host_views[slot][:] = block_ids
        t3 = time.perf_counter()
        self._block_ids_d.copy_(self._host_slots[slot], non_blocking=True)
        t4 = time.perf_counter()
        event.record()
        t5 = time.perf_counter()
        acc.n += 1
        acc.ready += 1 if ready else 0
        acc.query_ms += (t1 - t0) * 1000.0
        acc.sync_ms += (t2 - t1) * 1000.0
        acc.fill_ms += (t3 - t2) * 1000.0
        acc.copy_ms += (t4 - t3) * 1000.0
        acc.record_ms += (t5 - t4) * 1000.0
        if acc.n >= _PROBE_EVERY:
            _probe_report(acc)

    def pack(self, block_ids, device_buf: torch.Tensor) -> None:
        if not self._prepare(block_ids, device_buf):
            return
        _pack_chunk_major_kernel[self._grid](
            device_buf,
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            self._tile_job,
            self._tile_pos,
            NUM_SEGMENTS=self._num_segments,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=_NUM_WARPS,
        )

    def unpack(self, block_ids, device_buf: torch.Tensor) -> None:
        if not self._prepare(block_ids, device_buf):
            return
        _unpack_chunk_major_kernel[self._grid](
            device_buf,
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            self._tile_job,
            self._tile_pos,
            NUM_SEGMENTS=self._num_segments,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=_NUM_WARPS,
        )


# Plans are keyed by the segment geometry and the chunk shape -- everything
# _build_meta derives them from except block_ids. Only a couple of shapes occur
# in practice -- a full staging group plus whatever tail a store leaves -- but
# cap the cache anyway so an unusual traffic pattern cannot grow it without
# bound.
_PLAN_CACHE: dict[tuple, _ChunkMajorPlan] = {}
_MAX_PLANS = 64


def _plan_key(segment_tensors, counts) -> tuple | None:
    """Identify one segment geometry and chunk shape, in constant time.

    Naming every segment costs one ``data_ptr`` and one ``int`` per segment,
    and at M3's 180 segments that is 360 Python-level conversions on every
    staging group. It is also pure overhead: the codec builds its segment list
    once in ``__init__`` and never mutates it. Measured at that geometry, the
    full key costs 0.019 ms per group with the GIL idle and 0.277 ms with it
    contended -- which is the state offload actually runs in -- against
    0.003 ms for this.

    Live tensors have distinct addresses, so the segment count together with
    the first, middle and last segment pointers names one live segment list.
    Two lists could only collide by sharing those three tensors and their
    length while differing in between, which the codec cannot produce: it
    derives the whole list from a single kv_caches mapping. A reallocated KV
    cache misses, so a stale plan is never used to pack from freed storage.

    Returns None for an empty segment list, leaving ``_build_meta`` to raise.
    """
    n = len(segment_tensors)
    if n == 0:
        return None
    return (
        n,
        segment_tensors[0].data_ptr(),
        segment_tensors[n // 2].data_ptr(),
        segment_tensors[-1].data_ptr(),
        counts,
    )


def _get_plan(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    device_buf: torch.Tensor,
) -> _ChunkMajorPlan:
    counts = tuple(int(n) for n in chunk_block_counts)
    key = _plan_key(segment_tensors, counts)
    plan = _PLAN_CACHE.get(key) if key is not None else None
    if plan is None:
        # block_ids is only length-checked by _build_meta, so a placeholder of
        # the right length is enough to get every other tensor built and every
        # argument validated; the tensor it returns becomes the plan's device
        # landing buffer for block_ids.
        nblocks = sum(counts)
        meta = _build_meta(
            segment_tensors,
            segment_block_bytes,
            counts,
            [0] * nblocks,
            device_buf,
        )
        if len(_PLAN_CACHE) >= _MAX_PLANS:
            _PLAN_CACHE.clear()
        plan = _ChunkMajorPlan(meta, len(segment_tensors), nblocks)
        _PLAN_CACHE[key] = plan
    return plan


def fused_pack_chunk_major(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf,
) -> None:
    _get_plan(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        device_buf,
    ).pack(block_ids, device_buf)


def fused_unpack_chunk_major(
    device_buf,
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
) -> None:
    _get_plan(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        device_buf,
    ).unpack(block_ids, device_buf)
