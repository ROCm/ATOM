# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoRIIO over the GPU scale-up fabric (UALink): ``moriio_backend`` ``fabric``.

- Both ranks back their paged KV pool with fabric-exportable VMM memory from
  mori's allocator (``mori.io.fabric_alloc``), through
  ``KVConnectorBase.allocate_kv_pool``, so registering it publishes a fabric
  handle that the peer imports and maps.
- The connector registers the pool once, as one MoRIIO buffer: mori imports and
  maps a peer's whole allocation per registered descriptor, so registering each
  PAGE region on its own would map the pool once per region.
- Every PAGE region is addressed as (offset in the pool, bytes per block), and a
  request moves as one ``batch_write`` on the single (local pool, remote pool)
  session. With ROCm/mori#1021 mori copies up to 512 segments per launch of its
  fabric copy kernel and runs them concurrently, so the destinations of one
  batch must not overlap: no block may be written twice.
- A region sharded by block id (`KVTransferRegion.shard_units`: MiniMax-M3's
  index cache under ``ATOM_M3_INDEX_SHARD``) holds 1/n of the blocks on each
  rank; rank i pushes only its own (`shard_pairs`) into rank i, whose share is
  the same blocks because both ends allocate a logical block in the same lane.

The pure helpers here (`page_table`, `page_batch`) take plain numbers, so the
region checks and the batch plan run without mori or a GPU.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import time
from collections.abc import Sequence
from typing import Any

import msgpack
import numpy as np
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import coalesce_contiguous

logger = logging.getLogger("atom")

# MemoryDesc's msgpack array (mori include/mori/io/common.hpp MSGPACK_DEFINE),
# the same positions before and after ROCm/mori#1021 (which appends a field).
_DESC_SIZE = 5
_DESC_FABRIC_HANDLE = 10
_DESC_VPOD_ID = 11


def is_fabric_backend(kv_transfer_config: dict) -> bool:
    """Whether ``moriio_backend`` is ``fabric``; raises on a value that is
    neither that nor ``rdma`` (the default), so a typo does not quietly run
    RDMA."""
    value = kv_transfer_config.get("moriio_backend", "rdma")
    backend = str(value).strip().lower()
    if backend not in ("rdma", "fabric"):
        raise ValueError(f"moriio_backend must be 'rdma' or 'fabric', got {value!r}")
    return backend == "fabric"


class _FabricBuffer:
    """Owns one allocation; torch keeps it alive for as long as the tensor."""

    def __init__(self, ptr: int, nbytes: int):
        self._ptr = ptr
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            from mori.io import fabric_free

            fabric_free(self._ptr)


# The pool this process allocated, as (base, nbytes).
_pool_range: tuple[int, int] | None = None


def allocate_fabric_pool(nbytes: int, device: Any) -> torch.Tensor:
    """A zeroed uint8 tensor of ``nbytes`` in fabric-exportable memory."""
    global _pool_range
    from mori.io import fabric_alloc

    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    started = time.perf_counter()
    with torch.cuda.device(index):
        ptr = fabric_alloc(nbytes, index)
        if not ptr:
            raise RuntimeError(
                f"mori fabric_alloc failed for {nbytes} B on device {index}: the "
                "device cannot create fabric-exportable memory"
            )
        tensor = torch.as_tensor(_FabricBuffer(ptr, nbytes), device=device)
        tensor.zero_()
    _pool_range = (ptr, nbytes)
    logger.info(
        "MoRIIO fabric: KV pool %.2f GiB at %#x on device %d (%.1f ms)",
        nbytes / (1 << 30),
        ptr,
        index,
        (time.perf_counter() - started) * 1e3,
    )
    return tensor


def page_table(
    regions: Sequence[Any],
    num_blocks: int,
    pool: tuple[int, int] | None = None,
) -> tuple[int, int, list[tuple[int, int, str, int, int]]]:
    """``(pool base, pool bytes, [(offset, bytes per block, role, shard rank,
    shard world)])`` of ``regions``.

    ``regions`` are the published PAGE regions (``KVTransferRegion``), ``pool``
    the fabric allocation they must live in, by default the one this process
    made (`allocate_fabric_pool`). Raises unless every region is forward
    indexed, holds exactly ``num_blocks`` blocks (its share of them, sharded),
    and lies inside the pool, and unless no two regions overlap: a region
    outside the pool has no fabric handle to reach it, and two overlapping ones
    would write the same bytes twice in one batch.
    """
    if pool is None:
        pool = _pool_range
    if pool is None:
        raise RuntimeError(
            "moriio_backend fabric: this process allocated no fabric KV pool; "
            "the paged pool must come from MoRIIOConnector.allocate_kv_pool"
        )
    base, size = pool
    entries = []
    for region in regions:
        role = region.semantic_role or ""
        if region.reverse_indexed:
            raise NotImplementedError(
                f"moriio_backend fabric addresses PAGE blocks forward only; "
                f"region {role!r} is reverse indexed"
            )
        world = int(region.shard_world)
        entries.append(
            (
                int(region.base_addr) - base,
                int(region.unit_bytes),
                role,
                int(region.shard_rank),
                world,
            )
        )
        if num_blocks % world or int(region.total_bytes) != (
            num_blocks // world * int(region.unit_bytes)
        ):
            raise ValueError(
                f"PAGE region {role!r} holds {region.total_bytes} B, not 1/{world} "
                f"of {num_blocks} blocks of {region.unit_bytes} B"
            )
    check_page_table(entries, num_blocks, size, "local")
    return base, size, entries


def check_page_table(
    entries: Sequence[tuple[int, int, str, int, int]],
    num_blocks: int,
    pool_bytes: int,
    side: str,
) -> None:
    """Raise unless every ``(offset, unit, role, shard rank, shard world)``
    region of ``num_blocks`` blocks (its share of them) lies inside
    ``pool_bytes`` and no two of them overlap."""
    if not entries:
        raise ValueError(f"{side} PAGE table is empty")
    if num_blocks <= 0:
        raise ValueError(f"{side} PAGE table has {num_blocks} blocks")
    spans = []
    for offset, unit, role, _rank, world in entries:
        end = offset + num_blocks // world * unit
        if unit <= 0 or offset < 0 or end > pool_bytes:
            raise ValueError(
                f"{side} PAGE region {role!r} [{offset}, {end}) of {num_blocks} "
                f"blocks of {unit} B is not inside its {pool_bytes} B fabric pool"
            )
        spans.append((offset, end, role))
    spans.sort()
    for (_, end, role), (start, _, next_role) in itertools.pairwise(spans):
        if start < end:
            raise ValueError(f"{side} PAGE regions {role!r} and {next_role!r} overlap")


def page_batch(
    local_regions: Sequence[tuple[int, int]],
    remote_regions: Sequence[tuple[int, int]],
    src_blocks: Sequence[int],
    dst_blocks: Sequence[int],
    local_num_blocks: int,
    remote_num_blocks: int,
) -> tuple[list[int], list[int], list[int]]:
    """Pool-relative ``(local offsets, remote offsets, sizes)`` of one push.

    Moves block ``src_blocks[i]`` of every local region, given as ``(offset in
    the pool, bytes per block)``, into block ``dst_blocks[i]`` of the remote
    region at the same position. Runs of blocks contiguous on both sides go as
    one segment per region. Raises when the tables do not pair, when a block
    id is outside its side's pool (an id past the end would address the next
    region), or when ``dst_blocks`` names a block twice (the copies of one
    fabric batch run concurrently).
    """
    if len(local_regions) != len(remote_regions):
        raise ValueError(
            f"{len(local_regions)} local PAGE regions against "
            f"{len(remote_regions)} remote ones"
        )
    src, dst = _checked_blocks(
        src_blocks, dst_blocks, local_num_blocks, remote_num_blocks
    )
    run_src, run_dst, run_len = coalesce_contiguous(src, dst, np.ones_like(src))
    local_offs, remote_offs, sizes = [], [], []
    for i, ((l_off, l_unit), (r_off, r_unit)) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if l_unit != r_unit:
            raise ValueError(
                f"PAGE region {i}: {l_unit} B per block here, {r_unit} B remote"
            )
        local_offs.append(l_off + run_src * l_unit)
        remote_offs.append(r_off + run_dst * l_unit)
        sizes.append(run_len * l_unit)
    if not sizes:
        return [], [], []
    return (
        np.concatenate(local_offs).tolist(),
        np.concatenate(remote_offs).tolist(),
        np.concatenate(sizes).tolist(),
    )


def _checked_blocks(
    src_blocks: Sequence[int],
    dst_blocks: Sequence[int],
    local_num_blocks: int,
    remote_num_blocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The block ids as arrays; raises unless they pair one to one, lie inside
    their pools, and name no destination twice."""
    if len(src_blocks) != len(dst_blocks):
        raise ValueError(
            f"{len(src_blocks)} source blocks for {len(dst_blocks)} destinations"
        )
    src = np.asarray(src_blocks, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst_blocks, dtype=np.int64).reshape(-1)
    for name, ids, limit in (
        ("source", src, local_num_blocks),
        ("destination", dst, remote_num_blocks),
    ):
        bad = ids[(ids < 0) | (ids >= limit)]
        if bad.size:
            raise ValueError(
                f"{name} blocks {bad[:8].tolist()} are outside its pool of "
                f"{limit} blocks"
            )
    if np.unique(dst).size != dst.size:
        values, counts = np.unique(dst, return_counts=True)
        raise ValueError(
            f"destination blocks {values[counts > 1][:8].tolist()} appear more "
            f"than once among the {dst.size} a fabric batch writes; its copies "
            "run concurrently, so they must not overlap"
        )
    return src, dst


def unit_batch(
    local_regions: Sequence[tuple[int, int]],
    remote_regions: Sequence[tuple[int, int]],
    targets: Sequence[int],
    cuts: Sequence[np.ndarray],
    src_blocks: Sequence[int],
    dst_blocks: Sequence[int],
    local_num_blocks: int,
    remote_num_blocks: int,
) -> tuple[list[int], list[int], list[int]]:
    """Pool-relative ``(local offsets, remote offsets, sizes)`` of a push paired
    by unit rather than by position.

    Local region ``i`` lands in remote region ``targets[i]``; ``cuts[i]`` is
    its ``[src offset, dst offset, bytes]`` rows within one block
    (`consumer_head_slices`), which must fill the remote block in order, so
    every byte of a destination block is written exactly once. A PP stage
    holds some of the consumer's regions and a TP1 stage cuts each consumer
    rank its KV heads. Segments contiguous on both sides are merged; the same
    block checks as `page_batch` apply.
    """
    if not (len(local_regions) == len(targets) == len(cuts)):
        raise ValueError(
            f"{len(local_regions)} local PAGE regions, {len(targets)} targets, "
            f"{len(cuts)} cuts"
        )
    if len(set(targets)) != len(targets):
        raise ValueError("two local PAGE regions land in one remote region")
    src, dst = _checked_blocks(
        src_blocks, dst_blocks, local_num_blocks, remote_num_blocks
    )
    local_offs, remote_offs, sizes = [], [], []
    for i, ((l_off, l_unit), c, rows) in enumerate(zip(local_regions, targets, cuts)):
        if not 0 <= c < len(remote_regions):
            raise ValueError(
                f"PAGE region {i} lands in remote region {c} of "
                f"{len(remote_regions)}"
            )
        r_off, r_unit = remote_regions[c]
        rows = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        ends = np.cumsum(rows[:, 2])
        if (
            rows[:, 0].min() < 0
            or (rows[:, 0] + rows[:, 2]).max() > l_unit
            or not np.array_equal(rows[:, 1], ends - rows[:, 2])
            or int(ends[-1]) != r_unit
        ):
            raise ValueError(
                f"PAGE region {i}: rows {rows.tolist()[:4]} do not fill remote "
                f"region {c}'s {r_unit}-byte block in order from a "
                f"{l_unit}-byte one"
            )
        seg_src = (l_off + src[:, None] * l_unit + rows[None, :, 0]).reshape(-1)
        seg_dst = (r_off + dst[:, None] * r_unit + rows[None, :, 1]).reshape(-1)
        seg_len = np.broadcast_to(rows[None, :, 2], (src.size, len(rows))).reshape(-1)
        seg_src, seg_dst, seg_len = coalesce_contiguous(seg_src, seg_dst, seg_len)
        local_offs.append(seg_src)
        remote_offs.append(seg_dst)
        sizes.append(seg_len)
    if not sizes or src.size == 0:
        return [], [], []
    return (
        np.concatenate(local_offs).tolist(),
        np.concatenate(remote_offs).tolist(),
        np.concatenate(sizes).tolist(),
    )


def shard_pairs(
    src_blocks: Sequence[int], dst_blocks: Sequence[int], rank: int, world: int
) -> tuple[list[int], list[int]]:
    """The ``(source units, destination units)`` of the pairs a region sharded
    ``world`` ways holds on rank ``rank`` at both ends: ``src % world == rank``,
    at unit ``src // world``. Raises when a destination's share differs from its
    source's: the consumer's allocator broke the lane rule, and its rank would
    keep the block's index in some other rank's lane."""
    src = np.asarray(src_blocks, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst_blocks, dtype=np.int64).reshape(-1)
    if src.size != dst.size:
        raise ValueError(f"{src.size} source blocks for {dst.size} destinations")
    if np.any(src % world != dst % world):
        bad = np.flatnonzero(src % world != dst % world)[:4].tolist()
        raise ValueError(
            f"blocks {src[bad].tolist()} -> {dst[bad].tolist()} change their "
            f"shard (block id mod {world}): the two ends do not allocate a "
            "logical block in the same lane"
        )
    mine = src % world == rank
    return (src[mine] // world).tolist(), (dst[mine] // world).tolist()


def desc_fabric_info(packed_desc: bytes) -> tuple[bool, int]:
    """``(has a fabric handle, vPOD id)`` of a packed MemoryDesc."""
    fields = msgpack.unpackb(packed_desc, raw=True)
    handle = fields[_DESC_FABRIC_HANDLE]
    return bool(handle) and any(handle), int(fields[_DESC_VPOD_ID])


def desc_size(packed_desc: bytes) -> int:
    """Bytes a packed MemoryDesc registers."""
    return int(msgpack.unpackb(packed_desc, raw=True)[_DESC_SIZE])
