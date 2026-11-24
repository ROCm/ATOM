# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoRIIO over the GPU scale-up fabric (AMD Helios UALoE): ``moriio_backend`` ``fabric``.

- Both ranks back their paged KV pool with fabric-exportable VMM memory from
  mori's allocator (``mori.io.fabric_alloc``), so registering it publishes a
  fabric handle that the peer imports and maps.
- The connector registers the pool once, as one MoRIIO buffer: mori imports and
  maps a peer's whole allocation per registered descriptor, so registering each
  of the model's PAGE regions on its own would map the pool once per region.
- Every PAGE region is then addressed as (offset in the pool, bytes per block),
  and a request moves as one batch on the single (local pool, remote pool)
  session; mori merges segments that are contiguous on both sides and copies
  up to 512 of them per launch of its fabric copy kernel. Blocks of one batch
  run concurrently, so a batch's destinations must not overlap: no block may
  appear twice on the side being written.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

import msgpack
import torch

logger = logging.getLogger("atom")

# MemoryDesc's msgpack array (mori include/mori/io/common.hpp MSGPACK_DEFINE).
_DESC_FABRIC_HANDLE = 10
_DESC_VPOD_ID = 11


def is_fabric_backend(kv_transfer_config: dict) -> bool:
    return str(kv_transfer_config.get("moriio_backend", "rdma")).lower() == "fabric"


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


def pool_span(regions: list[tuple[int, int]]) -> tuple[int, int]:
    """``(base, nbytes)`` to register for ``regions`` ((base, nbytes) each).

    The allocated pool when it holds every region, else the regions' hull,
    which must still lie in one fabric allocation for the export to cover it.
    """
    base = min(b for b, _ in regions)
    end = max(b + n for b, n in regions)
    if _pool_range is not None:
        pool_base, pool_size = _pool_range
        if pool_base <= base and end <= pool_base + pool_size:
            return pool_base, pool_size
        logger.warning(
            "MoRIIO fabric: PAGE regions [%#x, %#x) are not all inside the "
            "fabric KV pool [%#x, %#x)",
            base,
            end,
            pool_base,
            pool_base + pool_size,
        )
    return base, end - base


def desc_fabric_info(packed_desc: bytes) -> tuple[bool, int]:
    """``(has a fabric handle, vPOD id)`` of a packed MemoryDesc."""
    fields = msgpack.unpackb(packed_desc, raw=True)
    handle = fields[_DESC_FABRIC_HANDLE]
    return bool(handle) and any(handle), int(fields[_DESC_VPOD_ID])


def repeated_blocks(block_ids: list[int]) -> list[int]:
    """The block ids that appear more than once, sorted."""
    seen: set[int] = set()
    repeated: set[int] = set()
    for b in block_ids:
        if b in seen:
            repeated.add(b)
        seen.add(b)
    return sorted(repeated)


def block_runs(src: list[int], dst: list[int]) -> list[tuple[int, int, int]]:
    """``(src_block, dst_block, blocks)`` runs that are contiguous on both sides."""
    runs: list[tuple[int, int, int]] = []
    for s, d in zip(src, dst):
        if runs and s == runs[-1][0] + runs[-1][2] and d == runs[-1][1] + runs[-1][2]:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        else:
            runs.append((s, d, 1))
    return runs


def page_offsets(
    local_regions: list[tuple[int, int]],
    remote_regions: list[tuple[int, int]],
    runs: list[tuple[int, int, int]],
) -> tuple[list[int], list[int], list[int]]:
    """Pool-relative ``(local offsets, remote offsets, sizes)`` of ``runs`` in
    every region, regions given as ``(offset in the pool, bytes per block)``
    and paired by position; each run is ``(local block, remote block, blocks)``."""
    local_offs: list[int] = []
    remote_offs: list[int] = []
    sizes: list[int] = []
    for (l_off, unit), (r_off, _) in zip(local_regions, remote_regions):
        for lb, rb, n in runs:
            local_offs.append(l_off + lb * unit)
            remote_offs.append(r_off + rb * unit)
            sizes.append(n * unit)
    return local_offs, remote_offs, sizes


def page_digest(views: list[torch.Tensor], block_ids: list[int]) -> int:
    """Position-weighted byte digest of ``block_ids`` in every PAGE view.

    Equal on both ends exactly when every region's blocks hold the same bytes
    in the same order, so producer and consumer digests compare directly.
    """
    return sum(page_region_digests(views, block_ids))


def page_region_digests(views: list[torch.Tensor], block_ids: list[int]) -> list[int]:
    """Each PAGE view's term of `page_digest`, in view order."""
    device = views[0].device
    idx = torch.as_tensor(block_ids, dtype=torch.int64, device=device)
    terms = []
    for i, view in enumerate(views):
        rows = view.index_select(0, idx).reshape(-1).to(torch.int64)
        weights = (
            torch.arange(rows.numel(), dtype=torch.int64, device=device) + 7919 * i
        ) % 65521 + 1
        terms.append(torch.sum(rows * weights))
    return [int(t) for t in torch.stack(terms).tolist()]
