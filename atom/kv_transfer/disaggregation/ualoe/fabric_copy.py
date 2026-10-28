# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One Triton launch that copies a descriptor list between mapped pools.

Descriptors are cut into fixed-size pieces on the host, one program per
piece, each moving the widest element the addresses' alignment allows.

Measured over xGMI (MI308X, 32-256 MiB per launch): 41 / 51 / 54 GB/s at
16 KiB / 64 KiB / 1 MiB blocks, against 49 GB/s for torch's own peer
``copy_``. ``hipMemcpyBatchAsync`` pays ~3 us a descriptor (5 GB/s at 16 KiB),
too slow for KV blocks of tens of KiB.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

# bytes one program copies at most, and elements per inner step
_PIECE_BYTES = 64 * 1024
_BLOCK = 1024


@triton.jit
def _copy_pieces_kernel(
    table,
    ELEM_BYTES: tl.constexpr,
    PIECE_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program ``i`` copies ``table[i] = (src, dst, n_elems)``."""
    pid = tl.program_id(0)
    src = tl.load(table + pid * 3)
    dst = tl.load(table + pid * 3 + 1)
    n = tl.load(table + pid * 3 + 2)
    if ELEM_BYTES == 8:
        sp = src.to(tl.pointer_type(tl.int64))
        dp = dst.to(tl.pointer_type(tl.int64))
    elif ELEM_BYTES == 4:
        sp = src.to(tl.pointer_type(tl.int32))
        dp = dst.to(tl.pointer_type(tl.int32))
    else:
        sp = src.to(tl.pointer_type(tl.int8))
        dp = dst.to(tl.pointer_type(tl.int8))
    offs = tl.arange(0, BLOCK)
    for start in range(0, PIECE_ELEMS, BLOCK):
        idx = start + offs
        mask = idx < n
        tl.store(dp + idx, tl.load(sp + idx, mask=mask), mask=mask)


def element_bytes(src: np.ndarray, dst: np.ndarray, size: np.ndarray) -> int:
    """The widest of 8 / 4 / 1 bytes every address and size is a multiple of."""
    if not src.size:
        return 8
    bits = int(np.bitwise_or.reduce(np.concatenate([src, dst, size])))
    if bits % 8 == 0:
        return 8
    if bits % 4 == 0:
        return 4
    return 1


def split_pieces(
    src: np.ndarray, dst: np.ndarray, size: np.ndarray, piece: int
) -> np.ndarray:
    """``(src, dst, len)`` rows, each descriptor cut into ``piece``-byte runs."""
    counts = -(-size // piece)
    owner = np.repeat(np.arange(size.size), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    offset = (np.arange(owner.size) - first) * piece
    rows = np.empty((owner.size, 3), dtype=np.int64)
    rows[:, 0] = src[owner] + offset
    rows[:, 1] = dst[owner] + offset
    rows[:, 2] = np.minimum(piece, size[owner] - offset)
    return rows


def launch_copy(
    src_addrs: list[int],
    dst_addrs: list[int],
    sizes: list[int],
    stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Enqueue the copy on ``stream`` without blocking the host; returns the
    (pinned host, device) descriptor tables, which must stay alive until the
    stream has run the launch."""
    if not src_addrs:
        return None
    src = np.asarray(src_addrs, dtype=np.int64)
    dst = np.asarray(dst_addrs, dtype=np.int64)
    size = np.asarray(sizes, dtype=np.int64)
    elem = element_bytes(src, dst, size)
    rows = split_pieces(src, dst, size, _PIECE_BYTES)
    rows[:, 2] //= elem
    # Pinned and non-blocking: ``stream`` may be waiting on compute, and a
    # pageable upload would hold the host until that compute is done.
    host = torch.from_numpy(rows).pin_memory()
    with torch.cuda.stream(stream):
        table = host.to(stream.device, non_blocking=True)
        _copy_pieces_kernel[(rows.shape[0],)](
            table,
            ELEM_BYTES=elem,
            PIECE_ELEMS=_PIECE_BYTES // elem,
            BLOCK=_BLOCK,
        )
    return host, table
