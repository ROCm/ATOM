# SPDX-License-Identifier: MIT
"""DCP copies of PAGE bytes with physical token axes declared by the cache.

Supports split token axes in SHUFFLE K/V, scales and token-major index/NVFP4.
"""

from functools import lru_cache
from math import prod

import numpy as np

from atom.kv_transfer.disaggregation.sharded_transfer import coalesce_contiguous


def canonical_token_layout(layout, block_size, unit_bytes):
    if not isinstance(layout, (tuple, list)) or len(layout) != 2:
        raise ValueError("DCP PAGE transfer needs a physical token layout")
    shape, axes = layout
    if not isinstance(shape, (tuple, list)) or not isinstance(axes, (tuple, list)):
        raise TypeError("DCP token layout must contain shape and token axes")
    shape, axes = tuple(shape), tuple(axes)
    if (
        not 1 <= len(shape) <= 8
        or any(type(n) is not int or n <= 0 for n in shape)
        or not axes
        or any(type(a) is not int or not 0 <= a < len(shape) for a in axes)
        or len(set(axes)) != len(axes)
        or prod(shape) != unit_bytes
        or prod(shape[a] for a in axes) != block_size
    ):
        raise ValueError(
            f"Invalid DCP token layout {layout!r}: {block_size} tokens, {unit_bytes} B"
        )
    return shape, axes


@lru_cache(maxsize=64)
def _token_bytes(shape, axes):
    other = tuple(a for a in range(len(shape)) if a not in axes)
    offsets = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    result = offsets.transpose(axes + other).reshape(prod(shape[a] for a in axes), -1)
    result.flags.writeable = False
    return result


def dcp_page_descriptors(plan, dst_block_ids, src_base, dst_base, unit_bytes, layout):
    """Yield byte descriptors bounded to one destination PAGE at a time.

    Both endpoint layouts must already be checked identical. Missing source
    blocks leave destination padding untouched. No descriptor crosses a PAGE,
    hence none crosses a PAGE-aligned RDMA memory-region boundary.
    S == block_size takes the whole-page fast path. Smaller interleaves sort
    one page's physical bytes and merge only adjacent source AND destination
    addresses. Scratch space does not grow with context length or layer count.
    """
    shape, axes = canonical_token_layout(layout, plan.block_size, unit_bytes)
    if len(dst_block_ids) != plan.dst_pages:
        raise ValueError("DCP destination block count disagrees with shard plan")
    if any(int(b) < 0 for b in dst_block_ids):
        raise ValueError("DCP destination block ids must be nonnegative")
    offsets = (
        None if plan.interleave_size == plan.block_size else _token_bytes(shape, axes)
    )
    n = plan.block_size // plan.interleave_size
    for page, dst_id in enumerate(dst_block_ids):
        start, stop = page * n, (page + 1) * n
        keep = np.flatnonzero(plan.valid[start:stop]) + start
        if not keep.size:
            continue
        src_ids = plan.src_block_id_per_run[keep]
        if (src_ids < 0).any():
            raise ValueError("DCP source block ids must be nonnegative")
        if offsets is None:
            yield (
                src_base + src_ids * unit_bytes,
                np.array([dst_base + int(dst_id) * unit_bytes], dtype=np.int64),
                np.array([unit_bytes], dtype=np.int64),
            )
            continue
        within = np.arange(plan.interleave_size, dtype=np.int64)
        src = (
            src_ids[:, None, None] * unit_bytes
            + offsets[plan.src_token[keep, None] + within]
        ).ravel()
        dst = (
            int(dst_id) * unit_bytes + offsets[plan.dst_token[keep, None] + within]
        ).ravel()
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        split = np.r_[True, src[1:] // unit_bytes != src[:-1] // unit_bytes]
        src, dst, sizes = coalesce_contiguous(
            src, dst, np.ones(src.size, dtype=np.int64), split
        )
        yield src_base + src, dst_base + dst, sizes
