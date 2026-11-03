# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cut a TP1 producer's per-head KV units into a TP-sharded consumer's."""


def consumer_head_slices(src_bytes, dst_bytes, src_runs, dst_runs, rank, tp):
    """``(src offset, dst offset, bytes)`` per run, for consumer ``rank`` of ``tp``.

    The source (a TP1 PP stage) holds every KV head; the consumer holds its
    share of them, or one replica each when ``tp`` exceeds the head count
    (`MhaKvPool.from_hf_config`). ``None`` runs on both ends is a unit that is
    not per head, moved whole.
    """
    if src_runs is None or dst_runs is None:
        if src_runs != dst_runs or src_bytes != dst_bytes:
            raise ValueError(f"{src_runs}/{src_bytes} B vs {dst_runs}/{dst_bytes} B")
        return [(0, 0, src_bytes)]
    (runs, heads), (dst_runs_, dst_heads) = src_runs, dst_runs
    if min(runs, heads, dst_heads) <= 0 or not 0 <= rank < tp:
        raise ValueError(f"bad geometry {src_runs}/{dst_runs} for rank {rank}/{tp}")
    width = src_bytes // (runs * heads)
    if (
        runs != dst_runs_
        or src_bytes != runs * heads * width
        or dst_bytes != runs * dst_heads * width
    ):
        raise ValueError(f"{src_runs}/{src_bytes} B vs {dst_runs}/{dst_bytes} B")
    if heads == dst_heads * tp:
        first = rank * dst_heads
    elif dst_heads == 1 and tp % heads == 0:
        first = rank // (tp // heads)
    else:
        raise ValueError(f"{heads} heads do not split over TP {tp} as {dst_heads}")
    size = dst_heads * width
    return [(r * heads * width + first * width, r * size, size) for r in range(runs)]
