# SPDX-License-Identifier: MIT
"""Byte slices for TP1 prefill into TP-sharded MHA decode pools."""


def head_slices(src_bytes, dst_bytes, src_shape, dst_shape, consumer_rank, consumer_tp):
    """(source offset, destination offset, bytes) within one scheduler PAGE.

    A sparse M3 page contains eight independently head-major physical pages;
    dense/draft pages have one. Indexer keys are replicated, never head-sharded.
    No quantization or reordering within a head is performed.
    """
    if src_bytes <= 0 or dst_bytes <= 0 or not 0 <= consumer_rank < consumer_tp:
        raise ValueError("invalid PAGE byte count or consumer TP rank")
    if src_shape is None or dst_shape is None:
        if src_shape != dst_shape or src_bytes != dst_bytes:
            raise ValueError("incompatible replicated PAGE geometry")
        return [(0, 0, src_bytes)]
    groups, heads = src_shape
    dst_groups, dst_heads = dst_shape
    if (
        groups <= 0
        or heads <= 0
        or dst_heads <= 0
        or groups != dst_groups
        or src_bytes % (groups * heads)
        or dst_bytes % (groups * dst_heads)
    ):
        raise ValueError("invalid head-major PAGE geometry")
    width = src_bytes // (groups * heads)
    if width != dst_bytes // (groups * dst_heads):
        raise ValueError("incompatible bytes per KV head")
    if heads >= consumer_tp and heads == dst_heads * consumer_tp:
        first_head = consumer_rank * dst_heads
    elif heads < consumer_tp and consumer_tp % heads == 0 and dst_heads == 1:
        first_head = consumer_rank // (consumer_tp // heads)
    else:
        raise ValueError("TP1 producer head count does not match consumer TP shards")
    size = dst_heads * width
    return [
        (g * heads * width + first_head * width, g * size, size) for g in range(groups)
    ]
