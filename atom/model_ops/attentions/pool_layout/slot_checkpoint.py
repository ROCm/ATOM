# SPDX-License-Identifier: MIT
"""Full KDA checkpoint copies; torch-only so byte correctness is CPU-testable."""

import torch


def copy_kda_checkpoint_slots(conv, state, spec, store_ops, restore_ops):
    """Copy immutable slot images, preserving all layers and both dtypes.

    This is intentionally separate from PAGE descriptors: no PAGE unit is
    allocated or addressed for these operations. Scheduler leases keep every
    source and destination alive until the enclosing batch completes.
    """
    ops = [op for op in (*store_ops, *restore_ops) if op.slot_pair is not None]
    if not ops:
        return
    if spec is None or spec.image_bytes != spec.slot_bytes:
        raise RuntimeError("slot checkpoints require a complete KDA image")
    if conv.shape[1] != state.shape[1]:
        raise RuntimeError("KDA checkpoint planes have different slot counts")
    image_bytes = sum(
        cache[:, 0].numel() * cache.element_size() for cache in (conv, state)
    )
    if image_bytes != spec.image_bytes:
        raise RuntimeError("KDA checkpoint tensors do not match the image size")
    pairs = [op.slot_pair for op in ops]
    for op, (src, dst) in zip(ops, pairs):
        if op.layout_id != spec.layout_id or op.total_bytes != spec.image_bytes:
            raise RuntimeError("KDA slot checkpoint layout or size mismatch")
        if op.unit_ids:
            raise RuntimeError("a slot checkpoint must not also name PAGE units")
        if any(slot < 0 or slot >= conv.shape[1] for slot in (src, dst)):
            raise RuntimeError("state checkpoint slot is out of range")
    src_ids = {src for src, _ in pairs}
    dst_ids = [dst for _, dst in pairs]
    if src_ids.intersection(dst_ids) or len(set(dst_ids)) != len(dst_ids):
        raise RuntimeError("state checkpoint copies alias a source or destination")
    dsts, srcs = [], []
    for src, dst in pairs:
        for cache in (conv, state):
            dsts.append(cache[:, dst])
            srcs.append(cache[:, src])
    torch._foreach_copy_(dsts, srcs)
