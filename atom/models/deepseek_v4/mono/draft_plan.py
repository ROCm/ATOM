# SPDX-License-Identifier: MIT
"""Host draft metadata plan for native strided window allocations.

This is an eager publication building block, not an engine lifecycle owner.
It does not certify that target context writes have completed or lease slots.
"""

from dataclasses import dataclass

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import WindowParams


@dataclass(frozen=True)
class DraftBlockPlan:
    width: int
    positions: tuple[int, ...]
    dest_rows: tuple[int, ...]
    indices: tuple[tuple[int, ...], ...]


def plan_draft_block(
    anchors, slots, window: WindowParams, *, width, history, pool_rows
):
    """Ordered valid-window ++ complete-block rows, independently per request.

    All addresses are checked before publication, including wrap/strided runs.
    Aliasing live requests or overwriting still-visible history is rejected.
    The caller passes actual physical slots and allocation-relative geometry.
    """
    anchors, slots = tuple(anchors), tuple(slots)
    if not anchors or len(anchors) != len(slots):
        raise ValueError("one anchor and physical slot per live draft request required")
    if any(type(v) is not int or v < 0 for v in (*anchors, *slots)):
        raise ValueError("anchors and physical slots must be nonnegative integers")
    if any(type(v) is not int or v <= 0 for v in (width, history, pool_rows)):
        raise ValueError("draft width/history/pool capacity must be positive integers")
    if (
        type(window.ring_start) is not int
        or window.ring_start < 0
        or any(
            type(v) is not int or v <= 0
            for v in (
                window.slot_rows,
                window.ring_slots,
                window.ring_stride,
                window.run_rows,
            )
        )
    ):
        raise ValueError("invalid native window allocation geometry")
    if width > history or window.ring_slots < history + width:
        raise ValueError(
            "native ring must hold visible history and the whole draft block"
        )
    if len(set(slots)) != len(slots):
        raise ValueError("live draft requests cannot share a physical slot")
    capacity = ((history + width + 63) // 64) * 64
    positions, destinations, indices, owners = [], [], [], set()
    for anchor, slot in zip(anchors, slots):
        start = max(0, anchor - history + 1)
        visible = list(range(start, anchor + width + 1))
        keys = tuple(window.index(slot, pos) for pos in visible)
        if any(not 0 <= row < pool_rows or row >= 2**31 for row in keys):
            raise ValueError(
                "draft window/block row exceeds native pool or int32 capacity"
            )
        if len(keys) != len(set(keys)) or owners.intersection(keys):
            raise ValueError("draft history/block or live request rows alias")
        owners.update(keys)
        draft_positions = tuple(range(anchor + 1, anchor + width + 1))
        if draft_positions[-1] >= 2**31:
            raise ValueError("draft position exceeds device int32 capacity")
        positions.extend(draft_positions)
        destinations.extend(keys[-width:])
        padded = keys + (-1,) * (capacity - len(keys))
        indices.extend([padded] * width)
    return DraftBlockPlan(width, tuple(positions), tuple(destinations), tuple(indices))
