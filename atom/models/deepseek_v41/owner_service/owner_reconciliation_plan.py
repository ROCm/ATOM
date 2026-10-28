"""Native-reader dependencies for persistent request-owned PAGE and STATE."""


def reconciliation_requests(
    dirty, *, request_ids=(), slots=(), pages=(), all_dirty=False
):
    """Select whole dirty epochs; unrelated owner state remains authoritative.

    A native request reads its own STATE and all PAGE units in its block table.
    Checkpoint/relocation readers can instead name physical source slots/units.
    Selecting an epoch publishes its complete recorded write set, preserving
    the existing broadcast arithmetic and deterministic collective order.
    """
    ids, state_slots, page_units = set(request_ids), set(slots), set(pages)
    return tuple(
        request_id
        for request_id, (_, slot, written) in sorted(dirty.items())
        if all_dirty
        or request_id in ids
        or slot in state_slots
        or not page_units.isdisjoint(written)
    )
