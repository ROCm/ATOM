# SPDX-License-Identifier: MIT
"""I3 (DESIGN_v3 4.9) for the V4.1 mono K2b: for any routing, every ug flag a
down round polls is one an ug task writes this step and announces that slot's
MID rows, and those rows cover every pick of the slot. The functions under
test are the ones the kernel calls (``mono/moe_plan.py``); the route table is
built by ``stage_route``'s rules."""

import random

import pytest

from atom.models.deepseek_v41.mono import moe_plan as mplan
from atom.models.deepseek_v41.mono.kernels.dims import UG_PART, Dims


def route_table(counts):
    """``stage_route``'s table from each used expert's pick count, in slot order:
    slot_first, tile_first (one past the last slot: the totals) and tile_slot."""
    slot_first, tile_first, tile_slot = [0], [0], []
    for u, n in enumerate(counts):
        slot_first.append(slot_first[-1] + n)
        tiles = mplan.pick_tiles(n)
        tile_first.append(tile_first[-1] + tiles)
        tile_slot += [u] * tiles
    return slot_first, tile_first, tile_slot


def routings(tokens, experts, topk, n=60):
    """Random routings (each token picks ``topk`` distinct experts), the
    concentrated one (every token the same experts) among them."""
    rng = random.Random(tokens * 1000 + experts)
    out = [[tokens] * topk]
    for _ in range(n):
        pool = rng.randrange(topk, experts + 1)
        hits = {}
        for _ in range(tokens):
            for e in rng.sample(range(pool), topk):
                hits[e] = hits.get(e, 0) + 1
        out.append([hits[e] for e in sorted(hits)])
    return out


def ug_writes(counts, ug_parts, multi):
    """flag -> (slot, its pick rows) of every ug task the step runs."""
    slot_first, tile_first, tile_slot = route_table(counts)
    units = tile_first[-1] if multi else len(counts)
    writes = {}
    for task in range(mplan.ug_tasks(units, ug_parts)):
        unit, part = task // ug_parts, task % ug_parts
        assert mplan.ug_flag(unit, part, ug_parts) == task
        if multi:
            u = tile_slot[unit]
            first = mplan.tile_pick0(slot_first[u], unit, tile_first[u])
            rows = range(first, min(first + mplan.TILE, slot_first[u + 1]))
        else:
            u = unit
            rows = range(slot_first[u], slot_first[u + 1])
        writes[task] = (u, set(rows))
    return writes


def down_polls(counts, u, ug_parts, multi, slot_tiles, polled_tile=mplan.polled_tile):
    """The flags a down round polls for slot u (``_down_ug_flag``)."""
    _, tile_first, _ = route_table(counts)
    if not multi:
        return {mplan.ug_flag(u, p, ug_parts) for p in range(ug_parts)}
    tiles = tile_first[u + 1] - tile_first[u]
    return {
        mplan.ug_flag(polled_tile(tile_first[u], tiles, j), p, ug_parts)
        for j in range(slot_tiles)
        for p in range(ug_parts)
    }


SHAPES = [(384, 6), (128, 3)]  # the target's routed experts, the draft's


def parts(tp, ug_groups):
    return mplan.ug_parts(Dims(tp).inter_real // UG_PART, ug_groups)


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("ug_groups", [1, 2])
@pytest.mark.parametrize("tokens", [6, 12, 16, 18, 24, 40, 48])
@pytest.mark.parametrize("experts,topk", SHAPES)
def test_every_polled_flag_is_written_for_its_slot(
    tp, ug_groups, tokens, experts, topk
):
    ug_parts = parts(tp, ug_groups)
    multi = tokens > mplan.TILE
    slot_tiles = mplan.pick_tiles(tokens)
    for counts in routings(tokens, experts, topk):
        slot_first, _, _ = route_table(counts)
        writes = ug_writes(counts, ug_parts, multi)
        for u in range(len(counts)):
            polled = down_polls(counts, u, ug_parts, multi, slot_tiles)
            assert polled <= set(writes), (counts, u)
            assert {writes[f][0] for f in polled} == {u}, (counts, u)
            rows = set().union(*(writes[f][1] for f in polled))
            assert rows >= set(range(slot_first[u], slot_first[u + 1])), (counts, u)


def test_unclamped_tile_polls_the_next_slots_flags():
    """Positive control: a slot's j-th tile without the clamp to its last runs
    into the next slot's tiles past one token tile."""

    def unclamped(tile_first, tiles, j):
        return tile_first + j

    ug_parts, tokens = parts(4, 2), 24
    counts = [1, tokens]  # slot 0: one tile of the two a slot may have
    writes = ug_writes(counts, ug_parts, multi=True)
    polled = down_polls(
        counts, 0, ug_parts, True, mplan.pick_tiles(tokens), polled_tile=unclamped
    )
    assert {writes[f][0] for f in polled if f in writes} != {0}
