# SPDX-License-Identifier: MIT
"""The rules both recurrent-state planners share, run against each of them.

`BoundaryPlanner` holds them once so the KDA leg and the V4.1 checkpoint ride
cannot drift apart on them: a load's destination is one block per state group
at the boundary's row, every group or none, and the hit cap declines a hit
armed for another request. Where the two legs differ by design -- which way a
cap fired outside any lookup fails -- the difference is pinned here too.

No vLLM: both planners are importable without it.
"""

import logging
from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.boundary_planner import BoundaryPlanner
from atom.plugin.vllm.kv_transfer.kda_state import KdaBoundaryPlanner
from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner

GRID = 256
#: Two state groups, so a half restore is expressible.
GROUPS = (1, 2)


def _kda():
    return KdaBoundaryPlanner(
        group_ids=GROUPS,
        mamba_block_size=GRID,
        hash_block_size=GRID,
        chunk_size=GRID,
        world_size=1,
    )


def _v41():
    return V41BoundaryPlanner(
        hash_block_size=GRID,
        boundary_grid=GRID,
        chunk_size=GRID,
        checkpoint_group_ids=GROUPS,
    )


#: (factory, whether a cap fired outside any lookup lets the hit through)
EACH_PLANNER = pytest.mark.parametrize(
    "make, unarmed_passes", [(_kda, True), (_v41, False)], ids=["kda", "v41"]
)


@EACH_PLANNER
def test_both_legs_are_boundary_planners(make, unarmed_passes):
    planner = make()
    assert isinstance(planner, BoundaryPlanner)
    assert planner.boundary_grid == GRID
    assert planner.boundary_group_ids == GROUPS


@EACH_PLANNER
@pytest.mark.parametrize(
    "computed, group_blocks, expected",
    [
        # Row (n - 1) // grid of every state group, in group order.
        (512, ([90, 91], [3, 4], [7, 8]), (4, 8)),
        (256, ([90], [3, 4], [7, 8]), (3, 7)),
        # Not on the grid: no row is the resuming state.
        (500, ([90, 91], [3, 4], [7, 8]), ()),
        (0, ([90, 91], [3, 4], [7, 8]), ()),
        # One group short of the row fails the other with it.
        (768, ([90, 91, 92], [3, 4, 5], [7, 8]), ()),
        # So does one group's null block.
        (512, ([90, 91], [3, 4], [7, 0]), ()),
        # And a group with no block table at all.
        (512, ([90, 91], [3, 4]), ()),
        (512, (), ()),
    ],
)
def test_a_boundary_lands_in_every_state_group_or_none(
    make, unarmed_passes, computed, group_blocks, expected
):
    assert make()._boundary_blocks(group_blocks, computed) == expected


@EACH_PLANNER
def test_an_unarmed_cap_fails_the_way_its_leg_chose(make, unarmed_passes):
    """KDA passes it through; V4.1 declines it, since an untested hit restores
    PAGE with no state behind it."""
    planner = make()
    seq = SimpleNamespace(id="r")
    assert planner.cap_hit(seq, 512) == (512 if unarmed_passes else 0)
    # A hit of nothing is returned as it is, armed or not.
    assert planner.cap_hit(seq, 0) == 0
    assert planner.cap_hit(seq, -1) == -1


@EACH_PLANNER
def test_a_cap_armed_for_another_request_declines(make, unarmed_passes, caplog):
    planner = make()
    request = SimpleNamespace(request_id="r", num_tokens=4096, block_hashes=())
    planner.begin_lookup(request)
    try:
        with caplog.at_level(logging.WARNING, logger="atom"):
            assert planner.cap_hit(SimpleNamespace(id="someone-else"), 512) == 0
    finally:
        planner.end_lookup()
    assert (
        f"{planner.leg_name}: hit cap armed for r but called for someone-else"
        in caplog.text
    )


@EACH_PLANNER
def test_end_lookup_disarms_the_cap(make, unarmed_passes):
    """Cleared unconditionally, so a later hit has nothing stale to read."""
    planner = make()
    planner.begin_lookup(SimpleNamespace(request_id="r", num_tokens=4096))
    planner.end_lookup()
    assert planner.cap_hit(SimpleNamespace(id="r"), 512) == (
        512 if unarmed_passes else 0
    )
