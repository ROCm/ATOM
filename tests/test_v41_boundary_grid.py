"""A STATE boundary sits on the PAGE grid; a block hash is indexed by vLLM's unit.

These were one number, and they coincide while there is one kv cache group.
A second group makes the hash unit -- which is engine-wide -- the smallest
group's: with DSpark, 32 against the proxy's 256. The planner then accepted
a boundary at 96 tokens, which is a legal 32-grid point, is not a PAGE edge,
and floors to zero chunks at the caller. The hit it was capping was 1280,
and the tier supplied nothing all run.
"""

import pytest

from atom.plugin.vllm.kv_transfer.v41_state import (
    V41BoundaryPlanner,
    boundary_prefix_hash,
)


def _planner(hash_unit, grid, interval=4096):
    p = V41BoundaryPlanner.__new__(V41BoundaryPlanner)
    p.hash_block_size = hash_unit
    p.boundary_grid = grid
    p.state_interval = interval
    return p


def test_the_anchor_lands_on_the_page_grid_not_the_hash_grid():
    p = _planner(32, 256)

    class _Req:
        num_prompt_tokens = 1345

    assert p._prompt_anchor(_Req()) == 1280          # was 1344 on the 32 grid


def test_a_hash_is_indexed_in_vllms_unit():
    """The other half: `boundary_hash` must keep using the hash unit, or it
    reads the wrong entry of a list vLLM built."""
    p = _planner(32, 256)
    hashes = list(range(50))
    # 1280 tokens is entry 39 at 32 tokens each. Compared against the hash
    # of that entry rather than the index, because the key is derived.
    assert p.boundary_hash(hashes, 1280) == boundary_prefix_hash(hashes[39])
    # Off the hash grid there is no key at all.
    assert p.boundary_hash(hashes, 1300) is None


def test_the_grid_must_be_a_multiple_of_the_hash_unit():
    """Otherwise a boundary on the grid could not be hashed -- the two halves
    would disagree about which positions exist."""
    with pytest.raises(ValueError, match="not a multiple of the hash block"):
        V41BoundaryPlanner(
            hash_block_size=256,
            boundary_grid=96,
            chunk_size=256,
            state_interval=4096,
            max_num_batched_tokens=8192,
            world_size=1,
        )


def test_they_stay_equal_when_there_is_one_group():
    """The single-group case must be untouched: grid defaults to the hash
    unit, which is what every run without a draft model does."""
    p = _planner(256, 256)

    class _Req:
        num_prompt_tokens = 1345

    assert p._prompt_anchor(_Req()) == 1280
    assert p.boundary_hash(list(range(10)), 1280) == boundary_prefix_hash(4)
