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

    assert p._prompt_anchor(_Req()) == 1280  # was 1344 on the 32 grid


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


def test_the_prefill_clip_aligns_to_the_grid_not_the_hash_unit():
    """Fourth site of the same conflation.

    `apply_vllm_v41_prefill_alignment_patch` clips a prefill chunk so the
    frontier lands on a boundary. Clipping to the hash unit lands it
    somewhere that is a legal 32-grid point and not a PAGE edge, so the
    prompt-end anchor -- the one boundary a later request with the same
    prompt actually asks about -- is never stored.
    """
    import inspect

    from atom.plugin.vllm import scheduler as sched

    src = inspect.getsource(sched.apply_vllm_v41_prefill_alignment_patch)
    assert (
        "boundary_grid" in src
    ), "the prefill clip must align to the grid a boundary sits on"
    # And the fallback keeps single-group runs working unchanged.
    assert "planner.hash_block_size" in src


def _ride_planner(keep_tail):
    from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner

    p = V41BoundaryPlanner(
        hash_block_size=256,
        boundary_grid=256,
        chunk_size=256,
        state_interval=256,
        max_num_batched_tokens=8192,
        world_size=1,
        checkpoint_group_ids=(1,),
        rides_page=True,
        keep_a_prefill_tail=keep_tail,
    )
    return p


@pytest.mark.parametrize("keep_tail, expected", [(False, 1024), (True, 768)])
def test_a_speculative_ride_never_resumes_one_token_short(keep_tail, expected):
    """vLLM pads a request admitted with one token left into a verify row.

    `pad_spec_decode` turns it into `1 + num_spec_tokens` rows of placeholder
    drafts, and only refuses to do so after a prefill was scheduled -- not
    before one. V4.1 refuses the mixed step: measured as `(1024, 6, 5)` beside
    `(768, 200, 0)`. Two or more tokens left is a plain prefill instead.
    """
    from types import SimpleNamespace

    p = _ride_planner(keep_tail)
    request = SimpleNamespace(request_id="r", num_tokens=1025, block_hashes=())
    p.begin_lookup(request)
    try:
        got = p.cap_hit(SimpleNamespace(id="r"), 1024)
    finally:
        p.end_lookup()
    assert got == expected


def test_a_longer_tail_is_left_alone():
    from types import SimpleNamespace

    p = _ride_planner(True)
    request = SimpleNamespace(request_id="r", num_tokens=1100, block_hashes=())
    p.begin_lookup(request)
    try:
        assert p.cap_hit(SimpleNamespace(id="r"), 1024) == 1024
    finally:
        p.end_lookup()
