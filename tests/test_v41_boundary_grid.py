"""A STATE boundary sits on the PAGE grid; a block hash is indexed by vLLM's unit.

These were one number, and they coincide while there is one kv cache group.
A second group makes the hash unit -- which is engine-wide -- the smallest
group's: with DSpark, 32 against the proxy's 256. The planner then accepted
a boundary at 96 tokens, which is a legal 32-grid point, is not a PAGE edge,
and floors to zero chunks at the caller. The hit it was capping was 1280,
and the tier supplied nothing all run.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer import v41_state
from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner


def _planner(hash_unit=256, grid=256, *, keep_tail=False, can_serve=False):
    return V41BoundaryPlanner(
        hash_block_size=hash_unit,
        boundary_grid=grid,
        chunk_size=256,
        state_checkpoints_can_serve=can_serve,
        checkpoint_group_ids=(1,),
        keep_a_prefill_tail=keep_tail,
    )


def _cap(planner, hit, *, num_tokens=4096, armed_for="r"):
    request = SimpleNamespace(request_id=armed_for, num_tokens=num_tokens)
    planner.begin_lookup(request)
    try:
        return planner.cap_hit(SimpleNamespace(id="r"), hit)
    finally:
        planner.end_lookup()


def test_the_grid_must_be_a_multiple_of_the_hash_unit():
    """Otherwise a boundary on the grid could not be hashed -- the two halves
    would disagree about which positions exist."""
    with pytest.raises(ValueError, match="not a multiple of the hash block"):
        _planner(256, 96)


def test_a_ride_needs_a_checkpoint_group_to_name():
    with pytest.raises(ValueError, match="no checkpoint group"):
        V41BoundaryPlanner(hash_block_size=256, chunk_size=256)


def test_a_hit_is_capped_to_the_page_grid_not_the_hash_unit():
    """The DSpark case: hash unit 32, grid 256. 1300 is on neither, 1280 is a
    PAGE edge, and 1312 would be a legal 32-grid point that is not one."""
    p = _planner(32, 256)
    assert _cap(p, 1300) == 1280
    assert _cap(p, 1312) == 1280
    assert p.counters()["cap_kept"] == 2


def test_a_hit_below_one_page_is_declined():
    p = _planner()
    assert _cap(p, 200) == 0
    assert p.counters()["cap_declined"] == 1


def test_an_unarmed_cap_declines_the_hit():
    """Fail closed: an unarmed hook used to serve the hit whole, restoring PAGE
    to a position whose STATE was never written."""
    p = _planner()
    assert p.cap_hit(SimpleNamespace(id="r"), 1024) == 0
    assert p.counters()["cap_unarmed"] == 1


def test_an_unarmed_cap_leaves_a_zero_hit_alone():
    assert _planner().cap_hit(SimpleNamespace(id="r"), 0) == 0


def test_a_cap_armed_for_another_request_declines():
    assert _cap(_planner(), 1024, armed_for="someone-else") == 0


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
        "planner.boundary_grid" in src
    ), "the prefill clip must align to the grid a boundary sits on"


@pytest.mark.parametrize("keep_tail, expected", [(False, 1024), (True, 768)])
def test_a_speculative_ride_never_resumes_one_token_short(keep_tail, expected):
    """vLLM pads a request admitted with one token left into a verify row.

    `pad_spec_decode` turns it into `1 + num_spec_tokens` rows of placeholder
    drafts, and only refuses to do so after a prefill was scheduled -- not
    before one. V4.1 refuses the mixed step: measured as `(1024, 6, 5)` beside
    `(768, 200, 0)`. Two or more tokens left is a plain prefill instead.
    """
    p = _planner(keep_tail=keep_tail)
    assert _cap(p, 1024, num_tokens=1025) == expected
    assert p.counters()["cap_prefill_tail"] == int(keep_tail)


def test_a_longer_tail_is_left_alone():
    assert _cap(_planner(keep_tail=True), 1024, num_tokens=1100) == 1024


def test_a_local_hit_is_declined_unless_a_checkpoint_serves_it(monkeypatch):
    """vLLM's own prefix hit never reaches the connector. Served without a
    checkpoint behind it, the forward finds a state slot nothing wrote."""
    monkeypatch.setattr(v41_state, "_CHECKPOINT_LOCAL_HITS_ENABLED", True)
    request = SimpleNamespace(request_id="r")
    assert _planner(can_serve=False).cap_local_hit(request, 1280) == 0
    p = _planner(can_serve=True)
    assert p.cap_local_hit(request, 1280) == 1280
    assert p.counters()["local_cap_checkpoint"] == 1


def test_local_hits_stay_off_without_the_gate(monkeypatch):
    monkeypatch.setattr(v41_state, "_CHECKPOINT_LOCAL_HITS_ENABLED", False)
    p = _planner(can_serve=True)
    assert p.cap_local_hit(SimpleNamespace(request_id="r"), 1280) == 0
    assert p.counters()["local_cap_declined"] == 1


@pytest.mark.parametrize(
    "computed, group_blocks, expected",
    [
        (512, ([3, 4], [7, 8]), True),  # row 1 of group 1 is block 8
        (500, ([3, 4], [7, 8]), False),  # not a PAGE edge
        (768, ([3, 4], [7, 8]), False),  # past the allocated rows
        (512, ([3, 4], [7, 0]), False),  # NULL block: not a destination
        (512, ([3, 4],), False),  # the checkpoint group is missing
    ],
)
def test_a_load_names_its_checkpoint_blocks_or_nothing(
    computed, group_blocks, expected
):
    p = _planner()
    request = SimpleNamespace(request_id="r")
    assert p.resolve_load(request, computed, group_blocks=group_blocks) is expected
    recorded = p._ride_boundaries.get("r", {})
    assert recorded == ({computed: (8,)} if expected else {})
    p.forget_request("r")
    assert "r" not in p._ride_boundaries
