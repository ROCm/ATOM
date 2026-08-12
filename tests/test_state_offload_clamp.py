# SPDX-License-Identifier: MIT
# P <= L. Violating it is silent wrong output, so it gets its own test file.

from atom.kv_transfer.offload.state_tier import clamp_state_boundary


def test_a_state_boundary_within_the_loaded_kv_is_kept():
    assert clamp_state_boundary(4, 8) == 4


def test_a_state_boundary_past_the_loaded_kv_is_cut_to_it():
    """State claims to have seen [0,P) but [L,P) KV does not exist. The forward
    would produce wrong output and raise nothing."""
    assert clamp_state_boundary(8, 4) == 4


def test_equal_is_the_ideal_and_is_kept():
    assert clamp_state_boundary(4, 4) == 4


def test_no_kv_loaded_clamps_to_zero_which_means_recompute():
    """0 is always a valid boundary — a request starting from scratch needs no
    prior state. This is the existing path, not a new failure mode."""
    assert clamp_state_boundary(8, 0) == 0


def test_negatives_floor_at_zero():
    """Either argument. A negative L is the more dangerous one: the inner
    min() carries it straight through, and a negative boundary reaching the
    forward is not a shorter prefix but an out-of-range one."""
    assert clamp_state_boundary(-1, 8) == 0
    assert clamp_state_boundary(8, -1) == 0


from atom.kv_transfer.offload.state_tier import _JointPark


def test_a_request_needing_both_wakes_on_neither_alone():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=True)
    park.settle_state("r", ok=True)
    assert park.take_ready() == (set(), set())
    park.settle_kv("r", ok=True)
    assert park.take_ready() == ({"r"}, set())


def test_either_failing_fails_the_pair():
    """Half a load is not a partial success: the state would claim a prefix
    whose KV never arrived."""
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=True)
    park.settle_state("r", ok=False)
    park.settle_kv("r", ok=True)
    assert park.take_ready() == (set(), {"r"})


def test_a_kv_only_request_is_unchanged():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=False)
    park.settle_kv("r", ok=True)
    assert park.take_ready() == ({"r"}, set())


def test_take_ready_drains():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=False)
    park.settle_kv("r", ok=True)
    park.take_ready()
    assert park.take_ready() == (set(), set())
