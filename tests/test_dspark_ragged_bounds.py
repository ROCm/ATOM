# SPDX-License-Identifier: MIT
"""DSpark ragged verify length is bounded on BOTH sides.

Violating either bound desyncs the batch layout from what the scheduler
reserved, and that surfaces far downstream as an out-of-range token id in the
draft's Markov transition-table lookup (a device-side ASSERT_TRAP attributed to
whatever kernel happened to be in flight), never as a failure where the length
is chosen. So pin both bounds here, on the CPU.

Observed on DeepSeek-V4-Pro-DSpark with `ragged: true` and the compiled draft:
the ell map is read without syncing, so in steady state it is two steps old,
while the caller's boundary guard only compares step N-1's request set with step
N's. A request present in both may have had a longer segment back at N-2, and
its stale ell then exceeds what the scheduler gave it now.
"""

from atom.spec_decode.dspark_scheduler import flat_bucket_fits, ragged_verify_len

FULL_Q = 6


def test_stale_ell_cannot_grow_past_scheduled_len():
    # This seq was already shrunk to 2 by an earlier step; a STALE ell of 5 would
    # ask for 6 and spill into the next seq's segment.
    assert ragged_verify_len(5, FULL_Q, 0, 2) == 2


def test_fresh_ell_still_shrinks():
    assert ragged_verify_len(2, FULL_Q, 0, FULL_Q) == 3
    assert ragged_verify_len(0, FULL_Q, 0, FULL_Q) == 1


def test_missing_ell_verifies_full_length():
    # No ell yet (new request, or its copy still in flight) -> never under-verify.
    assert ragged_verify_len(None, FULL_Q, 0, FULL_Q) == FULL_Q
    # ...but still bounded by what was actually scheduled.
    assert ragged_verify_len(None, FULL_Q, 0, 3) == 3


def test_length_covers_bonus_tokens():
    # Lower bound: must cover max_num_bonus even when ell is smaller, or the
    # anchor falls outside the shrunk segment.
    assert ragged_verify_len(0, FULL_Q, 3, FULL_Q) == 4


def test_upper_bound_never_breaks_the_lower_bound():
    """The trap a naive one-sided clamp falls into: capping at scheduled_len
    must not push the length below max_num_bonus + 1. When the bounds cross
    there is no representable length and the caller must stay rectangular."""
    assert ragged_verify_len(1, FULL_Q, 3, 2) is None
    # Exactly at the bound is fine.
    assert ragged_verify_len(1, FULL_Q, 3, 4) == 4


def test_clamped_to_full_q_and_min_one():
    assert ragged_verify_len(99, FULL_Q, 0, FULL_Q) == FULL_Q
    assert ragged_verify_len(-5, FULL_Q, 0, FULL_Q) == 1


def test_nonpositive_scheduled_len_imposes_no_bound():
    # 0 means "no scheduler bound to honor" -- do not collapse the length to 0.
    assert ragged_verify_len(2, FULL_Q, 0, 0) == 3


def test_both_bounds_hold_across_the_grid():
    for ell in (None, -1, 0, 1, 3, 5, 50):
        for max_nb in (0, 1, 4):
            for sched in (1, 2, 3, FULL_Q):
                li = ragged_verify_len(ell, FULL_Q, max_nb, sched)
                if li is None:
                    assert sched < max_nb + 1, (ell, max_nb, sched)
                    continue
                assert max_nb + 1 <= li <= FULL_Q, (ell, max_nb, sched, li)
                assert li <= sched, (ell, max_nb, sched, li)


# ---------------------------------------------------------------------------
# The ragged shrink is only safe if the REPLAY can follow it down.


def test_full_cudagraphs_cannot_represent_a_ragged_shrink():
    """No captured flat bucket set -> no representable shrink.

    This is the configuration that trapped: under FULL (non-PIECEWISE)
    cudagraphs nothing flat is captured, `_dynamic_num_tokens_pad` returns None,
    every caller falls back to `bs * max_seqlen_q`, and the replay runs over
    more tokens than the rebuild populated. The tail holds the previous step's
    ids, which the draft's Markov lookup then indexes out of range.
    """
    assert flat_bucket_fits(3, 6, []) is False
    assert flat_bucket_fits(3, 6, None) is False


def test_bucket_must_cover_the_total_and_divide_by_q():
    buckets = [6, 12, 24]
    # 3 real tokens at q=6 -> bucket 6 holds them and 6 % 6 == 0.
    assert flat_bucket_fits(3, 6, buckets) is True
    assert flat_bucket_fits(6, 6, buckets) is True
    # Bigger than every captured bucket -> nothing can hold it.
    assert flat_bucket_fits(25, 6, buckets) is False
    # Covers the total but is not q-divisible -> the per-seq rows would not tile.
    assert flat_bucket_fits(3, 6, [10]) is False
    # Degenerate q never matches (guards the b % q modulo).
    assert flat_bucket_fits(3, 0, buckets) is False


def test_predicate_agrees_with_dynamic_num_tokens_pad():
    """Must stay in step with the lookup it mirrors: yes exactly when that
    lookup finds a bucket, no exactly when it returns the None that triggers the
    unsafe `bs * max_seqlen_q` fallback."""
    buckets = [6, 12, 24]

    def pad_lookup(total, q):
        for b in buckets:
            if b >= total and q > 0 and b % q == 0:
                return b
        return None

    for total in (0, 1, 3, 6, 7, 12, 25):
        for q in (0, 1, 3, 6):
            assert flat_bucket_fits(total, q, buckets) is (
                pad_lookup(total, q) is not None
            ), (total, q)
