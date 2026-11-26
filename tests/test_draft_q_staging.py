# SPDX-License-Identifier: MIT
"""`DraftQStaging`: what lossless draft staging writes into each row.

What staging owes the kernels: every row the captured block head runs names a
destination (a slot its own request owns, or -1), every verifying request names
the rows its drafts were drawn into (or -1), and the draft's temperature and
greedy mask and filters follow the target's. Numpy only, so this runs in the
CPU unit job; `tests/test_spec_lossless_staging.py` covers the proposer wiring
around it.
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from atom.spec_decode.draft_q_slots import DraftQSlots
from atom.spec_decode.draft_q_staging import DraftQStaging

MAX_BS = 4
T = 3
EPS = 1e-10  # SAMPLER_EPS


def staging(slots=2 * MAX_BS):
    arrays = {
        "seed": np.zeros(1, np.int32),
        "q_row_base": np.full(MAX_BS, -1, np.int32),
        "verify_q_row_base": np.full(MAX_BS, -1, np.int32),
    }
    return (
        DraftQStaging(DraftQSlots(slots, T), greedy_threshold=EPS, **arrays),
        SimpleNamespace(**arrays),
    )


def batch(
    req_ids,
    temperatures=None,
    *,
    top_ks=None,
    top_ps=None,
    dummy=False,
    final_chunk=None,
):
    n = len(req_ids)
    temps = np.ones(n, np.float32) if temperatures is None else temperatures
    return SimpleNamespace(
        req_ids=list(req_ids),
        temperatures=np.asarray(temps, np.float32),
        # `ScheduledBatch`'s spelling: -1 and 1.0 are "off".
        top_ks=np.asarray([-1] * n if top_ks is None else top_ks, np.int32),
        top_ps=np.asarray([1.0] * n if top_ps is None else top_ps, np.float32),
        is_dummy_run=dummy,
        is_final_chunk=final_chunk,
    )


def stage(st, b, *, verifiable=True, verifies=True):
    """One forward's staging."""
    return st.stage(b, len(b.req_ids), drafts_verifiable=verifiable, verifies=verifies)


def test_a_verify_finds_the_rows_its_drafts_were_written_to():
    st, a = staging()
    stage(st, batch(["a", "b", "c"]))
    drafted = dict(zip("abc", a.q_row_base[:3].tolist()))
    assert a.verify_q_row_base[:3].tolist() == [-1, -1, -1], "nothing drafted yet"
    # Reordered, with a request that never drafted.
    stage(st, batch(["c", "x", "a"]))
    assert a.verify_q_row_base[:3].tolist() == [drafted["c"], -1, drafted["a"]]
    # The same requests draft back into the rows they own.
    assert a.q_row_base[0] == drafted["c"] and a.q_row_base[2] == drafted["a"]


def test_a_dummy_forward_claims_nothing_and_verifies_nothing():
    """Warmup and DP dummies carry ids 0..n-1 and -1; none may enter the map,
    or a real request 0 would later verify against a dummy's q."""
    st, a = staging()
    rows = stage(st, batch([0, 1, -1], dummy=True), verifiable=False, verifies=False)
    assert a.q_row_base[:3].tolist() == [-1, -1, -1]
    assert rows.verify_q_row_base is None
    assert len(st.slots) == 0
    stage(st, batch([0]))
    assert a.verify_q_row_base[0] == -1


def test_an_align_only_propose_writes_no_q_and_keeps_other_slots():
    st, a = staging()
    stage(st, batch(["d"]))
    d = a.q_row_base[0]
    rows = stage(st, batch(["m1", "m2"]), verifiable=False, verifies=False)
    assert a.q_row_base[:2].tolist() == [-1, -1]
    assert rows.q_row_base == 2
    assert "m1" not in st.slots and "m2" not in st.slots
    stage(st, batch(["d"]))
    assert a.verify_q_row_base[0] == d


def test_a_request_re_prefilled_after_its_slot_expired_reads_its_new_slot():
    """Re-prefilled in chunks after its slot expired, a request finds no q
    until its final chunk claims a slot, and its first verify reads that
    slot."""
    st, a = staging(slots=1)
    stage(st, batch(["a"]))
    stage(st, batch(["b"]))  # a expires
    stage(st, batch(["a"]), verifiable=False, verifies=False)  # middle chunk
    stage(st, batch(["a"], final_chunk=[True]), verifies=False)
    claimed = a.q_row_base[0]
    stage(st, batch(["a"]))
    assert a.verify_q_row_base[0] == claimed


def test_a_mid_prompt_member_expired_by_its_own_batch_loses_nothing():
    """A skipped member keeps its map entry through the claims, so a new
    request in the same batch may expire it. Its drafts are dropped until its
    final chunk, which claims again before its first verify reads."""
    st, a = staging(slots=2)
    stage(st, batch(["m"]))
    stage(st, batch(["b"]))
    stage(
        st,
        batch(["m", "c"], final_chunk=[False, True]),
        verifies=False,
    )
    assert "m" not in st.slots
    assert a.q_row_base[0] == -1
    stage(st, batch(["m"], final_chunk=[True]), verifies=False)
    claimed = a.q_row_base[0]
    assert claimed >= 0
    stage(st, batch(["m"]))
    assert a.verify_q_row_base[0] == claimed


def test_an_all_greedy_step_does_not_touch_the_slots():
    """The runner passes ``verifies`` False for an all-greedy batch, whose
    exact-match verify reads no q: no lookup, no host write, no verify row
    published. The next sampled verify still finds its q."""
    st, a = staging()
    stage(st, batch(["s", "g"], [1.0, 0.0]))
    drafted = a.q_row_base[0]
    before = a.verify_q_row_base.copy()
    with mock.patch.object(
        st.slots, "verify_bases", wraps=st.slots.verify_bases
    ) as lookup:
        rows = stage(st, batch(["g"], [0.0]), verifies=False)
        lookup.assert_not_called()
        assert rows.verify_q_row_base is None
        np.testing.assert_array_equal(a.verify_q_row_base, before)
        rows = stage(st, batch(["s", "g"], [1.0, 0.0]))
        lookup.assert_called_once()
    assert rows.verify_q_row_base == 2
    assert a.verify_q_row_base[:2].tolist() == [drafted, -1]


def test_a_greedy_request_claims_no_slot_and_writes_no_q():
    """Its verify is exact match and never reads q; a sampled request beside
    it, filtered or not, still gets the rows its q is read from."""
    st, a = staging()
    for top_k in (-1, 5):
        stage(st, batch(["g", "s"], [0.0, 1.0], top_ks=[top_k, top_k]))
        assert a.q_row_base[0] == -1 and a.q_row_base[1] >= 0
        assert "g" not in st.slots and "s" in st.slots
    stage(st, batch(["g", "s"], [0.0, 1.0]))
    assert a.verify_q_row_base[0] == -1
    assert a.verify_q_row_base[1] == a.q_row_base[1]


def test_a_request_still_mid_prompt_claims_no_slot():
    """In a prefill batch where some requests finish their prompt and others
    do not, the unfinished ones' drafts are dropped, so they claim nothing."""
    st, a = staging()
    stage(
        st,
        batch(["done", "mid", "done2"], final_chunk=[True, False, True]),
        verifies=False,
    )
    assert a.q_row_base[1] == -1
    assert a.q_row_base[0] >= 0 and a.q_row_base[2] >= 0
    assert "mid" not in st.slots
    stage(st, batch(["mid"], final_chunk=[True]), verifies=False)
    assert "mid" in st.slots and a.q_row_base[0] >= 0


@pytest.mark.parametrize(
    "b, verifiable, draws_q",
    [
        (batch(["s", "g"], [1.0, 0.0]), True, True),
        (batch(["g", "h"], [0.0, 0.0]), True, False),
        (batch(["e"], [EPS]), True, False),
        (batch(["a", "b"]), False, False),
        (batch(["m", "n"], final_chunk=[False, False]), True, False),
        (batch(["d", "m"], final_chunk=[True, False]), True, True),
        (batch([]), True, False),
    ],
    ids=[
        "mixed",
        "all-greedy",
        "eps-temperature",
        "dummy-or-align-only",
        "all-mid-prompt",
        "one-final-chunk",
        "empty",
    ],
)
def test_a_step_draws_q_only_where_some_row_names_a_destination(b, verifiable, draws_q):
    """`draws_q` is what picks the greedy block: False exactly when no row of
    the step was handed a q destination."""
    st, a = staging()
    stage(st, b, verifiable=verifiable, verifies=False)
    assert st.draws_q is draws_q
    assert draws_q is bool((a.q_row_base >= 0).any())


def test_clearing_the_destinations_draws_no_q():
    st, _ = staging()
    stage(st, batch(["a"]))
    assert st.draws_q
    st.clear_destinations()
    assert not st.draws_q


def test_a_smaller_batch_clears_the_larger_ones_pad_rows():
    """The block runs at `running_bs >= scheduled_bs`; a pad row still naming a
    previous batch's slot would write its q into a live request's rows."""
    st, a = staging()
    assert stage(st, batch(list("abcd"))).q_row_base == 4
    rows = stage(st, batch(list("ab")))
    assert a.q_row_base[2:].tolist() == [-1, -1]
    assert rows.q_row_base == 4, "the cleared tail is published too"
    assert stage(st, batch(list("ab"))).q_row_base == 2


def test_the_philox_key_advances_once_per_step():
    st, a = staging()
    keys = []
    for _ in range(3):
        stage(st, batch(["a"]))
        keys.append(int(a.seed[0]))
    assert keys == [0, 1, 2]


def test_an_empty_batch_publishes_only_the_key_and_any_stale_tail():
    st, a = staging()
    stage(st, batch(["a", "b"]))
    rows = stage(st, batch([]))
    assert rows.seed == 1 and rows.verify_q_row_base is None
    assert rows.q_row_base == 2 and a.q_row_base[:2].tolist() == [-1, -1]


def test_a_propose_needs_a_staging_of_its_own():
    """Same batch size twice in a row is not enough: each propose must consume
    a staging of its own, or it would draft through stale row bases."""
    st, _ = staging()
    with pytest.raises(RuntimeError, match="fresh DSparkLossless.stage"):
        st.consume()
    for _ in range(2):
        stage(st, batch(["a", "b"]))
        st.consume()
        with pytest.raises(RuntimeError, match="fresh DSparkLossless.stage"):
            st.consume()


def test_clearing_the_destinations_keeps_the_slot_map():
    """What a draft-graph re-warm needs: no row writes q, yet requests that
    span it keep the slots their q lives in."""
    st, a = staging()
    stage(st, batch(["a", "b", "c"]))
    st.clear_destinations()
    assert a.q_row_base.tolist() == [-1] * MAX_BS
    assert {"a", "b", "c"} <= set(st.slots._slot_of)
    # The next step after it clears only what it must.
    assert stage(st, batch(["a"])).q_row_base == 1
