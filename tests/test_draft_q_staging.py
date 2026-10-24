# SPDX-License-Identifier: MIT
"""`DraftQStaging`: what lossless draft staging writes into each row.

What staging owes the kernels: every row the captured block head runs names a
destination (a slot its own request owns, or -1), every verifying request names
the rows its drafts were drawn into (or -1), and the draft's temperature and
greedy mask follow the target's. Numpy only, so this runs in the CPU unit job;
`tests/test_spec_lossless_staging.py` covers the proposer wiring around it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from atom.spec_decode.draft_q_slots import DraftQSlots, DraftQStaging

MAX_BS = 4
T = 3
EPS = 1e-10  # SAMPLER_EPS


def staging(slots=2 * MAX_BS):
    arrays = {
        "temperatures": np.ones(MAX_BS, np.float32),
        "greedy_mask": np.ones(MAX_BS, np.int32),
        "seed": np.zeros(1, np.int32),
        "dest_row_base": np.full(MAX_BS, -1, np.int32),
        "verify_row_base": np.full(MAX_BS, -1, np.int32),
    }
    return (
        DraftQStaging(DraftQSlots(slots, T), greedy_threshold=EPS, **arrays),
        SimpleNamespace(**arrays),
    )


def batch(req_ids, temperatures=None, *, dummy=False, final_chunk=None):
    temps = np.ones(len(req_ids), np.float32) if temperatures is None else temperatures
    return SimpleNamespace(
        req_ids=list(req_ids),
        temperatures=np.asarray(temps, np.float32),
        is_dummy_run=dummy,
        is_final_chunk=final_chunk,
    )


def stage(st, b, *, has_filter=False, verifiable=True):
    return st.stage(
        b, len(b.req_ids), has_filter=has_filter, drafts_verifiable=verifiable
    )


def test_a_verify_finds_the_rows_its_drafts_were_written_to():
    st, a = staging()
    stage(st, batch(["a", "b", "c"]))
    drafted = dict(zip("abc", a.dest_row_base[:3].tolist()))
    assert a.verify_row_base[:3].tolist() == [-1, -1, -1], "nothing drafted yet"
    # Reordered, with a request that never drafted.
    stage(st, batch(["c", "x", "a"]))
    assert a.verify_row_base[:3].tolist() == [drafted["c"], -1, drafted["a"]]
    # The same requests draft back into the rows they own.
    assert a.dest_row_base[0] == drafted["c"] and a.dest_row_base[2] == drafted["a"]


def test_a_dummy_forward_claims_nothing_and_verifies_nothing():
    """Warmup and DP dummies carry ids 0..n-1 and -1; none may enter the map,
    or a real request 0 would later verify against a dummy's q."""
    st, a = staging()
    stage(st, batch([0, 1, -1], dummy=True), verifiable=False)
    assert a.dest_row_base[:3].tolist() == [-1, -1, -1]
    assert a.verify_row_base[:3].tolist() == [-1, -1, -1]
    assert len(st.slots) == 0
    stage(st, batch([0]))
    assert a.verify_row_base[0] == -1


def test_an_align_only_propose_writes_no_q_and_keeps_other_slots():
    st, a = staging()
    stage(st, batch(["d"]))
    d = a.dest_row_base[0]
    rows = stage(st, batch(["m1", "m2"]), verifiable=False)
    assert a.dest_row_base[:2].tolist() == [-1, -1]
    assert rows["dest_row_base"] == 2
    assert "m1" not in st.slots and "m2" not in st.slots
    stage(st, batch(["d"]))
    assert a.verify_row_base[0] == d


def test_a_greedy_request_claims_no_slot_and_writes_no_q():
    """Its verify is exact match and never reads q; a sampled request beside
    it, filtered batch or not, still gets the rows its q is read from."""
    st, a = staging()
    for has_filter in (False, True):
        stage(st, batch(["g", "s"], [0.0, 1.0]), has_filter=has_filter)
        assert a.dest_row_base[0] == -1 and a.dest_row_base[1] >= 0
        assert "g" not in st.slots and "s" in st.slots
    stage(st, batch(["g", "s"], [0.0, 1.0]))
    assert a.verify_row_base[0] == -1
    assert a.verify_row_base[1] == a.dest_row_base[1]


def test_a_request_still_mid_prompt_claims_no_slot():
    """In a prefill batch where some requests finish their prompt and others
    do not, the unfinished ones' drafts are dropped, so they claim nothing."""
    st, a = staging()
    stage(st, batch(["done", "mid", "done2"], final_chunk=[True, False, True]))
    assert a.dest_row_base[1] == -1
    assert a.dest_row_base[0] >= 0 and a.dest_row_base[2] >= 0
    assert "mid" not in st.slots
    stage(st, batch(["mid"], final_chunk=[True]))
    assert "mid" in st.slots and a.dest_row_base[0] >= 0


def test_a_smaller_batch_clears_the_larger_ones_pad_rows():
    """The block runs at `running_bs >= scheduled_bs`; a pad row still naming a
    previous batch's slot would write its q into a live request's rows."""
    st, a = staging()
    assert stage(st, batch(list("abcd")))["dest_row_base"] == 4
    rows = stage(st, batch(list("ab")))
    assert a.dest_row_base[2:].tolist() == [-1, -1]
    assert rows["dest_row_base"] == 4, "the cleared tail is published too"
    assert stage(st, batch(list("ab")))["dest_row_base"] == 2


def test_a_filtered_batch_drafts_greedily():
    """Only the mask changes: the delta q comes from the greedy draw itself,
    so the temperatures stay the requests' own (clamped) ones."""
    st, a = staging()
    stage(st, batch(["a", "b"], [0.7, 1.3]), has_filter=True)
    assert a.greedy_mask[:2].tolist() == [1, 1]
    np.testing.assert_allclose(a.temperatures[:2], [0.7, 1.3], rtol=1e-6)


def test_an_unfiltered_batch_keeps_each_rows_temperature_and_greedy_predicate():
    st, a = staging()
    temps = np.array([0.0, EPS, 0.7], np.float32)
    stage(st, batch(["a", "b", "c"], temps))
    np.testing.assert_array_equal(a.temperatures[:3], np.maximum(temps, EPS))
    # The predicate `Sampler` tests, so draft and target agree on greedy.
    assert a.greedy_mask[:3].tolist() == [1, 1, 0]


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
    assert rows["seed"] == 1
    assert rows["temperatures"] is None and rows["verify_row_base"] is None
    assert rows["dest_row_base"] == 2 and a.dest_row_base[:2].tolist() == [-1, -1]


def test_a_propose_needs_a_staging_of_its_own():
    """Same batch size twice in a row is not enough: each propose must consume
    a staging of its own, or it would draft through stale row bases."""
    st, _ = staging()
    with pytest.raises(AssertionError, match="fresh stage_draft_sampling_host"):
        st.consume()
    for _ in range(2):
        stage(st, batch(["a", "b"]))
        st.consume()
        with pytest.raises(AssertionError):
            st.consume()


def test_clearing_the_destinations_keeps_the_slot_map():
    """What a draft-graph re-warm needs: no row writes q, yet requests that
    span it keep the slots their q lives in."""
    st, a = staging()
    stage(st, batch(["a", "b", "c"]))
    st.clear_destinations()
    assert a.dest_row_base.tolist() == [-1] * MAX_BS
    assert {"a", "b", "c"} <= set(st.slots._slot_of)
    # The next step after it clears only what it must.
    assert stage(st, batch(["a"]))["dest_row_base"] == 1
