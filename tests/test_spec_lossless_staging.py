# SPDX-License-Identifier: MIT
"""ATOM_SPEC_LOSSLESS_SAMPLING's host side: arming, per-step staging, and the
verify policy, run through the real `DSparkProposer` methods on CPU buffers.

What staging owes the kernels: every row the captured block head runs names a
destination (a slot its own request owns, or -1), every verifying request names
the rows its drafts were drawn into (or -1), and all of it rides the protected
`sampling` publication, so a host running ahead cannot rewrite a row base the
GPU has yet to read.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the proposer imports the AITER sampler ops")

from atom.model_ops.rejection_sampler import RejectionSampler
from atom.model_ops.sampler import SAMPLER_EPS
from atom.spec_decode import dspark_proposer
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.dspark_proposer import DSparkProposer

MAX_BS = 4
T = 3
VOCAB = 16
LOSSLESS = (
    "draft_temperatures",
    "draft_greedy_mask",
    "draft_sample_seed",
    "draft_q_row_base",
    "verify_q_row_base",
)


class FakeDraftModel:
    supports_lossless_draft = True
    vocab_size = VOCAB

    def __init__(self):
        self.markov_head = SimpleNamespace(
            markov_w1=nn.Embedding(VOCAB, 8), markov_w2=nn.Embedding(VOCAB, 8)
        )
        self.bound = None

    def bind_draft_sampling(self, sampling):
        self.bound = sampling


def proposer(monkeypatch, *, lossless=True, pp=1, pcp=1, model=None):
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1" if lossless else "0")
    monkeypatch.setattr(dspark_proposer, "get_pcp_world_size", lambda: pcp)
    # CPU stand-in weights; the GPU arming check has its own test below.
    monkeypatch.setattr(
        dspark_proposer, "dspark_markov_sample_unsupported", lambda w1, w2: None
    )
    prop = DSparkProposer.__new__(DSparkProposer)
    prop.config = SimpleNamespace(max_num_seqs=MAX_BS, pipeline_parallel_size=pp)
    prop.device = torch.device("cpu")
    prop.runner = SimpleNamespace(rank=1)
    prop.metadata_buffers = {}
    prop.mtp_k = T
    prop.model = FakeDraftModel() if model is None else model
    prop._init_lossless_sampling()
    return prop


def group(prop):
    names = list(prop.metadata_buffers)
    return SimpleNamespace(
        counts=[None] * len(names), indices={n: i for i, n in enumerate(names)}
    )


def batch(req_ids, temperatures=None, *, dummy=False):
    temps = np.ones(len(req_ids), np.float32) if temperatures is None else temperatures
    return SimpleNamespace(
        req_ids=list(req_ids),
        temperatures=np.asarray(temps, np.float32),
        is_dummy_run=dummy,
    )


def stage(prop, b, *, has_filter=False, verifiable=True):
    g = group(prop)
    prop.stage_draft_sampling_host(
        b, len(b.req_ids), g, has_filter=has_filter, drafts_verifiable=verifiable
    )
    return {name: g.counts[g.indices[name]] for name in LOSSLESS}


def dest(prop):
    return prop._draft_row_base_buf.np.tolist()


def verify(prop):
    return prop._verify_row_base_buf.np.tolist()


# ---- arming -----------------------------------------------------------------


def test_everything_the_kernels_read_rides_the_sampling_publication(monkeypatch):
    prop = proposer(monkeypatch)
    for name in LOSSLESS:
        assert prop.metadata_buffers[name].publication_group == "sampling", name
    sampling = prop.model.bound
    assert sampling.q_row_base is prop._draft_row_base_buf.gpu
    assert sampling.q_store.shape == (2 * MAX_BS * T, VOCAB)
    # Unpublished rows are harmless from the start: no destination, T=1.
    assert dest(prop) == [-1] * MAX_BS and verify(prop) == [-1] * MAX_BS
    assert prop._draft_temps_buf.np.tolist() == [1.0] * MAX_BS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lossless": False},
        {"pp": 2},
        {"pcp": 2},
        {"model": SimpleNamespace(supports_lossless_draft=False)},
    ],
    ids=["flag-off", "pp", "pcp", "unported-flavor"],
)
def test_the_greedy_pair_stays_when_not_armed(monkeypatch, kwargs):
    prop = proposer(monkeypatch, **kwargs)
    assert prop.draft_sampling is None and prop.draft_probs is None
    assert not set(LOSSLESS) & set(prop.metadata_buffers)
    assert prop.lossless_verify_inputs(2, all_greedy=False, has_filter=False) == (
        None,
        None,
    )
    assert prop.kv_budget_reserve_bytes() == 0
    # Staging is a no-op rather than an error.
    prop.stage_draft_sampling_host(
        batch(["a"]), 1, None, has_filter=False, drafts_verifiable=True
    )


def test_a_head_the_fused_op_cannot_run_keeps_the_greedy_pair(monkeypatch):
    prop = proposer(monkeypatch, lossless=False)
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    monkeypatch.setattr(
        dspark_proposer,
        "dspark_markov_sample_unsupported",
        lambda w1, w2: "markov weights are torch.float32/torch.float32",
    )
    prop._init_lossless_sampling()
    assert prop.draft_sampling is None and prop.model.bound is None


def test_the_scratch_reserve_is_counted_when_armed(monkeypatch):
    assert proposer(monkeypatch).kv_budget_reserve_bytes() > 0


# ---- staging ----------------------------------------------------------------


def test_a_verify_finds_the_rows_its_drafts_were_written_to(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b", "c"]))
    drafted = dict(zip("abc", dest(prop)[:3]))
    assert verify(prop)[:3] == [-1, -1, -1], "nothing drafted before step 1"
    # Reordered, with a request that never drafted.
    stage(prop, batch(["c", "x", "a"]))
    assert verify(prop)[:3] == [drafted["c"], -1, drafted["a"]]
    # The same requests draft back into the rows they own.
    assert dest(prop)[0] == drafted["c"] and dest(prop)[2] == drafted["a"]


def test_a_dummy_forward_claims_nothing_and_verifies_nothing(monkeypatch):
    """Warmup and DP dummies carry ids 0..n-1 and -1; none may enter the map,
    or a real request 0 would later verify against a dummy's q."""
    prop = proposer(monkeypatch)
    stage(prop, batch([0, 1, -1], dummy=True), verifiable=False)
    assert dest(prop)[:3] == [-1, -1, -1] and verify(prop)[:3] == [-1, -1, -1]
    assert len(prop._q_slots) == 0
    # A real request 0 afterwards has no q to verify.
    stage(prop, batch([0]))
    assert verify(prop)[0] == -1


def test_an_align_only_propose_writes_no_q_and_keeps_other_slots(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["d"]))
    (d,) = dest(prop)[:1]
    counts = stage(prop, batch(["m1", "m2"]), verifiable=False)
    assert dest(prop)[:2] == [-1, -1]
    assert counts["draft_q_row_base"] == 2
    assert "m1" not in prop._q_slots and "m2" not in prop._q_slots
    stage(prop, batch(["d"]))
    assert verify(prop)[0] == d


def test_a_smaller_batch_clears_the_larger_ones_pad_rows(monkeypatch):
    """The block runs at `running_bs >= scheduled_bs`; a pad row still naming a
    previous batch's slot would write its q into a live request's rows."""
    prop = proposer(monkeypatch)
    assert stage(prop, batch(list("abcd")))["draft_q_row_base"] == 4
    counts = stage(prop, batch(list("ab")))
    assert dest(prop)[2:] == [-1, -1]
    assert counts["draft_q_row_base"] == 4, "the cleared tail is published too"
    assert stage(prop, batch(list("ab")))["draft_q_row_base"] == 2


def test_a_filtered_batch_drafts_greedily(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"], [0.7, 1.3]), has_filter=True)
    assert prop._draft_temps_buf.np[:2].tolist() == [np.float32(SAMPLER_EPS)] * 2
    assert prop._draft_greedy_buf.np[:2].tolist() == [1, 1]


def test_an_unfiltered_batch_keeps_each_rows_temperature_and_greedy_predicate(
    monkeypatch,
):
    prop = proposer(monkeypatch)
    temps = np.array([0.0, SAMPLER_EPS, 0.7], np.float32)
    stage(prop, batch(["a", "b", "c"], temps))
    np.testing.assert_array_equal(
        prop._draft_temps_buf.np[:3], np.maximum(temps, SAMPLER_EPS)
    )
    # The predicate `Sampler` applies, so draft and target agree on greedy.
    assert prop._draft_greedy_buf.np[:3].tolist() == [1, 1, 0]


def test_the_philox_key_advances_once_per_step(monkeypatch):
    prop = proposer(monkeypatch)
    keys = []
    for _ in range(3):
        stage(prop, batch(["a"]))
        keys.append(int(prop._draft_seed_buf.np[0]))
    assert keys == [0, 1, 2]


def test_an_empty_batch_publishes_only_the_key_and_any_stale_tail(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"]))
    counts = stage(prop, batch([]))
    assert counts["draft_sample_seed"] == 1
    assert counts["draft_temperatures"] is None
    assert counts["verify_q_row_base"] is None
    assert counts["draft_q_row_base"] == 2 and dest(prop)[:2] == [-1, -1]


# ---- verify policy ----------------------------------------------------------


def test_only_an_unfiltered_sampled_batch_reaches_the_ratio_rule(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"]))
    q, base = prop.lossless_verify_inputs(2, all_greedy=False, has_filter=False)
    assert q is prop._q_store and base.shape == (2,)
    for all_greedy, has_filter in ((True, False), (False, True), (True, True)):
        assert prop.lossless_verify_inputs(
            2, all_greedy=all_greedy, has_filter=has_filter
        ) == (None, None)


@pytest.mark.parametrize(
    "all_greedy, has_q, draws",
    [(True, False, False), (False, False, True), (False, True, False)],
)
def test_the_target_draw_is_taken_exactly_when_exact_match_needs_it(
    all_greedy, has_q, draws
):
    q = torch.zeros(1) if has_q else None
    assert RejectionSampler.needs_target_draw(all_greedy, q) is draws


# ---- exclusion with relaxed MTP ----------------------------------------------


class _Bare(Drafter):
    def _resolve_mtp_k(self):
        return 1

    def propose(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("lossless, relaxed", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_relaxed_mtp_and_lossless_sampling_refuse_each_other_first(
    monkeypatch, lossless, relaxed
):
    """Refused before anything else and for every drafter flavor, so whether
    the pair errors cannot depend on PP, PCP, or the draft model."""
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", str(lossless))
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", str(relaxed))
    drafter = _Bare.__new__(_Bare)
    if lossless and relaxed:
        with pytest.raises(ValueError, match="ATOM_ENABLE_RELAXED_MTP"):
            Drafter.__init__(drafter, None, None, None)
    else:
        # Past the check, construction fails on the missing config instead.
        with pytest.raises(AttributeError):
            Drafter.__init__(drafter, None, None, None)
