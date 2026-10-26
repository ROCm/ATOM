# SPDX-License-Identifier: MIT
"""ATOM_SPEC_LOSSLESS_SAMPLING's proposer wiring: arming and its fallbacks, the
publication the staged buffers ride, the draft-graph warmup, and the verify
policy, run through the real `DSparkProposer` methods on CPU buffers.

Imports the proposer, hence aiter, so the CPU unit job skips this file; the
row-by-row staging policy itself is `DraftQStaging`, tested without aiter in
`tests/test_draft_q_staging.py`.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the proposer imports the AITER sampler ops")

from atom.model_ops.rejection_sampler import RejectionSampler
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
    "draft_top_k",
    "draft_top_p",
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


def proposer(monkeypatch, *, lossless=True, pp=1, pcp=1, model=None, **config):
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1" if lossless else "0")
    monkeypatch.setattr(dspark_proposer, "get_pcp_world_size", lambda: pcp)
    # CPU stand-in weights; the GPU arming check has its own test below.
    monkeypatch.setattr(
        dspark_proposer, "dspark_markov_sample_unsupported", lambda w1, w2: None
    )
    prop = DSparkProposer.__new__(DSparkProposer)
    prop.config = SimpleNamespace(
        max_num_seqs=MAX_BS, pipeline_parallel_size=pp, **config
    )
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


def batch(req_ids, temperatures=None, *, dummy=False, final_chunk=None):
    temps = np.ones(len(req_ids), np.float32) if temperatures is None else temperatures
    return SimpleNamespace(
        req_ids=list(req_ids),
        temperatures=np.asarray(temps, np.float32),
        is_dummy_run=dummy,
        is_final_chunk=final_chunk,
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
    assert sampling.top_k is prop._draft_top_k_buf.gpu
    assert sampling.top_p is prop._draft_top_p_buf.gpu
    # Unpublished rows are harmless from the start: no destination, T=1, no
    # filter.
    assert dest(prop) == [-1] * MAX_BS and verify(prop) == [-1] * MAX_BS
    assert prop._draft_temps_buf.np.tolist() == [1.0] * MAX_BS
    assert prop._draft_top_k_buf.gpu.tolist() == [0] * MAX_BS
    assert prop._draft_top_p_buf.gpu.tolist() == [1.0] * MAX_BS


@pytest.mark.parametrize("topk, width", [(None, VOCAB), ("5", 5), ("99999", VOCAB)])
def test_the_q_store_is_k_wide_and_never_wider_than_the_vocab(monkeypatch, topk, width):
    if topk is not None:
        monkeypatch.setenv("ATOM_SPEC_LOSSLESS_Q_TOPK", topk)
    sampling = proposer(monkeypatch).model.bound
    store = sampling.q_store
    assert store.ids.shape == store.probs.shape == (2 * MAX_BS * T, width)
    assert store.meta.shape == (2 * MAX_BS * T, 2)
    # The draft's dense scores are per batch row, whatever K is.
    assert sampling.s_scratch.shape == (MAX_BS, VOCAB)


def test_a_q_store_of_no_entries_is_refused(monkeypatch):
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_Q_TOPK", "0")
    with pytest.raises(ValueError, match="ATOM_SPEC_LOSSLESS_Q_TOPK"):
        proposer(monkeypatch)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lossless": False},
        {"pp": 2},
        {"pcp": 2},
        {"model": SimpleNamespace(supports_lossless_draft=False)},
        {"kv_transfer_config": {"kv_role": "kv_producer"}},
        {"kv_transfer_config": {"connectors": [{"kv_role": "kv_producer"}]}},
        {"enable_rapidserve": True, "disagg_is_decode": False},
    ],
    ids=[
        "flag-off",
        "pp",
        "pcp",
        "unported-flavor",
        "pd-producer",
        "pd-producer-multi",
        "rapidserve-prefill",
    ],
)
def test_the_greedy_pair_stays_when_not_armed(monkeypatch, kwargs):
    prop = proposer(monkeypatch, **kwargs)
    assert prop.draft_sampling is None and prop._q_store is None
    assert not set(LOSSLESS) & set(prop.metadata_buffers)
    assert prop.lossless_verify_inputs(2, has_filter=False) == (None, None, None)
    # Staging is a no-op rather than an error.
    prop.stage_draft_sampling_host(
        batch(["a"]), 1, None, has_filter=False, drafts_verifiable=True
    )


@pytest.mark.parametrize(
    "config",
    [
        {"kv_transfer_config": {"kv_role": "kv_consumer"}},
        {"enable_rapidserve": True, "disagg_is_decode": True},
    ],
    ids=["pd-consumer", "rapidserve-decode"],
)
def test_the_verifying_side_of_a_disaggregated_pair_still_arms(monkeypatch, config):
    assert proposer(monkeypatch, **config).draft_sampling is not None


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


def test_the_ops_scratch_is_allocated_once_when_armed(monkeypatch):
    """Next to the q store, before the KV cache is sized: the profiling run
    sees it and no step allocates."""
    prop = proposer(monkeypatch)
    assert prop.draft_sampling.scratch is prop._draft_scratch
    assert prop._draft_scratch.f32.numel() and prop._verify_scratch.f32.numel()


# ---- staging ----------------------------------------------------------------


def test_a_draft_graph_warmup_writes_through_no_row_base(monkeypatch):
    """A recapture after a rollout sleep/wake warms the block directly, not
    through propose; whatever bases the last served step staged must not
    reach it, or the synthetic batch overwrites live requests' q."""
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b", "c"]))
    assert any(base >= 0 for base in dest(prop)[:3])
    seen = {}

    def base_warmup(self, build_context, stream, end_trace=None):
        seen["np"] = self._draft_row_base_buf.np.tolist()
        seen["gpu"] = self._draft_row_base_buf.gpu.tolist()

    monkeypatch.setattr(dspark_proposer.Drafter, "warmup_draft_graphs", base_warmup)
    prop.warmup_draft_graphs(None, None)
    assert seen["np"] == [-1] * MAX_BS and seen["gpu"] == [-1] * MAX_BS
    # The slot map is untouched: those requests still own their q rows.
    assert {"a", "b", "c"} <= set(prop._staging.slots._slot_of)


def test_an_empty_batch_publishes_only_the_key_and_any_stale_tail(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"]))
    counts = stage(prop, batch([]))
    assert counts["draft_sample_seed"] == 1
    assert counts["draft_temperatures"] is None
    assert counts["verify_q_row_base"] is None
    assert counts["draft_q_row_base"] == 2 and dest(prop)[:2] == [-1, -1]


# ---- verify policy ----------------------------------------------------------


def test_only_an_unfiltered_batch_reaches_the_ratio_rule(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"]))
    q, base, scratch = prop.lossless_verify_inputs(2, has_filter=False)
    # The sparse store itself, and the draft's score scratch to spread a
    # rejecting request's row into: one dense row per batch row, V wide.
    assert q.store is prop._q_store and q.dense is prop._s_scratch
    assert q.dense.shape == (MAX_BS, VOCAB)
    assert base.shape == (2,) and scratch is prop._verify_scratch
    assert prop.lossless_verify_inputs(2, has_filter=True) == (None, None, None)


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
