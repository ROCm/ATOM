# SPDX-License-Identifier: MIT
"""A model-owned draft must never be assigned the ordinary drafter's stale q."""

from types import SimpleNamespace as NS

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield NS(
            runner=load("atom.model_engine.model_runner"),
            prob=load("atom.model_ops.dspark_prob_sample"),
            proposer=load("atom.spec_decode.dspark_proposer"),
            context=load("atom.utils.forward_context"),
        )


def state(modules):
    # Ordinary CPU buffers exercise the real ownership map without requiring
    # the pinned-memory allocator used by the GPU production setup.
    return modules.prob.ProbDraftState(
        q=torch.zeros(3, 2, 4),
        ids=torch.full((3, 2), -1, dtype=torch.int32),
        temps=torch.ones(3),
        seed=torch.zeros(1, dtype=torch.int64),
        src_cpu=[torch.empty(3, dtype=torch.int32) for _ in range(2)],
        src_gpu=torch.empty(3, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "source", ["ordinary", "step_override", "FORWARD", "LEAN", "DEVICE"]
)
def test_propose_only_publishes_q_for_ordinary_drafts(modules, monkeypatch, source):
    mod = modules.runner
    st = state(modules)
    st.record_owner([7, 8])
    prop = modules.proposer.DSparkProposer.__new__(modules.proposer.DSparkProposer)
    prop._prob_state, prop._verify_scheduler = st, None
    prop.prepare_inputs = lambda count: torch.tensor([0, 1])
    ids = torch.tensor([[1, 2], [2, 3]], dtype=torch.int32)

    def propose(**kwargs):
        if source == "ordinary":
            st.begin_draft(torch.ones(2), 2)
            st.ids[:2].copy_(ids)
        return ids

    prop.propose = propose
    service = getattr(
        modules.context.RoundService, source, modules.context.RoundService.NONE
    )
    context = NS(
        attn_metadata=NS(served_by_round=service),
        context=NS(positions=torch.arange(2)),
    )
    monkeypatch.setattr(mod, "get_forward_context", lambda: context)
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    runner = mod.ModelRunner.__new__(mod.ModelRunner)
    runner.drafter = prop
    runner._served = object() if source == "step_override" else None
    runner.attn_metadata_builder = NS(commit_speculative_state=lambda *args: None)
    runner.tokenID_processor = NS(prepare_draft_ids=lambda batch, tokens: tokens)
    batch = NS(next_token_ids=None, total_seqs_num=2, req_ids=[7, 8])

    actual = runner.propose_draft_token_ids(
        batch, torch.arange(2), torch.zeros(2, 1), torch.ones(2), None, None
    )
    assert actual is ids
    # Reordering and a newly admitted request are mapped by request ID.
    if source == "ordinary":
        assert st.src_for([8, 99, 7], 3).tolist() == [1, -1, 0]
    else:
        assert st.owner is None
        assert st.src_for([8, 99, 7], 3).tolist() == [-1, -1, -1]
        assert (st.ids == -1).all()  # the served draft never wrote these buffers


@pytest.mark.parametrize("align_only, has_draft", [(True, True), (False, False)])
def test_propose_without_usable_draft_clears_owner(
    modules, monkeypatch, align_only, has_draft
):
    mod = modules.runner
    st = state(modules)
    st.record_owner([7])
    prop = modules.proposer.DSparkProposer.__new__(modules.proposer.DSparkProposer)
    prop._prob_state, prop._verify_scheduler = st, None
    prop.prepare_inputs = lambda count: torch.tensor([0])
    prop.propose = lambda **kwargs: torch.ones(1, 2) if has_draft else None
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: NS(attn_metadata=None, context=NS(positions=torch.tensor([0]))),
    )
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    runner = mod.ModelRunner.__new__(mod.ModelRunner)
    runner.drafter, runner._served = prop, None
    runner.attn_metadata_builder = NS(commit_speculative_state=lambda *args: None)
    runner.tokenID_processor = NS(prepare_draft_ids=lambda batch, tokens: tokens)
    result = runner.propose_draft_token_ids(
        NS(next_token_ids=None, total_seqs_num=1, req_ids=[7]),
        torch.tensor([0]),
        torch.zeros(1, 1),
        torch.tensor([1]),
        None,
        None,
        align_only=align_only,
    )
    assert result is None
    assert st.src_for([7], 1).tolist() == [-1]


@pytest.mark.parametrize(
    "disabled",
    [
        "none",
        "feature",
        "top_k",
        "synthetic",
        "pcp",
        "step_override",
        "FORWARD",
        "LEAN",
        "DEVICE",
    ],
)
def test_ratio_state_gate(modules, monkeypatch, disabled):
    mod = modules.runner
    st = state(modules)
    runner = mod.ModelRunner.__new__(mod.ModelRunner)
    runner.drafter = NS(prob_state=None if disabled == "feature" else st)
    runner._served = object() if disabled == "step_override" else None
    runner.rejection_sampler = NS(
        synthetic_acceptance_rates=[] if disabled == "synthetic" else None
    )
    service = getattr(
        modules.context.RoundService, disabled, modules.context.RoundService.NONE
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: NS(attn_metadata=NS(served_by_round=service)),
    )
    monkeypatch.setattr(
        mod, "get_pcp_world_size", lambda: 2 if disabled == "pcp" else 1
    )
    result = runner._dspark_prob_state(
        torch.tensor([1]) if disabled == "top_k" else None
    )
    assert result is (st if disabled == "none" else None)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("lossless", [False, True])
@pytest.mark.parametrize("standalone", [False, True])
def test_lossless_and_legacy_probability_state_are_exclusive(
    modules, monkeypatch, legacy, lossless, standalone
):
    mod = modules.proposer
    prop = mod.DSparkProposer.__new__(mod.DSparkProposer)
    prop._prob_state = None
    prop.lossless = object() if lossless else None
    prop.speculative_config = NS(use_dspark_with_draft=lambda: standalone)
    # Loading a backbone is outside this CPU test: only the head capability
    # is read by the real proposer's probability-state arming decision.
    prop.model = type("ProbModel", (), {"_head_and_sample_prob": lambda self: None})()
    monkeypatch.setattr(mod.Drafter, "load_model", lambda self, target: None)
    monkeypatch.setattr(mod.envs, "ATOM_DSPARK_PROB_DRAFT", legacy)
    st = state(modules)
    prop._init_prob_state = lambda: setattr(prop, "_prob_state", st)
    prop.load_model(object())
    assert prop.prob_state is (
        st if legacy and not lossless and not standalone else None
    )
