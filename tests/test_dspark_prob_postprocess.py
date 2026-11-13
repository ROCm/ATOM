# SPDX-License-Identifier: MIT
"""Probability-ratio verification must honor masks and retain raw logprobs."""

from types import SimpleNamespace as NS

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield (
            load("atom.model_engine.model_runner"),
            load("atom.model_ops.dspark_prob_sample"),
        )


@pytest.mark.parametrize("probabilistic", [False, True])
def test_structured_verify_keeps_raw_logprobs(modules, monkeypatch, probabilistic):
    mod, prob = modules
    metadata = NS(
        input_ids=torch.tensor([1, 1]),
        target_logits_indices=torch.tensor([0]),
        bonus_logits_indices=torch.tensor([1]),
        num_spec_steps=1,
        cu_num_draft_tokens=torch.tensor([1]),
        num_draft_tokens_np=[1],
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: NS(spec_decode_metadata=metadata, attn_metadata=None),
    )
    monkeypatch.setattr(mod, "get_tp_group", lambda: NS(world_size=1))
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(prob, "PROB_DEBUG", 0)

    batch = NS(
        total_seqs_num=1,
        total_tokens_num=2,
        req_ids=[7],
        any_logprobs=True,
        return_logprobs=[True],
        logprob_requested=[0],
        logprob_rows=None,
        top_logprobs=[1],
        structured_requests={0: object()},
        required_channel=None,
        num_spec_step=1,
    )
    runner = object.__new__(mod.ModelRunner)
    runner.forward_done_event = NS(record=lambda *args: None)
    runner._served = None
    # object.__new__ skips __init__, so the runner has no bonus side stream:
    # with _bonus_stream None the draw stays inline even though the
    # probabilistic step asks for the overlap.
    runner._bonus_stream = runner._bonus_ready = None
    observed = {}
    prefixes = {0: object()}

    def apply(logits, batch, metadata):
        # Token zero has the greatest raw score but the grammar disallows it.
        logits[:, 0] = -torch.inf
        return prefixes

    def commit(ids, batch, actual_prefixes):
        assert actual_prefixes is prefixes
        observed["committed"] = ids.clone()

    runner.structured_output = NS(vocab_size=3, apply=apply, commit=commit, errors={})

    class Sampler:
        def __call__(self, logits, **kwargs):
            return logits.argmax(-1)

        def sample_verification_tokens(self, logits, *args):
            observed["ordinary_verification"] = True
            return logits.argmax(-1)

    runner.sampler = Sampler()

    def verify(target_logits, bonus, target_ids=None):
        assert torch.isneginf(target_logits[:, 0]).all()
        draft = target_logits.argmax(-1) if target_ids is None else target_ids
        return torch.stack((draft, bonus), dim=1), torch.tensor([[1], [0], [1]])

    def ratio_verify(*args, **kwargs):
        observed["ratio_verification"] = True
        observed["in_place"] = kwargs.get("in_place")
        # The ratio test may read the target rows of the full logits in place.
        logits = args[4]
        if kwargs.get("in_place"):
            logits = logits.index_select(0, args[1])
        return verify(logits, args[5])

    def ordinary_verify(
        metadata, target_logits, bonus, *, target_token_ids, ratio=None
    ):
        return verify(target_logits, bonus, target_token_ids)

    monkeypatch.setattr(prob, "prob_rejection_sample", ratio_verify)
    runner.rejection_sampler = NS(
        synthetic_acceptance_rates=None, forward=ordinary_verify
    )
    state = NS(src_for=lambda requests, count: torch.tensor([0]))
    runner.drafter = NS(
        prob_state=state if probabilistic else None,
        lossless=None,
        context_kv_only=False,
    )
    monkeypatch.setattr(runner, "propose_draft_token_ids", lambda *args: None)

    def prepare(batch, ids, event, logprobs, **kwargs):
        observed["logprobs"] = logprobs.clone()
        return {7: ids[0].tolist()}, {}

    runner.tokenID_processor = NS(
        is_deferred_out=False,
        prev_batch=None,
        prepare_sampled_ids=prepare,
        input_ids=NS(gpu=torch.zeros(3, dtype=torch.int64)),
    )
    raw = torch.tensor([[5.0, 4.0, 0.0], [6.0, 3.0, 0.0]])
    temperatures = torch.ones(1)
    runner.postprocess(
        batch,
        raw.clone(),
        temperatures,
        None,
        None,
        False,
        hidden_states=torch.zeros(2, 1),
    )

    assert observed["committed"].tolist() == [[1, 1]]
    assert observed.get("ratio_verification", False) is probabilistic
    assert observed.get("ordinary_verification", False) is not probabilistic
    assert (runner._dspark_draft_temps is temperatures) is probabilistic
    # No required channel, so the ratio test reads the full logits in place
    # rather than a copy of the target rows.
    if probabilistic:
        assert observed["in_place"] is True
    payload = observed["logprobs"][0]
    torch.testing.assert_close(payload[:, 0], raw.log_softmax(-1)[:, 1])
    # Raw top candidate and sampled-token rank survive grammar masking.
    assert payload[:, 2].view(torch.int32).tolist() == [0, 0]
    assert payload[:, 3].view(torch.int32).tolist() == [2, 2]


def test_required_channel_verifies_the_masked_copy(modules, monkeypatch):
    """With a required channel the ratio verification must not read `logits`
    in place: the channel mask writes into the compacted target rows, and the
    verification has to see that masked copy."""
    mod, prob = modules
    metadata = NS(
        input_ids=torch.tensor([1, 1]),
        target_logits_indices=torch.tensor([0]),
        bonus_logits_indices=torch.tensor([1]),
        num_spec_steps=1,
        cu_num_draft_tokens=torch.tensor([1]),
        num_draft_tokens_np=[1],
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: NS(spec_decode_metadata=metadata, attn_metadata=None),
    )
    monkeypatch.setattr(mod, "get_tp_group", lambda: NS(world_size=1))
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(prob, "PROB_DEBUG", 0)

    batch = NS(
        total_seqs_num=1,
        total_tokens_num=2,
        req_ids=[7],
        any_logprobs=False,
        return_logprobs=[False],
        logprob_requested=[],
        logprob_rows=None,
        top_logprobs=[0],
        structured_requests={},
        required_channel=object(),
        num_spec_step=1,
    )
    runner = object.__new__(mod.ModelRunner)
    runner.forward_done_event = NS(record=lambda *args: None)
    runner._served = None
    runner._bonus_stream = runner._bonus_ready = None
    runner.structured_output = None
    observed = {}

    def constrain(batch, bonus_logits, md=None, target_logits=None):
        # The channel forbids token 0 in the compacted target rows.
        assert target_logits is not None
        target_logits[:, 0] = -torch.inf
        observed["masked"] = target_logits
        # None: no channel states to advance after the step.

    monkeypatch.setattr(runner, "_constrain_required_channel", constrain)

    class Sampler:
        def __call__(self, logits, **kwargs):
            return logits.argmax(-1)

    runner.sampler = Sampler()

    def ratio_verify(*args, **kwargs):
        observed["in_place"] = kwargs.get("in_place")
        observed["verified"] = args[4]
        draft = args[4].argmax(-1)
        return torch.stack((draft, args[5]), dim=1), torch.tensor([[1], [0], [1]])

    monkeypatch.setattr(prob, "prob_rejection_sample", ratio_verify)
    runner.rejection_sampler = NS(synthetic_acceptance_rates=None)
    runner.drafter = NS(
        prob_state=NS(src_for=lambda requests, count: torch.tensor([0])),
        lossless=None,
        context_kv_only=False,
    )
    monkeypatch.setattr(runner, "propose_draft_token_ids", lambda *args: None)

    def prepare(batch, ids, event, logprobs, **kwargs):
        observed["committed"] = ids.clone()
        return {7: ids[0].tolist()}, {}

    runner.tokenID_processor = NS(
        is_deferred_out=False,
        prev_batch=None,
        prepare_sampled_ids=prepare,
        input_ids=NS(gpu=torch.zeros(3, dtype=torch.int64)),
    )
    raw = torch.tensor([[5.0, 4.0, 0.0], [6.0, 3.0, 0.0]])
    runner.postprocess(
        batch,
        raw.clone(),
        torch.ones(1),
        None,
        None,
        False,
        hidden_states=torch.zeros(2, 1),
    )

    assert observed["in_place"] is False
    assert observed["verified"] is observed["masked"]
    assert observed["verified"].shape == (1, 3)
    assert observed["committed"].tolist() == [[1, 0]]
