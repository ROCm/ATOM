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
        return verify(args[4], args[5])

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
    payload = observed["logprobs"][0]
    torch.testing.assert_close(payload[:, 0], raw.log_softmax(-1)[:, 1])
    # Raw top candidate and sampled-token rank survive grammar masking.
    assert payload[:, 2].view(torch.int32).tolist() == [0, 0]
    assert payload[:, 3].view(torch.int32).tolist() == [2, 2]
