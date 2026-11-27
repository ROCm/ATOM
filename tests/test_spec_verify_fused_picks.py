# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""A sampled verify step's picks in one aiter launch (`Sampler.verify_picks`).

Which steps take it, and that it draws the noise the two-call path draws.
Runner wiring only; the kernel's picks are aiter's op test.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from aiter_stub import atom_without_aiter

with atom_without_aiter():
    from atom.model_engine import model_runner as runner_module
    from atom.model_ops import sampler as sampler_module
    from atom.model_ops.sampler import SAMPLER_EPS, Sampler


def test_only_the_unfiltered_temperature_path_fuses():
    sampler = Sampler()
    assert sampler.fuses_verify_picks(None, None)
    assert not sampler.fuses_verify_picks(torch.tensor([4, 4]), None)
    assert not sampler.fuses_verify_picks(None, torch.tensor([0.9, 0.9]))
    assert not sampler.fuses_verify_picks(4, 0.9)


def test_verify_picks_draws_the_two_calls_noise_in_their_order(monkeypatch):
    """Verification rows' [rows, V] draw first, then the requests' [bs, V]:
    what `sample_verification_tokens` and the bonus call draw, so a seeded
    stream gives the same noise either way."""
    seen = {}

    # eps / greedy_eps are keyword-only in aiter's contract: a positional
    # pass would fail here as it does against the routed op.
    def op(
        out_t,
        out_b,
        logits,
        expo_t,
        expo_b,
        temps,
        target,
        bonus,
        cu,
        *,
        eps,
        greedy_eps,
    ):
        seen.update(
            out_t=out_t,
            out_b=out_b,
            logits=logits,
            expo_t=expo_t.clone(),
            expo_b=expo_b.clone(),
            temps=temps,
            target=target,
            bonus=bonus,
            cu=cu,
            eps=eps,
            greedy_eps=greedy_eps,
        )
        out_t.fill_(7)
        out_b.fill_(9)

    monkeypatch.setattr(sampler_module, "spec_verify_pick", op)
    logits = torch.randn(7, 16)
    target = torch.tensor([0, 1, 3, 4, 5])
    bonus = torch.tensor([2, 6])
    cu = torch.tensor([2, 5], dtype=torch.int32)
    temps = torch.tensor([0.7, SAMPLER_EPS])

    torch.manual_seed(1234)
    target_ids, bonus_ids = Sampler().verify_picks(logits, target, bonus, cu, temps)
    torch.manual_seed(1234)
    want_t = torch.empty(5, 16).exponential_(1)
    want_b = torch.empty(2, 16).exponential_(1)

    torch.testing.assert_close(seen["expo_t"], want_t, rtol=0, atol=0)
    torch.testing.assert_close(seen["expo_b"], want_b, rtol=0, atol=0)
    # The step's logits, read in place; indices and temperatures as given.
    assert seen["logits"] is logits and seen["temps"] is temps
    assert seen["target"] is target and seen["bonus"] is bonus and seen["cu"] is cu
    assert seen["eps"] == SAMPLER_EPS and seen["greedy_eps"] == SAMPLER_EPS
    assert target_ids is seen["out_t"] and bonus_ids is seen["out_b"]
    assert target_ids.dtype == bonus_ids.dtype == torch.int32
    assert target_ids.tolist() == [7] * 5 and bonus_ids.tolist() == [9, 9]


class _Reached(Exception):
    pass


def _postprocess(
    monkeypatch,
    *,
    all_greedy=False,
    top_ks=None,
    top_ps=None,
    ratio=None,
    constraints=None,
    n_target=2,
    tp=1,
):
    """Run the verify branch up to the rejection sampler and report the calls."""
    bs = 2
    temperatures = torch.ones(bs)
    spec = SimpleNamespace(
        bonus_logits_indices=torch.tensor([1, 3]),
        target_logits_indices=torch.tensor([0, 2])[:n_target],
        cu_num_draft_tokens=torch.tensor([1, n_target], dtype=torch.int32),
    )
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    group = SimpleNamespace(
        world_size=tp, broadcast=mock.Mock(side_effect=lambda t, src: t + 100)
    )
    monkeypatch.setattr(runner_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        runner_module.TokenConstraints, "apply", lambda *args: constraints
    )
    lossless = SimpleNamespace(verify_inputs=mock.Mock(return_value=ratio))
    real = Sampler()
    sampler = mock.Mock(return_value=torch.full((bs,), 5, dtype=torch.int32))
    sampler.fuses_verify_picks = real.fuses_verify_picks
    sampler.verify_picks = mock.Mock(
        return_value=(
            torch.full((n_target,), 3, dtype=torch.int32),
            torch.full((bs,), 4, dtype=torch.int32),
        )
    )
    sampler.sample_verification_tokens = mock.Mock(
        return_value=torch.full((n_target,), 6, dtype=torch.int32)
    )
    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(zero_spec_counts=torch.zeros(8)),
        sampler=sampler,
        rejection_sampler=SimpleNamespace(forward=mock.Mock(side_effect=_Reached)),
        drafter=SimpleNamespace(lossless=lossless),
        _served=None,
        structured_output=None,
    )
    logits = torch.zeros(4, 8)
    with pytest.raises(_Reached):
        runner_module.ModelRunner.postprocess(
            runner,
            SimpleNamespace(
                total_seqs_num=bs,
                return_logprobs=[False] * bs,
                any_logprobs=False,
                required_channel=None,
                structured_requests=None,
                num_spec_step=2,
                temperatures=temperatures,
            ),
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            runner_module.LmHeadOutput(None, logits),
            hidden_states=None,
        )
    forward = runner.rejection_sampler.forward.call_args
    return SimpleNamespace(
        sampler=sampler,
        logits=logits,
        spec=spec,
        temperatures=temperatures,
        bonus_ids=forward.args[2],
        target_ids=forward.kwargs["target_token_ids"],
        broadcast=group.broadcast,
    )


def test_a_sampled_unfiltered_step_makes_both_picks_in_one_call(monkeypatch):
    run = _postprocess(monkeypatch)
    run.sampler.verify_picks.assert_called_once()
    args = run.sampler.verify_picks.call_args.args
    assert args[0] is run.logits
    assert args[1] is run.spec.target_logits_indices
    assert args[2] is run.spec.bonus_logits_indices
    assert args[3] is run.spec.cu_num_draft_tokens
    assert args[4] is run.temperatures
    assert not run.sampler.sample_verification_tokens.called
    assert not run.sampler.called
    assert run.target_ids.tolist() == [3, 3] and run.bonus_ids.tolist() == [4, 4]


def test_tp_ranks_take_rank0_target_ids_after_the_fused_picks(monkeypatch):
    run = _postprocess(monkeypatch, tp=2)
    run.sampler.verify_picks.assert_called_once()
    run.broadcast.assert_called_once()
    assert run.target_ids.tolist() == [103, 103]
    assert run.bonus_ids.tolist() == [4, 4]


@pytest.mark.parametrize(
    "kwargs, draws_target",
    [
        ({"all_greedy": True}, False),
        ({"top_ks": torch.tensor([4, 4])}, True),
        ({"top_ps": torch.tensor([0.9, 0.9])}, True),
        ({"ratio": object()}, False),
        ({"n_target": 0}, True),
        (
            {
                "constraints": SimpleNamespace(
                    logits=torch.zeros(2, 8), target=torch.zeros(2, 8)
                )
            },
            True,
        ),
    ],
    ids=["all-greedy", "top-k", "top-p", "lossless-ratio", "no-target", "grammar"],
)
def test_every_other_step_keeps_the_two_calls(monkeypatch, kwargs, draws_target):
    run = _postprocess(monkeypatch, **kwargs)
    assert not run.sampler.verify_picks.called
    assert run.sampler.sample_verification_tokens.called is draws_target
    run.sampler.assert_called_once()
    assert run.bonus_ids.tolist() == [5, 5]
