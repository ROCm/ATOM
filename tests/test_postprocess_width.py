# SPDX-License-Identifier: MIT
"""What `postprocess` hands the sampler, in rows.

The LM head emits one row per sequence the step FORWARDED. On a prefill that is
`running_bs` -- the ladder-rounded, DP-agreed batch -- while the sampler's
per-request parameters are `scheduled_bs` wide, because they describe requests.
Every step between the two has to happen on the same side of the cut.

Skipped where `model_engine.model_runner` cannot import (it needs aiter at
module load); nothing here touches a GPU.
"""

import types

import numpy as np
import pytest
import torch

# Not `importorskip("aiter")`: the non-GPU runner has `aiter` as a namespace
# package, so that import succeeds and `model_runner` still fails on a symbol.
# Naming the module we actually need catches both, and `exc_type` is what makes
# a non-ModuleNotFoundError ImportError a skip rather than a collection error --
# one of which takes the whole suite down with it. See tests/import_guard.py.
mod = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="model_runner imports aiter at module load",
    exc_type=ImportError,
)


def _batch(seqs, mask=None):
    from atom.model_engine.token_logprobs import LogprobRows, host_logprob_rows

    mask = [False] * seqs if mask is None else mask
    requested = tuple(i for i, enabled in enumerate(mask) if enabled)
    rows = (
        LogprobRows(*(torch.from_numpy(part) for part in host_logprob_rows(requested)))
        if requested
        else None
    )
    return types.SimpleNamespace(
        total_seqs_num=seqs,
        total_tokens_num=seqs,
        req_ids=list(range(seqs)),
        return_logprobs=mask,
        any_logprobs=any(mask),
        logprob_requested=requested,
        logprob_rows=rows,
        required_channel=None,
        num_spec_step=0,
    )


def _runner(seen):
    """A ModelRunner carrying only what the no-spec branch of postprocess reads."""
    runner = object.__new__(mod.ModelRunner)
    runner.rank = 0

    def _sampler(logits, temperatures, top_ks, top_ps, all_greedy, **kw):
        seen["logits_rows"] = logits.shape[0]
        seen["temperature_rows"] = temperatures.shape[0]
        return torch.zeros(logits.shape[0], dtype=torch.int)

    runner.sampler = _sampler
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    runner.tokenID_processor = types.SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        zero_spec_counts=torch.zeros(128, dtype=torch.int32),
        prepare_sampled_ids=lambda *a, **k: ({}, {}),
    )
    return runner


def test_the_sampler_gets_as_many_rows_as_it_has_parameters(monkeypatch):
    """A prefill forwards more sequences than it scheduled, and only one of the
    two numbers describes a request.

    `decide` rounds every step onto the capture ladder, so a 3-sequence prefill
    runs 4 rows, and `prepare_prefill` pads `cu_seqlens_q` to match -- which is
    what the LM head slices by. `prepare_sample` sizes `temperatures` at the
    scheduled count and never pads. Sampling before the cut divides
    `[4, V]` by `[3, 1]`, which raises on the top-k/top-p path and reads past
    the parameter buffer on the temperature one.
    """
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    scheduled_bs, running_bs, vocab = 3, 4, 8
    out = _runner(seen).postprocess(
        batch=_batch(scheduled_bs),
        logits=torch.zeros(running_bs, vocab),
        temperatures=torch.ones(scheduled_bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert seen["logits_rows"] == seen["temperature_rows"] == scheduled_bs
    assert out.num_rejected.tolist() == [0] * scheduled_bs


def test_a_step_that_padded_nothing_is_unaffected(monkeypatch):
    """The cut is a slice, so the common case has to stay a no-op."""
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    bs = 5
    _runner(seen).postprocess(
        batch=_batch(bs),
        logits=torch.zeros(bs, 8),
        temperatures=torch.ones(bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert seen["logits_rows"] == bs


def test_the_logprob_gather_reads_the_cut_logits(monkeypatch):
    """`log_probs.gather` indexes with the sampled ids, which are per-request.

    Left on the padded logits it is `[running_bs, V]` gathered by `[scheduled_bs,
    1]` -- a second break from the same uncut tensor, and the one a client asking
    for logprobs hits.
    """
    seen = {}
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )

    scheduled_bs, running_bs = 3, 4
    batch = _batch(scheduled_bs, [True] * scheduled_bs)
    captured = {}
    runner = _runner(seen)
    runner.tokenID_processor.prepare_sampled_ids = lambda b, ids, ev, lp, **_: (
        captured.update(logprobs=lp) or ({}, {})
    )

    runner.postprocess(
        batch=batch,
        logits=torch.zeros(running_bs, 8),
        temperatures=torch.ones(scheduled_bs),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    assert seen["logits_rows"] == seen["temperature_rows"] == scheduled_bs
    values, candidates, ranks = _payload_tensors(captured["logprobs"])
    assert values.shape == (scheduled_bs, 1, 1)
    assert candidates.shape == (scheduled_bs, 1, 0)
    assert ranks.shape == (scheduled_bs, 1)
    torch.testing.assert_close(
        values.flatten(), torch.full((scheduled_bs,), -__import__("math").log(8))
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("top_k", [0, 2])
def test_deferred_tp_broadcasts_complete_probability_payload(
    monkeypatch, rank, deferred, enabled, top_k
):
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    broadcasts = []

    def broadcast(tensor, src):
        # Distinct received values prove output uses the broadcast result.
        received = tensor.clone()
        if tensor.ndim >= 2:
            received[..., 0] += 10
            received[..., -1:].view(torch.int32).add_(100)
        broadcasts.append((tensor, received, src))
        return received

    monkeypatch.setattr(
        mod,
        "get_tp_group",
        lambda: types.SimpleNamespace(world_size=2, broadcast=broadcast),
    )
    batch = _batch(2, [False, enabled])
    batch.top_logprobs = [0, top_k]
    runner = _runner({})
    runner.rank = rank
    runner.tokenID_processor.is_deferred_out = deferred
    captured = {}

    class Transferred(Exception):
        pass

    def transfer(batch, ids, event, payload, **kwargs):
        captured["payload"] = payload
        raise Transferred

    runner.tokenID_processor.prepare_sampled_ids = transfer
    with pytest.raises(Transferred):
        runner.postprocess(
            batch, torch.zeros(2, 8), torch.ones(2), None, None, False, None
        )
    probability_calls = [entry for entry in broadcasts if entry[0].ndim >= 2]
    if not enabled:
        assert captured["payload"] is None
        assert not probability_calls
    elif not deferred:
        assert captured["payload"] is not None
        assert not probability_calls
    else:
        assert len(probability_calls) == 1
        payload = captured["payload"]
        for tensor, received, source in probability_calls:
            assert source == 0
            assert received is payload
        values, candidates, ranks = _payload_tensors(payload)
        assert values.shape == (1, 1, top_k + 1)
        assert candidates.shape == (1, 1, top_k)
        assert ranks.shape == (1, 1)
        assert ranks.item() == 101


@pytest.mark.parametrize(
    "with_schema,with_channel", [(False, True), (True, False), (True, True)]
)
def test_a_masked_row_still_reports_the_models_logprob(
    monkeypatch, with_schema, with_channel
):
    """A required channel masks the logits in place. The logprob a client asked
    for is the model's own, as on every other request, not the forced row's
    0.0."""
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    batch = _batch(2, [True, True])
    batch.required_channel = object() if with_channel else None
    batch.top_logprobs = [3, 3]
    captured = {}
    runner = _runner({})

    if with_schema:
        # A batch may contain both schema and required-channel requests.
        # The second mask must reuse the snapshot from before the first mask.
        batch.structured_requests = {1: ("schema", (), ())}

        def mask_schema_rows(logits, *_):
            logits[1, 4:] = float("-inf")
            return {}

        runner.structured_output = types.SimpleNamespace(
            vocab_size=8, apply=mask_schema_rows, errors={}
        )

    def force_column_zero(_batch, logits, *_args):
        if with_channel:
            logits[:, 1:] = float("-inf")  # the sampler samples id 0

    runner._constrain_required_channel = force_column_zero
    runner.tokenID_processor.prepare_sampled_ids = lambda b, ids, ev, lp, **_: (
        captured.update(logprobs=lp) or ({}, {})
    )
    logits = torch.randn(2, 8)
    expected = torch.log_softmax(logits, dim=-1)[:, 0]

    runner.postprocess(
        batch=batch,
        logits=logits.clone(),
        temperatures=torch.ones(2),
        top_ks=None,
        top_ps=None,
        all_greedy=False,
        hidden_states=None,
    )

    values, candidates, ranks = _payload_tensors(captured["logprobs"])
    torch.testing.assert_close(values[:, 0, 0], expected)
    expected_top, expected_ids = torch.topk(torch.log_softmax(logits, dim=-1), 3)
    torch.testing.assert_close(values[:, 0, 1:], expected_top)
    torch.testing.assert_close(candidates[:, 0].long(), expected_ids)
    torch.testing.assert_close(ranks[:, 0].long(), (logits > logits[:, :1]).sum(-1) + 1)


def test_prefill_really_does_run_wider_than_it_scheduled():
    """The premise, from the decider itself -- not a number invented here."""
    mode = mod.ForwardMode.decide(
        batch=types.SimpleNamespace(
            has_prefill=True,
            total_tokens_num_prefill=9,
            total_tokens_num=9,
            total_seqs_num=3,
            is_dummy_run=False,
            final_prefill_requests=lambda: 0,
        ),
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        step_needs_eager=False,
        capture_sizes=np.array([1, 2, 4, 8], dtype=np.int32),
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=1,
        running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
    )

    assert (mode.scheduled_bs, mode.running_bs) == (3, 4)


def _payload_tensors(payload):
    k = (payload.shape[-1] - 2) // 2
    return (
        payload[..., : k + 1],
        payload[..., k + 1 : 2 * k + 1].view(torch.int32),
        payload[..., -1].view(torch.int32),
    )
