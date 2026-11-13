# SPDX-License-Identifier: MIT
"""What `postprocess` hands the sampler, in rows.

The LM head emits one row per sequence the step FORWARDED. On a prefill that is
`running_bs` -- the ladder-rounded, DP-agreed batch -- while the sampler's
per-request parameters are `scheduled_bs` wide, because they describe requests.
Every step between the two has to happen on the same side of the cut.

The runner module is imported against AITER/Triton stand-ins on a CPU job, so
the wiring runs there too; nothing here touches a GPU.
"""

import types

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

mod = None  # bound by `_runner_module` for the whole file


@pytest.fixture(scope="module", autouse=True)
def _runner_module():
    """The real runner module; AITER and Triton are stood in for on a CPU job."""
    global mod
    with atom_without_aiter() as load:
        mod = load("atom.model_engine.model_runner")
        yield mod


@pytest.fixture(autouse=True)
def isolated_topology(monkeypatch):
    # These runner-wiring tests do not enable sequence/context parallelism.
    # Individual TP tests set their own group without inheriting global config.
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def _without_context_parallelism(monkeypatch):
    # These row/broadcast tests use TP only and build no global engine config.
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)


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
        structured_requests=None,
        num_spec_step=0,
    )


def _runner(seen):
    """A ModelRunner carrying only what the no-spec branch of postprocess reads."""
    runner = object.__new__(mod.ModelRunner)
    runner.rank = 0
    runner._served = None
    runner.structured_output = None

    def _sampler(logits, temperatures, top_ks, top_ps, all_greedy, **kw):
        seen["logits_rows"] = logits.shape[0]
        seen["temperature_rows"] = temperatures.shape[0]
        return torch.zeros(logits.shape[0], dtype=torch.int)

    runner.sampler = _sampler
    runner.forward_done_event = types.SimpleNamespace(record=lambda stream=None: None)
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

    def force_column_zero(_runner, _batch, logits, *_args):
        if with_channel:
            logits[:, 1:] = float("-inf")  # the sampler samples id 0

    monkeypatch.setattr(
        "atom.model_engine.token_constraints.apply_required_channel", force_column_zero
    )
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


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("schema", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_channel_constraint_preserves_raw_logprobs(
    monkeypatch, speculative, schema, enabled
):
    from atom.model_engine.token_logprobs import LogprobRows, host_logprob_rows

    md = (
        types.SimpleNamespace(
            target_logits_indices=torch.tensor([0, 1, 3, 4]),
            bonus_logits_indices=torch.tensor([2, 5]),
            cu_num_draft_tokens=torch.tensor([2, 4]),
            num_draft_tokens_np=np.array([2, 2]),
        )
        if speculative
        else None
    )
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=md),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    batch = _batch(2, [enabled, False])
    batch.top_logprobs = [3, 0]
    batch.required_channel = object()

    def force_channel(_runner, _batch, bonus, _metadata, target, preserve_masks=False):
        if schema:
            assert preserve_masks.dtype == torch.bool
            assert preserve_masks.tolist() == [True, False]
        else:
            assert preserve_masks is False
        for logits in (bonus, target):
            if logits is not None:
                logits[:, 1:] = -float("inf")
        # This fixture constrains both sampled rows, matching the active-state
        # return value used to compose channel and other per-request masks.
        return (None, torch.zeros(_batch.total_seqs_num, dtype=torch.int32))

    monkeypatch.setattr(
        "atom.model_engine.token_constraints.apply_required_channel", force_channel
    )
    if enabled:
        batch.logprob_rows = LogprobRows(
            *(
                torch.from_numpy(a)
                for a in host_logprob_rows(
                    [0],
                    np.array([3, 3]) if speculative else None,
                    3 if speculative else 1,
                )
            )
        )
    runner = _runner({})
    runner.forward_vars = {
        "positions": types.SimpleNamespace(gpu=torch.arange(6 if speculative else 2)),
        "cu_seqlens_q": types.SimpleNamespace(
            gpu=torch.tensor([0, 3, 6] if speculative else [0, 1, 2])
        ),
    }
    if schema:
        batch.structured_requests = {0: ("schema", (), ())}

        def grammar_mask(logits, *_):
            logits[:, 4:] = -float("inf")
            return {}

        runner.structured_output = types.SimpleNamespace(
            vocab_size=8, apply=grammar_mask, errors={}
        )
    if speculative:

        def verify(_md, target, bonus, **_):
            sampled = torch.cat(
                (target.argmax(-1).reshape(2, 2), bonus[:, None]), dim=1
            )
            return sampled, torch.tensor([[2, 2], [0, 0], [0, 0]])

        runner.rejection_sampler = types.SimpleNamespace(forward=verify)
    seen = {}

    def transfer(_batch, ids, _event, payload, **kwargs):
        seen.update(
            ids=ids.clone(), payload=payload, failed=kwargs["constraint_failed"]
        )
        return {}, None

    runner.tokenID_processor.prepare_sampled_ids = transfer
    if not enabled:
        monkeypatch.setattr(
            mod,
            "capture_token_logits",
            lambda *a, **kw: pytest.fail("disabled snapshot"),
        )
        monkeypatch.setattr(
            mod,
            "gather_token_logprobs",
            lambda *a, **kw: pytest.fail("disabled scoring"),
        )
    raw = (
        torch.arange((6 if speculative else 2) * 8, dtype=torch.float32).view(-1, 8)
        / 10
    )
    runner.postprocess(batch, raw.clone(), torch.ones(2), None, None, True, None)
    assert torch.all(seen["ids"] == 0)
    assert seen["failed"] is None or not seen["failed"].any()
    if not enabled:
        assert seen["payload"] is None
        return
    values, ids, ranks = _payload_tensors(seen["payload"])
    reference = torch.log_softmax(raw[: 3 if speculative else 1], dim=-1)
    torch.testing.assert_close(values[0, :, 0], reference[:, 0])
    top_values, top_ids = reference.topk(3)
    torch.testing.assert_close(values[0, :, 1:], top_values)
    torch.testing.assert_close(ids[0].long(), top_ids)
    assert torch.all(ranks == 8)


@pytest.mark.parametrize("context_kv_only", [False, True])
def test_a_context_only_producer_leaves_dp_draft_lockstep(context_kv_only):
    """No DP peer is owed the collectives of a propose nobody runs."""
    runner = object.__new__(mod.ModelRunner)
    runner.drafter = types.SimpleNamespace(context_kv_only=context_kv_only)
    runner.tokenID_processor = types.SimpleNamespace(is_pipeline_parallel=False)
    runner.config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_size=2)
    )
    assert runner._dp_draft_lockstep_active() is not context_kv_only


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("producer", [False, True])
def test_a_producer_skips_propose_on_either_output_path(
    monkeypatch, deferred, producer
):
    """Either output path skips propose on a producer, keeps the step's
    bookkeeping, and still commits the target's tentative step (V4.1)."""
    forward_mode = types.SimpleNamespace(running_tokens=2)
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            spec_decode_metadata=None,
            attn_metadata="step",
            context=types.SimpleNamespace(forward_mode=forward_mode),
        ),
    )
    monkeypatch.setattr(
        mod, "get_tp_group", lambda: types.SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod, "sp_is_enabled", lambda: False)
    runner = _runner({})
    runner._served = None
    runner.forward_done_event = types.SimpleNamespace(record=lambda stream=None: None)
    runner.drafter = types.SimpleNamespace(
        context_kv_only=producer, prepare_inputs=torch.arange
    )
    proposed, sent, commits = [], [], []
    runner.attn_metadata_builder = types.SimpleNamespace(
        commit_speculative_state=lambda md, rows: commits.append((md, rows.tolist()))
    )
    drafts = np.array([[5], [6]], dtype=np.int32)
    runner.propose_draft_token_ids = lambda *a: proposed.append(a) or drafts
    runner.tokenID_processor.__dict__.update(
        is_deferred_out=deferred,
        is_pd_producer=producer,
        prev_rejected_num=np.array([1, 2], dtype=np.int32),
        prev_bonus_num=np.array([3, 4], dtype=np.int32),
        spec_status_queue=types.SimpleNamespace(
            send=lambda counts, ready: sent.append(counts)
        ),
        input_ids=types.SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int32)),
    )

    out = runner.postprocess(
        _batch(2), torch.zeros(2, 8), torch.ones(2), None, None, False, None
    )

    assert len(proposed) == (0 if producer else 1)
    assert out.draft_token_ids is (None if producer else drafts)
    # Propose commits for the others; it is faked here.
    assert commits == ([("step", [0, 1])] if producer else [])
    if deferred:
        assert len(sent) == 1
        assert out.num_rejected.tolist() == [1, 2]
        assert runner.tokenID_processor.prev_token_ids.shape == (2,)
