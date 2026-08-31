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


def _batch(seqs):
    return types.SimpleNamespace(
        total_seqs_num=seqs,
        total_tokens_num=seqs,
        req_ids=list(range(seqs)),
        return_logprobs=[False] * seqs,
    )


def _runner(seen):
    """A ModelRunner carrying only what the no-spec branch of postprocess reads."""
    runner = object.__new__(mod.ModelRunner)

    def _sampler(logits, temperatures, top_ks, top_ps, all_greedy, **kw):
        seen["logits_rows"] = logits.shape[0]
        seen["temperature_rows"] = temperatures.shape[0]
        return torch.zeros(logits.shape[0], dtype=torch.int)

    runner.sampler = _sampler
    runner.forward_done_event = types.SimpleNamespace(record=lambda: None)
    runner.tokenID_processor = types.SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        default_num_rejected_tokens=torch.zeros(64, dtype=torch.int32),
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
    batch = _batch(scheduled_bs)
    batch.return_logprobs = [True] * scheduled_bs
    captured = {}
    runner = _runner(seen)
    runner.tokenID_processor.prepare_sampled_ids = lambda b, ids, ev, lp: (
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

    assert captured["logprobs"].shape == (scheduled_bs,)


def test_prefill_really_does_run_wider_than_it_scheduled():
    """The premise, from the decider itself -- not a number invented here."""
    mode = mod.ForwardMode.decide(
        batch=types.SimpleNamespace(
            total_tokens_num_prefill=9,
            total_tokens_num=9,
            total_seqs_num=3,
            is_dummy_run=False,
        ),
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        capture_sizes=np.array([1, 2, 4, 8], dtype=np.int32),
        captured_tokens=None,
        is_block_drafter=False,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=1,
    )

    assert (mode.scheduled_bs, mode.running_bs) == (3, 4)
