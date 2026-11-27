# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""`ATOM_DP_LM_HEAD_MODE=argmax`: which heads shard the vocab, which runners
take the greedy-ids head (`_init_dp_argmax_head`), and which requests are
admitted (`InputOutputProcessor`)."""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.sampling_params import SamplingParams


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield (
            load("atom.model_engine.model_runner"),
            load("atom.model_ops.embed_head"),
            load("atom.model_engine.llm_engine"),
        )


def _head(head_type, shards=True):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.dp_shards_vocab = shards
    return head


def _runner(mr, heads, spec=None):
    runner = object.__new__(mr.ModelRunner)
    runner.verify_head = None
    runner.config = SimpleNamespace(speculative_config=spec)
    runner.model = SimpleNamespace(modules=lambda: iter(heads))
    runner.sampler = sampler = mock.Mock(return_value="sampled")
    return runner, sampler


# ------------------------------------------------------------ head shard ---
@pytest.mark.parametrize(
    "mode, tp, world, vocab, expect",
    [
        ("argmax", 1, 4, 129280, True),
        ("all2all", 1, 4, 129280, True),
        ("allgather", 1, 4, 129280, True),
        ("default", 1, 4, 129280, False),
        ("argmax", 2, 4, 129280, False),  # not pure DP
        ("argmax", 1, 1, 129280, False),  # DP off
        ("argmax", 1, 3, 129280, False),  # the vocab does not split evenly
    ],
)
def test_dp_shards_vocab(monkeypatch, modules, mode, tp, world, vocab, expect):
    _, eh, _ = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", mode)
    monkeypatch.setattr(eh, "get_dp_group", lambda: SimpleNamespace(world_size=world))
    assert eh.dp_shards_vocab(tp, vocab) is expect


# ---------------------------------------------------------------- runner ---
def test_the_runner_takes_the_head_at_startup(modules):
    mr, _, _ = modules
    assert "self._init_dp_argmax_head()" in inspect.getsource(mr.ModelRunner.__init__)


@pytest.mark.parametrize("subclass", [False, True])
def test_argmax_mode_takes_the_one_dp_sharded_head(monkeypatch, modules, subclass):
    """A subclass such as DeepSeek-V4's `ParallelHead` is taken too: the
    exchange is chosen inside `ParallelLMHead.forward`. The sampler is bound
    to pass the head's ids through."""
    mr, eh, _ = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    head_type = eh.ParallelLMHead
    if subclass:
        head_type = type("ParallelHead", (eh.ParallelLMHead,), {})
    head = _head(head_type)
    runner, sampler = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert isinstance(runner.sampler, mr.GreedyIdsOrSample)
    ids = torch.tensor([5.0, 7.0])
    out = runner.sampler(ids, None)
    assert out.dtype == torch.int32 and out.tolist() == [5, 7]
    assert out.data_ptr() != ids.data_ptr()  # a new int32 tensor
    assert runner.sampler(torch.zeros(2, 8), None) == "sampled"
    sampler.assert_called_once()
    assert runner.sampler.keep_ties is sampler.keep_ties  # the rest is the sampler


@pytest.mark.parametrize(
    "case", ["verify_head", "speculative", "unsharded", "two_heads", "no_head"]
)
def test_argmax_mode_fails_at_startup(monkeypatch, modules, case):
    mr, eh, _ = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    heads = {
        "unsharded": [_head(eh.ParallelLMHead, shards=False)],
        "two_heads": [_head(eh.ParallelLMHead), _head(eh.ParallelLMHead)],
        "no_head": [],
    }.get(case, [_head(eh.ParallelLMHead)])
    runner, _ = _runner(mr, heads, spec=object() if case == "speculative" else None)
    if case == "verify_head":
        runner.verify_head = mock.Mock()
    with pytest.raises(ValueError, match="ATOM_DP_LM_HEAD_MODE=argmax"):
        runner._init_dp_argmax_head()


@pytest.mark.parametrize("mode", ["all2all", "allgather", "default"])
def test_other_modes_take_no_head_and_keep_the_sampler(monkeypatch, modules, mode):
    """Off, nothing is checked or taken, even where argmax would fail."""
    mr, _, _ = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", mode)
    runner, sampler = _runner(mr, [], spec=object())
    runner._init_dp_argmax_head()
    assert runner.sampler is sampler


# ------------------------------------------------------------- admission ---
@pytest.mark.parametrize(
    "params, serves",
    [
        ({"temperature": 0.0, "top_k": 1}, True),
        ({"temperature": 0.6}, False),
        ({"temperature": 0.0, "logprobs": True}, False),
        ({"temperature": 0.0, "json_schema": "{}"}, False),
        ({"temperature": 0.0, "structured_tool_format": "kimi"}, False),
        ({"temperature": 0.0, "required_channel": object()}, False),
        ({"temperature": 0.0, "presence_penalty": 1.5}, False),
    ],
)
def test_serves_greedy_ids(modules, params, serves):
    _, _, le = modules
    assert le.serves_greedy_ids(SamplingParams(**params)) is serves


def _processor(le, monkeypatch, mode):
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", mode)
    monkeypatch.setattr(
        le.SegmentTokenizer, "maybe_create", staticmethod(lambda *a, **k: None)
    )
    config = SimpleNamespace(
        speculative_config=None,
        hf_config=SimpleNamespace(model_type="deepseek_v3"),
        enable_prefix_caching=False,
    )
    return le.InputOutputProcessor(config, tokenizer=None, block_size=16)


def test_preprocess_fanout_rejects_a_sampled_request_under_argmax(monkeypatch, modules):
    _, _, le = modules
    processor = _processor(le, monkeypatch, "argmax")
    with pytest.raises(ValueError, match="greedy requests only"):
        processor.preprocess_fanout([1, 2, 3], SamplingParams(temperature=0.6))
    (seq,) = processor.preprocess_fanout([1, 2, 3], SamplingParams(temperature=0))
    assert list(seq.token_ids) == [1, 2, 3]
