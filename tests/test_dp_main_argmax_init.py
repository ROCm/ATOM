# SPDX-License-Identifier: MIT
"""`ATOM_DP_LM_HEAD_MODE=argmax`: which runners take the head
(`_init_dp_argmax_head`) and which requests are admitted."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.sampling_params import SamplingParams


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner"), load("atom.model_ops.embed_head")


def _head(head_type, shards=True):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.dp_shards_vocab = lambda: shards
    return head


def _runner(mr, heads, spec=None):
    runner = object.__new__(mr.ModelRunner)
    runner.compute_head = None
    runner.config = SimpleNamespace(speculative_config=spec)
    runner.model = SimpleNamespace(modules=lambda: iter(heads))
    return runner


@pytest.mark.parametrize("subclass", [False, True])
def test_argmax_mode_takes_the_one_dp_sharded_head(monkeypatch, modules, subclass):
    """A subclass such as DeepSeek-V4's `ParallelHead` is taken too: the
    exchange is chosen inside `ParallelLMHead.forward`."""
    mr, eh = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    head_type = eh.ParallelLMHead
    if subclass:
        head_type = type("ParallelHead", (eh.ParallelLMHead,), {})
    head = _head(head_type)
    runner = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is head


@pytest.mark.parametrize(
    "case", ["compute_head", "speculative", "unsharded", "two_heads", "no_head"]
)
def test_argmax_mode_fails_at_startup(monkeypatch, modules, case):
    mr, eh = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    heads = {
        "unsharded": [_head(eh.ParallelLMHead, shards=False)],
        "two_heads": [_head(eh.ParallelLMHead), _head(eh.ParallelLMHead)],
        "no_head": [],
    }.get(case, [_head(eh.ParallelLMHead)])
    runner = _runner(mr, heads, spec=object() if case == "speculative" else None)
    if case == "compute_head":
        runner.compute_head = mock.Mock()
    with pytest.raises(ValueError, match="ATOM_DP_LM_HEAD_MODE=argmax"):
        runner._init_dp_argmax_head()


@pytest.mark.parametrize("mode", ["all2all", "allgather", "default"])
def test_other_modes_take_no_head(monkeypatch, modules, mode):
    """Off, nothing is checked or taken, even where argmax would fail."""
    mr, _ = modules
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", mode)
    runner = _runner(mr, [], spec=object())
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 0.6},
        {"temperature": 0.0, "logprobs": True},
        {"temperature": 0.0, "json_schema": "{}"},
        {"temperature": 0.0, "structured_tool_format": "kimi"},
        {"temperature": 0.0, "required_channel": object()},
    ],
)
def test_argmax_mode_rejects_a_request_it_cannot_serve(monkeypatch, params):
    sp = SamplingParams(**params)
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    with pytest.raises(ValueError, match="greedy requests only"):
        sp.validate_greedy_head()
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "allgather")
    sp.validate_greedy_head()


def test_argmax_mode_admits_a_greedy_request(monkeypatch):
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "argmax")
    SamplingParams(temperature=0.0, top_k=1).validate_greedy_head()
