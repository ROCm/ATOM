# SPDX-License-Identifier: MIT
"""Which runners take the DP greedy argmax head (`_init_dp_argmax_head`)."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner"), load("atom.model_ops.embed_head")


@pytest.fixture(autouse=True)
def _allgather(monkeypatch):
    monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "allgather")


def _head(head_type, shards=True):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.dp_shards_vocab = lambda: shards
    return head


def _runner(mr, heads, dp=4, spec=None):
    runner = object.__new__(mr.ModelRunner)
    runner.compute_head = None
    runner.config = SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )
    runner.model = SimpleNamespace(modules=lambda: iter(heads))
    return runner


def test_one_dp_sharded_head_is_taken(modules):
    mr, eh = modules
    head = _head(eh.ParallelLMHead)
    runner = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is head


@pytest.mark.parametrize(
    "case",
    ["all2all", "compute_head", "speculative", "dp1", "unsharded", "two_heads"],
)
def test_no_head_is_taken(monkeypatch, modules, case):
    mr, eh = modules
    if case == "all2all":
        monkeypatch.setenv("ATOM_DP_LM_HEAD_MODE", "all2all")
    heads = [_head(eh.ParallelLMHead, shards=case != "unsharded")]
    if case == "two_heads":
        heads.append(_head(eh.ParallelLMHead))
    runner = _runner(
        mr,
        heads,
        dp=1 if case == "dp1" else 4,
        spec=object() if case == "speculative" else None,
    )
    if case == "compute_head":
        runner.compute_head = mock.Mock()
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None


def test_a_head_subclass_is_taken(modules):
    """A subclass such as DeepSeek-V4's `ParallelHead` is taken: the exchange
    is chosen inside `ParallelLMHead.forward`."""
    mr, eh = modules

    class ParallelHead(eh.ParallelLMHead):
        pass

    head = _head(ParallelHead)
    runner = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is head


class _FakeReqGraph:
    def __init__(self):
        self.calls = []

    def record(self, forward, key=None, phase=None):
        self.calls.append(("record", key, phase, self.context.dp_all_greedy))

    def replay(self, key=None, phase=None):
        self.calls.append(("replay", key, phase))
        return phase


@pytest.mark.parametrize("taken", [True, False])
def test_a_decode_bucket_records_the_greedy_phase_with_the_flag(modules, taken):
    """With the head taken, a bucket is recorded again under
    `DP_GREEDY_PHASE`, the context saying greedy only for that recording."""
    mr, _ = modules
    runner = object.__new__(mr.ModelRunner)
    runner.dp_argmax_head = object() if taken else None
    runner.req_graph = graph = _FakeReqGraph()
    graph.context = context = SimpleNamespace(dp_all_greedy=False)
    seen = []
    runner._head_output = lambda rows: seen.append(context.dp_all_greedy)
    runner._record_decode_bucket(lambda: None, "key", torch.zeros(2, 4), context)
    expect = [("record", "key", None, False)]
    if taken:
        expect.append(("record", "key", mr.DP_GREEDY_PHASE, True))
    assert graph.calls == expect
    assert seen == ([True] if taken else [])
    assert context.dp_all_greedy is False


@pytest.mark.parametrize("greedy", [True, False])
def test_a_greedy_step_replays_the_greedy_phase(modules, greedy):
    mr, _ = modules
    runner = object.__new__(mr.ModelRunner)
    runner.req_graph = graph = _FakeReqGraph()
    phase = mr.DP_GREEDY_PHASE if greedy else None
    head = runner._replay_decode_bucket("key", SimpleNamespace(dp_all_greedy=greedy))
    assert head == phase
    assert graph.calls == [("replay", "key", phase)]


@pytest.mark.parametrize("greedy", [True, False])
def test_the_context_takes_the_groups_greedy_verdict(modules, greedy):
    """A step's `Context` reads `dp_all_greedy` from its `ForwardMode`."""
    mr, _ = modules
    mode = SimpleNamespace(dp_all_greedy=greedy)
    context = mr.Context(positions=torch.zeros(1), forward_mode=mode)
    assert context.dp_all_greedy is greedy
    assert mr.Context(positions=torch.zeros(1)).dp_all_greedy is False
