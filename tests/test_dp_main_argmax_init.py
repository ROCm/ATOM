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


def _head(head_type, shards=True):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.dp_shards_vocab = lambda: shards
    return head


def _runner(mr, heads, dp=4, spec=None, compute_logits=None):
    runner = object.__new__(mr.ModelRunner)
    runner.compute_head = None
    runner.config = SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )
    runner.model = SimpleNamespace(
        modules=lambda: iter(heads), compute_logits=compute_logits
    )
    return runner


def test_one_dp_sharded_head_is_taken(modules):
    mr, eh = modules
    head = _head(eh.ParallelLMHead)
    runner = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is head


@pytest.mark.parametrize(
    "case", ["compute_head", "speculative", "dp1", "unsharded", "two_heads"]
)
def test_no_head_is_taken(modules, case):
    mr, eh = modules
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


def test_a_head_subclass_keeps_the_models_preprocessing(modules):
    """A subclass such as DeepSeek-V4's `ParallelHead` is taken: the exchange
    is chosen inside `ParallelLMHead.forward`, so the head output is still
    the model's `compute_logits` (here, its mHC reduction and norm)."""
    mr, eh = modules

    class ParallelHead(eh.ParallelLMHead):
        pass

    head = _head(ParallelHead)
    residual = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    def compute_logits(hidden_states):
        merged = hidden_states.sum(dim=1)
        return torch.nn.functional.normalize(merged, dim=-1)

    project = mock.Mock(side_effect=compute_logits)
    runner = _runner(mr, [head], compute_logits=project)
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is head
    torch.testing.assert_close(runner._head_output(residual), compute_logits(residual))
    project.assert_called_once_with(residual)
