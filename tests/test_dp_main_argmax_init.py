# SPDX-License-Identifier: MIT
"""CPU regression for models whose LM-head subclass needs preprocessing."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner"), load("atom.model_ops.embed_head")


def _empty_head(head_type):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    return head


def _runner(mr, heads, compute_logits=None):
    runner = object.__new__(mr.ModelRunner)
    runner.compute_head = None
    runner.model = SimpleNamespace(
        modules=lambda: iter(heads), compute_logits=compute_logits
    )
    return runner


def test_parallel_head_subclass_keeps_model_preprocessing(modules, monkeypatch):
    mr, eh = modules
    monkeypatch.setattr(mr, "envs", SimpleNamespace(ATOM_DP_MAIN_ARGMAX=True))

    class ParallelHead(eh.ParallelLMHead):
        def dp_argmax_token(self, hidden_states):
            raise AssertionError("raw mHC residual must not reach the argmax head")

    head = _empty_head(ParallelHead)
    residual = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    def compute_logits(hidden_states):
        merged = hidden_states.sum(dim=1)
        return torch.nn.functional.normalize(merged, dim=-1)

    project = mock.Mock(side_effect=compute_logits)
    runner = _runner(mr, [head], project)
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None
    torch.testing.assert_close(runner._head_output(residual), compute_logits(residual))
    project.assert_called_once_with(residual)


@pytest.mark.parametrize("enabled", [False, True])
def test_only_one_base_head_can_enable_argmax(modules, monkeypatch, enabled):
    mr, eh = modules
    monkeypatch.setattr(mr, "envs", SimpleNamespace(ATOM_DP_MAIN_ARGMAX=enabled))
    head = _empty_head(eh.ParallelLMHead)
    runner = _runner(mr, [head])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is (head if enabled else None)

    runner.compute_head = mock.Mock()
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None

    runner.compute_head = None
    runner.drafter = object()
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None

    runner.drafter = None
    runner.model.modules = lambda: iter([head, _empty_head(eh.ParallelLMHead)])
    runner._init_dp_argmax_head()
    assert runner.dp_argmax_head is None
