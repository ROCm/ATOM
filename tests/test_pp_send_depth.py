# SPDX-License-Identifier: MIT
"""A PP stage leaves at most ATOM_PP_SEND_DEPTH sends unfinished."""

import pytest

pytest.importorskip("aiter")

from atom.model_engine import model_runner as runner_module
from atom.utils import envs


def _runner(monkeypatch, depth):
    log = []
    ids = iter(range(100))

    def send(model_output, rank_local_keys=()):
        works = [next(ids)]
        log.append(("send", works[0]))
        return works

    def commit(works):
        log.append(("commit", works[0]))
        works.clear()

    monkeypatch.setattr(runner_module, "async_send_intermediate_tensors", send)
    monkeypatch.setattr(runner_module, "commit_pp_send_work", commit)
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner._pp_pending_sends = runner_module.deque()
    runner._pp_send_depth = depth
    runner._pp_send_needs_sparse = False
    return runner, log


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_each_send_first_finishes_the_one_depth_sends_back(monkeypatch, depth):
    runner, log = _runner(monkeypatch, depth)
    for _ in range(5):
        runner._send_pp_output(object())
        assert len(runner._pp_pending_sends) <= depth

    expected = []
    for k in range(5):
        if k >= depth:
            expected.append(("commit", k - depth))
        expected.append(("send", k))
    assert log == expected

    assert runner.flush_pp_send() is True
    assert log[len(expected) :] == [("commit", k) for k in range(5 - depth, 5)]
    assert not runner._pp_pending_sends


def test_depth_and_in_flight_envs(monkeypatch):
    monkeypatch.delenv("ATOM_PP_SEND_DEPTH", raising=False)
    monkeypatch.delenv("ATOM_PP_MAX_IN_FLIGHT", raising=False)
    assert envs.ATOM_PP_SEND_DEPTH == 1
    assert envs.ATOM_PP_MAX_IN_FLIGHT is None

    monkeypatch.setenv("ATOM_PP_SEND_DEPTH", "2")
    monkeypatch.setenv("ATOM_PP_MAX_IN_FLIGHT", "8")
    assert envs.ATOM_PP_SEND_DEPTH == 2
    assert envs.ATOM_PP_MAX_IN_FLIGHT == 8

    for name in ("ATOM_PP_SEND_DEPTH", "ATOM_PP_MAX_IN_FLIGHT"):
        monkeypatch.setenv(name, "0")
        with pytest.raises(ValueError):
            getattr(envs, name)
