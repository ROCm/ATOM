# SPDX-License-Identifier: MIT
"""The DSpark collectives and delayed copies shared by the planner, the online
STS collector and the online/startup SPS paths."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import torch

from atom.spec_decode.dspark_comm import DSparkComm
from atom.spec_decode.dspark_delayed import DelayedRows


class Event:
    def __init__(self):
        self.ready = False
        self.waited = 0

    def record(self, _stream=None):
        pass

    def query(self):
        return self.ready

    def synchronize(self):
        self.waited += 1
        self.ready = True


def _cpu_cuda(monkeypatch):
    stream = NS(wait_stream=lambda _: None)
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda *a, pin_memory=False, **k: real_empty(*a, **k)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    return stream


def test_a_lone_worker_needs_no_process_group():
    comm = DSparkComm()
    assert (comm.tp_size, comm.dp_size, comm.tp_leader) == (1, 1, True)
    rows = torch.arange(6.0).reshape(2, 3)
    assert comm.gather_dp(comm.gather_tp(rows)).shape == (1, 1, 2, 3)
    assert comm.all_max([1.0, 2.5]) == [1.0, 2.5]
    assert comm.all_true(True) and not comm.all_true(False)
    comm.agree({"q": [1, 3, 6]})


def test_copies_are_read_once_landed_and_tagged(monkeypatch):
    stream = _cpu_cuda(monkeypatch)
    rows = DelayedRows(lambda: stream, 2)
    assert rows.push(5, [7, 8], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert rows.take(4, wait=False) is None  # another forward's copy
    assert rows.take(5, wait=False) is None  # still in flight
    ids, values = rows.take(5, [7, 8], wait=True)
    assert ids == (7, 8) and values.tolist() == [[1, 2], [3, 4]]
    assert rows.take(5, [8, 7], wait=True) is None  # a different request set
    ids, _ = rows.take(5, wait=False, release=True)
    assert not rows.pending and len(rows.free) == 1


def test_busy_and_invalidated_copies_keep_their_buffers(monkeypatch):
    stream = _cpu_cuda(monkeypatch)
    rows = DelayedRows(lambda: stream, 2, max_pending=2)
    assert rows.push(1, [1], torch.ones(1, 2))
    assert rows.push(2, [1], torch.ones(1, 2))
    assert not rows.push(3, [1], torch.ones(1, 2))  # bounded
    rows.invalidate()
    assert rows.take(1, wait=True) is None
    rows.reap(before=10)  # events never completed: nothing is released
    assert len(rows.pending) == 2 and not rows.free
    for copy in rows.pending:
        copy.event.ready = True
    rows.reap(before=10)
    assert not rows.pending and len(rows.free) == 2
