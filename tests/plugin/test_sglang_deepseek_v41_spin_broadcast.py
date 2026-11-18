# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 on the SGLang plugin: the scheduler's request broadcast polls
its gloo collective instead of sleeping (`v41_spin_broadcast_patch`)."""

import pickle
from collections import deque

import pytest
import torch

from atom.plugin.sglang.patches import v41_spin_broadcast_patch as patch


class _Work:
    def __init__(self, polls_before_done):
        self.left = polls_before_done
        self.polls = 0
        self.waited = False

    def is_completed(self):
        self.polls += 1
        if self.left == 0:
            return True
        self.left -= 1
        return False

    def wait(self):
        assert self.left == 0, "wait() before the collective completed"
        self.waited = True


class _Wire:
    """One broadcast channel between a src and its receivers: the src's
    tensors in order; a receiver's broadcast copies the next one in."""

    def __init__(self, polls=3):
        self.sent = deque()
        self.works = []
        self.calls = []
        self.polls = polls

    def broadcast(self, tensor, src, group=None, async_op=False):
        self.calls.append((tuple(tensor.shape), tensor.dtype, src, group, async_op))
        if group == "src":
            self.sent.append(tensor.clone())
        else:
            tensor.copy_(self.sent.popleft())
        work = _Work(self.polls)
        self.works.append(work)
        return work


@pytest.fixture
def wire(monkeypatch):
    w = _Wire()
    monkeypatch.setattr(torch.distributed, "broadcast", w.broadcast)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    return w


def _original(*args):
    raise AssertionError(f"SGLang's broadcast_pyobj called with {args}")


@pytest.mark.parametrize("data", [[], [{"rid": "a", "ids": [1, 2, 3]}, "abort"]])
def test_polls_and_keeps_sglangs_protocol(wire, data):
    spin = patch.make_spin_broadcast_pyobj(_original)
    assert spin(data, 0, "src", src=0) is data
    # SGLang's wire format: an int64 size, then the pickled bytes if any.
    sizes = [c for c in wire.calls if c[1] == torch.long]
    assert len(sizes) == 1 and all(c[4] for c in wire.calls)
    if data:
        assert int(wire.sent[0][0]) == len(pickle.dumps(data))
        assert bytes(wire.sent[1].numpy()) == pickle.dumps(data)
    received = spin(None, 1, "dst", src=0)
    assert received == data
    # Every collective was polled to completion, then waited on.
    assert all(w.polls == 4 and w.waited for w in wire.works)


def test_the_poll_yields_the_gil_between_polls(monkeypatch, wire):
    # Between polls the waiting thread releases the GIL (os.sched_yield), so a
    # HiCache-style helper thread is not held off until the switch interval.
    yields = []
    monkeypatch.setattr(patch, "_yield", lambda: yields.append(1))
    spin = patch.make_spin_broadcast_pyobj(_original)
    spin([{"rid": "a"}], 0, "src", src=0)
    # Two collectives (size, payload), each incomplete for 3 polls.
    assert len(wire.works) == 2 and len(yields) == 2 * wire.polls


def test_the_yield_is_sched_yield():
    import os

    if hasattr(os, "sched_yield"):
        assert patch._yield is os.sched_yield


def test_other_groups_keep_sglangs_function(monkeypatch, wire):
    seen = []

    def original(*args):
        seen.append(args)
        return "sglang"

    spin = patch.make_spin_broadcast_pyobj(original)
    assert spin([1], 0, None, 0, True) == "sglang"
    assert spin([1], 0, "src", 0, False) == "sglang"
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    assert spin([1], 0, "src", 0, True) == "sglang"
    assert len(seen) == 3 and not wire.calls


def test_a_failed_collective_raises(monkeypatch):
    class Failed(_Work):
        def wait(self):
            raise RuntimeError("gloo timeout")

    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: Failed(0))
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    spin = patch.make_spin_broadcast_pyobj(_original)
    with pytest.raises(RuntimeError, match="gloo timeout"):
        spin([], 0, "src", src=0)


@pytest.fixture
def receiver(monkeypatch):
    rr = pytest.importorskip(
        "sglang.srt.managers.scheduler_components.request_receiver"
    )
    monkeypatch.setattr(rr, "broadcast_pyobj", rr.broadcast_pyobj)
    return rr


def test_install_swaps_the_receivers_broadcast_once(monkeypatch, receiver):
    monkeypatch.delenv(patch.V41_SPIN_BROADCAST_ENV, raising=False)
    original = receiver.broadcast_pyobj
    if patch._upstream_shape_error(receiver) is not None:
        assert patch.install_v41_spin_broadcast_patch() is False
        assert receiver.broadcast_pyobj is original
        return
    assert patch.install_v41_spin_broadcast_patch() is True
    installed = receiver.broadcast_pyobj
    assert installed._atom_v41_spin_broadcast and installed.__wrapped__ is original
    assert patch.install_v41_spin_broadcast_patch() is True
    assert receiver.broadcast_pyobj is installed


def test_kill_switch(monkeypatch, receiver):
    monkeypatch.setenv(patch.V41_SPIN_BROADCAST_ENV, "0")
    original = receiver.broadcast_pyobj
    assert patch.install_v41_spin_broadcast_patch() is False
    assert receiver.broadcast_pyobj is original


def test_other_sglang_versions_keep_blocking(monkeypatch, receiver):
    monkeypatch.delenv(patch.V41_SPIN_BROADCAST_ENV, raising=False)
    monkeypatch.setattr(patch, "_sglang_version", lambda: "0.5.99")
    original = receiver.broadcast_pyobj
    assert patch.install_v41_spin_broadcast_patch() is False
    assert receiver.broadcast_pyobj is original


def test_upstream_signature_change_is_refused(monkeypatch):
    monkeypatch.setattr(patch, "_sglang_version", lambda: "0.5.22")

    class Module:
        @staticmethod
        def broadcast_pyobj(data, rank, group=None):
            return data

    assert "takes" in patch._upstream_shape_error(Module)
    assert "missing" in patch._upstream_shape_error(object())
