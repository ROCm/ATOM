# SPDX-License-Identifier: MIT
"""A PP frame's metadata crosses stages as one gloo message, and the receiving
stage posts its next frame's metadata receive before the sending stage sends
that frame, so asking for a frame waits on no gloo round trip.

`Wire` stands in for torch.distributed's point-to-point calls between two
ranks in this process; `test_frames_over_gloo` runs the same frames over a
real two-process gloo group.
"""

import faulthandler
import multiprocessing as mp
import os
from collections import defaultdict, deque
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from aiter_stub import stubbed_aiter

with stubbed_aiter("aiter.ops", "aiter.ops.communication"):
    from atom.distributed import pp_comm
    from atom.models.utils import IntermediateTensors


class _Work:
    def __init__(self, on_wait=None):
        self.on_wait = on_wait

    def wait(self):
        if self.on_wait is not None:
            self.on_wait()
            self.on_wait = None
        return True


class Wire:
    """In-order point-to-point queues per (group, tag), and a call log."""

    def __init__(self):
        self.queues = defaultdict(deque)
        self.log = []
        self.peers = {"to": set(), "from": set()}

    def isend(self, tensor, dst, group, tag=0):
        self.log.append(("isend", group, tag, tensor.numel()))
        self.peers["to"].add(dst)
        self.queues[group, tag].append(tensor.clone())
        return _Work()

    def irecv(self, tensor, src, group, tag=0):
        self.log.append(("irecv", group, tag, tensor.numel()))
        self.peers["from"].add(src)
        return _Work(lambda: tensor.copy_(self.queues[group, tag].popleft()))

    def recv(self, tensor, src, group, tag=0):
        self.log.append(("recv", group, tag, tensor.numel()))
        self.peers["from"].add(src)
        tensor.copy_(self.queues[group, tag].popleft())


def _stage(monkeypatch, wire, rank, ag_group=None):
    pp = SimpleNamespace(
        rank_in_group=rank,
        world_size=2,
        ranks=[4, 7],
        cpu_group="cpu",
        device_group="device",
    )
    monkeypatch.setattr(pp_comm, "get_pp_group", lambda: pp)
    monkeypatch.setattr(pp_comm, "pp_send_allgather_group", lambda: ag_group)
    # These CPU tensors take the path a GPU stage's tensors take.
    monkeypatch.setattr(pp_comm, "_on_device_group", lambda device: True)
    for name in ("isend", "irecv", "recv"):
        monkeypatch.setattr(pp_comm.torch.distributed, name, getattr(wire, name))


def _frame(rows):
    return {
        "hidden_states": torch.randn(rows, 6).to(torch.bfloat16),
        "residual": torch.randn(rows, 6).to(torch.bfloat16),
        "sparse_kv_indices": torch.arange(rows * 3, dtype=torch.int32),
        "not_a_proxy_key": torch.ones(2),
    }


def _assert_received(got, sent):
    assert set(got.tensors) == set(sent) - {"not_a_proxy_key"}
    for key, value in got.tensors.items():
        assert value.dtype == sent[key].dtype and value.shape == sent[key].shape
        assert torch.equal(value, sent[key])


@pytest.fixture(autouse=True)
def no_posted_receive(monkeypatch):
    monkeypatch.setattr(pp_comm, "_next_meta", None)


@pytest.mark.parametrize("rows", [1, 5, 0])
def test_frame_is_one_message_then_its_tensors(monkeypatch, rows):
    wire = Wire()
    sent = _frame(rows)
    _stage(monkeypatch, wire, 0)
    # The model runner sets a default device; the metadata stays on the CPU.
    with torch.device("meta"):
        works = pp_comm.async_send_intermediate_tensors(IntermediateTensors(sent))
    pp_comm.commit_pp_send_work(works)
    # One metadata message, then one send per proxy tensor that is not empty.
    sends = [entry for entry in wire.log if entry[0] == "isend"]
    assert sends[0] == ("isend", "cpu", pp_comm._META_TAG, pp_comm._META_BYTES)
    assert [entry[1] for entry in sends[1:]] == (["device"] * 3 if rows else [])

    _stage(monkeypatch, wire, 1)
    with torch.device("meta"):
        got = pp_comm.recv_intermediate_tensors()
    _assert_received(got, sent)
    # Group-relative next and previous stage, as global ranks.
    assert wire.peers == {"to": {7}, "from": {4}}


def test_next_metadata_receive_is_posted_before_the_frame_is_sent(monkeypatch):
    wire = Wire()
    frames = [_frame(rows) for rows in (3, 7, 2)]
    for frame in frames:
        _stage(monkeypatch, wire, 0)
        pp_comm.async_send_intermediate_tensors(IntermediateTensors(frame))
        _stage(monkeypatch, wire, 1)
        _assert_received(pp_comm.recv_intermediate_tensors(), frame)
        assert pp_comm._next_meta is not None

    meta = ("cpu", pp_comm._META_TAG, pp_comm._META_BYTES)
    posted = [i for i, e in enumerate(wire.log) if e[0] == "irecv" and e[1:] == meta]
    sent = [i for i, e in enumerate(wire.log) if e[0] == "isend" and e[1:] == meta]
    # The first frame's receive is posted on demand; every later one is posted
    # by the receive before it, ahead of its frame's send.
    assert len(posted) == len(frames) + 1
    assert all(p < s for p, s in zip(posted[1:], sent[1:]))


def test_missing_residual_and_plain_values_cross(monkeypatch):
    wire = Wire()
    sent = {"hidden_states": torch.randn(4, 6), "residual": None}
    _stage(monkeypatch, wire, 0)
    pp_comm.send_intermediate_tensors(IntermediateTensors(sent))
    _stage(monkeypatch, wire, 1)
    got = pp_comm.recv_intermediate_tensors()
    assert got.tensors["residual"] is None
    assert torch.equal(got["hidden_states"], sent["hidden_states"])


def test_send_allgather_ships_one_shard_and_gathers_it(monkeypatch):
    sent = _frame(4)  # every proxy tensor splits into 4 shards
    proxies = [sent[k] for k in ("hidden_states", "residual", "sparse_kv_indices")]
    shards = []

    class TP:
        world_size, rank_in_group = 4, 1

        def all_gather(self, part, dim):
            assert dim == 0
            shards.append(part.clone())
            # Every rank's shard, as each rank of the sending stage sent its own.
            return proxies[len(shards) - 1].reshape(-1)

    wire = Wire()
    _stage(monkeypatch, wire, 0, TP())
    works = pp_comm.async_send_intermediate_tensors(IntermediateTensors(sent))
    expected = [t.reshape(4, -1)[1] for t in proxies]
    assert [torch.equal(w.payload, e) for w, e in zip(works[1:], expected)] == [
        True
    ] * 3

    _stage(monkeypatch, wire, 1, TP())
    _assert_received(pp_comm.recv_intermediate_tensors(), sent)
    assert [torch.equal(s, e) for s, e in zip(shards, expected)] == [True] * 3


def _gloo_stage(rank, path, frames, results):
    faulthandler.dump_traceback_later(60, exit=True)  # a hang fails, with stacks
    dist.init_process_group(
        "gloo", store=dist.FileStore(path, 2), rank=rank, world_size=2
    )
    group = dist.group.WORLD
    pp = SimpleNamespace(
        rank_in_group=rank, world_size=2, ranks=[0, 1], cpu_group=group
    )
    pp.device_group = group
    pp_comm.get_pp_group = lambda: pp
    pp_comm.pp_send_allgather_group = lambda: None
    try:
        if rank == 1:
            # What the receive before the first frame leaves posted.
            pp_comm._next_meta = pp_comm._post_meta_recv(group, 0)
        sent_ahead = []
        for frame in frames:
            dist.barrier()
            if rank == 0:
                works = pp_comm.async_send_intermediate_tensors(
                    IntermediateTensors(frame)
                )
                # The receiver is parked in the barrier below, not receiving:
                # its posted receive is what lets the metadata go out.
                works[0].work.wait(timedelta(seconds=20))
                sent_ahead.append(True)
                dist.barrier()
                # A gloo send is waited on once: a second wait waits for
                # another send.
                pp_comm.commit_pp_send_work(works[1:])
            else:
                dist.barrier()
                _assert_received(pp_comm.recv_intermediate_tensors(), frame)
        if rank == 0:
            results.put(sent_ahead)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_frames_over_gloo(tmp_path):
    torch.manual_seed(0)
    frames = [_frame(rows) for rows in (2, 9, 1)]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    path = os.fspath(tmp_path / "store")
    procs = [
        ctx.Process(target=_gloo_stage, args=(rank, path, frames, results))
        for rank in (0, 1)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(90)
    for proc in procs:
        proc.kill()
    assert [proc.exitcode for proc in procs] == [0, 0]
    assert results.get(timeout=5) == [True] * len(frames)
