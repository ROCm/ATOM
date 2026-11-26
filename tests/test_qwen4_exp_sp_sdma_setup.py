# SPDX-License-Identifier: MIT
"""SDMA setup over a 2-rank gloo group with a stand-in aiter `SdmaComm`: a
failure on one rank makes every rank fall back to RCCL, in step (no rank left
in a collective the others skipped), with mori's queues freed where they were
set up."""

import socket
import sys
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import gpu_stack_import

WORLD = 2


def _fake_comm_module(fail: str, events: list) -> types.ModuleType:
    class SdmaComm:
        def __init__(self, *, group, num_queues, device):
            self.rank = dist.get_rank(group)
            self.world = dist.get_world_size(group)
            self._group = group
            if fail == "init" and self.rank == 1:
                raise RuntimeError("no SDMA bitcode")

        def reserve(self, name, nbytes):
            pass

        def commit(self):
            events.append("commit")
            dist.barrier(group=self._group)
            queues = 0 if fail == "queues" and self.rank == 1 else 1
            self.dev_comm = types.SimpleNamespace(sdma_num_queue=queues)
            return self

        # The ranks that did get a queue lay out their buffers before agreeing.
        def offset(self, name):
            return 0

        @property
        def arena(self):
            return types.SimpleNamespace(base_ptrs=(0,) * self.world)

        def view(self, name, shape, dtype):
            return torch.zeros(shape, dtype=dtype)

        def close(self):
            events.append("close")
            dist.barrier(group=self._group)

    module = types.ModuleType("aiter.ops.communication")
    module.SdmaComm = SdmaComm
    return module


def _rank(rank, port, fail, results):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    events = []
    sys.modules["aiter.ops.communication"] = _fake_comm_module(fail, events)
    from atom.model_ops.qwen4_exp import sp_sdma

    sp_sdma._local_unavailable_reason = lambda: None
    sp_sdma._build_kernels = lambda rank, world: (None,) * 5
    torch.cuda.current_device = lambda: 0
    collectives = sp_sdma.get_sdma_collectives(64, 8, torch.bfloat16, dist.group.WORLD)
    # Every rank is still in step: the next collective completes.
    flag = [None] * WORLD
    dist.all_gather_object(flag, collectives is None)
    results[rank] = (flag, events)
    dist.destroy_process_group()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(
    "fail,events",
    [
        # Rank 1 cannot construct it: nobody enters the collective commit.
        ("init", []),
        # Rank 1 got no SDMA queue: every rank frees its committed setup.
        ("queues", ["commit", "close"]),
    ],
)
def test_one_rank_failing_falls_back_on_all(monkeypatch, fail, events):
    gpu_stack_import("atom.model_ops.qwen4_exp.sp_sdma")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    results = mp.Manager().dict()
    ctx = mp.spawn(_rank, args=(_port(), fail, results), nprocs=WORLD, join=False)
    for _ in range(60):  # a hang shows up as a timeout, not a stuck run
        if ctx.join(timeout=1):
            break
    else:
        for p in ctx.processes:
            p.kill()
        pytest.fail("a rank is stuck in a setup collective")
    assert dict(results) == {r: ([True] * WORLD, events) for r in range(WORLD)}
