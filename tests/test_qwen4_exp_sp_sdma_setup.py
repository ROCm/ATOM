# SPDX-License-Identifier: MIT
"""SDMA setup over a 2-rank gloo group with a stand-in `mori.cco`: rank 0
broadcasts its uid or its `get_unique_id` error, so every rank leaves the
broadcast and raises alike (then falls back to RCCL) instead of hanging."""

import socket
import sys
import types

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import gpu_stack_import

WORLD = 2


def _fake_mori(uid_fails: bool) -> types.ModuleType:
    class Communicator:
        @staticmethod
        def get_unique_id():
            if uid_fails:
                raise RuntimeError("no bootstrap interface")
            return b"uid-from-rank-0"

        @staticmethod
        def init(world, rank, uid, per_rank_vmm):
            # Stop here: report what this rank received.
            raise RuntimeError(f"init {uid!r}")

    cco = types.ModuleType("mori.cco")
    cco.Communicator = Communicator
    cco.UniqueId = types.SimpleNamespace(from_bytes=lambda b: b)
    cco.GDA_CONNECTION_NONE = 0
    cco.CCODevCommRequirements = object
    return cco


def _rank(rank, port, uid_fails, results):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    sys.modules["mori"] = types.ModuleType("mori")
    sys.modules["mori.cco"] = _fake_mori(uid_fails)
    from atom.model_ops.qwen4_exp import sp_sdma

    try:
        sp_sdma._MoriSdma(dist.group.WORLD, None)
        error = None
    except RuntimeError as e:
        error = str(e)
    # Every rank is still in step: the next collective completes.
    flag = [None] * WORLD
    dist.all_gather_object(flag, error)
    results[rank] = flag
    dist.destroy_process_group()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize(
    "uid_fails,expected",
    [
        (True, "mori get_unique_id failed on rank 0: no bootstrap interface"),
        (False, "init b'uid-from-rank-0'"),
    ],
)
def test_uid_broadcast_carries_rank0_error(monkeypatch, uid_fails, expected):
    gpu_stack_import("atom.model_ops.qwen4_exp.sp_sdma")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    results = mp.Manager().dict()
    ctx = mp.spawn(_rank, args=(_port(), uid_fails, results), nprocs=WORLD, join=False)
    for _ in range(60):  # a hang shows up as a timeout, not a stuck run
        if ctx.join(timeout=1):
            break
    else:
        for p in ctx.processes:
            p.kill()
        pytest.fail("a rank is stuck in the uid broadcast")
    assert dict(results) == {r: [expected] * WORLD for r in range(WORLD)}
