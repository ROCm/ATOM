# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The IPC PP boundary (`atom.distributed.pp_ipc`) between two processes on one
or two GPUs: bit-exact transfers, and replaced events destroyed, not piled up."""

import importlib
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

H, AUX = 256, 3 * 256
# A short-row stress: PP_IPC_RING_TRANSFERS=1000000 PP_IPC_RING_ROWS=16 PP_IPC_RING_BUSY=0
MAX_ROWS = int(os.environ.get("PP_IPC_RING_ROWS", "512"))
BUSY = os.environ.get("PP_IPC_RING_BUSY", "1") == "1"


def _make(n: int, rows: int, has_aux: bool):
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + n)

    def rand(cols):
        x = torch.randn(rows, cols, generator=g, device="cuda")
        return x.to(torch.bfloat16)

    return rand(H), rand(H), rand(AUX) if has_aux else None


def _plan(n: int):
    return 1 + (n * 37) % MAX_ROWS, n % 7 != 3


def _shm_counts() -> tuple[int, int]:
    """This process's event segments (hip_<pid>_<n>) and HIP's /tmp/event* files."""
    own = f"hip_{os.getpid()}_"
    shm = sum(1 for f in os.listdir("/dev/shm") if f.startswith(own))
    tmp = sum(1 for f in os.listdir("/tmp") if f.startswith("event"))
    return shm, tmp


def _hip_maps() -> int:
    """This process's mappings of interprocess-event segments."""
    with open("/proc/self/maps") as f:
        return sum("/dev/shm/hip_" in line for line in f)


def _worker(rank: int, store: str, n_transfers: int, result: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    # `aiter.dist` the attribute is torch.distributed; import the module.
    ps = importlib.import_module("aiter.dist.parallel_state")

    group = SimpleNamespace(
        rank_in_group=rank, world_size=2, ranks=[0, 1], cpu_group=dist.group.WORLD
    )
    ps.get_pp_group = lambda: group
    from atom.distributed import pp_ipc
    from atom.models.utils import IntermediateTensors

    busy = torch.randn(2048, 2048, device="cuda")
    dist.barrier()
    shm0 = _shm_counts()
    if rank == 1:
        end = pp_ipc.PPIpcReceiver(MAX_ROWS, H, torch.bfloat16, AUX)
    else:
        end = pp_ipc.PPIpcSender()
    bad = torch.zeros((), dtype=torch.long, device="cuda")
    max_retired = 0
    maps = [_hip_maps()]
    t0 = time.time()
    for n in range(n_transfers):
        rows, has_aux = _plan(n)
        hidden, residual, aux = _make(n, rows, has_aux)
        if rank == 0:
            if BUSY and n % 3 == 0:
                busy @ busy
            end.send(
                IntermediateTensors({"hidden_states": hidden, "residual": residual}),
                aux,
            )
            del hidden, residual, aux  # the copy stream keeps them alive
        else:
            it, got = end.recv()
            for _ in range((1 + n % 3) * BUSY):
                busy @ busy  # the "forward", under the aux transfer
            end.finish_aux()
            bad += (it.tensors["hidden_states"] != hidden).sum()
            bad += (it.tensors["residual"] != residual).sum()
            if aux is None:
                bad += int(got is not None)
            else:
                bad += (got != aux).sum()
        if n % 8 == 7:
            # The runner's staging ring bounds the CPU lead the same way.
            torch.cuda.synchronize()
        max_retired = max(max_retired, len(end.retired))
        if n % 200 == 199 and rank == 0 and BUSY:
            time.sleep(0.05)  # let the receiver drain, then refill the ring
        if (n + 1) % max(1, n_transfers // 10) == 0:
            maps.append(_hip_maps())
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    dist.barrier()
    shm1 = _shm_counts()
    dist.barrier()
    end.close()  # a graceful exit leaves no event segment behind
    dist.barrier()
    shm2 = _shm_counts()
    created = (
        end.consumed.created if rank == 1 else end.ready.created + end.aux_ready.created
    )
    line = (
        f"rank={rank} transfers={n_transfers} bad={int(bad)} created={created} "
        f"max_retired={max_retired} hip_maps={'/'.join(map(str, maps))} "
        f"shm_hip={shm0[0]}->{shm1[0]}->{shm2[0]} "
        f"tmp_event={shm0[1]}->{shm1[1]} elapsed={elapsed:.2f}s\n"
    )
    with open(result, "a") as f:
        f.write(line)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_ring_transfers_bit_exact_and_replaced_events_destroyed(tmp_path):
    pytest.importorskip("aiter")
    n = int(os.environ.get("PP_IPC_RING_TRANSFERS", "1500"))
    result = tmp_path / "result.txt"
    proc = subprocess.run(
        [sys.executable, __file__, str(tmp_path / "store"), str(n), str(result)],
        capture_output=True,
        text=True,
        timeout=600 + n // 500,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    lines = result.read_text().splitlines()
    print("\n".join(lines))
    assert len(lines) == 2, lines
    for line in lines:
        fields = dict(kv.split("=", 1) for kv in line.split())
        assert fields["bad"] == "0", line
        # Each slot's events are replaced every EVENT_USES records ...
        assert int(fields["created"]) > n // 16, line
        # ... and destroyed once fenced, so few wait at a time and the
        # process maps no more event segments at the end than early on.
        assert int(fields["max_retired"]) <= 12, line
        maps = [int(v) for v in fields["hip_maps"].split("/")]
        assert max(maps[2:], default=0) <= maps[1] + 12, line
        shm = [int(v) for v in fields["shm_hip"].split("->")]
        assert shm[1] <= 3 * 3 and shm[2] == 0, line


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=(sys.argv[1], int(sys.argv[2]), sys.argv[3]), nprocs=2)
