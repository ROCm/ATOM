# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU coverage for rank-consistent candidate-mailbox setup and fallback."""

import socket
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

WORLD = 2


def _worker(rank, port, out):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
    )
    state = {"open_calls": 0}

    class HipRuntime:
        @staticmethod
        def hipMemset(*_args):
            return 0

    class Pool:
        def __init__(self, _device, _group):
            pass

        def create(self, _name, _nbytes, raw_cached):
            assert raw_cached
            return SimpleNamespace(data_ptr=1)

        def open_peer_buffers(self, _name):
            state["open_calls"] += 1
            return [1000 + peer for peer in range(WORLD)]

        def close_peer_buffers(self):
            pass

        def close(self):
            pass

    from atom.distributed import indexer_cp

    def runtime():
        return Pool, lambda: HipRuntime(), lambda _raw_cached: False

    indexer_cp._mailbox_runtime = runtime
    torch.cuda.synchronize = lambda *_args, **_kwargs: None

    def setup_failed():
        try:
            indexer_cp.CandidateMailbox(
                dist.group.WORLD, WORLD, max_rows=8, device=torch.device("cpu")
            )
        except RuntimeError:
            return True
        return False

    real_zeros = torch.zeros
    injected = False

    def zeros(*args, **kwargs):
        nonlocal injected
        if rank == 0 and not injected:
            injected = True
            raise torch.OutOfMemoryError("injected rank-local OOM")
        return real_zeros(*args, **kwargs)

    torch.zeros = zeros
    fallbacks = [setup_failed()]
    torch.zeros = real_zeros

    dist.barrier()
    if rank == 0:

        def missing_runtime():
            raise ImportError("injected missing private aiter symbol")

        indexer_cp._mailbox_runtime = missing_runtime
    fallbacks.append(setup_failed())

    dist.barrier()
    indexer_cp._mailbox_runtime = runtime
    mailbox = indexer_cp.CandidateMailbox(
        dist.group.WORLD, WORLD, max_rows=8, device=torch.device("cpu")
    )
    boundaries = (
        mailbox.fits(WORLD, 8, 1),
        mailbox.fits(WORLD, 8, mailbox.MAX_TOPK),
        not mailbox.fits(WORLD - 1, 8, 1),
        not mailbox.fits(WORLD, 9, 1),
        not mailbox.fits(WORLD, 8, 0),
        not mailbox.fits(WORLD, 8, mailbox.MAX_TOPK + 1),
    )
    out[rank] = (fallbacks, state["open_calls"], boundaries)
    dist.barrier()
    dist.destroy_process_group()


def test_setup_failures_are_agreed_before_collective_fallback():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(port, out), nprocs=WORLD, join=True)
    assert [out[rank] for rank in range(WORLD)] == [
        ([True, True], 1, (True,) * 6)
    ] * WORLD
