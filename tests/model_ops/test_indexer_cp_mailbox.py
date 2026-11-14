# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 indexer CP: the one-launch select / exchange / merge against
local top-k, an all-gather and the merge, on 4 GPUs."""

import socket

import pytest
import torch

pytest.importorskip("aiter", reason="requires the AITER runtime")
triton = pytest.importorskip("triton", reason="requires Triton")


def _mailbox_x4():
    if torch.cuda.device_count() < 4:
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() in ("gfx950", "gfx1250")


pytestmark = pytest.mark.skipif(
    not _mailbox_x4(), reason="requires 4 supported mailbox GPUs"
)

WORLD, QLEN, TOPK, MAX_LEN = 4, 4, 16, 1 << 20
CASES = [(1, 1000, 0, 1), (16, 60000, 1, 4), (64, 100000, 1, 4)]
# The first four calls cross both boundaries: 2**30 wraps the stored counter,
# and 2**30 == 1 (mod 32767) crosses the 15-bit tag boundary too.
COUNTER_START = (1 << 30) - 3


def _case(rank, mailbox, batch, context, init, keep):
    """Whether each call of one case matched: eager calls, then graph replays."""
    import torch.distributed as dist

    from atom.model_ops.minimax_m3 import indexer_candidate_exchange as ex

    blocks = triton.cdiv(MAX_LEN, 128)
    local = triton.cdiv(blocks, WORLD)
    tokens = batch * QLEN
    same = torch.Generator(device="cuda").manual_seed(batch)
    seq = torch.tensor(
        [max(QLEN, context - 911 * i) for i in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    table = torch.randperm(batch * blocks, device="cuda", generator=same)
    table = table.int().view(batch, blocks)
    mine = torch.Generator(device="cuda").manual_seed(batch * WORLD + rank)
    scores = [
        torch.randn((WORLD, tokens, local), device="cuda", generator=mine)
        for _ in range(2)
    ]

    def gathered(i):
        keys = ex.local_candidate_keys(
            scores[i], seq, TOPK, rank, WORLD, QLEN, blocks, init, keep
        )
        parts = [torch.empty_like(keys) for _ in range(WORLD)]
        dist.all_gather(parts, keys)
        recv = torch.stack([part[rank] for part in parts])
        return ex.merge_candidate_keys(recv, table, seq, TOPK, init, keep, QLEN)

    def one_launch(i):
        return ex.select_exchanged_candidates(
            scores[i], seq, table, TOPK, rank, WORLD, QLEN, init, keep, mailbox
        )

    expected = [tuple(t.clone() for t in gathered(i)) for i in range(2)]
    results = []
    # No barrier between calls: a rank runs up to one call ahead of a peer.
    for i in (0, 1, 1, 0, 0, 1, 0, 1):
        got = one_launch(i)
        results.append(
            all(torch.equal(a, b) for a, b in zip(got, expected[i], strict=True))
        )
    torch.cuda.synchronize()
    dist.barrier()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            order = (0, 1, 0)
            replayed = [one_launch(i) for i in order]
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            results.extend(
                all(torch.equal(a, b) for a, b in zip(got, expected[i], strict=True))
                for got, i in zip(replayed, order, strict=True)
            )
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    return results


def _worker(rank, port, out):
    import torch.distributed as dist

    from atom.distributed.indexer_cp import CandidateMailbox

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )
    cpu = dist.new_group(backend="gloo")
    mailbox = CandidateMailbox(cpu, WORLD, 256, torch.device("cuda", rank))
    mailbox.send_calls.fill_(COUNTER_START)
    mailbox.recv_calls.fill_(COUNTER_START)
    results = []
    for case in CASES:
        results += _case(rank, mailbox, *case)
    out[rank] = results
    dist.barrier()
    dist.destroy_process_group()


def test_one_launch_matches_gather_then_merge():
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_worker, args=(port, out), nprocs=WORLD, join=True)
    for rank in range(WORLD):
        results = out[rank]
        assert all(results), f"rank {rank}: {results}"
