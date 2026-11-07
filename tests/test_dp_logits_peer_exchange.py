# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Four-rank GPU regression test for the DP logits transport.

Run on four same-node gfx1250 GPUs with the ATOM/AITER runtime installed::

    PYTHONPATH=. torchrun --standalone --nproc-per-node=4 \
        tests/test_dp_logits_peer_exchange.py

Ordinary single-process pytest skips this test. EXCHANGE_TEST_OUT optionally
selects a directory for all-rank JSON receipts. Reference collectives stay
eager; only the candidate exchange is captured.
"""

import datetime
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist


def _capture(fn, stream):
    dist.barrier()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    dist.barrier()
    return graph, output


def _validate_peer_exchange():
    from aiter.dist.device_communicators.communicator_pynccl import (
        PyNcclCommunicator,
    )
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    from atom.distributed.dp_logits_exchange import PeerLogitsExchange

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    assert int(os.environ["WORLD_SIZE"]) == 4
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
    comm = PyNcclCommunicator(dist.group.WORLD, device)
    assert comm.available and not comm.disabled
    barriers = CustomAllreduce(dist.group.WORLD, device)
    assert not barriers.disabled and barriers._is_gfx1250 and barriers._same_node
    exchange = PeerLogitsExchange(
        dist.group.WORLD,
        device,
        384,
        32320,
        barrier_communicator=barriers,
    )
    stream = torch.cuda.Stream()
    generator = torch.Generator(device=device).manual_seed(91973 + rank)
    report = {"rank": rank, "tests": []}
    cases = []

    def fill(x):
        # Compare raw bits, including signed zero and NaN payloads. A floating
        # point equality check would reject identical transported NaNs.
        x.view(torch.int16).random_(-32768, 32768, generator=generator)

    def reference(x, local_rows):
        rows = x.shape[0] // 4
        received = torch.empty((4, *x.shape), device=device, dtype=x.dtype)
        comm.all_gather(received, x.contiguous())
        return (
            received[:, rank * rows : rank * rows + local_rows, :]
            .permute(1, 0, 2)
            .reshape(local_rows, 129280)
        )

    for rows in (384, 192, 144, 96, 6):
        local_rows = (
            rows if rows in (192, 144, 96) else (0 if rank == 0 else rows - rank)
        )
        storage = torch.empty(
            (4 * rows, 32320 + (16 if rows == 6 else 0)),
            dtype=torch.bfloat16,
            device=device,
        )
        x = storage[:, :32320]
        fill(x)
        expected = reference(x, local_rows)
        actual = exchange(x, local_rows)
        torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        graph, actual = _capture(
            lambda x=x, local_rows=local_rows: exchange(x, local_rows), stream
        )
        for iteration in range(7):
            fill(x)
            expected = reference(x, local_rows)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (
                rank,
                rows,
                iteration,
            )
        report["tests"].append(
            {
                "rows": rows,
                "local_rows": local_rows,
                "changed_input_replays_bitwise_exact": 7,
            }
        )
        cases.append((graph, actual, x, local_rows))

    # Different graphs reuse the same fixed peer pool in a different order.
    for index in (1, 2, 4, 0, 2, 3, 1, 0, 4):
        graph, actual, x, local_rows = cases[index]
        fill(x)
        expected = reference(x, local_rows)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (
            "interleaved_pool_reuse",
            rank,
            index,
        )
    report["interleaved_pool_replays_bitwise_exact"] = 9
    reports = [None] * 4
    dist.all_gather_object(reports, report)
    if rank == 0:
        if output_dir := os.environ.get("EXCHANGE_TEST_OUT"):
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            (root / "results.json").write_text(json.dumps(reports, indent=2))
        print("DP_LOGITS_PEER_EXCHANGE_ALL_RANKS_EXACT", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    for graph, *_ in cases:
        graph.reset()
    exchange.close()
    dist.destroy_process_group()


def test_dp_logits_peer_exchange():
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Run under a four-rank, same-node gfx1250 torchrun launcher")
    if not torch.cuda.is_available():
        pytest.skip("Peer logits exchange requires gfx1250 GPUs")
    _validate_peer_exchange()


if __name__ == "__main__":
    _validate_peer_exchange()
