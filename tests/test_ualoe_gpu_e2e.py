# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""UALoE data path on real GPUs, in the connector's direction: a producer
process exports a pool and writes blocks into it with its own kernels; a
consumer process maps the pool through ``PeerTable`` and copies the blocks
out with ``launch_copy``.

Both handle kinds the connector exports are covered, on two GPUs of one host:

* ``posix_fd`` -- dma-buf fds taken across processes with ``pidfd_getfd``.
* ``fabric``   -- ``hipMemHandleTypeFabric`` handles (ROCm >= 7.14). Skipped
  where the runtime cannot create one. Two processes on one host stand in for
  two hosts of a fabric domain; a cross-tray run needs two nodes.

Skipped without two GPUs. The 5 GiB case guards the chunking in
``shared_pool``: ROCm 7.2's dma-buf export / import maps the wrong memory once
a single allocation reaches 4 GiB, which corrupted every transfer out of a
full-size KV pool while small pools passed. For ``fabric`` it checks whether
those handles share the limit.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs",
)

BLOCK = 1 << 20
ROUNDS = 6
PER_ROUND = 64


def _producer(kind, pool_bytes, conn):
    from atom.kv_transfer.disaggregation.ualoe import shared_pool
    from atom.kv_transfer.disaggregation.ualoe.hip_vmm import HipError

    torch.cuda.set_device(0)
    try:
        pool = shared_pool.allocate(pool_bytes, torch.device("cuda:0"), kind)
    except HipError as exc:
        conn.send({"unsupported": str(exc)})
        return
    view = pool.tensor.view(-1, BLOCK)
    gen = torch.Generator().manual_seed(0)
    conn.send(pool.descriptor())
    for r in range(ROUNDS):
        value = 11 + r
        picks = torch.randperm(view.shape[0], generator=gen)[: PER_ROUND - 2]
        blocks = sorted({0, view.shape[0] - 1, *picks.tolist()})
        index = torch.tensor(blocks, device="cuda:0")
        view[index] = value
        # Read them back so the producer's L2 holds the new lines too.
        assert int(view[index].sum().item()) == value * BLOCK * len(blocks)
        torch.cuda.synchronize()
        conn.send((blocks, value))
        assert conn.recv() == "read"
    conn.send(None)
    conn.recv()
    shared_pool.release_all()


def _consumer(conn):
    from atom.kv_transfer.disaggregation.ualoe.fabric_copy import launch_copy
    from atom.kv_transfer.disaggregation.ualoe.peer_table import PeerTable

    desc = conn.recv()
    if desc is None:
        return
    torch.cuda.set_device(1)
    table = PeerTable()
    table.attach("producer", desc)
    stream = torch.cuda.Stream(device=1)
    results = []
    while (msg := conn.recv()) is not None:
        blocks, value = msg
        local = torch.zeros(len(blocks), BLOCK, dtype=torch.uint8, device="cuda:1")
        mapping = table.acquire("producer", 1)
        src = mapping.translate(
            [desc["base"] + b * BLOCK for b in blocks], [BLOCK] * len(blocks)
        )
        dst = [local.data_ptr() + i * BLOCK for i in range(len(blocks))]
        keep = launch_copy(src, dst, [BLOCK] * len(blocks), stream)
        stream.synchronize()
        del keep
        table.release(mapping)
        results.append(bool((local == value).all().item()))
        conn.send("read")
    table.close()
    conn.send(results)


@pytest.mark.parametrize("pool_gib", [1, 5])
@pytest.mark.parametrize("kind", ["posix_fd", "fabric"])
def test_consumer_reads_what_the_producer_wrote(kind, pool_gib):
    ctx = mp.get_context("spawn")
    p_end, p_peer = ctx.Pipe()
    c_end, c_peer = ctx.Pipe()
    producer = ctx.Process(target=_producer, args=(kind, pool_gib << 30, p_peer))
    consumer = ctx.Process(target=_consumer, args=(c_peer,))
    producer.start()
    consumer.start()
    desc = p_end.recv()
    if "unsupported" in desc:
        c_end.send(None)
        producer.join(60)
        consumer.join(60)
        pytest.skip(f"no {kind} handles on this runtime: {desc['unsupported']}")
    assert desc["kind"] == kind
    c_end.send(desc)
    while (msg := p_end.recv()) is not None:
        c_end.send(msg)
        p_end.send(c_end.recv())
    c_end.send(None)
    results = c_end.recv()
    p_end.send("done")
    producer.join(120)
    consumer.join(120)
    assert producer.exitcode == 0 and consumer.exitcode == 0
    assert results == [True] * ROUNDS
