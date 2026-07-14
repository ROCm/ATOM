# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Cross-process HIP VMM KV-transfer test: a producer exports a VMM buffer on
GPU 0, a consumer imports it on GPU 1 and peer-copies blocks, then verifies.
Requires >= 2 GPUs with VMM support; skips otherwise.
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp

NB, BE = 256, 4096  # 256 blocks x 4096 bf16 = 2 MiB; values < 128 are bf16-exact
NBYTES = NB * BE * 2
NCOPY = 64


def _producer(path, device, ready):
    from atom.kv_transfer.disaggregation.native import VmmBuffer

    torch.cuda.set_device(device)
    buf = VmmBuffer.alloc(NBYTES, device)
    kv = buf.tensor(torch.bfloat16, (NB, BE))
    for i in range(NB):
        kv[i].fill_(float(i % 128))
    torch.cuda.synchronize()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    socket.send_fds(conn, [b"vmm"], [buf.export_fd()])
    conn.recv(1)  # keep the buffer alive until the consumer is done
    conn.close()
    srv.close()


def _consumer(path, device, ready, result):
    from atom.kv_transfer.disaggregation.native import VmmBuffer

    torch.cuda.set_device(device)
    ready.wait(60)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(path)
    _, fds, _, _ = socket.recv_fds(s, 16, 1)

    peer = VmmBuffer.import_fd(fds[0], NBYTES, device).tensor(torch.bfloat16, (NB, BE))
    local = torch.empty(NB, BE, dtype=torch.bfloat16, device=device)

    # concurrent per-"request" peer copies across streams (hipMemcpyPeerAsync)
    streams = [torch.cuda.Stream(device=device) for _ in range(NCOPY)]
    for i, st in enumerate(streams):
        with torch.cuda.stream(st):
            local[i % NB].copy_(peer[(i * 7) % NB], non_blocking=True)
    for st in streams:
        st.synchronize()

    ok = all(
        local[i % NB][0].item() == float(((i * 7) % NB) % 128) for i in range(NCOPY)
    )
    result.put(bool(ok))
    s.send(b"d")
    s.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >= 2 GPUs",
)
def test_vmm_cross_process_transfer(tmp_path):
    from atom.kv_transfer.disaggregation.native import supported

    if not supported(0) or not supported(1):
        pytest.skip("HIP VMM not supported on these devices")

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    result = ctx.Queue()
    path = str(tmp_path / "vmm.sock")

    prod = ctx.Process(target=_producer, args=(path, 0, ready))
    cons = ctx.Process(target=_consumer, args=(path, 1, ready, result))
    prod.start()
    cons.start()
    cons.join(180)
    prod.join(30)

    assert cons.exitcode == 0, "consumer process crashed"
    assert prod.exitcode == 0, "producer process crashed"
    assert result.get(timeout=5) is True, "peer-copied data mismatch"


# --- fabric (scale-out / IFOE) variant: handle over TCP instead of fd ---------


def _producer_fabric(port, device, ready):
    from atom.kv_transfer.disaggregation.native import VmmBuffer

    torch.cuda.set_device(device)
    buf = VmmBuffer.alloc(NBYTES, device, fabric=True)
    kv = buf.tensor(torch.bfloat16, (NB, BE))
    for i in range(NB):
        kv[i].fill_(float(i % 128))
    torch.cuda.synchronize()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    conn.sendall(buf.export_fabric())  # 64-byte fabric handle
    conn.recv(1)  # keep alive until consumer is done
    conn.close()
    srv.close()


def _consumer_fabric(port, device, ready, result):
    from atom.kv_transfer.disaggregation.native import VmmBuffer

    torch.cuda.set_device(device)
    ready.wait(60)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    handle = b""
    while len(handle) < 64:
        handle += s.recv(64 - len(handle))

    peer = VmmBuffer.import_fabric(handle, NBYTES, device).tensor(
        torch.bfloat16, (NB, BE)
    )
    local = torch.empty(NB, BE, dtype=torch.bfloat16, device=device)
    for i in range(NCOPY):
        local[i % NB].copy_(peer[(i * 7) % NB])
    torch.cuda.synchronize()

    ok = all(
        local[i % NB][0].item() == float(((i * 7) % NB) % 128) for i in range(NCOPY)
    )
    result.put(bool(ok))
    s.send(b"d")
    s.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >= 2 GPUs",
)
def test_vmm_fabric_cross_process_transfer():
    """Fabric-handle path (scale-out / IFOE); skips on parts without fabric."""
    from atom.kv_transfer.disaggregation.native import supported_fabric

    if not supported_fabric(0) or not supported_fabric(1):
        pytest.skip("HIP VMM fabric handles not supported on these devices")

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    result = ctx.Queue()
    port = 53791

    prod = ctx.Process(target=_producer_fabric, args=(port, 0, ready))
    cons = ctx.Process(target=_consumer_fabric, args=(port, 1, ready, result))
    prod.start()
    cons.start()
    cons.join(180)
    prod.join(30)

    assert cons.exitcode == 0, "consumer process crashed"
    assert prod.exitcode == 0, "producer process crashed"
    assert result.get(timeout=5) is True, "fabric peer-copied data mismatch"
