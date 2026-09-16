#!/usr/bin/env python3
"""Verify same-host Mooncake GPU transfer correctness without TCP payload fallback.

Run in a HIP-enabled ATOM container on an otherwise idle pair of GPUs:
    python3 .github/scripts/atomesh/hip_smoke.py --source-gpu 0 --target-gpu 4

Control-plane RPC uses loopback. Payload-sized loopback traffic indicates a
TCP fallback, which older Mooncake builds permit even with protocol="hip".
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def loopback_received_bytes():
    for line in Path("/proc/net/dev").read_text().splitlines():
        if line.strip().startswith("lo:"):
            return int(line.split(":", 1)[1].split()[0])
    raise RuntimeError("Loopback traffic counters are required to detect TCP fallback")


def worker(role, gpu, pipe):
    try:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        import torch
        from mooncake.engine import TransferEngine

        from atom.kv_transfer.disaggregation.mooncake.mooncake_connector import (
            _configure_mooncake_transport,
            _engine_device_filter,
            _select_ib_device,
        )

        assert torch.version.hip
        torch.cuda.set_device(0)
        size = 64 * 1024 * 1024
        buffer = torch.full(
            (size,), 37 if role == "producer" else 0, dtype=torch.uint8, device="cuda"
        )
        torch.cuda.synchronize()
        _configure_mooncake_transport("hip")
        device_filter = _engine_device_filter(
            "hip", _select_ib_device("hip", "ionic_0", None)
        )
        engine = TransferEngine()
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "hip", device_filter) == 0
        assert engine.register_memory(buffer.data_ptr(), size) == 0
        pipe.send(
            {
                "engine": f"127.0.0.1:{engine.get_rpc_port()}",
                "ptr": buffer.data_ptr(),
                "gpu": gpu,
                "filter": device_filter,
            }
        )
        if role == "producer":
            peer = pipe.recv()
            for _ in range(3):
                assert (
                    engine.batch_transfer_sync_write(
                        peer["engine"], [buffer.data_ptr()], [peer["ptr"]], [size]
                    )
                    == 0
                )
            iterations = 100
            loopback_before = loopback_received_bytes()
            start = time.perf_counter()
            for _ in range(iterations):
                assert (
                    engine.batch_transfer_sync_write(
                        peer["engine"], [buffer.data_ptr()], [peer["ptr"]], [size]
                    )
                    == 0
                )
            elapsed = time.perf_counter() - start
            pipe.send(
                {
                    "bytes": size * iterations,
                    "seconds": elapsed,
                    "GB_per_second": size * iterations / elapsed / 1e9,
                    "loopback_received_bytes": loopback_received_bytes()
                    - loopback_before,
                }
            )
            pipe.recv()
            buffer.fill_(83)
            torch.cuda.synchronize()
            assert (
                engine.batch_transfer_sync_write(
                    peer["engine"], [buffer.data_ptr()], [peer["ptr"]], [size]
                )
                == 0
            )
            pipe.send("second-pattern-written")
            pipe.recv()
        else:
            for expected in (37, 83):
                assert pipe.recv() == "verify"
                torch.cuda.synchronize()
                assert bool(
                    torch.all(buffer == expected).item()
                ), f"GPU {gpu} payload mismatch"
                pipe.send({"verified_bytes": size, "pattern": expected})
            pipe.recv()
        assert engine.unregister_memory(buffer.data_ptr()) == 0
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
        raise


def receive(pipe):
    assert pipe.poll(180), "Worker timed out"
    value = pipe.recv()
    assert not (isinstance(value, dict) and "error" in value), value
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.source_gpu == args.target_gpu:
        parser.error("Select two distinct GPUs")
    ctx = mp.get_context("spawn")
    processes = []
    pipes = []
    try:
        for role, gpu in [("producer", args.source_gpu), ("consumer", args.target_gpu)]:
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(role, gpu, child))
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        source, target = pipes
        source_meta, target_meta = receive(source), receive(target)
        source.send(target_meta)
        metrics = receive(source)
        target.send("verify")
        first = receive(target)
        source.send("next-pattern")
        receive(source)
        target.send("verify")
        second = receive(target)
        source.send("done")
        target.send("done")
        for process in processes:
            process.join(30)
            assert process.exitcode == 0, process.exitcode
        assert metrics["loopback_received_bytes"] < metrics["bytes"] // 10, (
            (
                "Payload-sized loopback traffic: HIP is falling back to TCP. "
                "Use a Mooncake build that installs HIP transport."
            ),
            metrics,
        )
        print(
            "HIP_TRANSFER_PASS="
            + json.dumps(
                {
                    "source": source_meta,
                    "target": target_meta,
                    "benchmark": metrics,
                    "verification": [first, second],
                }
            ),
            flush=True,
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)


if __name__ == "__main__":
    main()
