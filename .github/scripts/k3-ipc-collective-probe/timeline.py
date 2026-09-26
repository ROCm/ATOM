"""Bounded counter-backed IPC/collective test; no model or per-step HIP events."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import timedelta
from pathlib import Path

import stress
from stress import DEPTH, ELEMENTS, WORLD, recv, save, trace
from timeline_backend import Channel, Native, preflight


def producer(rank, connection, port, variant, out):
    import torch
    import torch.distributed as dist
    import torch.multiprocessing

    config = json.loads((out.parent / "config.json").read_text())
    steps = config["steps"][variant]
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=90),
    )
    stream = torch.cuda.current_stream()
    native = Native(Path(config["build"]))
    writer = Channel(native, config["epoch"], rank, "p", True, stream)
    shared = torch.zeros((DEPTH, 4096), device=f"cuda:{rank}", dtype=torch.int32)
    source = torch.full(
        (ELEMENTS,), rank + 1, device=f"cuda:{rank}", dtype=torch.bfloat16
    )
    gathered = torch.empty(
        WORLD * ELEMENTS, device=f"cuda:{rank}", dtype=torch.bfloat16
    )
    connection.send({"kind": "tensor", "tensor": shared})
    assert recv(connection) == {"kind": "mapped"}
    reader = Channel(native, config["epoch"], rank, "c", False, stream)
    assert writer.value() == reader.value() == 0
    writer.path.unlink()  # Consumer has mapped it; the mapping survives unlink.
    connection.send({"kind": "counter_mapped"})
    for _ in range(3):
        dist.all_gather_into_tensor(gathered, source)
    torch.cuda.synchronize()
    connection.send({"kind": "start"})
    assert recv(connection) == {"kind": "started"}
    logfile = out / f"producer-{rank}.jsonl"
    pending = deque(maxlen=DEPTH)  # Tokens own no registration or GPU allocation.
    peak_pending = 0
    ready_queries = 0
    started = time.monotonic()

    def drain(expected):
        reply = recv(connection)
        assert reply["kind"] == "event" and reply["step"] == expected, reply
        trace(logfile, phase="import_enter", step=expected)
        token = reader.import_token(reply["handle"])
        assert token.generation == expected + 1
        trace(logfile, phase="import_exit", step=expected)
        token.wait(stream)
        pending.append((expected, token))

    for step in range(steps):
        if step >= DEPTH:
            drain(step - DEPTH)
        peak_pending = max(peak_pending, len(pending))
        if pending and pending[0][1].query():
            completed = pending[0][0]
            trace(logfile, phase="retire_enter", step=completed)
            pending.popleft()
            ready_queries += 1
            trace(logfile, phase="retire_exit", step=completed)
        dist.all_gather_into_tensor(gathered, source)
        shared[step % DEPTH].fill_(step + 1)
        trace(logfile, phase="record_enter", step=step)
        token = writer.record(stream)
        trace(logfile, phase="record_exit", step=step)
        connection.send({"kind": "event", "step": step, "handle": token.ipc_handle()})
        del token
    for step in range(max(0, steps - DEPTH), steps):
        drain(step)
    torch.cuda.synchronize()
    expected = torch.arange(1, WORLD + 1, device=f"cuda:{rank}", dtype=torch.bfloat16)
    correct = bool((gathered.view(WORLD, ELEMENTS) == expected[:, None]).all().item())
    assert correct, "Collective result mismatch"
    assert writer.value() == reader.value() == steps
    assert all(token.query() for _, token in pending)
    elapsed = time.monotonic() - started
    pending.clear()
    # All producer waits have completed. Consumer drains its stream before ACK.
    connection.send({"kind": "done"})
    assert recv(connection) == {"kind": "drained"}
    reader.close_after_drain(steps)
    writer.close_after_drain(steps)
    connection.send({"kind": "closed"})
    assert recv(connection) == {"kind": "released"}
    dist.destroy_process_group()
    return {
        "rank": rank,
        "steps": steps,
        "seconds": elapsed,
        "collective_correct": correct,
        "peak_pending_tokens": peak_pending,
        "ready_queries": ready_queries,
        "final_counter": steps,
        "writer": writer.stats(),
        "reader": reader.stats(),
    }


def consumer(connections, build, variant, out):
    import torch
    import torch.multiprocessing

    config = json.loads((out.parent / "config.json").read_text())
    steps = config["steps"][variant]
    torch.set_num_threads(1)
    native = Native(build)
    stop = threading.Event()
    progress, failures = {}, []
    results = [None] * WORLD

    def heartbeat():
        tick = 0
        while not stop.wait(1):
            tick += 1
            save(
                out / "heartbeat.json",
                {
                    "ticks": tick,
                    "mono_ns": time.monotonic_ns(),
                    "progress": dict(progress),
                },
            )

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()

    def work(rank, connection):
        try:
            torch.cuda.set_device(rank)
            stream = torch.cuda.Stream(device=rank)
            writer = Channel(native, config["epoch"], rank, "c", True, stream)
            errors = torch.zeros((), device=f"cuda:{rank}", dtype=torch.int64)
            message = recv(connection)
            assert message["kind"] == "tensor"
            shared = message.pop("tensor")
            reader = Channel(native, config["epoch"], rank, "p", False, stream)
            connection.send({"kind": "mapped"})
            assert recv(connection) == {"kind": "counter_mapped"}
            writer.path.unlink()
            assert recv(connection) == {"kind": "start"}
            torch.cuda.synchronize(rank)
            connection.send({"kind": "started"})
            logfile = out / f"consumer-{rank}.jsonl"
            started = time.monotonic()
            for step in range(steps):
                message = recv(connection)
                assert message["kind"] == "event" and message["step"] == step
                progress[rank] = {"step": step, "phase": "import"}
                trace(logfile, phase="import_enter", step=step)
                token = reader.import_token(message["handle"])
                assert token.generation == step + 1
                trace(logfile, phase="import_exit", step=step)
                with torch.cuda.stream(stream):
                    token.wait(stream)
                    errors.add_((shared[step % DEPTH] != step + 1).sum())
                    trace(logfile, phase="record_enter", step=step)
                    reply = writer.record(stream)
                    trace(logfile, phase="record_exit", step=step)
                    handle = reply.ipc_handle()
                    del reply
                trace(logfile, phase="retire_enter", step=step)
                del token
                trace(logfile, phase="retire_exit", step=step)
                progress[rank] = {"step": step, "phase": "retired"}
                connection.send({"kind": "event", "step": step, "handle": handle})
            assert recv(connection) == {"kind": "done"}
            stream.synchronize()
            errors_count = int(errors.item())
            assert errors_count == 0, errors_count
            assert writer.value() == reader.value() == steps
            # Both directions are idle before either side unregisters memory.
            reader.close_after_drain(steps)
            writer.close_after_drain(steps)
            connection.send({"kind": "drained"})
            assert recv(connection) == {"kind": "closed"}
            del shared
            connection.send({"kind": "released"})
            results[rank] = {
                "rank": rank,
                "steps": steps,
                "errors": errors_count,
                "seconds": time.monotonic() - started,
                "final_counter": steps,
                "writer": writer.stats(),
                "reader": reader.stats(),
            }
        except (
            RuntimeError,
            TimeoutError,
            EOFError,
            AssertionError,
            OSError,
            KeyError,
            ValueError,
            TypeError,
        ) as error:
            failures.append(
                {
                    "rank": rank,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )

    workers = [
        threading.Thread(target=work, args=(i, connection), daemon=True)
        for i, connection in enumerate(connections)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    stop.set()
    monitor.join()
    save(out / "consumer-threads.json", {"results": results, "failures": failures})
    assert not failures and all(results), failures
    return {"threads": results}


def main(args):
    preflight()
    if args.variant != "all":
        stress.producer, stress.consumer = producer, consumer
        stress.STEPS = int(args.variant.removeprefix("timeline"))
        stress.run_variant(args)
        return
    import torch

    assert torch.version.hip and torch.cuda.device_count() == WORLD
    args.out.mkdir(parents=True, exist_ok=True)
    args.build.mkdir(parents=True, exist_ok=True)
    rocm = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        "-D__HIP_PLATFORM_AMD__",
        f"-I{rocm}/include",
        str(Path(__file__).with_suffix(".cpp")),
        f"-L{rocm}/lib",
        "-lamdhip64",
        f"-Wl,-rpath,{rocm}/lib",
        "-o",
        str(args.build / "timeline.so"),
    ]
    subprocess.run(command, check=True, timeout=120)
    epoch = uuid.uuid4().hex
    variants = {"timeline640": 640, "timeline6400": 6400}
    save(
        args.out / "config.json",
        {
            "variants": list(variants),
            "steps": variants,
            "epoch": epoch,
            "build": str(args.build),
            "world_size": WORLD,
            "credit_depth": DEPTH,
            "elements": ELEMENTS,
            "model_loaded": False,
            "channel_count": 16,
            "bytes_per_channel": 4096,
            "registrations_per_process_channel": 1,
            "variant_timeout_seconds": 600,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "GPU_STREAMOPS_CP_WAIT": os.environ.get("GPU_STREAMOPS_CP_WAIT"),
            "blocking_wait": {
                key: os.environ.get(key)
                for key in ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT")
            },
            "compile_command": command,
        },
    )
    for variant in variants:
        with (args.out / f"{variant}.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--variant",
                    variant,
                    "--out",
                    str(args.out),
                    "--build",
                    str(args.build),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                rc = 124
            finally:
                stress.terminate_group(process)
                # Only this run's startup names; steady-state names were already unlinked.
                leftovers = list(Path("/dev/shm").glob(f"k3-timeline-{epoch}-*"))
                for path in leftovers:
                    path.unlink()
        save(
            args.out / f"{variant}.status.json",
            {
                "return_code": rc,
                "startup_names_removed": [str(path) for path in leftovers],
            },
        )
        print(f"IPC-TIMELINE variant={variant} rc={rc}", flush=True)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("all", "timeline640", "timeline6400"), default="all"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/k3-timeline-build"))
    main(parser.parse_args())
