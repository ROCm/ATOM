"""Bounded IPC lifetime/collective experiment; no model weights are loaded."""

import argparse
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import timedelta
from pathlib import Path

VARIANTS = (
    "original",
    "nogil_serial",
    "nogil",
    "original_retain",
    "nogil_serial_retain",
    "nogil_retain",
)
WORLD = 8
DEPTH = 16
STEPS = 640
ELEMENTS = 4755456
RING = 8192


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trace(path, **fields):
    with path.open("a") as stream:
        stream.write(json.dumps({"mono_ns": time.monotonic_ns(), **fields}) + "\n")


def recv(connection):
    if not connection.poll(120):
        raise TimeoutError("No IPC protocol progress for 120 seconds")
    return connection.recv()


def guarded(function, report, *args):
    try:
        result = function(*args)
        save(report, {"ok": True, **result})
    except BaseException as error:
        save(
            report,
            {"ok": False, "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


def producer(rank, connection, port, variant, out):
    import torch
    import torch.distributed as dist
    import torch.multiprocessing  # Registers tensor IPC reducers before using Pipe.

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=90),
    )
    shared = torch.zeros((DEPTH, 4096), device=f"cuda:{rank}", dtype=torch.int32)
    source = torch.full(
        (ELEMENTS,), rank + 1, device=f"cuda:{rank}", dtype=torch.bfloat16
    )
    gathered = torch.empty(
        WORLD * ELEMENTS, device=f"cuda:{rank}", dtype=torch.bfloat16
    )
    connection.send({"kind": "tensor", "tensor": shared})
    assert recv(connection) == {"kind": "mapped"}
    # Identical warmup and startup synchronization for every independent variant.
    for _ in range(3):
        dist.all_gather_into_tensor(gathered, source)
    torch.cuda.synchronize()
    connection.send({"kind": "start"})
    assert recv(connection) == {"kind": "started"}
    stream = torch.cuda.current_stream()
    exported = deque(maxlen=RING)
    pending = deque()
    retained = []
    logfile = out / f"producer-{rank}.jsonl"
    started = time.monotonic()

    def drain(expected):
        reply = recv(connection)
        assert reply["kind"] == "event" and reply["step"] == expected, reply
        trace(logfile, phase="import_enter", step=expected)
        event = torch.cuda.Event.from_ipc_handle(rank, reply["handle"])
        trace(logfile, phase="import_exit", step=expected)
        event.wait(stream)
        pending.append((expected, event))
        if variant.endswith("_retain"):
            retained.append(event)

    def retire_ready():
        if pending and pending[0][1].query():
            completed = pending[0][0]
            trace(logfile, phase="retire_enter", step=completed)
            pending.popleft()
            trace(logfile, phase="retire_exit", step=completed)

    for step in range(STEPS):
        if step >= DEPTH:
            drain(step - DEPTH)
        retire_ready()
        dist.all_gather_into_tensor(gathered, source)
        shared[step % DEPTH].fill_(step + 1)
        event = torch.cuda.Event(interprocess=True)
        trace(logfile, phase="record_enter", step=step)
        event.record(stream)
        trace(logfile, phase="record_exit", step=step)
        handle = event.ipc_handle()
        exported.append(event)
        connection.send({"kind": "event", "step": step, "handle": handle})
    for step in range(max(0, STEPS - DEPTH), STEPS):
        drain(step)
    torch.cuda.synchronize()
    expected = torch.arange(1, WORLD + 1, device=f"cuda:{rank}", dtype=torch.bfloat16)
    correct = bool((gathered.view(WORLD, ELEMENTS) == expected[:, None]).all().item())
    assert correct, "Collective result mismatch"
    elapsed = time.monotonic() - started
    # Retire while this rank has no remaining GPU operations or future collectives.
    pending.clear()
    retained.clear()
    exported.clear()
    connection.send({"kind": "done"})
    assert recv(connection) == {"kind": "released"}
    dist.destroy_process_group()
    return {
        "rank": rank,
        "steps": STEPS,
        "seconds": elapsed,
        "collective_correct": correct,
    }


def consumer(connections, build, variant, out):
    import torch
    import torch.multiprocessing  # Registers tensor IPC reducers before using Pipe.

    torch.set_num_threads(1)
    sys.path.insert(0, str(build))
    native = importlib.import_module("k3_hip_event_import")
    import_lock = threading.Lock()
    stop = threading.Event()
    progress = {}
    results = [None] * WORLD
    failures = []
    exported = deque(maxlen=RING)

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
            errors = torch.zeros((), device=f"cuda:{rank}", dtype=torch.int64)
            message = recv(connection)
            assert message["kind"] == "tensor"
            shared = message.pop("tensor")
            connection.send({"kind": "mapped"})
            assert recv(connection) == {"kind": "start"}
            torch.cuda.synchronize(rank)
            connection.send({"kind": "started"})
            logfile = out / f"consumer-{rank}.jsonl"
            retained = []
            imports = []
            started = time.monotonic()
            for step in range(STEPS):
                message = recv(connection)
                assert message["kind"] == "event" and message["step"] == step
                progress[rank] = {"step": step, "phase": "import"}
                trace(logfile, phase="import_enter", step=step)
                begin = time.monotonic()
                if variant.startswith("original"):
                    event = torch.cuda.Event.from_ipc_handle(rank, message["handle"])
                elif variant.startswith("nogil_serial"):
                    trace(logfile, phase="lock_wait_enter", step=step)
                    with import_lock:
                        trace(logfile, phase="lock_acquired", step=step)
                        trace(logfile, phase="native_enter", step=step)
                        event = native.from_ipc_handle(rank, message["handle"])
                        trace(logfile, phase="native_exit", step=step)
                else:
                    event = native.from_ipc_handle(rank, message["handle"])
                imports.append(time.monotonic() - begin)
                trace(logfile, phase="import_exit", step=step)
                progress[rank] = {"step": step, "phase": "enqueue"}
                with torch.cuda.stream(stream):
                    event.wait(stream)
                    errors.add_((shared[step % DEPTH] != step + 1).sum())
                    reply = torch.cuda.Event(interprocess=True)
                    trace(logfile, phase="record_enter", step=step)
                    reply.record(stream)
                    trace(logfile, phase="record_exit", step=step)
                    handle = reply.ipc_handle()
                    exported.append(reply)
                if variant.endswith("_retain"):
                    retained.append(event)
                progress[rank] = {"step": step, "phase": "retire"}
                trace(logfile, phase="retire_enter", step=step)
                del event
                trace(logfile, phase="retire_exit", step=step)
                progress[rank] = {"step": step, "phase": "retired"}
                connection.send({"kind": "event", "step": step, "handle": handle})
            assert recv(connection) == {"kind": "done"}
            stream.synchronize()
            errors_count = int(errors.item())
            assert errors_count == 0, errors_count
            retained.clear()
            del shared
            connection.send({"kind": "released"})
            results[rank] = {
                "rank": rank,
                "steps": STEPS,
                "errors": errors_count,
                "seconds": time.monotonic() - started,
                "max_import_seconds": max(imports),
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


def run_variant(args):
    import torch.multiprocessing as mp

    out = args.out / args.variant
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    pairs = [ctx.Pipe() for _ in range(WORLD)]
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    processes = [
        ctx.Process(
            target=guarded,
            args=(
                consumer,
                out / "consumer.json",
                [p[0] for p in pairs],
                args.build,
                args.variant,
                out,
            ),
        )
    ]
    processes.extend(
        ctx.Process(
            target=guarded,
            args=(
                producer,
                out / f"rank-{i}.json",
                i,
                pair[1],
                port,
                args.variant,
                out,
            ),
        )
        for i, pair in enumerate(pairs)
    )
    try:
        for process in processes:
            process.start()
        for pair in pairs:
            for connection in pair:
                connection.close()
        while any(process.is_alive() for process in processes):
            failed = [p for p in processes if p.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(
                    f"Child process failed: {[(p.pid, p.exitcode) for p in failed]}"
                )
            time.sleep(0.2)
        for process in processes:
            process.join()
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for pair in pairs:
            for connection in pair:
                connection.close()
        started = [p for p in processes if p.pid is not None]
        for process in started:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        for process in started:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in started:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 5
        for process in started:
            process.join(timeout=max(0, deadline - time.monotonic()))
    save(
        out / "summary.json",
        {
            "ok": True,
            "variant": args.variant,
            "world_size": WORLD,
            "steps_per_rank": STEPS,
            "credit_depth": DEPTH,
        },
    )


def terminate_group(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        # A dead group leader does not imply its children have exited.
        if sig == signal.SIGTERM:
            continue
    process.wait(timeout=5)


def main(args):
    if args.variant != "all":
        run_variant(args)
        return
    import torch
    from torch.utils.cpp_extension import ROCM_HOME, load

    assert torch.version.hip and torch.cuda.device_count() == WORLD
    args.out.mkdir(parents=True, exist_ok=True)
    args.build.mkdir(parents=True, exist_ok=True)
    native_source = Path(__file__).parents[1] / "k3-prefill-probe/import_event.cpp"
    load(
        name="k3_hip_event_import",
        sources=[str(native_source)],
        build_directory=str(args.build),
        with_cuda=False,
        extra_cflags=["-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM=1"],
        extra_include_paths=[str(Path(ROCM_HOME) / "include")],
        extra_ldflags=["-lc10_hip", "-ltorch_hip", f"-L{ROCM_HOME}/lib", "-lamdhip64"],
    )
    save(
        args.out / "config.json",
        {
            "variants": args.variants,
            "world_size": WORLD,
            "steps": STEPS,
            "credit_depth": DEPTH,
            "elements": ELEMENTS,
            "export_ring_size": RING,
            "variant_timeout_seconds": 240,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "retention_scope": "consumer and producer imported events until each rank is idle",
            "producer_import": "original torch in every variant",
            "producer_retirement": "one nonblocking pending query per step; release only when ready",
            "blocking_wait": {
                name: os.environ.get(name)
                for name in ("TORCH_NCCL_BLOCKING_WAIT", "NCCL_BLOCKING_WAIT")
            },
            "model_loaded": False,
        },
    )
    for variant in args.variants:
        with (args.out / f"{variant}.log").open("w") as stream:
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
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                rc = 124
            finally:
                terminate_group(process)
        save(args.out / f"{variant}.status.json", {"return_code": rc})
        print(f"IPC-STRESS variant={variant} rc={rc}", flush=True)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", *VARIANTS), default="all")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=Path("/tmp/k3-ipc-stress-build"))
    main(parser.parse_args())
