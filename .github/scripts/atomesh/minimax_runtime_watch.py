#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only Python thread snapshots and read-only host memory sampling."""

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

_STARTED_PID = None


def process_start(pid):
    # comm can contain spaces and parentheses; fields after it start at state.
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def start():
    global _STARTED_PID
    root = os.environ.get("ATOM_CI_RUNTIME_DIAGNOSTICS")
    if not root or _STARTED_PID == os.getpid():
        return
    role = os.environ.get("ATOM_CI_RUNTIME_ROLE", "unknown")
    folder = Path(root) / role
    folder.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    stack_path = folder / f"pid-{pid}.stacks.log"
    stack_path.touch()
    (folder / f"pid-{pid}.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "start_ticks": process_start(pid),
                "role": role,
                "stack_sampler": "python_thread",
                "unix_time": time.time(),
            }
        )
        + "\n"
    )
    _STARTED_PID = pid
    threading.Thread(
        target=watch_stacks, args=(stack_path,), daemon=True, name="ci-stack-sampler"
    ).start()


def dump_stacks(path):
    # Public Python APIs retain frame references and run with the GIL. Do not
    # use an asynchronous native frame walker: on Python 3.12 the first native
    # timeout dump coincided with SIGSEGV during threaded loading (38795325170).
    frames = sys._current_frames()
    lines = [f"unix_time={time.time()} pid={os.getpid()} sampler=python_thread\n"]
    try:
        for ident, frame in list(frames.items())[:128]:
            lines.append(f"Thread {ident}:\n")
            lines.extend(traceback.format_stack(frame, limit=40))
    finally:
        # The sampler's own frame is in this dictionary; break that cycle.
        frames.clear()
        frame = None
        del frames
    if path.stat().st_size > 8 << 20:
        path.replace(path.with_suffix(".previous.log"))
    with path.open("a") as stream:
        stream.write("".join(lines))


def watch_stacks(path):
    while True:
        time.sleep(120)
        try:
            dump_stacks(path)
        except OSError:
            return  # Diagnostics must not terminate a model worker.


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"unavailable: {exc}"


def sample(root):
    vmstat = read("/proc/vmstat")
    prefixes = (
        "compact_",
        "thp_",
        "pgscan_",
        "pgsteal_",
        "allocstall",
        "pgmajfault",
        "pswp",
        "unevictable_",
        "nr_mlock",
        "nr_foll_pin",
    )
    result = {
        "unix_time": time.time(),
        "vmstat": [line for line in vmstat.splitlines() if line.startswith(prefixes)],
        "meminfo": read("/proc/meminfo"),
        "memory_pressure": read("/proc/pressure/memory"),
        "buddyinfo": read("/proc/buddyinfo"),
        "numa": {
            p.parent.name: read(p)
            for p in Path("/sys/devices/system/node").glob("node*/meminfo")
        },
        "processes": [],
    }
    for path in sorted(root.glob("*/pid-*.json")):
        try:
            row = json.loads(path.read_text())
            pid = row["pid"]
            if process_start(pid) != row["start_ticks"]:
                continue  # Never inspect a reused PID belonging to another run.
            proc = Path("/proc") / str(pid)
            row["status"] = read(proc / "status")
            row["threads"] = {
                p.name: read(p / "wchan") for p in (proc / "task").iterdir()
            }
            result["processes"].append(row)
        except (OSError, ValueError, KeyError):
            continue
    return result


def empty_gather_kv(shape, *, dtype, device):
    """Observe allocator entry/return without synchronizing GPU work."""
    import torch

    root = os.environ.get("ATOM_CI_RUNTIME_DIAGNOSTICS")
    role = os.environ.get("ATOM_CI_RUNTIME_ROLE", "unknown")
    path = Path(root) / role / f"pid-{os.getpid()}.allocations.jsonl" if root else None
    started = time.monotonic()

    def record(stage):
        if path is None:
            return
        stats = torch.cuda.memory_stats(device)
        row = {
            "unix_time": time.time(),
            "elapsed_s": time.monotonic() - started,
            "stage": stage,
            "shape": list(shape),
            "dtype": str(dtype),
            "device": str(device),
            "allocator_config": {
                name: os.environ.get(name)
                for name in (
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "PYTORCH_HIP_ALLOC_CONF",
                    "PYTORCH_ALLOC_CONF",
                )
            },
            "memory": {
                name: stats.get(name)
                for name in (
                    "allocated_bytes.all.current",
                    "reserved_bytes.all.current",
                    "active_bytes.all.current",
                    "inactive_split_bytes.all.current",
                    "num_alloc_retries",
                    "num_ooms",
                    "num_device_alloc",
                    "num_device_free",
                )
            },
        }
        try:
            if path.exists() and path.stat().st_size > 8 << 20:
                path.replace(path.with_suffix(".previous.jsonl"))
            with path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        except OSError:
            pass  # A full diagnostics disk must not stop inference.

    record("before_k")
    k = torch.empty(shape, dtype=dtype, device=device)
    record("after_k_before_v")
    v = torch.empty(shape, dtype=dtype, device=device)
    record("after_v")
    return k, v


def ipc_compatible_kv_zeros(size, *, dtype, device):
    """Keep the startup-only shared KV pool exportable through HIP IPC.

    aiter supports expandable segments for inference, but the pinned Mooncake
    build also registers an IPC handle even when protocol=rdma. Only the paged
    pool needs native storage. This runs before request/offload threads start;
    never toggle the process-wide allocator during live inference.
    """
    import torch

    conf = next(
        (
            os.environ[name]
            for name in (
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_HIP_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
            )
            if os.environ.get(name)
        ),
        "",
    )
    fields = [
        field
        for field in conf.split(",")
        if field.partition(":")[0].strip() != "expandable_segments"
    ]
    expandable = any(
        field.replace(" ", "").lower() == "expandable_segments:true"
        for field in conf.split(",")
    )
    if not expandable:
        return torch.zeros(size, dtype=dtype, device=device)
    native = ",".join(
        [field for field in fields if field] + ["expandable_segments:False"]
    )
    print(
        f"[runtime-watch] Native IPC-compatible KV pool: {size} bytes on {device}",
        flush=True,
    )
    torch.cuda.memory._set_allocator_settings(native)
    try:
        return torch.zeros(size, dtype=dtype, device=device)
    finally:
        torch.cuda.memory._set_allocator_settings(conf)
        print(
            "[runtime-watch] Restored expandable segments for inference allocations",
            flush=True,
        )


def instrument_kv_pool(path):
    source = path.read_text()
    old = "            buf = torch.zeros(total, dtype=torch.uint8, device=self.device)"
    new = """            from atom._ci_runtime_watch import ipc_compatible_kv_zeros

            buf = ipc_compatible_kv_zeros(total, dtype=torch.uint8, device=self.device)"""
    if source.count(old) != 1:
        raise ValueError("Expected exactly one shared paged KV allocation")
    path.write_text(source.replace(old, new))


def instrument_gather(path):
    source = path.read_text()
    old = """        k_full = torch.empty(
            (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
        )
        v_full = torch.empty(
            (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
        )"""
    new = """        from atom._ci_runtime_watch import empty_gather_kv

        k_full, v_full = empty_gather_kv(
            (total_tokens, num_kv_heads, head_dim), dtype=k.dtype, device=k.device
        )"""
    if source.count(old) != 1:
        raise ValueError("Expected exactly one draft KV allocation block")
    path.write_text(source.replace(old, new))


def prepare(source, runtime, output):
    shutil.copytree(
        source / "atom",
        runtime / "atom",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    helper = Path(__file__).read_text()
    (runtime / "atom/_ci_runtime_watch.py").write_text(helper)
    init = runtime / "atom/__init__.py"
    init.write_text(
        "from atom._ci_runtime_watch import start as _ci_watch_start\n"
        "_ci_watch_start()\n" + init.read_text()
    )
    instrument_gather(runtime / "atom/model_ops/attention_mha.py")
    instrument_kv_pool(runtime / "atom/model_engine/model_runner.py")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime-watch-source.py").write_text(helper)
    (output / "runtime-overlay.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "runtime": str(runtime),
                "change": "Native shared KV pool; expandable inference allocations; Python stacks and allocation statistics",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.source:
        prepare(args.source, args.runtime, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "host-memory.jsonl"
    while True:
        if log.exists() and log.stat().st_size > 8 << 20:
            log.replace(log.with_suffix(".previous.jsonl"))
        with log.open("a") as stream:
            stream.write(json.dumps(sample(args.output)) + "\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
