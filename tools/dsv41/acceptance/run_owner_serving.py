"""Run real client generation, capacity screening and full GSM8K on one load."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from run_prefill_route_capture import bounded_log

HERE = Path(__file__).resolve().parent
OUTPUT = Path(os.environ["DSV41_OWNER_SERVING_OUTPUT"])
CONTROL = Path(os.environ["DSV41_OWNER_SERVING_CONTROL"])
MODEL = "/models/DeepSeek-V4.1-Flash"
PORT = 19125
URL = f"http://127.0.0.1:{PORT}"


def save(path, value):
    content = (json.dumps(value, indent=2) + "\n").encode()
    if len(content) > 2_000_000:
        raise RuntimeError("Manifest exceeded its bound")
    path.write_bytes(content)


def client_environment():
    environment = dict(os.environ)
    environment["DSV41_OWNER_SERVING"] = "0"
    environment.pop("ATOM_BENCHMARK_REQUESTS", None)
    return environment


def child(argv, label, server=None, timeout=1800):
    process = subprocess.Popen(
        argv, env=client_environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    drain = threading.Thread(
        target=bounded_log, args=(process.stdout, OUTPUT / f"{label}.log", 2_000_000)
    )
    drain.start()
    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError(f"Serving model exited during {label}")
            failures = list(OUTPUT.glob("owner_failure_rank*.json"))
            if server is not None and failures:
                raise RuntimeError(
                    f"Owner worker failed during {label}: {failures[0].read_text()}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"Client {label} exceeded its ceiling")
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            raise RuntimeError(f"Client {label} failed, see its log")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        drain.join(timeout=10)


def benchmark(
    label,
    mode,
    server,
    *,
    count=256,
    concurrency=256,
    input_length=8192,
    output_length=1024,
    seed=2026102518,
):
    CONTROL.write_text(mode + "\n")
    before = routes()
    argv = [
        sys.executable,
        "-m",
        "atom.benchmarks.benchmark_serving",
        "--model",
        MODEL,
        "--served-model-name",
        MODEL,
        "--backend",
        "vllm",
        "--base-url",
        URL,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_length),
        "--random-output-len",
        str(output_length),
        "--random-range-ratio",
        "0.8",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--trust-remote-code",
        "--use-chat-template",
        "--save-result",
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(count),
        "--num-warmups",
        "0",
        "--seed",
        str(seed),
        "--result-dir",
        str(OUTPUT),
        "--result-filename",
        label + ".json",
    ]
    print(f"OWNER_SERVING_CLIENT start {label} mode={mode} C={concurrency}", flush=True)
    child(argv, label, server)
    result = json.loads((OUTPUT / f"{label}.json").read_text())
    if (
        result["completed"] != count
        or any(result.get("errors", []))
        or result["total_output_tokens"] <= 0
    ):
        raise RuntimeError("Incomplete client-visible generation")
    after = routes()
    delta = [
        {
            key: rank["counts"].get(key, 0) - old.get("counts", {}).get(key, 0)
            for key in rank["counts"]
        }
        for rank, old in zip(after, before, strict=True)
    ]
    summary = {
        key: result[key]
        for key in (
            "completed",
            "duration",
            "total_input_tokens",
            "total_output_tokens",
            "mean_tpot_ms",
            "output_throughput",
        )
    }
    summary.update(mode=mode, concurrency=concurrency, route_delta=delta)
    save(OUTPUT / f"{label}_summary.json", summary)
    print(
        f"OWNER_SERVING_CLIENT done {label} TPOT={summary['mean_tpot_ms']:.6f}ms throughput={summary['output_throughput']:.3f}tok/s",
        flush=True,
    )
    return summary


def routes():
    return [
        json.loads(path.read_text()) if path.exists() else {"counts": {}}
        for path in (OUTPUT / f"route_rank{rank}.json" for rank in range(4))
    ]
