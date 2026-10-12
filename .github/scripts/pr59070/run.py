"""Full GSM8K on two independent normal K3 servers from one pinned build."""

import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

scripts = Path(__file__).resolve().parent
root = Path(os.environ["RUN_DIR"])
template = json.loads((scripts / "launch-template.json").read_text())
url = "http://127.0.0.1:2594"
processes = []


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)


def spawn(argv, log, env=None):
    with log.open("w") as output:
        process = subprocess.Popen(
            argv,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    processes.append(process)
    return process


def run(argv, log, server, timeout=14400):
    process = spawn(argv, log)
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        assert server.poll() is None, f"Server exited during {log.name}"
        assert time.monotonic() < deadline, f"Timeout during {log.name}"
        time.sleep(5)
    assert process.returncode == 0, f"{log.name}: rc={process.returncode}"


def gpu_snapshot(path, require_empty=False):
    deadline = time.monotonic() + 120
    while True:
        data = json.loads(
            subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--showpids", "--json"], text=True
            )
        )
        save(path, data)
        cards = [v for k, v in data.items() if k.startswith("card")]
        assert len(cards) == 8, data
        if not require_empty or all(
            int(c["VRAM Total Used Memory (B)"]) < 2 * 1024**3 for c in cards
        ):
            return
        assert time.monotonic() < deadline, "GPU memory remains after server cleanup"
        time.sleep(5)


def metrics(path):
    response = requests.get(url + "/metrics", timeout=30)
    response.raise_for_status()
    path.write_text(response.text)


def rpc():
    response = requests.post(
        url + "/collective_rpc", json={"method": "experiment_info"}, timeout=120
    )
    response.raise_for_status()
    return response.json()


def launch(mode, directory):
    env = dict(os.environ)
    env.update(template["environment"])
    env.update(
        K3_CONTEXT_MODE=mode,
        PYTHONPATH=str(scripts),
        VLLM_CACHE_ROOT=f"/tmp/pr59070-{mode}/vllm-cache",
        TORCHINDUCTOR_CACHE_DIR=f"/tmp/pr59070-{mode}/inductor-cache",
        TRITON_CACHE_DIR=f"/tmp/pr59070-{mode}/triton-cache",
    )
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        os.environ["MODEL_PATH"],
        *shlex.split(os.environ["EXTRA_SERVER_ARGS"]),
        "--host",
        "127.0.0.1",
        "--worker-extension-cls",
        "static_control.StaticControl",
        "--profiler-config",
        json.dumps(
            {
                "profiler": "torch",
                "torch_profiler_dir": str(directory / "profiles"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
                "torch_profiler_with_memory": False,
                "torch_profiler_use_gzip": True,
                "ignore_frontend": True,
                "detailed_trace_annotation": True,
            }
        ),
    ]
    assert "--enforce-eager" not in argv
    assert "--disable-custom-all-reduce" not in argv
    save(
        directory / "launch.json",
        {
            "argv": argv,
            "environment": {k: env[k] for k in template["environment"]},
            "source_sha": template["source_sha"],
            "mode": mode,
            "control": "Select superclass fallback before compilation"
            if mode == "fallback"
            else "Unmodified PR context method",
            "packages": {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
            },
        },
    )
    return spawn(argv, directory / "model-server.log", env)


try:
    gpu_snapshot(root / "gpu-before.json", require_empty=True)
    proxy = spawn(
        [sys.executable, str(scripts / "response_proxy.py")], root / "proxy.log"
    )
    for mode in ("fallback", "fused"):
        directory = root / mode
        directory.mkdir(exist_ok=True)
        assert not (root / f"{mode}-responses.jsonl").exists()
        gpu_snapshot(directory / "gpu-before.json", require_empty=True)
        print(f"[{mode}] launching {template['source_sha']}", flush=True)
        server = launch(mode, directory)
        try:
            deadline = time.monotonic() + 7200
            while True:
                assert server.poll() is None, f"{mode} server exited before readiness"
                assert time.monotonic() < deadline, "Server readiness timeout"
                try:
                    if requests.get(url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            info = rpc()
            save(directory / "execution-info.json", info)
            ranks = info["results"]
            assert {r["rank"] for r in ranks} == set(range(8))
            expected = "MLACommonBaseImpl" if mode == "fallback" else "AiterMLAImpl"
            assert all(r["mode"] == mode and r["layers"] for r in ranks)
            assert all(
                r["context_implementation"]
                == expected + "._context_parallel_compute_prefill_context"
                for r in ranks
            )
            gpu_snapshot(directory / "gpu-loaded.json")
            metrics(directory / "gsm8k-before.metrics")
            command = [
                sys.executable,
                "-m",
                "lm_eval",
                "run",
                "--model",
                "local-completions",
                "--model_args",
                f"model=Kimi-K3,base_url=http://127.0.0.1:2595/{mode}/v1/completions,tokenizer_backend=None,tokenized_requests=False,num_concurrent=16,max_retries=1,timeout=1800,max_length=1048576,max_gen_toks=8192",
                "--tasks",
                "gsm8k",
                "--num_fewshot",
                "5",
                "--batch_size",
                "1",
                "--gen_kwargs",
                "max_gen_toks=8192,temperature=0.0,do_sample=False",
                "--seed",
                "0,1234,1234,1234",
                "--log_samples",
                "--output_path",
                str(directory / "gsm8k"),
            ]
            assert "--limit" not in command
            save(
                directory / "gsm8k-command.json",
                {
                    "argv": command,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"[{mode}] full GSM8K started", flush=True)
            run(command, directory / "gsm8k.log", server)
            metrics(directory / "gsm8k-after.metrics")
            result_files = list((directory / "gsm8k").rglob("results_*.json"))
            assert len(result_files) == 1, result_files
            result = json.loads(result_files[0].read_text())
            assert result["n-samples"]["gsm8k"] == {"original": 1319, "effective": 1319}
            responses = [
                json.loads(line) for line in (root / f"{mode}-responses.jsonl").open()
            ]
            assert len(responses) == 1319 and all(r["status"] == 200 for r in responses)
            save(directory / "accuracy-completed.json", result["results"]["gsm8k"])
            print(mode, result["results"]["gsm8k"], flush=True)
            try:
                # Independent prefix workload verifies the optimized path and TTFT.
                run(
                    [
                        sys.executable,
                        str(scripts / "profile_prefill.py"),
                        "--url",
                        url,
                        "--output",
                        str(directory / "client"),
                    ],
                    directory / "prefill-benchmark.log",
                    server,
                )
                traces = list((directory / "profiles").rglob("*.pt.trace.json.gz"))
                assert len(traces) == 8, traces
                coverage = {}
                for trace in traces:
                    with gzip.open(trace, "rt") as source:
                        events = json.load(source)["traceEvents"]
                    count = sum("gather_kv_b_proj" in e.get("name", "") for e in events)
                    coverage[trace.name] = count
                save(directory / "prefill-path-coverage.json", coverage)
                assert all(
                    c > 0 if mode == "fused" else c == 0 for c in coverage.values()
                ), coverage
            except Exception as exc:
                # Preserve both full accuracy arms if the secondary trace check fails.
                save(directory / "prefix-validation-failed.json", {"error": repr(exc)})
                print(f"[{mode}] prefix validation failed: {exc}", flush=True)
            save(
                directory / "completed.json",
                {"time": datetime.now(timezone.utc).isoformat()},
            )
        finally:
            stop(server)
        gpu_snapshot(directory / "gpu-after.json", require_empty=True)
    assert not list(root.glob("*/prefix-validation-failed.json")), (
        "Prefix coverage incomplete"
    )
    phase = os.environ.get("ATOMESH_EXECUTION_PHASE", "combined")
    token = os.environ.get("ATOMESH_RUN_TOKEN", os.environ["SLURM_JOB_ID"])
    (root / f"phase-{phase}.complete").write_text(f"{token}:{phase}\n")
finally:
    for process in reversed(processes):
        stop(process)
    gpu_snapshot(root / "gpu-final.json", require_empty=True)
