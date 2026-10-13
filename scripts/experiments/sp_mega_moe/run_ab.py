#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Portable SP4 native / full MegaMoE serving comparison. See the handoff MD."""

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def common_env():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT),
        PYTHONUNBUFFERED="1",
        HIP_VISIBLE_DEVICES=env.get("HIP_VISIBLE_DEVICES", "0,1,2,3"),
        GPU_ARCHS="gfx950",
        HSA_ENABLE_IPC_MODE_LEGACY="1",
        GLOO_SOCKET_IFNAME=env.get("GLOO_SOCKET_IFNAME", "lo"),
        OMP_NUM_THREADS="1",
        ATOM_LOADER_NUM_THREADS="1",
        AITER_LOG_LEVEL="WARNING",
        AITER_QUICK_REDUCE_QUANTIZATION="INT4",
        AITER_QUICK_REDUCE_CAST_BF16_TO_FP16="1",
        ATOM_FORCE_ATTN_TRITON="1",
        ATOM_USE_TRITON_MOE="0",
        ATOM_USE_TRITON_MOE_DECODE="0",
        ATOM_MOE_GU_ITLV="0",
        ATOM_M3_TP_REPLICATED_O_PROJ="0",
        ATOM_SP_MOE_MEGA_MAX_LOCAL_TOKENS="1024",
        ATOM_SP_MOE_MEGA_ACT_DTYPE="fp4",
        ATOM_SP_MOE_MEGA_COMM_DTYPE="fp8",
        ATOM_SP_MOE_MEGA_ALLOW_ACTIVATION_CHANGE="0",
    )
    for key in ("AITER_MEGAMOE_TP_LB_MT", "GPTOSS_SWIGLU_MXFP4_BF16_BOUND"):
        env.pop(key, None)
    return env


def server_command(args, batch):
    command = [
        sys.executable,
        "-u",
        "-m",
        "atom.entrypoints.openai_server",
        "--host",
        "127.0.0.1",
        "--server-port",
        str(args.port),
        "--model",
        args.model,
        "--trust-remote-code",
        "--tensor-parallel-size",
        "1",
        "--sequence-parallel-size",
        "4",
        "--all2all-backend",
        "none",
        "--no-enable_prefix_caching",
        "--kv-cache-dtype",
        "fp8",
        "--index-cache-dtype",
        "fp8",
        "--block-size",
        "128",
        "--max-model-len",
        "132096",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        str(batch),
        "--gpu-memory-utilization",
        "0.8",
        "--disable-uvicorn-access-log",
        "--online_quant_config",
        json.dumps(
            {
                "global_quant_config": "ptpc_fp8",
                "exclude_layer": [
                    "lm_head",
                    "model.embed_tokens",
                    "vision_tower",
                    "multi_modal_projector",
                    "patch_merge_mlp",
                    "*block_sparse_moe",
                ],
            }
        ),
        "--hf-overrides",
        '{"use_index_cache":true,"index_topk_freq":4}',
    ]
    if args.profile_dir:
        command += ["--torch-profiler-dir", str(args.profile_dir.resolve())]
    return command


def client_command(args, directory):
    return [
        sys.executable,
        "-u",
        "-m",
        "atom.benchmarks.benchmark_serving",
        "--model",
        args.model,
        "--trust-remote-code",
        "--backend",
        "vllm",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--dataset-name",
        "random",
        "--random-input-len",
        "131072",
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "1",
        "--random-prefix-len",
        "0",
        "--num-prompts",
        "160",
        "--num-warmups",
        "16",
        "--max-concurrency",
        "16",
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--seed",
        "0",
        "--save-result",
        "--save-detailed",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--result-dir",
        str(directory),
        "--result-filename",
        "result.json",
    ]


def stop_group(process):
    if process is None:
        return
    # The parent may have exited while its workers still hold GPU memory.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        process.poll()  # Reap the group leader if it has exited.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(server, telemetry, port):
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        if server.poll() is not None or telemetry.poll() is not None:
            raise RuntimeError("Server or telemetry exited; inspect their logs")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=2
            ) as response:
                if json.load(response).get("data"):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(5)
    raise TimeoutError("Server readiness exceeded 40 minutes")


def validate_result(path, output_len):
    data = json.loads(path.read_text())
    if not (
        data["completed"] == 160
        and data["input_lens"] == [131072] * 160
        and data["output_lens"] == [output_len] * 160
        and not any(data["errors"])
        and data["duration"] > 0
    ):
        raise RuntimeError(f"Incomplete requests or unexpected lengths: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--batches", type=int, nargs="+", default=[4096, 2048])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-len", type=int, choices=[1, 100], default=100)
    parser.add_argument(
        "--serve", choices=["native", "mega"], help="Start one server only"
    )
    parser.add_argument("--profile-dir", type=Path, help="Only with --serve")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or min(args.batches) < 128:
        parser.error("rounds must be positive and batch budgets must be >=128")
    if args.serve and len(args.batches) != 1:
        parser.error("--serve requires exactly one --batches value")
    if args.profile_dir and not args.serve:
        parser.error("Use --serve for profiling; throughput runs have no profiler")
    args.output = args.output.resolve()
    env = common_env()
    cases = []
    for repeat in range(args.rounds if not args.serve else 1):
        modes = ("native", "mega") if repeat % 2 == 0 else ("mega", "native")
        for batch in args.batches:
            for mode in (args.serve,) if args.serve else modes:
                directory = args.output / f"round{repeat + 1}-batch{batch}-{mode}"
                cases.append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "directory": str(directory),
                        "server": server_command(args, batch),
                        "client": client_command(args, directory),
                    }
                )
    plan = {
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith(
                ("ATOM_", "AITER_", "HIP_", "HSA_", "GPU_", "GLOO_", "OMP_", "PYTHON")
            )
        },
        "mode_environment": {
            "native": {"ATOM_SP_MOE_MEGA": "0"},
            "mega": {"ATOM_SP_MOE_MEGA": "1"},
        },
        "cases": cases,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if port_open(args.port):
        raise RuntimeError(f"Port {args.port} is already occupied")
    if args.serve:
        env["ATOM_SP_MOE_MEGA"] = "1" if args.serve == "mega" else "0"
        print(shlex.join(cases[0]["server"]), flush=True)
        os.chdir(ROOT)
        os.execvpe(sys.executable, cases[0]["server"], env)

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    packages = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    (args.output / "host.json").write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "packages": packages,
                "atom_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "atom_status": subprocess.check_output(
                    ["git", "status", "--short"], cwd=ROOT, text=True
                ),
            },
            indent=2,
        )
        + "\n"
    )
    # Validate the pinned dependency and checkpoint before consuming GPU time.
    with (args.output / "source-check.json").open("w") as output:
        subprocess.run(
            [sys.executable, str(HERE / "verify_source.py"), "--model", args.model],
            env=env,
            stdout=output,
            check=True,
        )
    with (args.output / "gpu-mapping.json").open("w") as output:
        subprocess.run(
            [sys.executable, str(HERE / "telemetry.py"), "--mapping"],
            env=env,
            stdout=output,
            check=True,
        )

    def record(event, **fields):
        line = json.dumps(dict(time=time.time(), event=event, **fields))
        print(line, flush=True)
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    for case in cases:
        directory = Path(case["directory"])
        directory.mkdir()
        env["ATOM_SP_MOE_MEGA"] = "1" if case["mode"] == "mega" else "0"
        telemetry = server = client = None
        with (
            (directory / "server.log").open("w") as server_log,
            (directory / "telemetry.log").open("w") as telemetry_log,
            (directory / "client.log").open("w") as client_log,
        ):
            try:
                telemetry = subprocess.Popen(
                    [
                        sys.executable,
                        str(HERE / "telemetry.py"),
                        "--output",
                        str(directory / "telemetry.jsonl"),
                    ],
                    env=env,
                    stdout=telemetry_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                server = subprocess.Popen(
                    case["server"],
                    cwd=ROOT,
                    env=env,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                record("server_start", directory=str(directory), pid=server.pid)
                wait_ready(server, telemetry, args.port)
                record("benchmark_start", directory=str(directory))
                client = subprocess.Popen(
                    case["client"],
                    cwd=ROOT,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = client.wait(timeout=14400)
                if code or telemetry.poll() is not None or server.poll() is not None:
                    raise RuntimeError(
                        "Client, telemetry or server failed; inspect logs"
                    )
                validate_result(directory / "result.json", args.output_len)
                record("benchmark_passed", directory=str(directory))
            finally:
                for process in (client, server, telemetry):
                    stop_group(process)
                record(
                    "stopped", directory=str(directory), port_open=port_open(args.port)
                )
            if port_open(args.port):
                raise RuntimeError("Server port still open after cleanup")
    record("complete", cases=len(cases))


if __name__ == "__main__":
    main()
