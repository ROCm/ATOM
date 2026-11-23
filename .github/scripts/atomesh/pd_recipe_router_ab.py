"""Run the pinned PR 5090 worker contract and router acceptance probes."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def worker_plan(contract, role, model_path, port, host, gpus, inherited):
    spec = contract["roles"][role]
    if "gpu_ids" in spec:
        gpus = ",".join(str(index) for index in spec["gpu_ids"])
    fabric = {
        "socket-ifname": inherited["MORI_SOCKET_IFNAME"],
        "rdma-devices": inherited["NCCL_IB_HCA"],
        "rdma-gid-index": inherited["NCCL_IB_GID_INDEX"],
        "rdma-tc": inherited["NCCL_IB_TC"],
    }
    env = dict(inherited)
    for key, value in (contract["environment"] | spec["env"]).items():
        value = str(value)
        env[key] = (
            fabric[value.removeprefix("@fabric.")]
            if value.startswith("@fabric.")
            else value
        )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = gpus
    env["VLLM_HOST_IP"] = host
    if role == "prefill":
        env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = "7200"
    else:
        env.pop("VLLM_MOONCAKE_BOOTSTRAP_PORT", None)
    argv = ["vllm", "serve", model_path, "--host", "0.0.0.0", "--port", str(port)]
    for key, value in spec["args"].items():
        if value is False or value is None:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is not True:
            argv.append(
                json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            )
    return argv, env


def request(url, payload=None, stream=False):
    headers = {"Content-Type": "application/json", "X-Session-ID": "router-ab-probe"}
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=180
    ) as response:
        if not stream:
            return json.loads(response.read())
        chunks = []
        done = False
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
        assert chunks and done, "incomplete SSE response"
        assert all("error" not in chunk for chunk in chunks), chunks
        return chunks


def functional_probe(contract, router, output):
    model = contract["roles"]["decode"]["args"]["served-model-name"]
    result = {"models": request(router + "/v1/models"), "responses": []}
    assert any(item["id"] == model for item in result["models"]["data"]), result[
        "models"
    ]
    cases = [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Reply with a short greeting."}]},
            False,
        ),
        ("/v1/completions", {"prompt": "A short greeting:"}, False),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Reply with a short greeting."}]},
            True,
        ),
        (
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Use echo to say hello."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "Echo a word",
                            "parameters": {
                                "type": "object",
                                "properties": {"word": {"type": "string"}},
                                "required": ["word"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 128,
            },
            False,
        ),
    ]
    if contract.get("functional_only"):
        cases.append(
            (
                "/v1/completions",
                {"prompt": "Transfer this context: " + "context block " * 512},
                False,
            )
        )
    for endpoint, payload, stream in cases:
        payload.update(model=model, temperature=0, stream=stream)
        payload.setdefault("max_tokens", 16)
        response = request(router + endpoint, payload, stream)
        if not stream:
            assert response.get("choices") and "error" not in response, response
            if payload.get("prompt", "").startswith("Transfer this context:"):
                assert response["usage"]["prompt_tokens"] > 128, response["usage"]
        result["responses"].append(
            {"endpoint": endpoint, "stream": stream, "response": response}
        )
        output.write_text(json.dumps(result, indent=2) + "\n")


def cancellation_probe(contract, router, workers, output):
    model = contract["roles"]["decode"]["args"]["served-model-name"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count upwards one number per line."}],
        "max_tokens": 4096,
        "stream": True,
    }
    query = urllib.request.Request(
        router + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(query, timeout=180) as response:
        first = response.readline().decode()
    time.sleep(5)
    deadline = time.monotonic() + 30
    values = []
    while time.monotonic() < deadline:
        values = []
        for url in workers:
            with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
                metrics = response.read().decode()
            gauges = re.findall(
                r"^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.]+)",
                metrics,
                re.MULTILINE,
            )
            assert len(gauges) >= 2, f"Missing request gauges at {url}"
            values.extend(float(value) for value in gauges)
        if all(value == 0 for value in values):
            break
        time.sleep(1)
    record = {"first_sse_line": first, "running_and_waiting_after_disconnect": values}
    output.write_text(json.dumps(record, indent=2) + "\n")
    assert values and all(value == 0 for value in values), record


def scrape(urls, output, tag):
    for index, url in enumerate(urls):
        with urllib.request.urlopen(url + "/metrics", timeout=30) as response:
            (output / f"{tag}-worker-{index}.prom").write_bytes(response.read())


def transferred_bytes(metrics):
    values = re.findall(
        r"^vllm:mooncake_bytes_transferred_sum(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",
        metrics,
        re.MULTILINE,
    )
    if not values:
        raise ValueError("Missing Mooncake transferred-byte counter on prefill")
    return sum(float(value) for value in values)


def verify_kv_transfer(workers, output):
    before = transferred_bytes((output / "before-probe-worker-0.prom").read_text())
    delta = 0.0
    for _ in range(15):
        scrape(workers, output, "after-transfer")
        after = transferred_bytes((output / "after-transfer-worker-0.prom").read_text())
        delta = after - before
        if delta > 0:
            break
        time.sleep(1)
    record = {
        "metric": "vllm:mooncake_bytes_transferred_sum",
        "prefill_before": before,
        "prefill_after": after,
        "transferred_bytes_delta": delta,
        "passed": delta > 0,
    }
    (output / "kv-transfer.json").write_text(json.dumps(record, indent=2) + "\n")
    assert delta > 0, "No completed Mooncake byte transfer during functional probes"
    return record


def benchmark(contract, args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if contract.get("functional_only") and shutil.which("rocm-smi"):
        with (output / "gpu-memory.log").open("w") as memory_log:
            subprocess.run(
                ["rocm-smi", "--showmemuse"],
                stdout=memory_log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
            )
    scrape(args.workers, output, "before-probe")
    functional_probe(contract, args.router, output / "functional.json")
    transfer = (
        verify_kv_transfer(args.workers, output)
        if contract.get("functional_only")
        else None
    )
    cancellation_probe(
        contract, args.router, args.workers, output / "cancellation.json"
    )
    scrape(args.workers, output, "after-probe")
    if contract.get("functional_only"):
        (output / "functional-status.json").write_text(
            json.dumps(
                {
                    "http_probes": "passed",
                    "request_gauges_after_disconnect": "zero",
                    "performance_run": False,
                    "kv_transfer_evidence": transfer,
                },
                indent=2,
            )
            + "\n"
        )
        (output.parent / "perf-analysis.md").write_text(
            "# Functional run\n\nProtocol probes, SSE and request gauges after disconnect passed. "
            "Completed Mooncake transfer bytes increased; see functional/kv-transfer.json and the raw worker metrics. "
            "GPU memory evidence: functional/gpu-memory.log when rocm-smi is available. "
            "This run used srtctl orchestration via remote run-pd-ci. "
            "TTFT p50/p95/p99, ITL p50/p99, output throughput and AgentX Pareto/SLO are unavailable: "
            "no performance window was requested. Accuracy evaluation was deferred. "
            "Slurm and container cleanup are verified separately after this client exits.\n"
        )
        return 0
    shared = Path(os.environ["RUN_DIR"]) / "client-venv"
    python = shared / "bin/python"
    if not python.exists():
        created = subprocess.run(
            [sys.executable, "-m", "venv", str(shared)], check=False
        )
        if created.returncode:
            subprocess.run(
                [
                    "uv",
                    "venv",
                    "--seed",
                    "--clear",
                    "--python",
                    sys.executable,
                    str(shared),
                ],
                check=True,
            )
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"agentx-harness=={contract['agentx_version']}",
                "datasets>=4.7.0",
            ],
            check=True,
        )
    version = subprocess.check_output(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata; "
                "print(importlib.metadata.version('agentx-harness'))"
            ),
        ],
        text=True,
    ).strip()
    assert version == contract["agentx_version"], version
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "client-freeze.txt").write_text(freeze)
    scrape(args.workers, output, "before-benchmark")
    model = contract["roles"]["decode"]["args"]["served-model-name"]
    argv = [
        str(shared / "bin/agentx"),
        "profile",
        "--url",
        args.router,
        "--model",
        model,
        "--tokenizer",
        args.model_path,
        "--concurrency",
        str(contract["concurrency"]),
        "--benchmark-duration",
        str(contract["duration"]),
        "--tokenizer-trust-remote-code",
        "--server-metrics",
        *[url + "/metrics" for url in args.workers],
        "--output-artifact-dir",
        str(output / "aiperf_artifacts"),
        "--public-dataset",
        contract["dataset"],
    ]
    env = dict(os.environ) | contract["benchmark_env"]
    # Preserve the recipe/client defaults in both arms, including session headers.
    env.pop("AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID", None)
    (output / "benchmark-command.json").write_text(
        json.dumps(
            {
                "argv": argv,
                "recipe_env": contract["benchmark_env"],
                "new_session_header": False,
            },
            indent=2,
        )
    )
    result = subprocess.run(argv, env=env, check=False)
    scrape(args.workers, output, "final")
    for name, url in (("workers", args.router + "/workers"),):
        try:
            (output / f"router-{name}.json").write_text(
                json.dumps(request(url), indent=2)
            )
        except (OSError, ValueError) as error:
            (output / f"router-{name}.error.txt").write_text(str(error))
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "benchmark"])
    parser.add_argument("--contract", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=["prefill", "decode"])
    parser.add_argument("--port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--gpus")
    parser.add_argument("--router")
    parser.add_argument("--workers", nargs="+")
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text())
    if args.mode == "benchmark":
        return benchmark(contract, args)
    argv, env = worker_plan(
        contract,
        args.role,
        args.model_path,
        args.port,
        args.host,
        args.gpus,
        os.environ,
    )
    declared = {
        key: env[key]
        for key in contract["environment"] | contract["roles"][args.role]["env"]
    }
    for key in ("HIP_VISIBLE_DEVICES", "VLLM_HOST_IP", "VLLM_MOONCAKE_BOOTSTRAP_PORT"):
        if key in env:
            declared[key] = env[key]
    import resource

    limits = {
        "memlock_soft_hard_bytes": list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    }
    Path(args.output).write_text(
        json.dumps(
            {
                "argv": argv,
                "env": declared,
                "source": contract["source"],
                "limits": limits,
            },
            indent=2,
        )
    )
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    raise SystemExit(main())
