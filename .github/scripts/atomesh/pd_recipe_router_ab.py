"""Run the pinned PR 5090 recipe unchanged behind either router."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def worker_plan(contract, role, model_path, port, host, gpus, inherited):
    spec = contract["roles"][role]
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
    for endpoint, payload, stream in cases:
        payload.update(model=model, temperature=0, stream=stream)
        payload.setdefault("max_tokens", 16)
        response = request(router + endpoint, payload, stream)
        if not stream:
            assert response.get("choices") and "error" not in response, response
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


def benchmark(contract, args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
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
            "import importlib.metadata; print(importlib.metadata.version('agentx-harness'))",
        ],
        text=True,
    ).strip()
    assert version == contract["agentx_version"], version
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "client-freeze.txt").write_text(freeze)
    scrape(args.workers, output, "before-probe")
    functional_probe(contract, args.router, output / "functional.json")
    cancellation_probe(
        contract, args.router, args.workers, output / "cancellation.json"
    )
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
    # Preserve the recipe/client defaults for both arms; do not add a session header for ATOMesh.
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
        except Exception as error:
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
