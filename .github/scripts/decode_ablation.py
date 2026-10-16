"""Focused TP4/C1 decode study; synthetic prompts, not a published AgentX score."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import image_ab as base

PHASES = {
    "official37": ("official", None),
    "current38": ("current", None),
    "current37": ("current", "3.7.0+amd.rocm7.2.0.git89002410"),
    "current38_repeat": ("current", None),
}
LENGTHS = [65536, 262144, 524288]
OUTPUT = 4096
REPEATS = 6


def write(path, obj):
    base.write(path, obj)


def api(path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base.BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def make_prompts():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_PATH, trust_remote_code=True)
    # Public, deterministic text. Reuse the exact saved token IDs in every phase.
    paragraphs = [
        f"Record {i}: A distributed inference service processes a request in order. "
        f"The cache stores previous tokens. Worker {i % 4} checks the input, computes "
        f"attention, selects experts, and returns a result. The observed value is "
        f"{(i * 7919) % 100003}. Explain the behavior using the supplied records.\n"
        for i in range(12000)
    ]
    tokens = tokenizer.encode("".join(paragraphs), add_special_tokens=False)
    assert len(tokens) > max(LENGTHS)
    for length in LENGTHS:
        path = Path(f"/study/prompts/{length}.json")
        write(path, tokens[:length])
    write(
        "/study/prompts/manifest.json",
        {
            str(n): hashlib.sha256(
                Path(f"/study/prompts/{n}.json").read_bytes()
            ).hexdigest()
            for n in LENGTHS
        },
    )


def prepare():
    base.prepare()
    folder = base.root()
    shutil.copy2(__file__, folder / "code/decode_ablation.py")
    (folder / "prompts").mkdir()
    base.call(
        [
            "docker",
            "exec",
            base.container_name("client"),
            "/study/client-venv/bin/python",
            "/study/code/decode_ablation.py",
            "prompts",
        ]
    )
    write(
        folder / "decode-plan.json",
        {
            "phases": PHASES,
            "contexts": LENGTHS,
            "output_tokens": OUTPUT,
            "repeats": REPEATS,
            "client": "stdlib urllib SSE; frozen baseline Python",
            "server_args": base.SERVER_ARGS,
            "runner": os.environ.get("RUNNER_NAME"),
            "scope": "Focused synthetic prompt decode; no AgentX score or corpus claim",
            "profiles": "After unprofiled repetitions, 128 output tokens at 256K context",
            "control": "current37 changes only triton and triton-kernels in current image",
        },
    )


def serve(phase):
    image_label, triton_override = PHASES[phase]
    folder = Path("/study") / phase
    installed = json.loads(
        base.output([sys.executable, "-m", "pip", "list", "--format=json"])
    )
    before = {d["name"].lower().replace("_", "-"): d["version"] for d in installed}
    if triton_override:
        with (folder / "triton-install.log").open("w") as stream:
            base.call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-cache-dir",
                    "--index-url",
                    "https://pypi.amd.com/triton/release/rocm-7.2.0/simple/",
                    "triton==" + triton_override,
                    "triton-kernels==1.0.0+amd.rocm7.2.0.git89002410",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    after = json.loads(
        base.output([sys.executable, "-m", "pip", "list", "--format=json"])
    )
    after = {d["name"].lower().replace("_", "-"): d["version"] for d in after}
    changes = {
        k: [before.get(k), after.get(k)]
        for k in before.keys() | after.keys()
        if before.get(k) != after.get(k)
    }
    write(folder / "package-changes.json", changes)
    write(folder / "packages.json", after)
    if triton_override:
        assert set(changes) == {"triton", "triton-kernels"}, changes
        assert after["triton"] == triton_override
    else:
        assert not changes, changes
    source = Path(importlib.util.find_spec("atom").origin).resolve().parent.parent
    sha = base.output(
        [
            "git",
            "-c",
            f"safe.directory={source}",
            "-C",
            str(source),
            "rev-parse",
            "HEAD",
        ]
    ).strip()
    assert sha == base.SHAS[image_label], sha
    identities = {}
    for name in ["atom", "aiter"]:
        spec = importlib.util.find_spec(name)
        location = Path(spec.origin).resolve()
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={location.parent.parent}",
                "-C",
                str(location.parent.parent),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        identities[name] = {
            "source": str(location),
            "git_sha": result.stdout.strip(),
            "git_returncode": result.returncode,
        }
    write(
        folder / "software.json",
        {"phase": phase, "atom_sha": sha, "identities": identities, "packages": after},
    )
    shutil.copy2(Path(base.MODEL_PATH) / "config.json", folder / "model-config.json")
    for cache in ["/root/.cache/atom", "/root/.cache/triton", "/root/.triton/cache"]:
        if Path(cache).exists():
            shutil.rmtree(cache)
    args = base.SERVER_ARGS + ["--torch-profiler-dir", str(folder / "traces")]
    write(folder / "server-args.json", args)
    os.execv(
        sys.executable,
        [sys.executable, "-u", "-m", "atom.entrypoints.openai_server", *args],
    )


def one_request(tokens, output, request_key):
    before = api("/debug/mtp_stats")
    payload = {
        "model": base.MODEL,
        "prompt_token_ids": tokens,
        "temperature": 0,
        "max_tokens": output,
        "ignore_eos": True,
        "stream": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base.BASE_URL + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter_ns()
    chunks = []
    usage = None
    text_hash = hashlib.sha256()
    done = False
    with urllib.request.urlopen(req, timeout=900) as response:
        headers_at = time.perf_counter_ns()
        for line in response:
            now = time.perf_counter_ns()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("text", "")
                if content:
                    chunks.append(now)
                    text_hash.update(content.encode())
    end = time.perf_counter_ns()
    after = api("/debug/mtp_stats")
    assert done and usage and chunks, (done, usage, len(chunks))
    assert usage["prompt_tokens"] == len(tokens), usage
    assert usage["completion_tokens"] == output, usage
    dist0, dist1 = before["distribution"], after["distribution"]
    distribution = {
        k: dist1.get(k, 0) - dist0.get(k, 0) for k in dist0.keys() | dist1.keys()
    }
    steps = sum(distribution.values())
    assert steps > 0 and all(v >= 0 for v in distribution.values())
    decode_ns = chunks[-1] - chunks[0]
    row = {
        "key": request_key,
        "usage": usage,
        "mtp_before": before,
        "mtp_after": after,
        "mtp_steps": steps,
        "accepted_distribution": distribution,
        "mean_tokens_per_mtp_step": sum(
            (int(k) + 1) * v for k, v in distribution.items()
        )
        / steps,
        "ttft_ms": (chunks[0] - start) / 1e6,
        "decode_ms": decode_ns / 1e6,
        "itl_ms": decode_ns / 1e6 / (output - 1),
        "decode_ms_per_mtp_step": decode_ns / 1e6 / steps,
        "latency_ms": (end - start) / 1e6,
        "headers_ms": (headers_at - start) / 1e6,
        "content_chunks": len(chunks),
        "output_text_sha256": text_hash.hexdigest(),
        "chunk_offsets_ns": [n - start for n in chunks],
        "started_epoch": time.time() - (end - start) / 1e9,
    }
    return row


def client(phase):
    folder = Path("/study") / phase
    prompts = {
        n: json.loads(Path(f"/study/prompts/{n}.json").read_text()) for n in LENGTHS
    }
    with (folder / "requests.jsonl").open("x") as stream:

        def send(n, out, key):
            row = one_request(prompts[n], out, key)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(
                key,
                {k: row[k] for k in ("itl_ms", "mtp_steps", "decode_ms_per_mtp_step")},
                flush=True,
            )

        # Compile all live-context paths and populate the prefix cache before timing.
        for n in LENGTHS:
            for i in range(2):
                send(n, OUTPUT, f"warmup/{n}/{i}")
        # Balanced forward/reverse ordering of contexts through repetitions.
        for repeat in range(REPEATS):
            for n in LENGTHS if repeat % 2 == 0 else LENGTHS[::-1]:
                send(n, OUTPUT, f"measure/{n}/{repeat}")
        # Profiling is separate and never included in the latency estimates.
        write(folder / "profile-start.json", api("/start_profile", {}))
        try:
            send(262144, 128, "profile/262144/0")
        finally:
            write(folder / "profile-stop.json", api("/stop_profile", {}, timeout=900))
    write(
        folder / "completed.json",
        {
            "phase": phase,
            "completed_at": time.time(),
            "measured_requests": REPEATS * len(LENGTHS),
            "scope": "synthetic focused decode, not AgentX",
        },
    )


def run(phase):
    folder = base.root() / phase
    folder.mkdir(exist_ok=False)
    name = base.container_name("server")
    image_label, _ = PHASES[phase]
    base.docker_start(name, base.IMAGES[image_label], gpu=True)
    process = None
    try:
        base.wait_for_free_gpus(name, folder)
        with (folder / "server.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    name,
                    "python3",
                    "/study/code/decode_ablation.py",
                    "serve",
                    phase,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 2100
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server failed, exit={process.returncode}; inspect server.log"
                    )
                try:
                    models = api("/v1/models", timeout=3)
                    assert any(x["id"] == base.MODEL for x in models["data"]), models
                    write(folder / "models.json", models)
                    break
                except (OSError, ValueError):
                    time.sleep(5)
            else:
                raise RuntimeError("Server readiness exceeded 35 minutes")
            memory = base.gpu_memory(name)
            write(folder / "gpu-loaded.json", memory)
            assert all(v > 10 * 1024**3 for v in memory.values()), memory
            with (
                (folder / "client.log").open("w") as client_log,
                (folder / "gpu-samples.log").open("w") as gpu_log,
            ):
                replay = subprocess.Popen(
                    [
                        "docker",
                        "exec",
                        base.container_name("client"),
                        "/study/client-venv/bin/python",
                        "/study/code/decode_ablation.py",
                        "client",
                        phase,
                    ],
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 2700
                try:
                    while replay.poll() is None:
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(
                                "Server exited or focused client exceeded 45 minutes"
                            )
                        gpu_log.write(f"UTC epoch {time.time()}\n")
                        gpu_log.flush()
                        subprocess.run(
                            [
                                "docker",
                                "exec",
                                name,
                                "rocm-smi",
                                "--showuse",
                                "--showmemuse",
                                "--showpower",
                                "--showclocks",
                                "--showtemp",
                            ],
                            stdout=gpu_log,
                            stderr=subprocess.STDOUT,
                            timeout=20,
                            check=True,
                        )
                        time.sleep(15)
                    if replay.returncode:
                        raise RuntimeError(
                            f"Client failed, exit={replay.returncode}; inspect client.log"
                        )
                finally:
                    if replay.poll() is None:
                        subprocess.run(
                            [
                                "docker",
                                "stop",
                                "--time",
                                "10",
                                base.container_name("client"),
                            ],
                            check=False,
                        )
                        replay.wait(timeout=30)
            traces = list((folder / "traces").rglob("*.pt.trace.json.gz"))
            if len(traces) != 4:
                raise RuntimeError(
                    f"Expected a trace for each TP rank, found {len(traces)}"
                )
            print(
                f"{phase}: measured 18 requests and captured all four TP traces",
                flush=True,
            )
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False)
        if process:
            process.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=["prepare", "prompts", "serve", "run", "client", "cleanup"]
    )
    parser.add_argument("phase", nargs="?", choices=list(PHASES))
    args = parser.parse_args()
    if args.operation in {"serve", "run", "client"} and not args.phase:
        parser.error("This operation requires a phase")
    if args.operation == "cleanup":
        base.cleanup()
    elif args.operation == "prompts":
        make_prompts()
    else:
        operation = globals()[args.operation]
        operation(args.phase) if args.phase else operation()
