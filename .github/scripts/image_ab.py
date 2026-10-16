"""Run two image-native ATOM servers against one frozen AgentX client on one runner."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

IMAGES = {
    "official": "rocm/atom-dev:nightly_202610121516@sha256:a60811a8467b43e3e50e9e9eb8cec34f68b7bca03f8c6dbd6a0ff813dde49d4c",
    "current": "rocm/atom-dev:nightly_202610151507@sha256:aa6754256dd68009240e54206265e5250395374640f07757f2fdeac6170097cf",
}
SHAS = {
    "official": "a0527c9dbfb801e22cd29882454df6f6d21cdd91",
    "current": "e6182e08b85f0e58bc15f7393ef57899220bfbaf",
}
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
MODEL_PATH = "/models/" + MODEL
PORT = 18764
BASE_URL = f"http://127.0.0.1:{PORT}"
CAPTURES = "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,48,64,128]"
SERVER_ARGS = [
    "--model",
    MODEL_PATH,
    "--served-model-name",
    MODEL,
    "--trust-remote-code",
    "--host",
    "0.0.0.0",
    "--server-port",
    str(PORT),
    "-tp",
    "4",
    "--kv_cache_dtype",
    "bf16",
    "--index-cache-dtype",
    "fp4",
    "--gpu-memory-utilization",
    "0.9",
    "--max-num-seqs",
    "128",
    "--max-num-batched-tokens",
    "16384",
    "--attn-prefill-chunk-size",
    "16384",
    "--enable_prefix_caching",
    "--block-size",
    "16",
    "--state-checkpoint-interval-tokens",
    "8192",
    "--level",
    "3",
    "--cudagraph-mode",
    "FULL",
    "--cudagraph-capture-sizes",
    CAPTURES,
    "--method",
    "dspark",
    "--num-speculative-tokens",
    "5",
    "--spec-decode-acceptance-length",
    "3.51",
    "--tool-call-parser",
    "dsml_v41",
]


def call(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def output(args):
    return subprocess.check_output(args, text=True)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def container_name(kind):
    return "tp4-c1-ab-" + os.environ["GITHUB_RUN_ID"] + "-" + kind


def root():
    identity = os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    return Path(os.environ["GITHUB_WORKSPACE"]) / "tp4-c1-results" / identity


def docker_start(name, image, gpu=False):
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--network=host",
        "--ipc=host",
        "--entrypoint",
        "bash",
        "-v",
        f"{root()}:/study",
        "-v",
        "/models:/models:ro",
        "-w",
        "/tmp",
        "-e",
        "HF_TOKEN",
        "-e",
        "HF_HOME=/study/hf-cache",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
    ]
    if gpu:
        args += [
            "--device=/dev/kfd",
            "--device=/dev/dri",
            "--group-add",
            "video",
            "--security-opt",
            "seccomp=unconfined",
            "--cap-add=SYS_PTRACE",
            "-e",
            "HIP_VISIBLE_DEVICES=0,1,2,3",
            "-e",
            "OMP_NUM_THREADS=4",
            "-e",
            "ATOM_NUMA_BIND=0",
            "-e",
            "ATOM_DISABLE_MMAP=true",
            "-e",
            "AITER_LOG_LEVEL=WARNING",
            "-e",
            "NCCL_SOCKET_IFNAME=lo",
            "-e",
            "MORI_SOCKET_IFNAME=lo",
        ]
    args += [image, "-lc", "exec sleep infinity"]
    call(args)


def gpu_memory(name):
    doc = json.loads(
        output(["docker", "exec", name, "rocm-smi", "--showmeminfo", "vram", "--json"])
    )
    used = {}
    for key, values in doc.items():
        if key in {"card0", "card1", "card2", "card3"}:
            used[key] = int(values["VRAM Total Used Memory (B)"])
    if len(used) != 4:
        raise RuntimeError(f"Cannot verify four selected GPUs: {doc}")
    return used


def prepare():
    folder = root()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "code").mkdir()
    shutil.copy2(__file__, folder / "code/image_ab.py")
    write(
        folder / "plan.json",
        {
            "images": IMAGES,
            "atom_shas": SHAS,
            "client": "agentx-harness==1.0.10",
            "client_python": "3.11",
            "model": MODEL,
            "tp": 4,
            "concurrency": 1,
            "duration_seconds": 3600,
            "server_args": SERVER_ARGS,
            "gpu_ids": [0, 1, 2, 3],
            "runner": os.environ.get("RUNNER_NAME"),
            "direct_server_endpoint": True,
            "note": "Common client and local direct endpoint isolate image changes; SA used AToMesh.",
        },
    )
    for label, image in IMAGES.items():
        print(f"Pulling pinned {label} image", flush=True)
        call(["docker", "pull", image])
        write(
            folder / f"{label}-image.json",
            json.loads(
                output(["docker", "image", "inspect", "--format", "{{json .}}", image])
            )
            | {"Config": None, "ContainerConfig": None},
        )
    client = container_name("client")
    docker_start(client, IMAGES["official"])
    call(
        [
            "docker",
            "exec",
            client,
            "python3",
            "-c",
            "from pathlib import Path; p=Path('/models/deepseek-ai/DeepSeek-V4.1-Flash'); "
            "assert (p/'config.json').is_file(), 'Predownloaded model missing'; "
            "assert list(p.glob('*.safetensors')), 'Model weights missing'",
        ]
    )
    with (folder / "client-install.log").open("w") as stream:
        call(
            [
                "docker",
                "exec",
                client,
                "bash",
                "-lc",
                "set -euo pipefail\n"
                "command -v uv >/dev/null || python3 -m pip install uv==0.9.2\n"
                "uv venv --python 3.11 /study/client-venv\n"
                'uv pip install --python /study/client-venv/bin/python agentx-harness==1.0.10 "datasets>=4.7.0"\n'
                "/study/client-venv/bin/agentx --version\n"
                "uv pip freeze --python /study/client-venv/bin/python > /study/client-requirements.lock\n",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    print("One common AgentX 1.0.10 / Python 3.11 client is ready", flush=True)


def serve(label):
    # Fresh, task-owned container only. Do not touch any host/user cache or process.
    cache = Path("/root/.cache/atom")
    if cache.exists():
        shutil.rmtree(cache)
    source = Path(importlib.util.find_spec("atom").origin).resolve()
    repo = source.parent.parent
    sha = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    versions = {}
    for dist in ("atom", "torch", "triton", "amd-aiter"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    if sha.returncode != 0 or sha.stdout.strip() != SHAS[label]:
        raise RuntimeError(
            f"Image code mismatch: {source}, SHA={sha.stdout!r}, expected={SHAS[label]}"
        )
    write(
        f"/study/{label}/software.json",
        {"source": str(source), "sha": sha.stdout.strip(), "versions": versions},
    )
    config = Path(MODEL_PATH) / "config.json"
    shutil.copy2(config, f"/study/{label}/model-config.json")
    os.execv(
        sys.executable,
        [sys.executable, "-u", "-m", "atom.entrypoints.openai_server", *SERVER_ARGS],
    )


def run(label):
    folder = root() / label
    folder.mkdir(exist_ok=False)
    name = container_name("server")
    client = container_name("client")
    docker_start(name, IMAGES[label], gpu=True)
    process = None
    try:
        before = gpu_memory(name)
        write(folder / "gpu-before.json", before)
        if any(value > 1024**3 for value in before.values()):
            raise RuntimeError(
                f"Selected GPUs already occupied; refusing to interrupt existing work: {before}"
            )
        with (folder / "server.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    name,
                    "python3",
                    "/study/code/image_ab.py",
                    "serve",
                    label,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            ready_until = time.monotonic() + 1800
            while time.monotonic() < ready_until:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{label} server exited with {process.returncode}; inspect server.log"
                    )
                try:
                    with urllib.request.urlopen(
                        BASE_URL + "/v1/models", timeout=3
                    ) as response:
                        models = json.load(response)
                    if not any(item["id"] == MODEL for item in models["data"]):
                        raise RuntimeError(f"Unexpected served model: {models}")
                    write(folder / "models.json", models)
                    break
                except (OSError, ValueError):
                    time.sleep(5)
            else:
                raise RuntimeError("Server readiness exceeded 30 minutes")
            loaded = gpu_memory(name)
            write(folder / "gpu-loaded.json", loaded)
            if not all(value > 10 * 1024**3 for value in loaded.values()):
                raise RuntimeError(f"Model is not loaded on all four GPUs: {loaded}")
            print(
                f"{label}: model loaded on all four GPUs; starting 3600-second C1 replay",
                flush=True,
            )
            command = [
                "docker",
                "exec",
                "-e",
                "AIPERF_HTTP_TCP_USER_TIMEOUT=900000",
                "-e",
                "AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300",
                "-e",
                "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0",
                "-e",
                "AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800",
                "-e",
                "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800",
                client,
                "/study/client-venv/bin/agentx",
                "profile",
                "--url",
                BASE_URL,
                "--model",
                MODEL,
                "--tokenizer",
                MODEL_PATH,
                "--concurrency",
                "1",
                "--benchmark-duration",
                "3600",
                "--tokenizer-trust-remote-code",
                "--server-metrics",
                BASE_URL + "/metrics",
                "--output-artifact-dir",
                "/study/" + label + "/aiperf",
                "--public-dataset",
                "semianalysis_cc_traces_weka_062126",
            ]
            write(folder / "client-command.json", command)
            with (
                (folder / "client.log").open("w") as client_log,
                (folder / "gpu-samples.log").open("w") as gpu_log,
            ):
                replay = subprocess.Popen(
                    command, stdout=client_log, stderr=subprocess.STDOUT
                )
                deadline = time.monotonic() + 7200
                try:
                    while replay.poll() is None:
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError(
                                "Server exited or replay exceeded its two-hour limit"
                            )
                        gpu_log.write(f"UTC epoch {time.time()}\n")
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
                        )
                        gpu_log.flush()
                        time.sleep(30)
                    if replay.returncode:
                        raise RuntimeError(
                            f"AgentX returned {replay.returncode}; inspect client.log"
                        )
                finally:
                    if replay.poll() is None:
                        subprocess.run(
                            ["docker", "stop", "--time", "10", client], check=False
                        )
                        replay.wait(timeout=30)
            export = json.loads(
                (folder / "aiperf/profile_export_aiperf.json").read_text()
            )
            if not (folder / "aiperf/profile_export.jsonl").is_file():
                raise RuntimeError("Missing original request records")
            records = [
                json.loads(line)
                for line in (folder / "aiperf/profile_export.jsonl")
                .read_text()
                .splitlines()
            ]
            good = [
                r
                for r in records
                if r["metadata"]["benchmark_phase"] == "profiling"
                and not r.get("error")
                and not r["metadata"].get("was_cancelled")
                and r.get("metrics", {}).get("request_latency")
            ]
            if (
                not good
                or export.get("metadata", {}).get("submission_valid") is not True
            ):
                raise RuntimeError(
                    "Replay has no successful profiling records or is not submission-valid"
                )
            span = (
                max(r["metadata"]["request_end_ns"] for r in good)
                - min(r["metadata"]["request_start_ns"] for r in good)
            ) / 1e9
            if span < 3420:
                raise RuntimeError(f"Incomplete measurement span: {span:.3f} seconds")
            write(
                folder / "completed.json",
                {
                    "finished_at": time.time(),
                    "version": export["aiperf_version"],
                    "successful_profiling_requests": len(good),
                    "measured_span_seconds": span,
                    "request_count": export.get("request_count"),
                    "duration": export.get("benchmark_duration"),
                    "metadata": export.get("metadata"),
                },
            )
            print(f"{label}: completed, original records saved", flush=True)
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False)
        if process:
            process.wait(timeout=30)


def cleanup():
    for kind in ("server", "client"):
        subprocess.run(["docker", "rm", "-f", container_name(kind)], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "serve", "run", "cleanup"])
    parser.add_argument("label", nargs="?", choices=list(IMAGES))
    args = parser.parse_args()
    if args.operation in ("serve", "run") and not args.label:
        parser.error("serve/run require a label")
    if args.operation == "prepare":
        prepare()
    elif args.operation == "serve":
        serve(args.label)
    elif args.operation == "run":
        run(args.label)
    else:
        cleanup()
