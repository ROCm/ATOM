"""Same-runner base/head prefill benchmark and full GSM8K for PRs 3311/3312."""

import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import torch

ROOT = Path("/workspace")
OUT = ROOT / "review-evidence"
PR = os.environ["PR"]
HEAD = os.environ["HEAD_SHA"]
BASE = os.environ["BASE_SHA"]
K3 = "moonshotai/Kimi-K3.1-Pro"
GLM = "zai-org/GLM-5.2-FP8"
PY = sys.executable


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def run(argv, log, *, env=None, timeout=3600):
    print("RUN", argv, flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            argv,
            check=False,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    print("EXIT", result.returncode, log, flush=True)
    return result.returncode


def model_path(model):
    target = Path("/models") / model
    if Path("/models").is_dir():
        index = target / "model.safetensors.index.json"
        if (target / "config.json").is_file() and index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if shards and all(
                (target / shard).is_file() and (target / shard).stat().st_size > 0
                for shard in shards
            ):
                save(
                    OUT / (model.replace("/", "-") + "-cache.json"),
                    {
                        "path": str(target),
                        "shards_present": len(shards),
                        "refreshed": False,
                    },
                )
                return str(target)
        env = dict(
            os.environ,
            MODEL_ID=model,
            TARGET_DIR=str(target),
            MODEL_DOWNLOAD_TIMEOUT="30m",
            MODEL_LOCK_WAIT_SECONDS="1800",
            MODEL_STAGING_ROOT=str(target.parent / ".atom-model-staging"),
            MODEL_LOCK_ROOT=str(target.parent / ".atom-model-locks"),
            MODEL_ENABLE_REMOTE_REVISION_CHECK="0",
        )
        rc = run(
            ["bash", ".github/scripts/download_model_with_lock.sh", model, str(target)],
            OUT / (model.replace("/", "-") + "-download.log"),
            env=env,
        )
        if rc:
            raise RuntimeError(f"model download failed: {model}")
        return str(target)
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def stop_server(server):
    for sig, seconds in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        try:
            os.killpg(server.pid, sig)
        except ProcessLookupError:
            break
        try:
            server.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            pass
    time.sleep(2)


def profile_gsm8k(argv, folder, env):
    requests.post("http://127.0.0.1:8000/start_profile", timeout=120).raise_for_status()
    finished = threading.Event()
    errors = []
    server_log = folder / "server.log"
    offset = server_log.stat().st_size

    def stop_after_first_completions():
        try:
            deadline = time.monotonic() + 180
            while not finished.wait(1) and time.monotonic() < deadline:
                with server_log.open() as stream:
                    stream.seek(offset)
                    completed = stream.read().count("POST /v1/completions")
                if completed >= 8:
                    break
            response = requests.post("http://127.0.0.1:8000/stop_profile", timeout=300)
            response.raise_for_status()
            save(folder / "profiler-response.json", response.json())
        except (OSError, requests.RequestException) as exc:
            errors.append(repr(exc))

    monitor = threading.Thread(target=stop_after_first_completions)
    monitor.start()
    try:
        rc = run(argv, folder / "gsm8k.log", env=env, timeout=4500)
    finally:
        finished.set()
        monitor.join(timeout=310)
    if monitor.is_alive() or errors:
        raise RuntimeError(f"GSM8K profiler failed: {errors}")
    traces = list((folder / "profiler").rglob("*.pt.trace.json.gz"))
    rank0 = [path for path in traces if "rank_0" in path.parts]
    labels = []
    for path in rank0:
        with gzip.open(path, "rt") as stream:
            events = json.load(stream)["traceEvents"]
        labels.extend(
            event["name"]
            for event in events
            if event.get("name", "").startswith("prefill[")
        )
    sizes = [
        int(match[1]) for label in labels if (match := re.search(r"\btok=(\d+)", label))
    ]
    save(
        folder / "gsm8k-prefill-coverage.json",
        {"labels": labels, "scheduled_tokens": sizes},
    )
    if not sizes or max(sizes) < 8192:
        raise RuntimeError(f"GSM8K did not witness a >=8192-token prefill: {sizes}")
    # Keep the raw rank-0 witness; the other seven ranks duplicate the window.
    for path in traces:
        if path not in rank0:
            path.unlink()
    return rc


def scenario(sha, name, model, *, mono="0", payload="0"):
    folder = OUT / name
    folder.mkdir()
    subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=/workspace",
            "-c",
            "core.filemode=false",
            "checkout",
            "--detach",
            sha,
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "safe.directory=/workspace",
            "-c",
            "core.filemode=false",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "atom",
        ],
        cwd=ROOT,
        check=True,
    )
    cache = Path.home() / ".cache" / "atom"
    if cache.is_dir():
        shutil.rmtree(cache)
    env = dict(os.environ, ATOM_MONO_ENABLE=mono, ATOM_DCP_PREFILL_A2A_FP8=payload)
    argv = [
        PY,
        "-m",
        "atom.entrypoints.openai_server",
        "--model",
        model,
        "--trust-remote-code",
        "-tp",
        "8",
        "--decode-context-parallel-size",
        "8",
        "--kv_cache_dtype",
        "fp8",
        "--index-cache-dtype",
        "fp8",
        "--block-size",
        "128",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "16384",
        "--gpu-memory-utilization",
        "0.90",
        "--no-enable_prefix_caching",
        "--server-port",
        "8000",
        "--torch-profiler-dir",
        str(folder / "profiler"),
        "--mark-trace",
    ]
    metadata = {
        "sha": sha,
        "argv": argv,
        "mono": mono,
        "payload_fp8": payload,
        "accuracy_contract": "FP8 MLA KV, FP8 index, TP8/DCP8; paired main baseline (not the recipe's BF16 KV score)",
        "model_config_sha256": hashlib.sha256(
            (Path(model) / "config.json").read_bytes()
        ).hexdigest(),
        "model_revision": (
            (Path(model) / ".hf-revision").read_text().strip()
            if (Path(model) / ".hf-revision").is_file()
            else None
        ),
    }
    save(folder / "contract.json", metadata)
    with (folder / "server.log").open("w") as log:
        server = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 2700
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"{name}: server exited {server.returncode}")
                try:
                    if requests.get("http://127.0.0.1:8000/health", timeout=5).ok:
                        response = requests.post(
                            "http://127.0.0.1:8000/v1/completions",
                            json={
                                "model": model,
                                "prompt": "The answer is",
                                "max_tokens": 1,
                                "temperature": 0,
                            },
                            timeout=300,
                        )
                        response.raise_for_status()
                        save(folder / "smoke.json", response.json())
                        break
                except requests.RequestException:
                    pass
                time.sleep(10)
            else:
                raise RuntimeError(f"{name}: server readiness timed out")
            bench = [
                PY,
                "-m",
                "atom.benchmarks.benchmark_serving",
                "--model",
                model,
                "--backend",
                "vllm",
                "--base-url",
                "http://127.0.0.1:8000",
                "--dataset-name",
                "random",
                "--random-input-len",
                "16384",
                "--random-output-len",
                "128",
                "--random-range-ratio",
                "1.0",
                "--max-concurrency",
                "8",
                "--num-prompts",
                "16",
                "--num-warmups",
                "2",
                "--request-rate",
                "inf",
                "--ignore-eos",
                "--trust-remote-code",
                "--save-result",
                "--result-dir",
                str(folder),
                "--result-filename",
                "benchmark.json",
            ]
            metadata["benchmark_rc"] = run(
                bench, folder / "benchmark.log", env=env, timeout=1800
            )
            evaluate = [
                "lm_eval",
                "--model",
                "local-completions",
                "--model_args",
                f"model={model},base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=1,max_gen_toks=4096,tokenized_requests=False,trust_remote_code=True",
                "--tasks",
                "gsm8k",
                "--num_fewshot",
                "5",
                "--seed",
                "0,1234,1234,1234",
                "--log_samples",
                "--output_path",
                str(folder / "gsm8k"),
            ]
            metadata["eval_argv"] = evaluate
            metadata["eval_rc"] = profile_gsm8k(evaluate, folder, env)
            results = list((folder / "gsm8k").rglob("results*.json"))
            if len(results) != 1:
                raise RuntimeError(f"{name}: expected one GSM8K result, got {results}")
            data = json.loads(results[0].read_text())
            metadata["gsm8k"] = data["results"]["gsm8k"]
            metadata["n_samples"] = data["n-samples"]["gsm8k"]
            assert metadata["n_samples"]["effective"] == 1319, metadata["n_samples"]
            benchmark = json.loads((folder / "benchmark.json").read_text())
            assert benchmark["completed"] == benchmark["num_prompts"] == 16
            assert metadata["benchmark_rc"] == metadata["eval_rc"] == 0
            minimum = 0.96 if model == MODELS[K3] else 0.94
            assert metadata["gsm8k"]["exact_match,flexible-extract"] >= minimum
            assert metadata["gsm8k"]["exact_match,strict-match"] >= minimum
            metadata["passed"] = True
        except (
            AssertionError,
            OSError,
            RuntimeError,
            KeyError,
            subprocess.SubprocessError,
        ) as exc:
            metadata["error"] = repr(exc)
            metadata["passed"] = False
        finally:
            stop_server(server)
            metadata["server_returncode_after_cleanup"] = server.poll()
            save(folder / "result.json", metadata)
    print(name, json.dumps(metadata), flush=True)
    return metadata["passed"]


if __name__ == "__main__":
    assert PR in ("3311", "3312")
    assert all(re.fullmatch("[0-9a-f]{40}", sha) for sha in (HEAD, BASE))
    OUT.mkdir(exist_ok=True)
    assert torch.cuda.device_count() == 8
    assert (
        Path(importlib.util.find_spec("atom").origin).resolve()
        == ROOT / "atom/__init__.py"
    )
    aiter_source = Path(os.environ.get("AITER_SOURCE_PATH", "/app/aiter-test"))
    aiter_sha = subprocess.check_output(
        ["git", "-C", str(aiter_source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert aiter_sha == os.environ["AITER_GIT_REF"]
    assert (
        Path(importlib.util.find_spec("aiter").origin).resolve()
        == aiter_source / "aiter/__init__.py"
    )
    manifest = {
        "pr": PR,
        "head": HEAD,
        "base": BASE,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "aiter_ref": os.environ["AITER_GIT_REF"],
        "gpu": [torch.cuda.get_device_name(i) for i in range(8)],
        "packages": {
            **{p: importlib.metadata.version(p) for p in ("amd-aiter", "lm_eval")},
            "triton": __import__("triton").__version__,
        },
    }
    save(OUT / "manifest.json", manifest)
    MODELS = {K3: model_path(K3)}
    if PR == "3311":
        MODELS[GLM] = model_path(GLM)
    patterns = (
        [
            "tests/models/test_kimi_k3_indexer_overlap.py",
            "tests/model_ops/test_sparse_mla_work_plan.py",
            "tests/model_ops/test_sparse_mla_prefill_gpu.py",
            "tests/model_ops/test_mla_absorb_qpack.py",
            "tests/model_ops/test_indexer_packed_concat.py",
            "tests/test_dcp_merge_ops.py",
            "tests/test_dcp_a2a_fused_quant.py",
        ]
        if PR == "3311"
        else [
            "tests/test_kimi_k3_dspark_gqa.py",
            "tests/models/test_kimi_k3_attn_res_rows.py",
            "tests/models/test_kimi_k3_moe_row_tail.py",
            "tests/models/test_kimi_k3_kda_in_proj_split.py",
            "tests/models/test_kimi_k3_mono_zero_addends.py",
            "tests/model_ops/test_kimi_k3_attn_res_bank.py",
        ]
    )
    test_rc = run(
        [
            PY,
            "-m",
            "pytest",
            *patterns,
            "-q",
            "--junitxml=" + str(OUT / "gpu-tests.xml"),
        ],
        OUT / "gpu-tests.log",
        timeout=2700,
    )
    passed = [
        scenario(BASE, "base-k3", MODELS[K3]),
        scenario(HEAD, "head-k3", MODELS[K3]),
    ]
    if PR == "3311":
        passed.append(scenario(HEAD, "head-k3-payload-fp8", MODELS[K3], payload="1"))
        passed.append(scenario(BASE, "base-glm52", MODELS[GLM]))
        passed.append(scenario(HEAD, "head-glm52", MODELS[GLM]))
    else:
        passed.append(scenario(HEAD, "head-k3-mono", MODELS[K3], mono="1"))
    save(
        OUT / "summary.json",
        {
            "test_rc": test_rc,
            "scenarios": passed,
            "passed": test_rc == 0 and all(passed),
        },
    )
    sys.exit(0 if test_rc == 0 and all(passed) else 1)
