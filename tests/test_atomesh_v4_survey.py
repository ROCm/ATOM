"""CPU-only public-contract tests for the frozen V4 survey harness."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_matrix_and_rendered_launch(tmp_path, monkeypatch):
    for key, value in {
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": str(tmp_path),
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_MODEL_ROOT": "/share_nfs/models",
        "ATOMESH_1P1D_NODES": "pit2-p03-g13,pit2-p03-g42",
        "ATOMESH_NODE_POOL": "pit2-p03-g13,pit2-p03-g42",
    }.items():
        monkeypatch.setenv(key, value)
    matrix = load_script("pd_matrix")
    cells = matrix.build_cells(
        matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml"),
        suite="vllm",
        model_filter={"DeepSeek-V4-Flash-vLLM-Survey"},
        case_filter=None,
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["num_nodes"] == 2
    assert cell["model_path"] == "/share_nfs/models/deepseek-ai/DeepSeek-V4-Flash"
    assert cell["isl"] == [255, 257, 513]
    assert cell["vllm"]["source"]["sha"] == "b22494cc0cb4bd9db4a62fb107d92429a4a3249d"
    identity = json.loads((SCRIPTS / cell["vllm"]["checkpoint_manifest"]).read_text())
    assert (
        identity["config_sha256"]
        == "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"
    )
    assert (
        identity["index_sha256"]
        == "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
    )
    inputs = json.loads(
        (ROOT / ".github/benchmark/rocm-pd-survey-v4-inputs.json").read_text()
    )
    assert inputs["case_names"] == cell["name"]
    assert inputs["publish_dashboard"] == "false"
    submit = (SCRIPTS / "pd_submit.sh").read_text()
    exports = submit.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    exported = subprocess.check_output(
        [sys.executable, "-c", exports],
        text=True,
        env={**os.environ, "CELL_JSON": json.dumps(cell)},
    )
    definitions = (
        (SCRIPTS / "pd_server_vllm.sh")
        .read_text()
        .split('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then')[0]
    )
    shell = (
        exported
        + "\n"
        + "\n".join(
            [
                "set -euo pipefail",
                f"export PATH={shlex.quote(str(Path(sys.executable).parent))}:$PATH",
                "host_ip=192.0.2.1; host_name=fixture; NODE0_ADDR=192.0.2.1; NODE_RANK=0",
                "HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
                "ATOMESH_SERVICE_PORT_OFFSET=0; ATOMESH_EXECUTION_PHASE=benchmark",
                f"ATOMESH_SCRIPT_DIR={shlex.quote(str(SCRIPTS))}",
                f"RUNTIME_LOG_DIR={shlex.quote(str(tmp_path))}; RUN_DIR=$RUNTIME_LOG_DIR",
                "PREFILL_TP_SIZE=$PREFILL_TP; DECODE_TP_SIZE=$DECODE_TP",
                'PREFILL_SERVER_ARGS="$EXTRA_SERVER_ARGS $PREFILL_EXTRA_SERVER_ARGS"',
                'DECODE_SERVER_ARGS="$EXTRA_SERVER_ARGS $DECODE_EXTRA_SERVER_ARGS"',
                "apply_role_env() { :; }; build_server_cache_env() { :; }",
                "dump_launch_info() { :; }; start_logged_process() { :; }",
                definitions,
                "start_vllm_server prefill prefill 2584",
                "start_vllm_server decode decode 2584",
            ]
        )
    )
    subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True)
    for role in ("prefill", "decode"):
        argv = json.loads((tmp_path / f"{role}.launch.json").read_text())["argv"]
        for flag, value in {
            "--tensor-parallel-size": "8",
            "--block-size": "256",
            "--dtype": "bfloat16",
            "--kv-cache-dtype": "fp8",
            "--max-num-batched-tokens": "256",
            "--kv-cache-memory-bytes": "4294967296",
            "--max-model-len": "2048",
        }.items():
            assert argv[argv.index(flag) + 1] == value
        assert "--attention_config.indexer_kv_dtype=fp8" in argv
        assert "--no-enable-prefix-caching" in argv
        assert "--decode-context-parallel-size" not in argv
        assert not any(x.startswith(("--quantization", "--hf-overrides")) for x in argv)
        transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        assert transfer["kv_connector"] == "NixlConnector"
        assert transfer["kv_load_failure_policy"] == "fail"
        assert transfer["kv_role"] == (
            "kv_producer" if role == "prefill" else "kv_consumer"
        )


@pytest.mark.parametrize("fault", [None, "ids_missing", "ids_mismatch"])
def test_v4_requests_preserve_handoff_and_require_exact_ids(
    tmp_path, monkeypatch, fault
):
    smoke = load_script("pd_m3_nixl_smoke")
    counters = {
        role: {
            "local_compute": 0,
            "local_cache_hit": 0,
            "external_kv_transfer": 0,
            "success": 0,
            "bytes": 0,
            "count": 0,
        }
        for role in ("prefill", "decode")
    }
    calls = []
    handoff = {
        "do_remote_prefill": True,
        "remote_engine_id": "p-engine",
        "remote_host": "prefill",
        "remote_port": 15559,
        "remote_block_ids": [[1, 2], [3]],
        "future_field": {"preserve": [5]},
    }

    def respond(request):
        role = request.url.host
        c = counters[role]
        if request.url.path == "/metrics":
            text = "\n".join(
                f'vllm:prompt_tokens_by_source_total{{source="{k}"}} {c[k]}'
                for k in smoke.SOURCES
            )
            text += f'\nvllm:request_success_total {c["success"]}'
            text += f'\nvllm:nixl_bytes_transferred_sum {c["bytes"]}'
            text += f'\nvllm:nixl_bytes_transferred_count {c["count"]}'
            text += "".join(f"\n{name} 0" for name in smoke.FAILURES)
            return httpx.Response(200, text=text)
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(1024))})
        body = json.loads(request.content)
        calls.append((role, body))
        assert body["return_token_ids"] is True
        n = len(body["prompt"])
        c["success"] += 1
        c["external_kv_transfer" if role == "decode" else "local_compute"] += n
        c["bytes"] += 4096 if role == "decode" else 0
        c["count"] += int(role == "decode")
        ids = [7] * body["max_tokens"]
        if role == "decode" and fault == "ids_mismatch":
            ids[-1] = 8
        result = {
            "choices": [
                {
                    "text": "identical even on ID mismatch",
                    "finish_reason": "length",
                    "token_ids": ids,
                    "prompt_token_ids": body["prompt"],
                }
            ],
            "usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]},
        }
        if role == "decode":
            assert body["kv_transfer_params"] == handoff
            if fault == "ids_missing":
                result["choices"][0].pop("token_ids")
        elif "kv_transfer_params" in body:
            result["kv_transfer_params"] = handoff
        return httpx.Response(200, json=result)

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        smoke.httpx,
        "AsyncClient",
        lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw),
    )
    args = argparse.Namespace(
        prefill="http://prefill",
        decode="http://decode",
        model="DeepSeek-V4-Flash",
        output=tmp_path,
        model_profile="v4",
    )
    if fault:
        with pytest.raises(AssertionError):
            asyncio.run(smoke.run(args))
        assert not (tmp_path / "complete.json").exists()
    else:
        asyncio.run(smoke.run(args))
        assert [len(body["prompt"]) for role, body in calls if role == "decode"] == [
            255,
            257,
            513,
        ]
        complete = json.loads((tmp_path / "complete.json").read_text())
        assert complete["lengths"] == [255, 257, 513]
        assert complete["status"] == "PENDING_REVIEW"
        for n in complete["lengths"]:
            assert (
                json.loads((tmp_path / f"request-{n}.json").read_text())[
                    "generated_token_ids_equal"
                ]
                is True
            )


def test_checkpoint_preflight_manifest_cli(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = b'{"architectures":["DeepseekV4ForCausalLM"],"expert_dtype":"fp4"}'
    index = b'{"weight_map":{"weight":"model-1.safetensors"}}'
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index)
    (model / "model-1.safetensors").write_bytes(b"12345678weights")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "tokenizer.json").write_bytes(b"12345678")
    manifest = tmp_path / "identity.json"
    expected = {
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "shards": {"model-1.safetensors": 15},
    }
    manifest.write_text(json.dumps(expected))
    output = tmp_path / "result.json"
    command = [
        sys.executable,
        str(SCRIPTS / "pd_survey_preflight.py"),
        str(model),
        str(output),
        "--manifest",
        str(manifest),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(output.read_text())
    assert report["status"] == "FILES_VISIBLE"
    assert report["checkpoint_identity"] == "STRUCTURE_MATCH_NOT_FULL_WEIGHT_HASH"
    (model / "config.json").write_bytes(config + b" ")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())["status"] == "BLOCKED_ENV"
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index + b" ")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "index_sha256 mismatch" in json.loads(output.read_text())["error"]
    (model / "model.safetensors.index.json").write_bytes(index)
    (model / "model-1.safetensors").write_bytes(b"different weight size")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "shard names/sizes mismatch" in json.loads(output.read_text())["error"]
    (model / "model-1.safetensors").unlink()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    (model / "model-1.safetensors").write_bytes(b"12345678weights")
    manifest.unlink()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())["status"] == "BLOCKED_ENV"


@pytest.mark.parametrize("manifest", ["v4-checkpoint-identity.json", ""])
def test_runner_and_each_node_block_before_submission_or_build(tmp_path, manifest):
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
    )
    steps = next(
        job["steps"]
        for job in workflow["jobs"].values()
        if any(
            step.get("name") == "Survey pre-submit visibility and queue check"
            for step in job.get("steps", [])
        )
    )
    gate = next(
        step
        for step in steps
        if step.get("name") == "Survey pre-submit visibility and queue check"
    )
    submit = next(
        step for step in steps if step.get("name") == "Run Slurm benchmark cell"
    )
    assert steps.index(gate) < steps.index(submit)
    code = gate["run"].split("python3 - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    (tmp_path / "survey-preflight").mkdir()
    (tmp_path / ".github/scripts").mkdir(parents=True)
    (tmp_path / ".github/scripts/atomesh").symlink_to(SCRIPTS, target_is_directory=True)
    cell = {
        "model_path": str(tmp_path / "missing-checkpoint"),
        "vllm": {"nixl_model_profile": "v4", "checkpoint_manifest": manifest},
    }
    env = {
        **os.environ,
        "CELL_JSON": json.dumps(cell),
        "ATOMESH_MODEL_ROOT": str(tmp_path),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    if manifest:
        report = json.loads((tmp_path / "survey-preflight/weights.json").read_text())
        assert report["status"] == "BLOCKED_ENV"
    definitions = (
        (SCRIPTS / "pd_server_vllm.sh")
        .read_text()
        .split('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then')[0]
    )
    # Exercise actual per-node install gate, intercept only forbidden build/network boundary.
    for rank in (0, 1):
        env.update(
            ATOMESH_VLLM_SOURCE_REPO="https://github.com/vllm-project/vllm",
            ATOMESH_VLLM_SOURCE_SHA="b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
            ATOMESH_VLLM_CLEAN_MAIN="1",
            ATOMESH_VLLM_NIXL_MODEL_PROFILE="v4",
            ATOMESH_VLLM_CHECKPOINT_MANIFEST=manifest,
            ATOMESH_VLLM_CONNECTOR="nixl",
            ATOMESH_VLLM_ROUTER_DISCOVERY_PORT="6300",
            ATOMESH_SERVICE_PORT_OFFSET="0",
            ATOMESH_SCRIPT_DIR=str(SCRIPTS),
            MODEL_PATH=cell["model_path"],
            RUNTIME_LOG_DIR=str(tmp_path),
            NODE_RANK=str(rank),
            host_ip="192.0.2.1",
        )
        shell = (
            "git() { printf forbidden-build; return 99; };\n"
            + definitions
            + "\ninstall_native_vllm"
        )
        result = subprocess.run(
            ["bash", "-c", shell], check=False, env=env, capture_output=True, text=True
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "forbidden-build" not in result.stdout
        if manifest:
            assert (
                json.loads(
                    (tmp_path / f"weights-preflight-rank-{rank}.json").read_text()
                )["status"]
                == "BLOCKED_ENV"
            )
