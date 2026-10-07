"""CPU-only checks for the survey DSpark draft visibility gate."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def checkpoint(path, *, draft=False, indexed=False):
    path.mkdir()
    config = {
        "architectures": ["KimiK3DSparkForCausalLM" if draft else "KimiK3ForCausalLM"],
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["lm_head", "layers.0.attn"],
            "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    weight = len(header).to_bytes(8, "little") + header + b"12345678"
    name = "model-1.safetensors" if indexed else "model.safetensors"
    (path / name).write_bytes(weight)
    if indexed:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": name}})
        )
    if not draft:
        (path / "tokenizer_config.json").write_text("{}")
        (path / "tokenizer.json").write_text("tokenizer bytes")
    return config


def cli(tmp_path, target, draft=None, *extra):
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        str(SCRIPTS / "pd_survey_preflight.py"),
        str(target),
        str(output),
    ]
    if draft is not None:
        command += ["--draft-model", str(draft)]
    result = subprocess.run(
        command + list(extra), check=False, capture_output=True, text=True
    )
    return result, json.loads(output.read_text()) if output.exists() else {}


@pytest.mark.parametrize("indexed", [False, True])
def test_draft_reuses_target_tokenizer_and_reports_original_metadata(tmp_path, indexed):
    target, draft = tmp_path / "target", tmp_path / "draft"
    target_config = checkpoint(target)
    draft_config = checkpoint(draft, draft=True, indexed=indexed)
    result, report = cli(tmp_path, target, draft)
    assert result.returncode == 0, result.stderr + result.stdout
    assert report["status"] == "FILES_VISIBLE"
    assert report["quantization_config"] == target_config["quantization_config"]
    evidence = report["draft"]
    assert evidence["status"] == "FILES_VISIBLE"
    assert evidence["tokenizer"] == "TARGET_TOKENIZER_REUSED"
    assert evidence["architectures"] == draft_config["architectures"]
    assert evidence["quantization_config"] == draft_config["quantization_config"]
    assert (
        evidence["config_sha256"]
        == hashlib.sha256((draft / "config.json").read_bytes()).hexdigest()
    )
    assert evidence["weights"] == {
        p.name: p.stat().st_size for p in draft.glob("*.safetensors")
    }
    assert (
        evidence["checkpoint_identity"] == "STRUCTURE_INSPECTED_NOT_IDENTITY_VERIFIED"
    )
    if indexed:
        assert (
            evidence["index_sha256"]
            == hashlib.sha256(
                (draft / "model.safetensors.index.json").read_bytes()
            ).hexdigest()
        )
    else:
        assert "index_sha256" not in evidence
    assert not list(draft.glob("tokenizer*"))


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "config_json",
        "config_type",
        "architecture",
        "empty_index",
        "index_json",
        "missing_shard",
        "empty_shard",
        "corrupt_shard",
        "truncated_shard",
    ],
)
def test_invalid_draft_blocks_valid_target(tmp_path, damage):
    target, draft = tmp_path / "target", tmp_path / "draft"
    checkpoint(target)
    checkpoint(draft, draft=True, indexed=True)
    if damage == "missing":
        draft = tmp_path / "missing"
    elif damage == "config_json":
        (draft / "config.json").write_text("{")
    elif damage == "config_type":
        (draft / "config.json").write_text("[]")
    elif damage == "architecture":
        (draft / "config.json").write_text('{"architectures":[]}')
    elif damage == "empty_index":
        (draft / "model.safetensors.index.json").write_text('{"weight_map":{}}')
    elif damage == "index_json":
        (draft / "model.safetensors.index.json").write_text("{")
    elif damage == "missing_shard":
        (draft / "model-1.safetensors").unlink()
    else:
        shard = draft / "model-1.safetensors"
        shard.write_bytes(
            {
                "empty_shard": b"",
                "corrupt_shard": b"not safetensors",
                "truncated_shard": shard.read_bytes()[:-1],
            }[damage]
        )
    result, report = cli(tmp_path, target, draft)
    assert result.returncode == 2
    assert report["status"] == report["draft"]["status"] == "BLOCKED_ENV"
    assert report["draft"]["error"]
    assert report["tokenizer_files"] == ["tokenizer.json"]


def test_target_only_contract_and_manifest_still_apply_with_draft(tmp_path):
    import importlib.util
    import inspect

    spec = importlib.util.spec_from_file_location(
        "draft_preflight", SCRIPTS / "pd_survey_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        str(inspect.signature(module.check_weights))
        == "(model, model_root=None, manifest=None)"
    )
    target, draft = tmp_path / "target", tmp_path / "draft"
    checkpoint(target, indexed=True)
    checkpoint(draft, draft=True)
    result, report = cli(tmp_path, target)
    assert result.returncode == 0
    assert report == module.check_weights(target)
    manifest = tmp_path / "manifest.json"
    expected = {
        "config_sha256": hashlib.sha256(
            (target / "config.json").read_bytes()
        ).hexdigest(),
        "index_sha256": hashlib.sha256(
            (target / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "shards": {
            "model-1.safetensors": (target / "model-1.safetensors").stat().st_size
        },
    }
    manifest.write_text(json.dumps(expected))
    result, report = cli(tmp_path, target, draft, "--manifest", str(manifest))
    assert result.returncode == 0
    assert report["checkpoint_identity"] == "STRUCTURE_MATCH_NOT_FULL_WEIGHT_HASH"
    (target / "config.json").write_text((target / "config.json").read_text() + " ")
    result, report = cli(tmp_path, target, draft, "--manifest", str(manifest))
    assert result.returncode == 2
    assert "config_sha256 mismatch" in report["error"]
    assert report["draft"]["status"] == "FILES_VISIBLE"
    (target / "tokenizer_config.json").unlink()
    result, report = cli(tmp_path, target, draft)
    assert result.returncode == 2
    assert "Missing tokenizer configuration" in report["error"]


def test_unmounted_target_root_stays_unknown_with_missing_draft(tmp_path):
    result, report = cli(
        tmp_path,
        tmp_path / "target",
        tmp_path / "draft",
        "--model-root",
        str(tmp_path / "unmounted"),
    )
    assert result.returncode == 2
    assert report["status"] == "UNKNOWN"
    assert report["draft"]["status"] == "BLOCKED_ENV"


def survey_cells():
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location(
        "draft_matrix", SCRIPTS / "pd_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    config = yaml.safe_load(
        (ROOT / ".github/benchmark/models_atomesh.yaml").read_text()
    )
    return matrix, config


def workflow_steps():
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/atomesh-benchmark.yaml").read_text()
    )
    return next(
        job["steps"]
        for job in workflow["jobs"].values()
        if any(step.get("id") == "machine-paths" for step in job.get("steps", []))
    )


def python_step(step):
    return step["run"].split("python3 - <<'PY'\n", 1)[1].split("\nPY", 1)[0]


def resolve_machine_paths(cell, monkeypatch, hostname="other-runner"):
    import socket
    import tempfile

    monkeypatch.setenv("CELL_JSON", json.dumps(cell))
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "output"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        exec(  # noqa: S102 - execute the checked-in workflow's CPU path resolver
            python_step(
                next(
                    step
                    for step in workflow_steps()
                    if step.get("id") == "machine-paths"
                )
            ),
            {},
        )
        return json.loads(output.read_text().split("=", 1)[1])


def test_only_survey_dspark_cell_gates_actual_speculative_paths(monkeypatch):
    import re
    import shlex

    for name in set(
        re.findall(
            r"\$\{(ATOMESH_\w+)\}",
            (ROOT / ".github/benchmark/models_atomesh.yaml").read_text(),
        )
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("ATOMESH_SLURM_SUBMIT_RUNNER", "atomesh-cicd")
    monkeypatch.setenv("ATOMESH_MODEL_ROOT", "/models with spaces")
    monkeypatch.setenv("ATOMESH_1P1D_NODES", "node-a,node-b")
    matrix, config = survey_cells()
    cells = matrix.build_cells(
        config,
        suite="vllm",
        model_filter=None,
        case_filter=None,
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    gated = [cell for cell in cells if cell.get("vllm", {}).get("draft_model_path")]
    assert len(gated) == 1
    cell = gated[0]
    cell = resolve_machine_paths(cell, monkeypatch)
    assert cell["name"] == "survey-k3-main-read-dspark3-1p1d-tp8-dcp8-eager"
    path = cell["vllm"]["draft_model_path"]
    assert path == "/models with spaces/Inferact/Kimi-K3-DSpark"
    for role in ("prefill", "decode"):
        argv = shlex.split(cell["service"][role]["extra_args"])
        speculative = json.loads(argv[argv.index("--speculative-config") + 1])
        assert speculative["model"] == path
        assert speculative["method"] == "dspark"
        assert not speculative.get("use_heterogeneous_vocab", False)
        assert "quantization" not in speculative
    remapped = resolve_machine_paths(gated[0], monkeypatch, "pit2-vm-amd-xl-02")
    assert (
        remapped["vllm"]["draft_model_path"]
        == "/share_nfs/models/Inferact/Kimi-K3-DSpark"
    )
    for role in ("prefill", "decode"):
        argv = shlex.split(remapped["service"][role]["extra_args"])
        assert json.loads(argv[1])["model"] == remapped["vllm"]["draft_model_path"]
    export_code = (
        (SCRIPTS / "pd_submit.sh")
        .read_text()
        .split("python3 - <<'PY'\n", 1)[1]
        .split("\nPY", 1)[0]
    )
    result = subprocess.run(
        [sys.executable, "-c", export_code],
        env={**os.environ, "CELL_JSON": json.dumps(remapped)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    exports = {}
    for line in result.stdout.splitlines():
        if line.startswith("export "):
            key, value = shlex.split(line)[1].split("=", 1)
            exports[key] = value
    assert (
        exports["ATOMESH_VLLM_DRAFT_MODEL_PATH"] == remapped["vllm"]["draft_model_path"]
    )
    for role in ("PREFILL", "DECODE"):
        argv = shlex.split(exports[f"{role}_EXTRA_SERVER_ARGS"])
        assert json.loads(argv[1])["model"] == exports["ATOMESH_VLLM_DRAFT_MODEL_PATH"]


@pytest.mark.parametrize("visible", [False, True])
def test_runner_and_both_nodes_gate_draft_before_submission_or_build(tmp_path, visible):
    target, draft = tmp_path / "target", tmp_path / "draft with spaces"
    checkpoint(target)
    if visible:
        checkpoint(draft, draft=True)
    steps = workflow_steps()
    gate = next(
        step
        for step in steps
        if step.get("name") == "Survey pre-submit visibility and queue check"
    )
    submit = next(
        step for step in steps if step.get("name") == "Run Slurm benchmark cell"
    )
    assert steps.index(gate) < steps.index(submit)
    (tmp_path / "survey-preflight").mkdir()
    (tmp_path / ".github/scripts").mkdir(parents=True)
    (tmp_path / ".github/scripts/atomesh").symlink_to(SCRIPTS, target_is_directory=True)
    env = {
        **os.environ,
        "CELL_JSON": json.dumps(
            {
                "model_path": str(target),
                "vllm": {"clean_main": 1, "draft_model_path": str(draft)},
            }
        ),
        "ATOMESH_MODEL_ROOT": str(tmp_path),
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    result = subprocess.run(
        [sys.executable, "-c", python_step(gate)],
        cwd=tmp_path,
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == visible, result.stderr + result.stdout
    report = json.loads((tmp_path / "survey-preflight/weights.json").read_text())
    assert report["draft"]["model_path"] == str(draft)
    assert report["status"] == ("FILES_VISIBLE" if visible else "BLOCKED_ENV")
    definitions = (
        (SCRIPTS / "pd_server_vllm.sh")
        .read_text()
        .split('\nif [[ -n "${ATOMESH_VLLM_SOURCE_SHA:-}" ]]; then')[0]
    )
    for rank in (0, 1):
        env.update(
            ATOMESH_VLLM_SOURCE_REPO="https://github.com/vllm-project/vllm",
            ATOMESH_VLLM_SOURCE_SHA="b22494cc0cb4bd9db4a62fb107d92429a4a3249d",
            ATOMESH_VLLM_CLEAN_MAIN="1",
            ATOMESH_VLLM_DRAFT_MODEL_PATH=str(draft),
            ATOMESH_VLLM_ROUTER_DISCOVERY_PORT="6300",
            ATOMESH_SERVICE_PORT_OFFSET="0",
            ATOMESH_SCRIPT_DIR=str(SCRIPTS),
            MODEL_PATH=str(target),
            RUNTIME_LOG_DIR=str(tmp_path),
            NODE_RANK=str(rank),
            host_ip="192.0.2.1",
        )
        # Exit the shell at the first build/network command; no installation or GPUs.
        shell = (
            "git() { printf build-boundary; exit 99; };\n"
            + definitions
            + "\ninstall_native_vllm"
        )
        result = subprocess.run(
            ["bash", "-c", shell], env=env, check=False, capture_output=True, text=True
        )
        assert result.returncode == (99 if visible else 2), (
            result.stderr + result.stdout
        )
        assert ("build-boundary" in result.stdout) == visible
        node_report = json.loads(
            (tmp_path / f"weights-preflight-rank-{rank}.json").read_text()
        )
        assert node_report == report
