"""CPU-only single-case configuration, provenance and fail-closed checks."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load("pd_prefix_read")
launcher = load("pd_c48_launch")
matrix = load("pd_matrix")


def test_single_case_preserves_reference(monkeypatch):
    for key, value in {
        "ATOMESH_MODEL_ROOT": "/mnt/models",
        "ATOMESH_1P1D_NODES": "",
        "ATOMESH_NODE_POOL": "",
        "ATOMESH_SLURM_PARTITION": "amd-spur",
        "ATOMESH_SLURM_ACCOUNT": "amd-frameworks",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/it-share/ATOMESH_LOG",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
    }.items():
        monkeypatch.setenv(key, value)
    cfg = yaml.safe_load((ROOT / ".github/benchmark/models_atomesh.yaml").read_text())
    variants = cfg["models"]["Kimi-K3-C48-Gap"]["suites"]["vllm"]
    baseline = next(x for x in variants if x["name"].endswith("-v0"))
    candidate = copy.deepcopy(
        next(x for x in variants if x["name"] == helper.CASE_NAME)
    )
    assert baseline["benchmark"]["cancel_drain_timeout"] == 300
    assert candidate["benchmark"]["cancel_drain_timeout"] == 10
    candidate["name"] = baseline["name"]
    candidate["benchmark"]["cancel_drain_timeout"] = 300
    candidate["env"]["common"]["ATOMESH_C48_CASE"] = "V0"
    assert candidate == baseline
    cells = matrix.build_cells(
        cfg,
        suite="vllm",
        model_filter=None,
        case_filter={helper.CASE_NAME},
        benchmark_kind_filter=None,
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["num_nodes"] == 2 and cell["nodes"] == []
    assert cell["concurrency"] == [48] and cell["run_eval"] is False
    assert cell["image"] == helper.IMAGE
    for role in ("prefill", "decode"):
        assert cell["service"][role]["tp"] == 8
        assert cell["service"][role]["dcp"] == 8
    assert cell["benchmark"]["benchmark_duration"] == 3600
    assert cell["benchmark"]["cancel_drain_timeout"] == 10


def test_manifest_and_serving_config():
    manifest = helper.verify_bundle(ROOT)
    assert manifest["readiness"] == helper.LOCAL_READY
    config = json.loads((ROOT / helper.PAYLOAD / "config.json").read_text())
    reference = json.loads(
        (ROOT / ".github/benchmark/c48-model-gap/V0.json").read_text()
    )
    for role in ("prefill", "decode"):
        args = (
            role,
            "/mnt/models/moonshotai/Kimi-K3",
            "10.0.0.1",
            "10.0.0.1",
            6100,
            36367,
        )
        actual, env = launcher.render(config, *args)
        expected, refenv = launcher.render(reference, *args)
        assert actual == expected
        assert env.pop("C48_PREFIX_READ_FIXED") == "1"
        assert refenv.pop("C48_PREFIX_READ_FIXED") == "0"
        assert env == refenv


def test_inputs_and_reviewed_local_readiness(monkeypatch):
    inputs = json.loads((ROOT / helper.PAYLOAD / "inputs.json").read_text())
    assert inputs["case_names"] == helper.CASE_NAME
    assert inputs["publish_dashboard"] == "false"
    assert inputs["run_all_models"] == "false"
    assert inputs["atomesh_1p1d_nodes"] == "auto"
    result = helper.require_ready()
    assert result["status"] == helper.LOCAL_READY
    assert result["gpu_verified"] is False
    assert result["runtime_gates_required"] is True
    manifest = helper.verify_bundle(ROOT)
    for invalid in (
        dict(manifest, readiness="BLOCKED"),
        dict(manifest, blockers=["pending review"]),
        dict(manifest, local_review={"passed": False}),
    ):
        monkeypatch.setattr(helper, "verify_bundle", lambda root, m=invalid: m)
        with pytest.raises(RuntimeError, match="BLOCKED"):
            helper.require_ready()


def test_readiness_cli_returns_json():
    result = subprocess.run(
        ["python3", str(SCRIPTS / "pd_prefix_read.py"), "require-ready"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["status"] == helper.LOCAL_READY


def test_overlay_requires_correct_hash_and_readonly_mount(tmp_path):
    target = tmp_path / helper.RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT / helper.PAYLOAD / "moriio_connector.py").read_bytes())
    mount = f"1 2 0:1 / {target} ro,relatime - ext4 /dev/x rw\n"
    assert helper.verify_overlay(tmp_path, mount)["readonly_mount_observed"]
    with pytest.raises(RuntimeError, match="readonly"):
        helper.verify_overlay(tmp_path, mount.replace("ro,relatime", "rw,relatime"))
    with pytest.raises(RuntimeError, match="readonly"):
        helper.verify_overlay(tmp_path, "")
    target.write_text("not the candidate")
    with pytest.raises(RuntimeError, match="hash"):
        helper.verify_overlay(tmp_path, mount)


def test_distinct_mounts():
    script = f"""set -euo pipefail
source {SCRIPTS}/pd_c48_port_mount.sh
source {SCRIPTS}/pd_prefix_read_mount.sh
docker_args=()
c48_append_port_mount {ROOT}
prefix_read_append_mount {ROOT}
printf '%s\\n' "${{docker_args[@]}}"
"""
    output = subprocess.check_output(["bash", "-c", script], text=True)
    assert output.count("readonly") == 5
    assert "moriio_common.py" in output and "moriio_connector.py" in output


@pytest.mark.parametrize(
    "available,rank,passes",
    [
        (1900000000000, 0, True),
        (1799000000000, 0, False),
        (100000000000, 1, True),
        (1000, 1, False),
    ],
)
def test_offload_memory_budget(available, rank, passes):
    meminfo = f"MemAvailable: {available // 1024} kB\n"
    if passes:
        helper.check_memory(meminfo, rank)
    else:
        with pytest.raises(RuntimeError, match="CPU memory"):
            helper.check_memory(meminfo, rank)


def test_storage_fail_closed(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        helper.check_capacity([tmp_path / "absent"], 1, 1)
    monkeypatch.setattr(
        helper.os,
        "statvfs",
        lambda p: SimpleNamespace(f_bavail=10, f_frsize=1, f_favail=0),
    )
    with pytest.raises(RuntimeError, match="storage"):
        helper.check_capacity([tmp_path], 1, 1)
    monkeypatch.setattr(
        helper.os,
        "statvfs",
        lambda p: SimpleNamespace(f_bavail=10, f_frsize=1, f_favail=10),
    )
    with pytest.raises(RuntimeError, match="storage"):
        helper.check_capacity([tmp_path], 11, 1)
    assert helper.check_capacity([tmp_path], 10, 10)


def test_cgroup_ancestor_budget(tmp_path):
    proc = tmp_path / "proc"
    mount = tmp_path / "cgroup"
    leaf = mount / "job"
    proc.mkdir()
    leaf.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/job\n")
    (mount / "memory.max").write_text("1800000000000")
    (mount / "memory.current").write_text("100000000000")
    (leaf / "memory.max").write_text("max")
    (leaf / "memory.current").write_text("1000000000")
    mounts = f"1 2 0:1 / {mount} rw - cgroup2 cgroup rw\n"
    rows = helper.cgroup_headroom(proc, mounts)
    assert rows[1]["headroom"] == 1700000000000
    with pytest.raises(RuntimeError, match="CPU memory"):
        helper.check_memory("MemAvailable: 2200000000 kB\n", 0, rows)
    assert helper.check_memory("MemAvailable: 2200000000 kB\n", 1, rows)
    with pytest.raises(RuntimeError, match="unavailable"):
        helper.cgroup_headroom(proc, "")


@pytest.mark.parametrize("cached,minimum", [(False, 200), (True, 40)])
def test_cache_aware_storage_threshold(monkeypatch, tmp_path, cached, minimum):
    class StopAfterStorage(Exception):
        pass

    monkeypatch.setattr(helper.subprocess, "check_output", lambda *a, **kw: "/docker\n")
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0 if cached else 1,
            stdout=json.dumps([{"RepoDigests": [helper.IMAGE]}]),
            stderr="No such image",
        ),
    )

    def capacity(paths, minimum_bytes, inodes):
        assert minimum_bytes == minimum * 1024**3 and inodes == 100000
        assert {str(p) for p in paths} == {
            "/docker",
            "/var/lib/containerd",
            "/tmp",
            str(tmp_path),
        }
        raise StopAfterStorage

    monkeypatch.setattr(helper, "check_capacity", capacity)
    with pytest.raises(StopAfterStorage):
        helper.host_preflight(tmp_path, tmp_path, 0, helper.IMAGE)


def test_missing_model_shards_fail(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}})
    )
    with pytest.raises(RuntimeError, match="missing"):
        helper.check_model(tmp_path)
    (tmp_path / "missing.safetensors").write_bytes(b"fixture")
    assert helper.check_model(tmp_path)["weight_contents_hashed"] is False


def test_unsharded_model_without_index(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="no safetensors"):
        helper.check_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty"):
        helper.check_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    result = helper.check_model(tmp_path)
    assert result["index_sha256"] is None
    assert result["shards"] == [{"path": "model.safetensors", "bytes": 7}]


def test_client_spawned_children_do_not_rerun_aiperf(tmp_path):
    client = tmp_path / "prefix-read-fixed/client"
    (client / "aiperf/timing").mkdir(parents=True)
    (client / "sitecustomize.py").write_text("ACTIVE = True\n")
    (client / "aiperf/__init__.py").write_text("")
    (client / "aiperf/timing/__init__.py").write_text("")
    (client / "aiperf/timing/phase_orchestrator.py").write_text("")
    # The weka trace loader reconstructs traces in a spawn-context pool.
    (client / "aiperf/__main__.py").write_text(
        "import multiprocessing\n"
        "from concurrent.futures import ProcessPoolExecutor\n"
        "print('aiperf-main', flush=True)\n"
        "ctx = multiprocessing.get_context('spawn')\n"
        "with ProcessPoolExecutor(1, mp_context=ctx) as pool:\n"
        "    print(list(pool.map(abs, [-1])), flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "pd_prefix_client.py")],
        env={"PYTHONPATH": str(client)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["aiperf-main", "[1]"]
