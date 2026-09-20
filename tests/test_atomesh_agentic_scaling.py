# SPDX-License-Identifier: MIT
"""Paired workload, launch placement and result-integrity regressions (no GPU)."""

import copy
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load("pd_matrix")
scaling = load("agentic_scaling")
IMAGE = "rocm/atom-dev:nightly@sha256:" + "a" * 64


@pytest.fixture
def cells(monkeypatch):
    env = {
        "ATOMESH_SLURM_ACCOUNT": "",
        "ATOMESH_SLURM_PARTITION": "",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/tmp/logs",
        "ATOMESH_MODEL_ROOT": "/models",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
        "ATOMESH_1P1D_NODES": "",
        "ATOMESH_SINGLE_NODE": "auto",
        "ATOMESH_NODE_POOL": "node-a,node-b,node-c",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = matrix.load_config(ROOT / ".github/benchmark/models_atomesh.yaml")

    def build(**kwargs):
        return matrix.build_cells(
            cfg,
            **{
                "suite": "agentic_scaling",
                "model_filter": None,
                "case_filter": None,
                "benchmark_kind_filter": None,
                "override_image": IMAGE,
                "override_benchmark_concurrency": None,
                "override_eval_concurrency": None,
                **kwargs,
            },
        )

    return cfg, build


def test_matrix_preserves_the_four_existing_baselines(cells):
    cfg, build = cells
    result = build()
    assert len(result) == 8
    original = {
        c["name"]: c for c in cfg["models"]["GLM-5.2-MXFP4"]["suites"]["nightly"]
    }
    for c, a, b in zip((32, 40, 48, 56), result[::2], result[1::2]):
        assert a["concurrency"] == [c]
        assert b["concurrency"] == [2 * c]
        assert (a["num_nodes"], b["num_nodes"]) == (1, 2)
        assert (a["scaling"]["active_gpus"], b["scaling"]["active_gpus"]) == (8, 16)
        assert b["pd_worker_layout"] == "paired_nodes"
        assert scaling.comparable_config(a) == scaling.comparable_config(b)
        source = original[a["name"]]
        for key, value in source["server"]["common_args"].items():
            assert a["server_args"][key] == value
        for key, value in source["benchmark"].items():
            assert a["benchmark"][key] == value
        assert a["benchmark"]["benchmark_duration"] == 3600
        assert a["benchmark"]["aiperf_use_preinstalled"] is True
        assert a["service"]["router"]["policy"] == "random"
        for key in ("policy", "prefill_policy", "decode_policy"):
            assert b["service"]["router"][key] == "kv_cache_aware"
        assert a["env"]["prefill"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "256"
        assert a["env"]["prefill"]["VLLM_PP_LAYER_PARTITION"] == "20,20,20,18"
        assert a["env"]["decode"]["HIP_VISIBLE_DEVICES"] == "4,5,6,7"
        assert "--pipeline-parallel-size 4" in a["service"]["prefill"]["extra_args"]
        assert (
            "--decode-context-parallel-size 4" in a["service"]["decode"]["extra_args"]
        )


def test_pair_filter_and_override_rejection(cells):
    _, build = cells
    name = build()[0]["name"]
    assert len(build(case_filter={name})) == 2
    for args in (
        {"case_filter": {"typo"}},
        {"override_benchmark_concurrency": [9]},
        {"override_eval_concurrency": [9]},
        {"model_filter": {"other"}},
    ):
        with pytest.raises(ValueError):
            build(**args)
    with pytest.raises(ValueError, match="equal P and D"):
        matrix.required_node_count("paired_nodes", {"workers": 2}, {"workers": 1})


def test_submission_exports_pp_geometry_and_no_install_mode(cells, tmp_path):
    _, build = cells
    source = (SCRIPTS / "pd_submit.sh").read_text()
    # Execute the real JSON->environment mapping used before sbatch.
    mapping = source.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = subprocess.run(
        ["python3", "-c", mapping],
        env={**os.environ, "CELL_JSON": json.dumps(build()[1])},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "export NUM_NODES=2\n" in result.stdout
    assert "export PREFILL_PP_SIZE=4\n" in result.stdout
    assert "export DECODE_PP_SIZE=1\n" in result.stdout
    assert "export ATOMESH_PREINSTALLED_ONLY=1\n" in result.stdout
    assert "export AIPERF_USE_PREINSTALLED=true\n" in result.stdout
    assert "export ROUTER_PREFILL_POLICY=kv_cache_aware\n" in result.stdout
    assert "export ROUTER_DECODE_POLICY=kv_cache_aware\n" in result.stdout


@pytest.mark.parametrize(
    "layout,rank,workers",
    [("single_node", 0, 1), ("paired_nodes", 0, 2), ("paired_nodes", 1, 2)],
)
def test_launches_disjoint_pp4_dcp4_pairs(tmp_path, layout, rank, workers):
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    # Run the launcher with process/HTTP boundaries stubbed. It still builds
    # both real command lines and applies the role environments and port math.
    setup, launch = source.rsplit("\nwrite_metadata\n", 1)
    setup = setup.replace("rm -rf /root/.cache/atom/* 2>/dev/null || true", ":")
    trace = tmp_path / "launch.jsonl"
    stubs = """
write_metadata() { :; }
dump_launch_info() { :; }
cleanup_processes() { :; }
wait_http() { :; }
wait_router_closed() { :; }
run_benchmark_and_eval() { touch "$TEST_BENCHMARK"; }
configure_cache_catalog() { :; }
python3() {
  # Catalog HTTP/identity validation is exercised separately; stub this I/O
  # boundary while retaining the router's actual policy argument construction.
  [[ "$1" != */agentic_routing.py ]] || return 0
  command python3 "$@"
}
start_logged_process() {
  printf -v "$1" '%s' 424242
  python3 - "$@" <<'PY'
import json, os, sys
with open(os.environ["TEST_TRACE"], "a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
                       "partition": os.environ.get("VLLM_PP_LAYER_PARTITION")}) + "\\n")
PY
}
"""
    env = {
        **os.environ,
        "MODEL_NAME": "test",
        "MODEL_PATH": "/models/test",
        "RUN_DIR": str(tmp_path),
        "ATOM_TORCH_PROFILER_DIR": str(tmp_path / "profiler"),
        "ATOMESH_PD_WORKER_LAYOUT": layout,
        "NODE_RANK": str(rank),
        "xP": str(workers),
        "yD": str(workers),
        "IPADDRS": ",".join(["10.0.0.1", "10.0.0.2"][:workers]),
        "NODE0_ADDR": "10.0.0.1",
        "PREFILL_TP_SIZE": "1",
        "PREFILL_PP_SIZE": "4",
        "DECODE_TP_SIZE": "4",
        "DECODE_PP_SIZE": "1",
        "ATOMESH_PREINSTALLED_ONLY": "0",
        "PREFILL_EXTRA_SERVER_ARGS": "--pipeline-parallel-size 4 --enforce-eager",
        "DECODE_EXTRA_SERVER_ARGS": "--decode-context-parallel-size 4",
        "ROUTER_POLICY": "kv_cache_aware" if workers == 2 else "random",
        "ROUTER_PREFILL_POLICY": "kv_cache_aware" if workers == 2 else "",
        "ROUTER_DECODE_POLICY": "kv_cache_aware" if workers == 2 else "",
        "ATOMESH_PREFILL_ENV_HIP_VISIBLE_DEVICES": "0,1,2,3",
        "ATOMESH_PREFILL_ENV_VLLM_PP_LAYER_PARTITION": "20,20,20,18",
        "ATOMESH_DECODE_ENV_HIP_VISIBLE_DEVICES": "4,5,6,7",
        "TEST_TRACE": str(trace),
        "TEST_BENCHMARK": str(tmp_path / "benchmark"),
    }
    for role in ("PREFILL", "DECODE"):
        env[f"ATOMESH_{role}_ENV_{role}_KV_TRANSFER_CONFIG"] = (
            '{"handshake_port":${HANDSHAKE_PORT},"proxy_ip":"${ROLE_IP}"}'
        )
    subprocess.run(
        ["bash", "-c", setup + stubs + launch],
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    p, d = records[:2]
    assert (p["gpu"], d["gpu"]) == ("0,1,2,3", "4,5,6,7")
    assert p["partition"] == "20,20,20,18"
    assert d["partition"] is None
    for entry, port in ((p, 6301), (d, 6305)):
        args = entry["args"]
        kv = json.loads(args[args.index("--kv-transfer-config") + 1])
        assert kv["handshake_port"] == port
        assert kv["proxy_ip"] == f"10.0.0.{rank + 1}"
    assert (tmp_path / "benchmark").exists() == (rank == 0)
    if rank == 0:
        assert records[2]["args"].count("--prefill") == workers
        assert records[2]["args"].count("--decode") == workers
        if workers == 2:
            args = records[2]["args"]
            for flag in ("--policy", "--prefill-policy", "--decode-policy"):
                assert args[args.index(flag) + 1] == "kv_cache_aware"
    else:
        assert len(records) == 2


def write_run(cell, root, speedup=1):
    (root / f"{cell['id']}.cell.json").write_text(json.dumps(cell))
    root = root / cell["id"]
    root.mkdir()
    metrics = {
        "max_concurrency": cell["concurrency"][0],
        "request_throughput": 10 * speedup,
        "output_throughput": 1000 * speedup,
        "successful_requests": 100,
        "request_error_rate_pct": 2.0,
        "benchmark_duration_s": 3600,
    }
    metrics.update(
        {f"p{p}_{key}_ms": 10 for key in ("ttft", "itl", "e2el") for p in (95, 99)}
    )
    files = {
        "pd-result.json": metrics,
        "job-result.json": {"result": {"state": "COMPLETED", "return_code": 0}},
        "mesh-build.json": {
            "profile": "release",
            "source_dirty": False,
            "commit": "reviewed-sha",
            "binary_sha256": "binary-digest",
        },
        "aiperf-version.json": {
            "version": "0.12.0",
            "source_sha256": "aiperf-sha",
            "mode": "preinstalled",
            "image": cell["image"],
        },
    }
    if cell["scaling"]["scale"] == 2:
        files["cache-routing-preflight.json"] = {"policy": "kv_cache_aware"}
        files["routing-decisions.json"] = {
            "policy": "kv_cache_aware",
            "selected": 95,
            "fallback": 5,
        }
    for name, payload in files.items():
        (root / name).write_text(json.dumps(payload))
    return root


def test_report_normalizes_by_two_times_resources(cells, tmp_path):
    _, build = cells
    pair = build()[:2]
    write_run(pair[0], tmp_path)
    write_run(pair[1], tmp_path, speedup=1.8)
    report = scaling.compare({"include": pair}, tmp_path)
    assert report["complete"]
    out = report["pairs"][0]["throughput"]["output_throughput"]
    assert out["speedup"] == 1.8
    assert out["per_gpu_efficiency"] == 0.9
    assert out["per_active_gpu"] == [125, 112.5]
    assert report["pairs"][0]["runs"][0]["failure_rate"] == 0.02
    assert "0.900" in scaling.markdown(report)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "failed",
        "nan",
        "no_errors",
        "wrong_concurrency",
        "version",
        "source",
        "settings",
        "duplicate",
        "excess_errors",
    ],
)
def test_report_rejects_incomplete_or_incomparable_runs(cells, tmp_path, fault):
    _, build = cells
    pair = copy.deepcopy(build()[:2])
    write_run(pair[0], tmp_path)
    scaled = write_run(pair[1], tmp_path, speedup=2)
    if fault == "missing":
        (scaled / "job-result.json").unlink()
    elif fault == "settings":
        pair[1]["benchmark"]["warmup_requests_per_lane"] = 1
    else:
        filename = {
            "failed": "job-result.json",
            "source": "mesh-build.json",
            "version": "aiperf-version.json",
        }.get(fault, "pd-result.json")
        path = scaled / filename
        payload = json.loads(path.read_text())
        if fault == "failed":
            payload["result"]["return_code"] = 1
        elif fault == "source":
            payload["commit"] = "other-sha"
        elif fault == "version":
            payload["source_sha256"] = "different-aiperf"
        elif fault == "nan":
            payload["output_throughput"] = float("nan")
        elif fault == "no_errors":
            del payload["request_error_rate_pct"]
        elif fault == "wrong_concurrency":
            payload["max_concurrency"] = 32
        elif fault == "excess_errors":
            payload["request_error_rate_pct"] = 11
        elif fault == "duplicate":
            (scaled / "pd-stale-result.json").write_text(path.read_text())
        path.write_text(json.dumps(payload))
    report = scaling.compare({"include": pair}, tmp_path)
    assert not report["complete"]
    assert report["pairs"][0]["status"] == "incomplete"
    assert "INCOMPLETE" in scaling.markdown(report)


def test_setup_refuses_prebuilt_mesh_override(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "setup_mesh.sh"),
            str(ROOT),
            str(tmp_path),
            IMAGE,
            "/dev/null",
            "test",
        ],
        env={
            **os.environ,
            "ATOMESH_PREINSTALLED_ONLY": "1",
            "ATOMESH_MESH_BINARY": "/prebuilt/mesh",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "must build Mesh" in result.stderr


def test_missing_preinstalled_aiperf_fails_without_installing():
    source = (SCRIPTS / "pd_server_atom.sh").read_text()
    function = source[
        source.index("ensure_aiperf() {") : source.index(
            "write_aiperf_dashboard_json() {"
        )
    ]
    result = subprocess.run(
        ["/bin/bash", "-c", "set -euo pipefail\n" + function + "\nensure_aiperf\n"],
        env={"PATH": "/missing-tools", "AIPERF_USE_PREINSTALLED": "true"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "requires AIPerf preinstalled" in result.stderr
