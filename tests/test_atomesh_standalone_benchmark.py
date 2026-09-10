"""Regression coverage for ATOMesh P/D mixed-deployment benchmarks."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATOMESH_SCRIPTS = ROOT / ".github" / "scripts" / "atomesh"
OBSERVABILITY_SCRIPTS = ATOMESH_SCRIPTS / "observability"
sys.path[:0] = [str(ATOMESH_SCRIPTS), str(OBSERVABILITY_SCRIPTS)]

import collect_metrics
import pd_matrix
import process_result


def test_glm52_standalone_agentic_cases(monkeypatch):
    env = {
        "ATOMESH_SLURM_ACCOUNT": "amd-tw",
        "ATOMESH_SLURM_PARTITION": "amd-tw",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd",
        "ATOMESH_LOG_ROOT": "/tmp/atomesh-logs",
        "ATOMESH_MODEL_ROOT": "/mnt/models",
        "ATOMESH_1P1D_NODES": "node-a,node-b",
        "ATOMESH_2P1D_NODES": "node-a,node-b",
        "ATOMESH_PD_RANK_MAPPING_POLICY": "none",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = pd_matrix.load_config(
        ROOT / ".github" / "benchmark" / "models_atomesh.yaml"
    )
    case_prefix = (
        "glm-52-mxfp4-standalone-tp4-dcp4-sharded-index-no-mtp-"
        "lmcache256-chunk256-agentic-1m-"
    )
    cases = {f"{case_prefix}c48", f"{case_prefix}c56"}
    cells = pd_matrix.build_cells(
        config,
        suite="nightly",
        model_filter={"GLM-5.2-MXFP4"},
        case_filter=cases,
        benchmark_kind_filter={"aiperf_agentic"},
        override_image=None,
        override_benchmark_concurrency=None,
        override_eval_concurrency=None,
    )

    assert {cell["concurrency"][0] for cell in cells} == {48, 56}
    for cell in cells:
        assert cell["deployment"] == "standalone"
        assert cell["num_nodes"] == 1
        assert cell["benchmark"]["benchmark_duration"] == 1800
        standalone = cell["service"]["standalone"]
        assert standalone["port"] == 8010
        assert standalone["tp"] == 4
        assert standalone["dcp"] == 4
        assert json.loads(standalone["cudagraph"]) == [
            1,
            2,
            *range(4, 2 * cell["concurrency"][0] + 1, 4),
        ]
        assert standalone["kv_transfer_config"] == (
            '{"kv_connector":"lmcache_offload","kv_role":"offload"}'
        )
        assert cell["env"]["standalone"]["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "256"
        assert cell["env"]["standalone"]["LMCACHE_CHUNK_SIZE"] == "256"
        assert cell["env"]["standalone"]["ATOM_ENABLE_METRICS_DEVICE_TIMER"] == "1"
        assert "method" not in cell["server_args"]


def test_standalone_metrics_and_dashboard_resources():
    config = collect_metrics.scrape_config(
        [], [], ["127.0.0.1:8010"], "127.0.0.1:29100"
    )
    roles = [
        group["labels"]["role"]
        for job in config["scrape_configs"]
        for group in job["static_configs"]
    ]
    assert roles == ["standalone", "router"]

    resources = process_result.topology_resources(
        {
            "deployment": "standalone",
            "display_topology": "STANDALONE-TP4-DCP4-SHARDED-INDEX-NO-MTP",
            "standalone_tp": 4,
            "standalone_dcp": 4,
        },
        {},
    )
    assert resources["total_gpu"] == 4
    assert resources["num_prefill_gpu"] == 4
    assert resources["num_decode_gpu"] == 4
