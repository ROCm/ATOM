# SPDX-License-Identifier: MIT
"""Calibration binding and evidence of actual new-policy selections."""

import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/atomesh/agentic_routing.py"
SPEC = importlib.util.spec_from_file_location("agentic_routing", SCRIPT)
routing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(routing)
IMAGE = "rocm/atom-dev@sha256:" + "a" * 64


@pytest.fixture
def bundle(monkeypatch):
    monkeypatch.setenv("ATOMESH_RUN_TOKEN", "this-job")
    monkeypatch.setenv("DOCKER_IMAGE", IMAGE)
    # Synthetic costs for protocol tests only; never used in GPU benchmarks.
    executions = {
        f"d{i}": {"layout_id": "decode-layout", "costs": {"decode_step_ms": 1}}
        for i in range(2)
    }
    for i in range(2):
        executions[f"p{i}"] = {
            "layout_id": "prefill-layout",
            "costs": {
                "prefill_curves": [
                    {"max_context_tokens": 1048576, "points": [[1048576, 20]]}
                ],
                "h2d_curve": [[1048576, 10]],
            },
            "transfer_paths": [
                {
                    "verified": True,
                    "protocol": "mooncake",
                    "destination_execution_id": f"d{j}",
                    "source_layout_id": "prefill-layout",
                    "destination_layout_id": "decode-layout",
                    "transfer_curve": [[1048576, 2]],
                }
                for j in range(2)
            ],
        }
    return {
        "schema_version": 1,
        "image": IMAGE,
        "nodes": ["node-a", "node-b"],
        "measurement_artifact_sha256": "b" * 64,
        "namespace_manifest": {
            "model_revision": "test-weights-sha",
            "tokenizer_revision": "test-tokenizer-sha",
            "template_revision": "test-template-sha",
            "kv_semantics": "test-fp8",
            "adapter_revision": None,
            "cache_salt": None,
            "multimodal_identity": None,
        },
        "calibration": {"executions": executions},
    }


def test_bundle_binds_image_and_all_local_cross_node_paths(bundle):
    assert routing.validate_bundle(bundle, IMAGE) is bundle
    assert routing.validate_bundle(bundle, IMAGE.replace("@", ":nightly@")) is bundle
    with pytest.raises(ValueError, match="No calibration was supplied"):
        routing.validate_bundle({}, IMAGE)
    with pytest.raises(ValueError, match="image digest"):
        routing.validate_bundle(bundle, "other@sha256:" + "c" * 64)
    for fault in ("unverified", "missing_link", "too_short", "nan"):
        invalid = copy.deepcopy(bundle)
        entry = invalid["calibration"]["executions"]["p0"]
        if fault == "unverified":
            entry["transfer_paths"][1]["verified"] = False
        elif fault == "missing_link":
            entry["transfer_paths"].pop()
        elif fault == "too_short":
            entry["costs"]["h2d_curve"] = [[256, 1]]
        else:
            entry["costs"]["h2d_curve"] = [[1048576, float("nan")]]
        with pytest.raises(ValueError):
            routing.validate_bundle(invalid, IMAGE)


def test_role_config_uses_fresh_ids_and_measured_gpu_host_placement(
    bundle, monkeypatch
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    config = routing.role_config(
        bundle, "prefill", 1, "node-b.example", "10.0.0.2", 18610
    )
    assert config["execution_id"] == "this-job/p1"
    assert config["catalog_url"] == "http://10.0.0.2:18610"
    with pytest.raises(ValueError, match="differs from live host"):
        routing.role_config(bundle, "prefill", 1, "node-a", "10.0.0.1", 18610)
    with pytest.raises(ValueError, match="GPU placement"):
        routing.role_config(bundle, "decode", 1, "node-b", "10.0.0.2", 18611)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6,7")
    assert (
        routing.role_config(bundle, "decode", 1, "node-b", "10.0.0.2", 18611)[
            "execution_id"
        ]
        == "this-job/d1"
    )


def test_router_rebinds_only_matching_live_catalogs(bundle, monkeypatch, tmp_path):
    def get_info(url):
        role, rank = url.split("//")[1][:2]
        return {
            "execution_id": f"this-job/{role}{rank}",
            "layout_id": "prefill-layout" if role == "p" else "decode-layout",
            "content_namespace": routing.namespace_digest(bundle["namespace_manifest"]),
            "capabilities": {"exact_prefix_reuse": True},
            "parallel_layout": {
                "pp_size": 4 if role == "p" else 1,
                "tp_size": 1 if role == "p" else 4,
                "dcp_size": 1 if role == "p" else 4,
            },
        }

    monkeypatch.setattr(routing, "get_json", get_info)
    output = tmp_path / "calibration.json"
    routing.prepare_router(
        bundle, ["http://p0", "http://p1"], ["http://d0", "http://d1"], output
    )
    result = json.loads(output.read_text())["executions"]
    assert set(result) == {
        f"this-job/{role}{rank}" for role in ("p", "d") for rank in range(2)
    }
    assert (
        result["this-job/p1"]["transfer_paths"][0]["destination_execution_id"]
        == "this-job/d0"
    )
    invalid = copy.deepcopy(bundle)
    invalid["calibration"]["executions"]["d1"]["layout_id"] = "wrong-layout"
    with pytest.raises(ValueError, match="calibrated identity/layout"):
        routing.prepare_router(
            invalid, ["http://p0", "http://p1"], ["http://d0", "http://d1"], output
        )


def test_actual_selection_counters_reject_all_fallback(monkeypatch, tmp_path):
    before = tmp_path / "before.json"
    before.write_text(json.dumps({"selected": 2, "fallback": 3}))
    output = tmp_path / "after.json"
    metrics = 'atomesh_kv_cache_routing_decisions_total{outcome="selected"} 11\natomesh_kv_cache_routing_decisions_total{outcome="fallback"} 4\n'
    monkeypatch.setattr(
        routing, "urlopen", lambda *a, **kw: io.BytesIO(metrics.encode())
    )
    routing.record_decisions("http://mesh/metrics", output, before)
    result = json.loads(output.read_text())
    assert result["selected"] == 9
    assert result["fallback"] == 1
    assert result["selected_fraction"] == 0.9
    metrics = metrics.replace('selected"} 11', 'selected"} 2')
    with pytest.raises(ValueError, match="no calibrated selections"):
        routing.record_decisions("http://mesh/metrics", output, before)
    assert json.loads(output.read_text())["selected_fraction"] == 0
