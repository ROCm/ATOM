# SPDX-License-Identifier: MIT
import asyncio
import gzip
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from atom.benchmarks.results import AGGREGATION_VERSION
from atom.benchmarks.results.aggregate import stats
from atom.benchmarks.results.bundle import (
    build_bundle,
    rebuild_bundle,
    recipe_fingerprint,
    run_index,
    summary_package,
    verify_bundle,
)
from atom.benchmarks.results.io import (
    artifact_path,
    iter_jsonl,
    read_json,
    write_json,
)
from atom.benchmarks.results.metadata import performance_env, select_devices
from atom.benchmarks.results.records import (
    RequestRecorder,
    full_response_itl,
    seconds,
    timestamp,
)
from atom.benchmarks.results.telemetry import derive_server_metrics, integrate_power

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[3]


def test_random_dataset_import_does_not_load_server_encoders():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectServerEncoder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "atom.entrypoints.openai.chat_encoders":
            raise ImportError("server encoders are unavailable in client-only CI")

sys.meta_path.insert(0, RejectServerEncoder())
from atom.benchmarks.benchmark_serving import sample_random_requests
assert callable(sample_random_requests)
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def make_bundle(tmp_path, kind="random", config=None, **kwargs):
    config = config or read_json(FIXTURES / f"{kind}-config.json")
    raw = tmp_path / "aiperf"
    raw.mkdir(exist_ok=True)
    write_json(raw / "profile_export_aiperf.json", {"submission_valid": True})
    write_json(raw / "server_metrics_export.json", {"fixture": True})
    output = tmp_path / "bundle"
    manifest = build_bundle(
        config,
        FIXTURES / f"{kind}.jsonl",
        output,
        record_format="aiperf" if kind == "agentic" else "atom",
        raw_dir=raw if kind == "agentic" else None,
        **kwargs,
    )
    return output, manifest


@pytest.mark.parametrize("kind", ["random", "agentic"])
def test_fixture_contract_and_counts(tmp_path, kind):
    root, manifest = make_bundle(tmp_path, kind)
    assert manifest["status"] == "complete"
    schemas = ROOT / "atom/benchmarks/results/schemas"
    for name, value in [
        ("manifest", manifest),
        ("config", read_json(root / "config.json")),
        ("summary", read_json(root / "summary.json")),
        ("validation", read_json(root / "validation.json")),
    ]:
        jsonschema.validate(value, read_json(schemas / f"{name}.schema.json"))
    rows = list(iter_jsonl(root / "requests/requests.jsonl.gz"))
    for row in rows:
        jsonschema.validate(row, read_json(schemas / "request.schema.json"))
    summary = read_json(root / "summary.json")
    assert summary["accounting"]["counts"]["total"] == 7
    assert summary["accounting"]["counts"]["profiled_success"] == 4
    timeline = read_json(root / "views/timeline/index.json")
    assert sum(p["count"] for p in timeline["parts"]) == 7
    assert verify_bundle(root)["point_id"] == manifest["point_id"]


def test_per_request_normalization_and_tp2(tmp_path):
    root, _ = make_bundle(tmp_path, "agentic")
    summary = read_json(root / "summary.json")
    metrics = summary["request_metrics"]
    assert metrics["latency"]["e2e_norm_intvty"]["p90"] == pytest.approx(1 / 0.37)
    assert metrics["latency"]["full_response_itl"]["count"] == 3  # OSL=1 excluded
    assert metrics["latency"]["full_response_intvty"]["p90"] == pytest.approx(1 / 0.38)
    assert metrics["throughput"]["duration_seconds"] == pytest.approx(7.3)
    assert metrics["throughput"]["output"]["tokens_per_second"] == pytest.approx(
        56 / 7.3
    )
    assert metrics["throughput"]["per_gpu"]["output_tput_tps"] == pytest.approx(
        56 / 7.3 / 2
    )
    export = read_json(root / "exports/inferencex-v3.json")
    assert export["num_gpus"] == 2
    assert export["num_requests_total"] == 5  # warmup/drain are not profiling
    assert export["num_requests_successful"] == 4


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"value": 1e9, "unit": "ns"}, 1),
        ({"value": 1e6, "unit": "us"}, 1),
        ({"value": 1000, "unit": "ms"}, 1),
        ({"value": 1, "unit": "seconds"}, 1),
    ],
)
def test_explicit_units(value, expected):
    assert seconds(value) == expected


def test_invalid_units_and_precision():
    with pytest.raises(ValueError):
        seconds({"value": 5, "unit": "minutes"})
    with pytest.raises(ValueError):
        timestamp(1.7e18)
    assert timestamp(1700000000000000123) == "1700000000000000123"
    assert stats([1, 2, 3, 4])["p90"] == pytest.approx(3.7)


def test_full_response_precedence():
    row = {
        "output_tokens": 10,
        "ttft_s": 1,
        "start_ns": "1000000000",
        "end_ns": "11000000000",
    }
    assert full_response_itl(row) == 1
    assert full_response_itl(dict(row, full_decode_duration_s=18)) == 2
    assert full_response_itl(dict(row, full_response_itl_s=3)) == 3
    assert full_response_itl(dict(row, output_tokens=1)) is None


def test_identity_immutable_and_rebuild(tmp_path):
    root, m = make_bundle(tmp_path, "agentic")
    with pytest.raises(FileExistsError):
        build_bundle(read_json(root / "config.json"), FIXTURES / "agentic.jsonl", root)
    rebuilt = rebuild_bundle(root, tmp_path / "rebuilt")
    assert rebuilt["point_id"] == m["point_id"]
    assert read_json(root / "summary.json") == read_json(
        tmp_path / "rebuilt/summary.json"
    )
    config = read_json(root / "config.json")
    config["source"]["attempt"] = 2
    other = build_bundle(
        config, root / "requests/requests.jsonl.gz", tmp_path / "attempt2"
    )
    assert other["point_id"] != m["point_id"]
    assert other["recipe_fingerprint"] == m["recipe_fingerprint"]
    config["recipe"]["server_argv"] += ["--cudagraph-capture-sizes", "[1,2,3]"]
    dense = build_bundle(
        config, root / "requests/requests.jsonl.gz", tmp_path / "dense"
    )
    assert dense["recipe_fingerprint"] != other["recipe_fingerprint"]


def test_curve_omits_physical_pci_and_concurrency(tmp_path):
    root, m = make_bundle(tmp_path)
    config = read_json(root / "config.json")
    config["workload"]["concurrency"] = 8
    config["hardware"]["devices"][0]["pci_bus_id"] = "0000:03:00.0"
    config["recipe"]["client_argv"] += ["--concurrency", "8"]
    other = build_bundle(config, FIXTURES / "random.jsonl", tmp_path / "other")
    assert other["curve_group_id"] == m["curve_group_id"]
    assert other["point_id"] != m["point_id"]


def test_recipe_keeps_client_replay_policy_but_omits_gpu_placement():
    config = read_json(FIXTURES / "agentic-config.json")
    environment = {
        "AIPERF_TRAJECTORY_START_MIN_RATIO": "0.25",
        "HIP_VISIBLE_DEVICES": "0,1",
    }
    config["recipe"]["client_environment"] = environment
    original = recipe_fingerprint(config)
    environment["HIP_VISIBLE_DEVICES"] = "6,7"
    assert recipe_fingerprint(config) == original
    environment["AIPERF_TRAJECTORY_START_MIN_RATIO"] = "0.50"
    assert recipe_fingerprint(config) != original


def test_previous_aggregation_bundle_verifies_and_upgrades(tmp_path):
    # Produced by the saved 1.0.0 implementation, including client replay env.
    original = tmp_path / "original"
    with zipfile.ZipFile(FIXTURES / "bundle-1.0.0.zip") as fixture:
        fixture.extractall(original)
    manifest = verify_bundle(original)
    assert manifest["aggregation_version"] == "1.0.0"
    assert manifest["point_id"] == (
        "8f7e0d6fea9cd5f0c7c82711205dcc1f635893f626a0472d46ba6b7691881cbe"
    )
    rebuilt = rebuild_bundle(original, tmp_path / "rebuilt")
    assert rebuilt["aggregation_version"] == AGGREGATION_VERSION
    assert rebuilt["point_id"] != manifest["point_id"]
    assert verify_bundle(original) == manifest


@pytest.mark.parametrize(
    "mutation", ["missing_phase", "reverse_time", "duplicate", "missing_output"]
)
def test_invalid_request_cannot_be_complete(tmp_path, mutation):
    rows = list(iter_jsonl(FIXTURES / "random.jsonl"))
    if mutation == "missing_phase":
        rows[1].pop("phase")
    if mutation == "reverse_time":
        rows[1]["end_ns"] = "1"
    if mutation == "duplicate":
        rows[1]["request_id"] = rows[0]["request_id"]
    if mutation == "missing_output":
        rows[1].pop("output_tokens")
    source = tmp_path / "invalid.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = build_bundle(
        read_json(FIXTURES / "random-config.json"), source, tmp_path / "bundle"
    )
    assert m["status"] == "partial"
    assert not read_json(tmp_path / "bundle/validation.json")["measurement_valid"]


def test_failure_missing_records_and_strict_cli(tmp_path):
    m = build_bundle(
        read_json(FIXTURES / "random-config.json"),
        None,
        tmp_path / "failed",
        exit_code=17,
    )
    assert m["status"] == "failed"
    assert not m["capabilities"]["normalized_interactivity"]
    assert read_json(tmp_path / "failed/validation.json")["exit_code"] == 17
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atom.benchmarks.results",
            "verify",
            str(tmp_path / "failed"),
            "--require-full",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2


def test_truncated_raw_is_retained(tmp_path):
    source = tmp_path / "broken.jsonl"
    source.write_text((FIXTURES / "random.jsonl").read_text() + '{"unfinished":')
    m = build_bundle(
        read_json(FIXTURES / "random-config.json"), source, tmp_path / "bundle"
    )
    assert m["status"] == "partial"
    with gzip.open(tmp_path / "bundle/raw/requests.jsonl.gz", "rt") as stream:
        assert stream.read() == source.read_text()
    assert not read_json(tmp_path / "bundle/validation.json")["measurement_valid"]


@pytest.mark.parametrize(
    "field", ["model.revision", "software.image_digest", "hardware.devices"]
)
def test_full_metadata_gate(tmp_path, field):
    config = read_json(FIXTURES / "random-config.json")
    section, key = field.split(".")
    config[section][key] = [] if key == "devices" else None
    root, m = make_bundle(tmp_path, config=config, require_full=True)
    assert m["status"] == "partial"
    assert read_json(root / "validation.json")["missing"]


def test_synthetic_performance_is_valid_but_not_submission_eligible(tmp_path):
    config = read_json(FIXTURES / "random-config.json")
    config["validity"]["synthetic"] = True
    root, m = make_bundle(tmp_path, config=config)
    assert m["status"] == "complete"
    validation = read_json(root / "validation.json")
    assert validation["measurement_valid"]
    assert not validation["submission_eligible"]
    assert validation["validity"]["synthetic"]
    assert (
        read_json(root / "exports/inferencex-v3.json")["benchmark_outcome"]["status"]
        == "success"
    )
    assert not read_json(root / "exports/inferencex-v3.json")["submission_valid"]


@pytest.mark.parametrize("failure", ["unsafe", "missing", "exit", "submission"])
def test_synthetic_cannot_hide_invalid_measurement(tmp_path, failure):
    config = read_json(FIXTURES / "random-config.json")
    config["validity"]["synthetic"] = True
    if failure == "unsafe":
        config["validity"]["unsafe_override"] = True
    elif failure == "missing":
        config["software"]["atom_sha"] = None
    elif failure == "submission":
        config["validity"]["submission_valid"] = False
    root, _ = make_bundle(
        tmp_path, config=config, exit_code=17 if failure == "exit" else 0
    )
    validation = read_json(root / "validation.json")
    assert not validation["measurement_valid"]
    assert not validation["submission_eligible"]


@pytest.mark.parametrize(
    "flag", ["--spec-decode-acceptance-length", "--spec_decode_acceptance_rate"]
)
def test_forced_acceptance_overrides_false_synthetic_declaration(flag):
    from atom.benchmarks.results.metadata import synthetic_settings

    settings = synthetic_settings(
        [flag, "0.2", flag + "=0.8"], {"ATOM_DSV41_BENCHMARK_SYNTHETIC": "0"}
    )
    assert settings["synthetic"]
    assert settings["synthetic_declaration_mismatch"]
    assert (
        settings["acceptance_length" if "length" in flag else "acceptance_rate"] == 0.8
    )


def test_rebuild_detects_previously_unmarked_forced_acceptance(tmp_path):
    config = read_json(FIXTURES / "random-config.json")
    config["recipe"]["server_argv"].extend(["--spec-decode-acceptance-length", "3.51"])
    root, _ = make_bundle(tmp_path, config=config)
    validation = read_json(root / "validation.json")
    assert validation["measurement_valid"]
    assert validation["validity"]["synthetic"]
    assert not validation["submission_eligible"]


def test_checksum_and_traversal(tmp_path):
    root, _ = make_bundle(tmp_path)
    (root / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        verify_bundle(root)
    for path in ("../secret", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError):
            artifact_path(root, path)
    (root / "symlink").symlink_to("/etc/passwd")
    with pytest.raises(ValueError):
        artifact_path(root, "symlink")


def test_summary_package_and_missing_cells(tmp_path):
    root, _ = make_bundle(tmp_path)
    output = tmp_path / "summaries/cell"
    summary_package(root, output)
    verify_bundle(output, summary_only=True)
    with pytest.raises(ValueError):
        verify_bundle(output)
    plan = tmp_path / "plan.json"
    write_json(
        plan, {"cells": [{"cell_id": "fixture-tp2-c2"}, {"cell_id": "missing-c4"}]}
    )
    run_index(
        tmp_path / "summaries", tmp_path / "index.json", summary_only=True, plan=plan
    )
    index = read_json(tmp_path / "index.json")
    assert index["uploaded_count"] == 1
    assert index["missing_cells"] == [{"cell_id": "missing-c4"}]


def test_summary_packaging_does_not_scan_raw_payload(tmp_path):
    root, _ = make_bundle(tmp_path)
    (root / "raw/requests.jsonl.gz").unlink()
    summary = tmp_path / "summary"
    summary_package(root, summary)
    verify_bundle(summary, summary_only=True)
    with pytest.raises(ValueError, match="Checksum"):
        verify_bundle(root)


def test_container_bundle_is_readable_by_host_user(tmp_path):
    old_umask = os.umask(0o077)
    try:
        root, _ = make_bundle(tmp_path)
    finally:
        os.umask(old_umask)
    for path in [root, *root.rglob("*")]:
        mode = path.stat().st_mode & 0o777
        assert mode == (0o755 if path.is_dir() else 0o644), path
    verify_bundle(root)


def test_power_clips_to_same_window_and_rejects_gaps(tmp_path):
    path = tmp_path / "gpu.jsonl"
    devices = [{"pci_bus_id": "a"}, {"pci_bus_id": "b"}]
    samples = [
        {"timestamp_ns": str(t * 10**9), "pci_bus_id": dev, "power_w": w}
        for t in (0, 1, 2, 3)
        for dev, w in [("a", 100), ("b", 200), ("unused", 900)]
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in samples))
    power = integrate_power(path, devices, "500000000", "2500000000")
    assert power["valid"] and power["energy_j"] == 600
    assert power["avg_total_gpu_power_w"] == 300
    assert not integrate_power(path, devices, "0", "4000000000")["valid"]
    assert not integrate_power(path, devices, "0", "3000000000", max_gap_s=0.5)["valid"]


def test_gpu_mapping_does_not_confuse_hip_and_drm():
    cards = [{"pci_bus_id": str(i), "uuid": str(i)} for i in range(8)]
    assert select_devices(cards, 2, {}) == ([], "unresolved_subset")
    mapped, source = select_devices(cards, 2, {}, ["7", "3"])
    assert [d["pci_bus_id"] for d in mapped] == ["7", "3"]
    assert source == "hip_logical_ordinals"
    with pytest.raises(ValueError):
        select_devices(cards, 2, {"ATOM_BENCHMARK_GPU_PCI_IDS": "0,0"})


def test_environment_is_explicit_allowlist(monkeypatch):
    monkeypatch.setenv("ATOM_PASSWORD", "secret")
    monkeypatch.setenv("AIPERF_PRIVATE_DATA", "secret")
    monkeypatch.setenv("ATOM_DSV41_BENCHMARK_SYNTHETIC", "1")
    env = performance_env()
    assert "ATOM_PASSWORD" not in env and "AIPERF_PRIVATE_DATA" not in env
    assert env["ATOM_DSV41_BENCHMARK_SYNTHETIC"] == "1"


def test_server_series_and_resets(tmp_path):
    path = tmp_path / "server.jsonl"
    rows = [
        {
            "timestamp_ns": str(t * 10**9),
            "text": f"atom:requests_running 2\natom:requests_finished_total {n}\natom:kv_cache_usage_ratio 0.5\n",
        }
        for t, n in [(0, 1), (1, 2), (2, 0), (3, 3)]
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    derived = derive_server_metrics(path, "0", "3000000000")
    assert derived["available"]
    assert derived["counter_resets"] == ["atom:requests_finished_total"]
    assert not derived["counter_deltas"]["atom:requests_finished_total"]["valid"]


def test_recorder_success_errors_cancel_and_token_count(tmp_path):
    async def run():
        recorder = RequestRecorder(
            tmp_path / "requests.jsonl",
            tokenizer=lambda *a, **kw: SimpleNamespace(input_ids=[1, 2, 3]),
        )
        request = SimpleNamespace(prompt_len=10, output_len=5)

        async def ok(**kwargs):
            return SimpleNamespace(
                success=True,
                error="",
                ttft=0.1,
                latency=0.5,
                output_tokens=None,
                itl=[0.2, 0.2],
                generated_text="not retained",
            )

        await recorder.call(ok, request, phase="warmup")

        async def failed(**kwargs):
            raise RuntimeError("fixture error")

        with pytest.raises(RuntimeError):
            await recorder.call(failed, request)

        async def cancelled(**kwargs):
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await recorder.call(cancelled, request)
        recorder.close()

    asyncio.run(run())
    rows = list(iter_jsonl(tmp_path / "requests.jsonl"))
    assert [r["status"] for r in rows] == ["success", "error", "cancelled"]
    assert (
        rows[0]["output_tokens"] == 3
        and rows[0]["token_count_source"] == "client_tokenizer"
    )
    assert "not retained" not in (tmp_path / "requests.jsonl").read_text()


def test_aiperf_runtime_identity_and_cancellation():
    from atom.benchmarks.results.records import from_aiperf

    row = from_aiperf(
        {
            "metadata": {
                "x_request_id": "unique-request",
                "x_correlation_id": "live-session",
                "conversation_id": "reused-dataset-conversation",
                "root_correlation_id": "root",
                "was_cancelled": True,
                "benchmark_phase": "profiling",
            },
            "metrics": {},
        },
        0,
    )
    assert row["request_id"] == "unique-request"
    assert row["session_id"] == "live-session"
    assert row["conversation_id"] == "reused-dataset-conversation"
    assert row["status"] == "cancelled"


def test_nested_aiperf_validity_and_count_mismatch(tmp_path):
    raw = tmp_path / "aiperf"
    raw.mkdir()
    write_json(
        raw / "profile_export_aiperf.json",
        {
            "metadata": {
                "submission_valid": False,
                "submission_invalid_reasons": ["short profiling"],
            },
            "request_count": {"avg": 99},
        },
    )
    write_json(raw / "server_metrics_export.json", {})
    m = build_bundle(
        read_json(FIXTURES / "agentic-config.json"),
        FIXTURES / "agentic.jsonl",
        tmp_path / "bundle",
        record_format="aiperf",
        raw_dir=raw,
    )
    report = read_json(tmp_path / "bundle/validation.json")
    assert m["status"] == "partial"
    assert not report["measurement_valid"]
    assert report["validity"]["submission_invalid_reasons"] == ["short profiling"]
    assert any("disagrees" in reason for reason in report["missing"])


def test_broken_telemetry_preserves_request_bundle(tmp_path):
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    for name in ("gpu_samples", "server_samples"):
        (telemetry / f"{name}.jsonl").write_text('{"incomplete":')
    root, m = make_bundle(tmp_path, telemetry_dir=telemetry)
    assert not m["capabilities"]["power"] and not m["capabilities"]["server_metrics"]
    assert (
        read_json(root / "summary.json")["accounting"]["counts"]["profiled_success"]
        == 4
    )
    verify_bundle(root)


def test_bundle_preserves_harness_summary(tmp_path):
    original = {"output_throughput": 123.0, "mean_itl_ms": 9.0}
    path = tmp_path / "result.json"
    write_json(path, original)
    root, _ = make_bundle(tmp_path, harness_summary=path)
    assert read_json(path) == original
    assert read_json(root / "raw/harness-summary.json") == original
    assert not (root / "exports/atom-dashboard.json").exists()


@pytest.mark.parametrize("field", ["model.key", "hardware.key"])
def test_unknown_inferencex_dimensions_do_not_invalidate_native_bundle(tmp_path, field):
    config = read_json(FIXTURES / "random-config.json")
    section, key = field.split(".")
    config[section][key] = "not-a-known-dimension"
    root, m = make_bundle(tmp_path, config=config)
    assert m["status"] == "complete"
    validation = read_json(root / "validation.json")
    assert validation["measurement_valid"]
    assert validation["export_errors"]["inferencex"]
    verify_bundle(root)
    assert not (root / "exports/inferencex-v3.json").exists()
