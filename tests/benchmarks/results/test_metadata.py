# SPDX-License-Identifier: MIT
"""Regression coverage for identities captured in container benchmark runs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from atom.benchmarks.results.bundle import build_bundle, rebuild_bundle, verify_bundle
from atom.benchmarks.results.io import read_json, write_json
from atom.benchmarks.results.metadata import capture_hf_dataset, git_sha

FIXTURES = Path(__file__).parent / "fixtures"


def test_git_sha_of_checkout_with_different_owner(tmp_path, monkeypatch):
    # Hosted runners may trust all directories in their system/global config.
    # Isolate those settings so the simulated ownership mismatch is effective.
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "0")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    expected = git_sha(tmp_path)
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    denied = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
    )
    assert denied.returncode != 0
    assert expected and git_sha(tmp_path) == expected
    assert git_sha(tmp_path / "missing") is None


def test_hf_cache_identity_without_inputs_json_roundtrips(tmp_path):
    raw = tmp_path / "raw"
    export = raw / "profile_export_aiperf.json"
    write_json(
        export,
        {
            "metadata": {
                "submission_valid": True,
                "dataset": {"hf_dataset_name": "fixture/traces", "hf_split": "train"},
            }
        },
    )
    cache = tmp_path / "cache"
    data = cache / "dataset/default/1.0/hash"
    write_json(data / "dataset_info.json", {"dataset_name": "traces"})
    (data / "traces-train.arrow").write_bytes(b"actual dataset content")
    identity = capture_hf_dataset(cache, export)
    # Cache lock files do not affect the dataset identity.
    (cache / "unrelated.lock").write_text("lock")
    assert capture_hf_dataset(cache, export) == identity
    write_json(raw / "dataset-identity.json", identity)
    write_json(raw / "server_metrics_export.json", {"fixture": True})
    config = read_json(FIXTURES / "agentic-config.json")
    config["workload"].pop("dataset_revision")
    bundle = tmp_path / "bundle"
    manifest = build_bundle(
        config,
        FIXTURES / "agentic.jsonl",
        bundle,
        record_format="aiperf",
        raw_dir=raw,
        require_full=True,
    )
    assert manifest["status"] == "complete"
    assert (
        read_json(bundle / "config.json")["workload"]["dataset_content_sha256"]
        == identity["dataset_content_sha256"]
    )
    assert (bundle / "raw/aiperf/dataset-identity.json.gz").is_file()
    assert verify_bundle(bundle)["point_id"] == manifest["point_id"]
    assert (
        rebuild_bundle(bundle, tmp_path / "rebuilt")["point_id"] == manifest["point_id"]
    )
    (data / "traces-train.arrow").write_bytes(b"changed dataset content")
    assert (
        capture_hf_dataset(cache, export)["dataset_content_sha256"]
        != identity["dataset_content_sha256"]
    )
    write_json(cache / "other/dataset_info.json", {})
    with pytest.raises(ValueError, match="exactly one dataset"):
        capture_hf_dataset(cache, export)


def test_hf_cache_cannot_claim_identity_without_data(tmp_path):
    export = tmp_path / "export.json"
    write_json(export, {"metadata": {"dataset": {"hf_dataset_name": "fixture/traces"}}})
    cache = tmp_path / "cache"
    write_json(cache / "dataset_info.json", {})
    with pytest.raises(ValueError, match="no Arrow"):
        capture_hf_dataset(cache, export)


def test_partial_cli_reports_missing_identity(tmp_path):
    config = read_json(FIXTURES / "random-config.json")
    config["software"]["atom_sha"] = None
    write_json(tmp_path / "config.json", config)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atom.benchmarks.results",
            "build",
            "--config",
            str(tmp_path / "config.json"),
            "--records",
            str(FIXTURES / "random.jsonl"),
            "--output",
            str(tmp_path / "bundle"),
            "--require-full",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["status"] == "partial"
    assert "missing software.atom_sha" in json.loads(result.stdout)["missing"]
    assert "Bundle incomplete: missing software.atom_sha" in result.stderr
