# SPDX-License-Identifier: MIT
"""Installed wheel identities must survive strict benchmark packaging."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import pytest

from atom.benchmarks.results import metadata
from atom.benchmarks.results.bundle import rebuild_bundle, verify_bundle
from atom.benchmarks.results.io import read_json, write_json

FIXTURES = Path(__file__).parent / "fixtures"
VERSION = "0.1.1.dev1+ga75ba53de"
DIGEST = "26ea7ce84b0b9e6e22235da060b91a5b59d121a228b5b4a58fed88d5ba867a8a"


@pytest.fixture
def installed_aiter(tmp_path, monkeypatch):
    directory = tmp_path / "amd_aiter.dist-info"
    directory.mkdir()
    original = importlib.metadata.distribution

    def install(version=VERSION, direct=None):
        (directory / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: amd-aiter\nVersion: {version}\n"
        )
        if direct is not None:
            write_json(directory / "direct_url.json", direct)
        distribution = importlib.metadata.PathDistribution(directory)
        monkeypatch.setattr(
            importlib.metadata,
            "distribution",
            lambda name: distribution if name == "amd-aiter" else original(name),
        )
        return distribution

    return install


@pytest.mark.parametrize(
    "archive,expected_hash",
    [
        ({"hashes": {"sha256": DIGEST}}, DIGEST),
        ({"hash": f"sha256={DIGEST}"}, DIGEST),
        ({"hashes": {"sha256": "invalid"}}, None),
        ({}, None),
    ],
)
def test_wheel_identity_comes_from_installed_distribution(
    installed_aiter, monkeypatch, archive, expected_hash
):
    installed_aiter(
        direct={
            "url": "file:///tmp/amd_aiter-0.1.1.dev1%2Bga75ba53de-cp312-cp312-linux_x86_64.whl",
            "archive_info": archive,
        }
    )
    monkeypatch.setattr(metadata, "git_sha", lambda path: pytest.fail(str(path)))
    identity = metadata.installed_package("amd-aiter")
    assert identity == {
        "version": VERSION,
        "sha": "a75ba53de",
        "sha_source": "package_version",
        "wheel_sha256": expected_hash,
    }


def test_index_wheel_without_direct_url_has_version_commit(installed_aiter):
    installed_aiter()
    identity = metadata.installed_package("amd-aiter")
    assert identity["sha"] == "a75ba53de"
    assert identity["wheel_sha256"] is None


@pytest.mark.parametrize("source", ["vcs", "local"])
def test_git_install_preserves_full_commit(
    installed_aiter, tmp_path, monkeypatch, source
):
    full_sha = "a75ba53de" + "1" * 31
    direct = (
        {"url": "https://github.com/ROCm/aiter", "vcs_info": {"commit_id": full_sha}}
        if source == "vcs"
        else {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    )
    installed_aiter(direct=direct)
    monkeypatch.setattr(metadata, "git_sha", lambda path: full_sha)
    identity = metadata.installed_package("amd-aiter")
    assert identity["sha"] == full_sha
    assert identity["sha_source"] == (
        "direct_url_vcs" if source == "vcs" else "local_git"
    )


@pytest.mark.parametrize(
    "version", ["0.1.1", "0.1.1+g123", "0.1.1+ga75ba53de.d20260925"]
)
def test_unknown_commit_does_not_borrow_checkout_sha(
    installed_aiter, monkeypatch, version
):
    installed_aiter(version=version)
    monkeypatch.setattr(metadata, "devices", list)
    monkeypatch.setattr(metadata, "hip_pci_ids", list)
    monkeypatch.setattr(metadata, "git_sha", lambda path: "f" * 40)
    software = metadata.capture_config("model", [], "random", 1)["software"]
    assert software["aiter_sha"] is None
    assert software["aiter_sha_source"] is None
    assert software["aiter_version"] == version


@pytest.mark.parametrize("client_exit", [0, 1])
def test_wheel_config_strict_cli_and_rebuild(
    installed_aiter, tmp_path, monkeypatch, client_exit
):
    installed_aiter(
        direct={
            "url": "https://example.com/amd_aiter.whl",
            "archive_info": {"hashes": {"sha256": DIGEST}},
        }
    )
    monkeypatch.setattr(metadata, "devices", list)
    monkeypatch.setattr(metadata, "hip_pci_ids", list)
    captured = metadata.capture_config("model", [], "agentic", 1)["software"]
    config = read_json(FIXTURES / "agentic-config.json")
    config["software"].update(
        {key: value for key, value in captured.items() if key.startswith("aiter_")}
    )
    write_json(tmp_path / "config.json", config)
    raw = tmp_path / "raw"
    write_json(raw / "profile_export_aiperf.json", {"submission_valid": True})
    write_json(raw / "server_metrics_export.json", {"fixture": True})
    bundle = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "atom.benchmarks.results",
            "build",
            "--config",
            str(tmp_path / "config.json"),
            "--records",
            str(FIXTURES / "agentic.jsonl"),
            "--record-format",
            "aiperf",
            "--raw-dir",
            str(raw),
            "--output",
            str(bundle),
            "--require-full",
            "--exit-code",
            str(client_exit),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == client_exit, result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["status"] == ("complete" if client_exit == 0 else "failed")
    assert "missing software.aiter_sha" not in outcome["missing"]
    assert read_json(bundle / "validation.json")["exit_code"] == client_exit
    assert verify_bundle(bundle)["point_id"] == outcome["point_id"]
    rebuilt = tmp_path / "rebuilt"
    assert rebuild_bundle(bundle, rebuilt)["point_id"] == outcome["point_id"]
    assert (
        read_json(rebuilt / "config.json")["software"]["aiter_wheel_sha256"] == DIGEST
    )
