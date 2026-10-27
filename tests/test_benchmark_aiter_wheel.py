"""CPU checks for optional, pinned benchmark AITER wheels."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_benchmark_aiter_wheel import (
    prepare_wheel,
    validate_request,
    wheel_metadata,
)

WHEEL = "amd_aiter-0.1.9+test-cp312-cp312-linux_x86_64.whl"
URL = "https://example.com/" + WHEEL


def wheel(path, name="amd-aiter"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "amd_aiter-0.1.9.dist-info/METADATA", f"Name: {name}\nVersion: 0.1.9+test\n"
        )


@pytest.mark.parametrize(
    "value",
    [
        "0.1.9",
        "http://example.com/" + WHEEL,
        "https://example.com/other.whl",
        "artifact:0",
        "artifact:bad",
        "https://user:secret@example.com/" + WHEEL,
    ],
)
def test_reject_ambiguous_or_invalid_wheel_requests(value):
    with pytest.raises(ValueError, match="aiter_wheel must be"):
        validate_request(value)


def test_blank_request_does_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("unexpected download")
    )
    assert prepare_wheel("", tmp_path / "wheels", tmp_path / "config") is None
    assert not list(tmp_path.iterdir())


def test_wheel_and_source_ref_conflict_before_matrix():
    from build_agentic_benchmark_matrix import build_configs

    with pytest.raises(ValueError, match="only one"):
        build_configs(inputs={"aiter_wheel": "latest", "aiter_commit": "main"})


@pytest.mark.parametrize("selector", ["latest", URL, "artifact:1234"])
def test_download_once_and_pin_replay_identity(tmp_path, monkeypatch, selector):
    output, config = tmp_path / "wheels", tmp_path / "config"
    config.mkdir()
    for name in ("run-config.json", "dispatch-inputs.json"):
        (config / name).write_text('{"profile":"nightly"}')
    calls = []

    def download(args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["check"] is True
        if args[0] == "curl":
            wheel(Path(args[args.index("--output") + 1]))
        else:
            env = kwargs["env"]
            wheel(Path(env["AITER_WHEEL_OUTPUT_DIR"]) / WHEEL)
            if selector == "latest":
                assert env["AITER_WHEEL_DOWNLOAD_MODE"] == "resolve"
                values = "aiter_wheel_url=" + URL + "\naiter_artifact_id=\n"
            else:
                assert env["AITER_WORKFLOW_ARTIFACT_ID"] == "1234"
                assert env["AITER_WORKFLOW_ARTIFACT_REPO"] == "ROCm/aiter"
                values = "aiter_wheel_url=\naiter_workflow_artifact_id=1234\n"
            Path(env["GITHUB_OUTPUT"]).write_text(values)

    monkeypatch.setattr(subprocess, "run", download)
    record = prepare_wheel(selector, output, config)
    assert len(calls) == 1
    assert record["version"] == "0.1.9+test"
    assert record["sha256"] == hashlib.sha256((output / WHEEL).read_bytes()).hexdigest()
    assert record["pinned"] == (
        "artifact:1234" if selector.startswith("artifact:") else validate_request(URL)
    )
    assert (
        json.loads((config / "dispatch-inputs.json").read_text())["aiter_wheel"]
        == record["pinned"]
    )
    assert (
        json.loads((config / "run-config.json").read_text())["aiter_wheel_resolution"]
        == record
    )
    assert (
        json.loads((config / "dispatch-inputs.json").read_text())["profile"]
        == "nightly"
    )
    subprocess.check_call(["sha256sum", "--check", "SHA256SUMS"], cwd=output)


def test_download_failure_never_records_a_pin(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(22, "curl")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        prepare_wheel(URL, tmp_path / "wheels", tmp_path / "config")
    assert not (tmp_path / "wheels/selection.json").exists()


def test_wrong_distribution_is_not_installable(tmp_path):
    path = tmp_path / WHEEL
    wheel(path, "unexpected-package")
    with pytest.raises(ValueError, match="not amd-aiter"):
        wheel_metadata(path)


def test_download_retry_preserves_failure_status(tmp_path):
    command = tmp_path / "curl"
    command.write_text("#!/bin/sh\nexit 22\n")
    command.chmod(0o755)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "download_aiter_wheel.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_TOKEN": "test-only",
            "ATOM_PYTHON_TAG": "cp312",
            "AITER_WHEEL_DOWNLOAD_MODE": "workflow_artifact",
            "AITER_WORKFLOW_ARTIFACT_REPO": "ROCm/aiter",
            "AITER_WORKFLOW_ARTIFACT_ID": "1234",
            "AITER_WHEEL_DOWNLOAD_MAX_ATTEMPTS": "1",
            "AITER_WHEEL_OUTPUT_DIR": str(tmp_path / "output"),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 22
    assert "Command failed after 1 attempts" in result.stderr


def test_copy_failure_stops_before_container_install(tmp_path):
    wheel(tmp_path / WHEEL)
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$CALLS"\nexit 17\n')
    docker.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "install_aiter_wheel.sh")],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CONTAINER_NAME": "test",
            "AITER_WHL_DIR": str(tmp_path),
            "CALLS": str(calls),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 17
    assert calls.read_text().splitlines() == ["cp"]


def test_agentic_workflow_guards_optional_download_and_shares_one_artifact():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (SCRIPTS.parent / "workflows/atom-agentic-benchmark.yaml").read_text()
    )
    dispatch = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
    assert dispatch["aiter_wheel"]["default"] == ""
    build = workflow["jobs"]["build-matrix"]
    steps = {s.get("name"): s for s in build["steps"]}
    for name in (
        "Download requested AITER wheel once",
        "Share pinned AITER wheel with all points",
    ):
        condition = steps[name]["if"]
        assert (
            "workflow_dispatch" in condition
            and "inputs.aiter_wheel != ''" in condition
            and "inputs.dry_run != true" in condition
        )
    assert (
        workflow["jobs"]["benchmark"]["with"]["aiter_wheel_artifact_id"]
        == "${{ needs.build-matrix.outputs.aiter_wheel_artifact_id }}"
    )
