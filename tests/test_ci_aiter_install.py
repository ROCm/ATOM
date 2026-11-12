"""CI retains its validated nightly wheel without weakening normal installs."""

import importlib.metadata
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"


def test_install_keeps_nightly_and_resolves_other_dependencies(monkeypatch):
    module = runpy.run_path(str(SCRIPTS / "install_atom_with_ci_aiter.py"))
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.1.1.dev1+g8c964fa61"
    )
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))
    module["main"]()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "amd-aiter>=0.1.28" in project["dependencies"]
    assert calls == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *[d for d in project["dependencies"] if not d.startswith("amd-aiter")],
        ],
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(ROOT)],
    ]


def test_install_requires_preinstalled_aiter(monkeypatch):
    module = runpy.run_path(str(SCRIPTS / "install_atom_with_ci_aiter.py"))

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("pip ran"))
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        module["main"]()


@pytest.mark.parametrize(
    "status, curl_status, commit, success",
    [
        ("ahead", 0, "a" * 40, True),
        ("identical", 0, "f" * 40, True),
        ("behind", 0, "a" * 40, False),
        ("diverged", 0, "a" * 40, False),
        ("null", 0, "a" * 40, False),
        ("ahead", 22, "a" * 40, False),
        ("ahead", 0, "", False),
    ],
)
def test_nightly_source_baseline(status, curl_status, commit, success):
    shell = shutil.which("bash") or shutil.which("sh")
    if shell is None:
        pytest.skip("requires bash")
    source = (SCRIPTS / "download_aiter_wheel.sh").read_text()
    function = source.split("verify_minimum_commit() {", 1)[1].split("\n}", 1)[0]
    # Exercise the actual shell guard; stub only the HTTP/JSON tools.
    script = """
set -eu
curl_with_retry() { return "$CURL_STATUS"; }
jq() { printf '%s\\n' "$COMPARE_STATUS"; }
verify_minimum_commit() {
""" + function + '\n}\nverify_minimum_commit "$CANDIDATE"\n'
    result = subprocess.run(
        [shell, "-c", script],
        env={
            **os.environ,
            "AITER_MINIMUM_COMMIT": "f" * 40,
            "CANDIDATE": commit,
            "COMPARE_STATUS": status,
            "CURL_STATUS": str(curl_status),
            "AUTH_HEADER": "test-only",
            "API_URL": "https://api.github.com",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == success, result.stderr


def test_wheel_consumers_validate_before_installing():
    yaml = pytest.importorskip("yaml")
    for name in ("atom-test.yaml", "atomesh-accuracy-validation.yaml"):
        workflow = yaml.safe_load((SCRIPTS.parent / "workflows" / name).read_text())
        download = workflow["jobs"]["download_aiter_wheel"]
        resolve = next(
            step
            for step in download["steps"]
            if step.get("run") == "bash .github/scripts/download_aiter_wheel.sh"
        )
        assert resolve["env"]["AITER_MINIMUM_COMMIT"] == (
            "f1965cd26021c43436ae2bd614846a5977c2636e"
        )
        consumers = [
            job
            for job in workflow["jobs"].values()
            if any(
                "install_atom_with_ci_aiter.py" in step.get("run", "")
                for step in job.get("steps", [])
            )
        ]
        assert consumers
        for job in consumers:
            assert "download_aiter_wheel" in job["needs"]
