"""CI retains its validated nightly wheel without weakening normal installs."""

import importlib.metadata
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts"


def _installer():
    if sys.version_info < (3, 11):
        pytest.importorskip("tomli", reason="CI installer needs tomli on Python 3.10")
    return runpy.run_path(str(SCRIPTS / "install_atom_with_ci_aiter.py"))


def test_install_keeps_nightly_and_resolves_other_dependencies(monkeypatch):
    module = _installer()
    calls = []
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.1.1.dev1+g8c964fa61"
    )
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))
    module["main"]()
    project = module["tomllib"].loads((ROOT / "pyproject.toml").read_text())["project"]
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
    module = _installer()

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("pip ran"))
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        module["main"]()


@pytest.fixture
def source_history(tmp_path):
    source = tmp_path / "source"
    subprocess.run(
        ["git", "init", "-b", "main", str(source)], check=True, capture_output=True
    )
    git = ["git", "-C", str(source)]
    subprocess.run([*git, "config", "user.name", "CI Test"], check=True)
    subprocess.run([*git, "config", "user.email", "ci@example.com"], check=True)
    revisions = []
    for message in ("before output APIs", "output APIs", "newer source"):
        subprocess.run(
            [
                *git,
                "-c",
                "commit.gpgsign=false",
                "commit",
                "--allow-empty",
                "-m",
                message,
            ],
            check=True,
            capture_output=True,
        )
        revisions.append(
            subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        )
    return source, revisions


@pytest.mark.parametrize("revision, accepted", [(0, False), (1, True), (2, True)])
def test_source_requires_minimum_ancestor(
    source_history, monkeypatch, revision, accepted
):
    source, revisions = source_history
    validate = _installer()["validate_aiter_source"]
    monkeypatch.setitem(validate.__globals__, "MINIMUM_AITER_COMMIT", revisions[1])
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", revisions[revision]],
        check=True,
        capture_output=True,
    )
    if accepted:
        validate(source)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            validate(source)


def test_source_validation_deepens_shallow_checkout(
    source_history, tmp_path, monkeypatch
):
    source, revisions = source_history
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "--depth=1", source.as_uri(), str(checkout)],
        check=True,
        capture_output=True,
    )
    git = ["git", "-C", str(checkout)]
    assert (
        subprocess.check_output(
            [*git, "rev-parse", "--is-shallow-repository"], text=True
        ).strip()
        == "true"
    )
    validate = _installer()["validate_aiter_source"]
    monkeypatch.setitem(validate.__globals__, "MINIMUM_AITER_COMMIT", revisions[1])
    validate(checkout)
    assert (
        subprocess.check_output(
            [*git, "rev-parse", "--is-shallow-repository"], text=True
        ).strip()
        == "false"
    )
    assert (
        subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        == revisions[2]
    )


def test_bad_source_stops_before_pip(monkeypatch):
    module = _installer()
    main = module["main"]

    def invalid(source):
        raise subprocess.CalledProcessError(1, "git merge-base")

    monkeypatch.setitem(main.__globals__, "validate_aiter_source", invalid)
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "0.1.1.dev1+g1234567"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("pip ran"))
    with pytest.raises(subprocess.CalledProcessError):
        main(Path("old-source"))


@pytest.mark.parametrize("plugin", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["benchmark", "accuracy-validation"])
def test_source_builders_preserve_validated_aiter(plugin, kind):
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (SCRIPTS.parent / "workflows" / f"atom-{plugin}-{kind}.yaml").read_text()
    )
    builders = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "COPY . /app/ATOM" in step.get("run", "")
    ]
    assert len(builders) == 1
    assert "pip install 'tomli; python_version < \"3.11\"'" in builders[0]
    assert "install_atom_with_ci_aiter.py --aiter-source /app/aiter-test" in builders[0]
    assert "RUN cd /app/ATOM && pip install -e ." not in builders[0]


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
