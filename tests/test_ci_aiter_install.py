"""CI installs its dependencies while leaving AITER separately managed."""

import importlib.metadata
import runpy
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


@pytest.mark.parametrize("aiter_installed", [True, False])
@pytest.mark.parametrize("extras", [(), ("tracing", "structured-output")])
def test_install_leaves_aiter_managed_separately(
    monkeypatch, capsys, aiter_installed, extras
):
    module = _installer()
    calls = []

    def version(name):
        assert name == "amd-aiter"
        if not aiter_installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return "0.1.1.dev1+g8c964fa61"

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", lambda args, **kw: calls.append(args))
    module["main"](extras)
    assert len(calls) == 2
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert not any(arg.startswith("amd-aiter") for arg in calls[0])
    assert "pybind11" in calls[0]
    assert ("llguidance==1.9.1" in calls[0]) == bool(extras)
    assert ("opentelemetry-api>=1.40,<2" in calls[0]) == bool(extras)
    assert calls[1] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-e",
        str(ROOT),
    ]
    assert capsys.readouterr().out == (
        "Keeping CI amd-aiter 0.1.1.dev1+g8c964fa61\n" if aiter_installed else ""
    )
