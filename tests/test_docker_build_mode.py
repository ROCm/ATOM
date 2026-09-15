import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "select_docker_build_mode.py"
)
SPEC = importlib.util.spec_from_file_location("select_docker_build_mode", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _labels():
    return {
        MODULE.LABEL_SOURCE: "https://github.com/ROCm/ATOM.git",
        MODULE.LABEL_REVISION: "d" * 40,
        MODULE.LABEL_FOUNDATION: "rocm/pytorch:release@sha256:foundation",
        MODULE.LABEL_AITER_SOURCE: "https://github.com/ROCm/aiter.git",
        MODULE.LABEL_AITER_REVISION: "a" * 40,
        MODULE.LABEL_RCCL_SOURCE: "https://github.com/ROCm/rccl.git",
        MODULE.LABEL_RCCL_REVISION: "b" * 40,
    }


def _expected():
    return dict(_labels())


def _choose(**overrides):
    arguments = {
        "requested_mode": "auto",
        "event_name": "workflow_dispatch",
        "base_labels": _labels(),
        "expected_labels": _expected(),
        "base_is_ancestor": True,
        "changed_files": ["atom/model_engine/model_runner.py"],
    }
    arguments.update(overrides)
    return MODULE.choose_build_mode(**arguments)


def test_auto_uses_incremental_for_python_only_change():
    decision = _choose()

    assert decision.mode == "incremental"


@pytest.mark.parametrize(
    "path",
    [
        "docker/atom_release.dockerfile",
        "pyproject.toml",
        "atom/mesh/src/main.rs",
        "atom/entrypoints/atomesh/server.py",
        "atom/model_ops/kernel.cu",
    ],
)
def test_auto_falls_back_to_full_for_build_sensitive_change(path):
    decision = _choose(changed_files=[path])

    assert decision.mode == "full"
    assert path in decision.reason


def test_auto_falls_back_when_metadata_is_missing():
    decision = _choose(base_labels={})

    assert decision.mode == "full"
    assert "missing metadata" in decision.reason


def test_auto_falls_back_when_base_revision_is_not_immutable():
    labels = _labels()
    labels[MODULE.LABEL_REVISION] = "main"

    decision = _choose(base_labels=labels)

    assert decision.mode == "full"
    assert "immutable ATOM commit" in decision.reason


def test_auto_falls_back_when_native_stack_changed():
    labels = _labels()
    labels[MODULE.LABEL_AITER_REVISION] = "c" * 40

    decision = _choose(base_labels=labels)

    assert decision.mode == "full"
    assert MODULE.LABEL_AITER_REVISION in decision.reason


def test_auto_falls_back_when_base_is_not_an_ancestor():
    decision = _choose(base_is_ancestor=False)

    assert decision.mode == "full"
    assert "not an ancestor" in decision.reason


def test_forced_incremental_rejects_unsafe_change():
    with pytest.raises(ValueError, match="forced incremental build is unsafe"):
        _choose(
            requested_mode="incremental",
            changed_files=["docker/atom_release.dockerfile"],
        )


def test_scheduled_release_stays_full():
    decision = _choose(event_name="schedule", changed_files=[])

    assert decision.mode == "full"
    assert "scheduled" in decision.reason


def test_explicit_full_build_wins():
    decision = _choose(requested_mode="full", changed_files=[])

    assert decision.mode == "full"
