# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the workflow's real Bash steps without a container runtime or GPU."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/atom-sglang-benchmark-gpu-shard.yaml"
)
BASH = shutil.which("bash")


@pytest.fixture
def shell(tmp_path):
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["benchmark"]["steps"]
    scripts = {step["name"]: step["run"] for step in steps if "run" in step}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "id").symlink_to(shutil.which("id"))
    github_env = tmp_path / "github_env"
    github_env.touch()
    calls = tmp_path / "calls"
    calls.touch()
    env = {
        "PATH": str(bin_dir),
        "GITHUB_ENV": str(github_env),
        "RUNNER_NAME": "test-mi308-runner",
        "CALL_LOG": str(calls),
        "CONTAINER_NAME": "atom_sglang_benchmark_mi308_0",
        "SGLANG_IMAGE_SOURCE": "prebuilt-scheduled-nightly",
        "SGLANG_IMAGE_TAG": "example/image:nightly",
    }

    def run(step_name, engines, **overrides):
        for name, status in engines.items():
            executable = bin_dir / name
            executable.write_text(
                f"#!{BASH}\n"
                f'echo "{name} $1" >> "$CALL_LOG"\n'
                f'echo "{name}: diagnostic on stdout"\n'
                f'echo "{name}: diagnostic on stderr" >&2\n'
                f"exit {status}\n"
            )
            executable.chmod(0o755)
        result = subprocess.run(
            [BASH, "--noprofile", "--norc", "-e", "-c", scripts[step_name]],
            env={**env, **overrides},
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result, github_env.read_text(), calls.read_text().splitlines()

    return run


@pytest.mark.parametrize(
    "engines,selected,group,expected_calls",
    [
        ({"podman": 0, "docker": 0}, "podman", "keep-groups", ["podman info"]),
        (
            {"podman": 125, "docker": 0},
            "docker",
            "video",
            ["podman info", "docker info"],
        ),
        ({"docker": 0}, "docker", "video", ["docker info"]),
    ],
)
def test_engine_selection_and_fallback(shell, engines, selected, group, expected_calls):
    result, exported, calls = shell("Detect container engine", engines)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"CONTAINER_ENGINE={selected}\n" in exported
    assert f"CONTAINER_GROUP_ADD=--group-add {group}\n" in exported
    assert calls == expected_calls
    if engines.get("podman") == 125:
        assert "podman: diagnostic on stderr" in result.stderr


@pytest.mark.parametrize("engines", [{}, {"podman": 125, "docker": 1}])
def test_unavailable_engines_fail_with_diagnostics(shell, engines):
    result, exported, _ = shell("Detect container engine", engines)
    assert result.returncode == 1
    assert exported == ""
    assert "test-mi308-runner" in result.stdout
    assert "Neither a working podman nor docker" in result.stdout
    for engine in ("podman", "docker"):
        if engine in engines:
            assert f"{engine}: diagnostic on stdout" in result.stdout
            assert f"{engine}: diagnostic on stderr" in result.stderr
        else:
            assert f"{engine} is not installed or not on PATH" in result.stdout


@pytest.mark.parametrize("overrides", [{}, {"CONTAINER_ENGINE": ""}])
def test_cleanup_without_initialized_engine_is_harmless(shell, overrides):
    result, _, calls = shell("Clean up SGLang benchmark container", {}, **overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Skipping cleanup" in result.stdout
    assert result.stderr == ""
    assert calls == []


@pytest.mark.parametrize("engine", ["podman", "docker"])
@pytest.mark.parametrize("image_source", ["prebuilt-scheduled-nightly", "rebuild"])
def test_cleanup_continues_after_container_errors(shell, engine, image_source):
    result, _, calls = shell(
        "Clean up SGLang benchmark container",
        {engine: 1},
        CONTAINER_ENGINE=engine,
        SGLANG_IMAGE_SOURCE=image_source,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = [f"{engine} exec", f"{engine} stop", f"{engine} rm"]
    if image_source == "rebuild":
        expected.append(f"{engine} rmi")
    assert calls == expected
