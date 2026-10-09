# SPDX-License-Identifier: MIT
"""The wheel-pin CLI validates every input before changing any Dockerfile."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/bump_lmcache_wheel_pin.py"
)
OLD_IMAGE = "rocm/atom-dev:lmcache-v0.1-g11111111-rocm-torch210@sha256:" + "a" * 64
NEW_IMAGE = "rocm/atom-dev:lmcache-v0.2-g22222222-rocm-torch210@sha256:" + "b" * 64
OLD_SHA = "c" * 64
NEW_SHA = "d" * 64


def dockerfiles(root):
    paths = [root / "docker/Dockerfile", root / "docker/atom_release.dockerfile"]
    paths[0].parent.mkdir()
    for path in paths:
        path.write_text(
            f'# {path.name}\nARG LMCACHE_WHEEL_IMAGE="{OLD_IMAGE}"\n'
            f"FROM scratch\nARG LMCACHE_WHEEL_SHA256={OLD_SHA}\n"
            "# Unrelated content must survive.\n"
        )
    return paths


def run_bump(root, *files, image=NEW_IMAGE, sha256=NEW_SHA):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--image", image, "--sha256", sha256, *files],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("key", ["LMCACHE_WHEEL_IMAGE", "LMCACHE_WHEEL_SHA256"])
@pytest.mark.parametrize("copies", [0, 2], ids=["missing", "duplicate"])
def test_later_layout_error_preserves_both_files(tmp_path, key, copies):
    paths = dockerfiles(tmp_path)
    lines = paths[1].read_text().splitlines(keepends=True)
    paths[1].write_text(
        "".join(
            line * copies if line.startswith(f"ARG {key}=") else line for line in lines
        )
    )
    before = [path.read_bytes() for path in paths]

    result = run_bump(tmp_path)

    assert result.returncode == 1
    assert f"expected one 'ARG {key}=', found {copies}" in result.stdout
    assert [path.read_bytes() for path in paths] == before
    assert ": pinned " not in result.stdout


def test_missing_later_file_preserves_the_first_file(tmp_path):
    first, second = dockerfiles(tmp_path)
    before = first.read_bytes()
    second.unlink()

    result = run_bump(tmp_path)

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert first.read_bytes() == before
    assert not second.exists()
    assert ": pinned " not in result.stdout


@pytest.mark.parametrize("explicit_files", [False, True], ids=["defaults", "explicit"])
def test_valid_bump_changes_only_pins_and_is_idempotent(tmp_path, explicit_files):
    paths = dockerfiles(tmp_path)
    before = [path.read_text() for path in paths]
    files = [str(path) for path in reversed(paths)] if explicit_files else []
    expected = [
        text.replace(OLD_IMAGE, NEW_IMAGE).replace(OLD_SHA, NEW_SHA) for text in before
    ]

    for _ in range(2):
        result = run_bump(tmp_path, *files)
        assert result.returncode == 0, result.stdout + result.stderr
        assert [path.read_text() for path in paths] == expected
        assert result.stdout.count(": pinned ") == 2


@pytest.mark.parametrize(
    "invalid", [{"image": "rocm/atom-dev:latest"}, {"sha256": "not-a-digest"}]
)
def test_invalid_pin_arguments_preserve_both_files(tmp_path, invalid):
    paths = dockerfiles(tmp_path)
    before = [path.read_bytes() for path in paths]

    result = run_bump(tmp_path, **invalid)

    assert result.returncode == 2
    assert [path.read_bytes() for path in paths] == before
    assert ": pinned " not in result.stdout
