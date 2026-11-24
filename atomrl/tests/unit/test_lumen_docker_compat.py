# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Behavioral tests for the Lumen CI container-launch wrapper."""

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "scripts"
    / "lumen-docker-compat.py"
)
_SPEC = importlib.util.spec_from_file_location("lumen_docker_compat", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_ci_image_source_removes_only_source_mounts(monkeypatch):
    monkeypatch.setenv("LUMEN_CI_USE_IMAGE_SOURCE", "1")
    args = [
        "run",
        "-v",
        "/checkout:/workspace/ATOM",
        "--volume=/checkout/atomrl/lumen:/workspace/Lumen:ro",
        "-v",
        "/data:/data:ro",
        "image",
    ]

    assert _MODULE.strip_source_mounts(args) == [
        "run",
        "-v",
        "/data:/data:ro",
        "image",
    ]


def test_non_ci_launch_preserves_source_mounts(monkeypatch):
    monkeypatch.delenv("LUMEN_CI_USE_IMAGE_SOURCE", raising=False)
    args = ["run", "-v", "/checkout:/workspace/ATOM", "image"]
    assert _MODULE.strip_source_mounts(args) == args
