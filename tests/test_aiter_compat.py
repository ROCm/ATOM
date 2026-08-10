# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import subprocess

import pytest

from atom import aiter_compat


def test_parse_gpu_archs_deduplicates_and_ignores_cpu_agent():
    assert aiter_compat._parse_gpu_archs(
        "gfx000\ngfx950\ngfx950\ngfx942:sramecc+\n"
    ) == ["gfx950", "gfx942"]


def test_detect_gpu_archs_uses_rocm_agent_enumerator(monkeypatch):
    monkeypatch.setattr(
        aiter_compat.shutil,
        "which",
        lambda command: (
            f"/usr/bin/{command}" if command == "rocm_agent_enumerator" else None
        ),
    )
    monkeypatch.setattr(
        aiter_compat.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode=0, stdout="gfx950\ngfx950\n", stderr=""
        ),
    )

    assert aiter_compat.detect_gpu_archs() == "gfx950"


def test_ensure_gpu_archs_env_preserves_explicit_value(monkeypatch):
    monkeypatch.setenv("GPU_ARCHS", "gfx942")
    monkeypatch.setattr(
        aiter_compat,
        "detect_gpu_archs",
        lambda: pytest.fail("explicit GPU_ARCHS must not trigger detection"),
    )

    assert aiter_compat.ensure_gpu_archs_env() == "gfx942"


def test_ensure_gpu_archs_env_sets_detected_value(monkeypatch):
    monkeypatch.delenv("GPU_ARCHS", raising=False)
    monkeypatch.setattr(aiter_compat, "detect_gpu_archs", lambda: "gfx950")

    assert aiter_compat.ensure_gpu_archs_env() == "gfx950"
    assert aiter_compat.os.environ["GPU_ARCHS"] == "gfx950"
