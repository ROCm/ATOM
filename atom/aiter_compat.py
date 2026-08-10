# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility helpers for AITER across ROCm and PyTorch releases."""

import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger("atom")

_ARCH_PATTERN = re.compile(r"\bgfx[0-9a-f]+\b", re.IGNORECASE)


def _parse_gpu_archs(output: str) -> list[str]:
    """Return unique, valid GPU architecture names in discovery order."""
    archs: list[str] = []
    for match in _ARCH_PATTERN.finditer(output):
        arch = match.group(0).lower()
        if arch == "gfx000" or arch in archs:
            continue
        archs.append(arch)
    return archs


def detect_gpu_archs() -> str | None:
    """Detect runtime GPU targets without assuming a system ROCm layout."""
    for command in ("rocm_agent_enumerator", "amdgpu-arch"):
        executable = shutil.which(command)
        if executable is None:
            continue
        try:
            result = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        archs = _parse_gpu_archs(result.stdout)
        if archs:
            return ";".join(archs)
    return None


def ensure_gpu_archs_env() -> str | None:
    """Populate ``GPU_ARCHS`` before AITER snapshots it during import.

    Python-managed ROCm distributions may provide ``rocm_agent_enumerator`` but
    omit ``amdgpu-arch`` and the traditional ``/opt/rocm`` tree. AITER's C++ JIT
    otherwise records an empty architecture and fails when compiling its first
    runtime template.
    """
    configured = os.environ.get("GPU_ARCHS", "").strip()
    if configured:
        return configured

    detected = detect_gpu_archs()
    if detected is not None:
        os.environ["GPU_ARCHS"] = detected
        logger.info("Detected GPU_ARCHS=%s for AITER runtime JIT", detected)
    return detected
