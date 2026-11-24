# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class CleanBuildPy(_build_py):
    """Remove stale package files before copying the current source tree."""

    def run(self):
        package_root = Path(self.build_lib) / "atomrl"
        if package_root.exists():
            shutil.rmtree(package_root)
        super().run()


def lumen_hip_extensions():
    """Build Lumen HIP extensions only when explicitly requested."""
    if os.environ.get("LUMEN_BUILD_EXTENSIONS") != "1":
        return [], None

    hip_roots = [
        os.environ.get("HIP_HOME"),
        os.environ.get("ROCM_HOME"),
        "/opt/rocm",
    ]
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        hipcc = next(
            (
                str(Path(root) / "bin" / "hipcc")
                for root in hip_roots
                if root and (Path(root) / "bin" / "hipcc").is_file()
            ),
            None,
        )
    if hipcc is None:
        raise RuntimeError(
            "LUMEN_BUILD_EXTENSIONS=1 requires hipcc from HIP_HOME, "
            "ROCM_HOME, /opt/rocm, or PATH"
        )

    try:
        import torch
        from torch.utils.cpp_extension import HIP_HOME, BuildExtension, CppExtension
    except ImportError as error:
        raise RuntimeError(
            "LUMEN_BUILD_EXTENSIONS=1 requires ROCm PyTorch with "
            "torch.utils.cpp_extension"
        ) from error

    if not getattr(torch.version, "hip", None) or not HIP_HOME:
        raise RuntimeError(
            "LUMEN_BUILD_EXTENSIONS=1 requires a ROCm PyTorch build and HIP_HOME"
        )
    torch_hipcc = Path(HIP_HOME) / "bin" / "hipcc"
    if not torch_hipcc.is_file() and not Path(hipcc).is_file():
        raise RuntimeError(
            "LUMEN_BUILD_EXTENSIONS=1 could not find hipcc under Torch HIP_HOME"
        )

    csrc = Path("atomrl") / "lumen" / "csrc"
    extensions = [
        CppExtension(
            name="atomrl.lumen.csrc._fused_quant_transpose",
            sources=[str(csrc / "fused_quant_transpose.cu")],
            extra_compile_args=["-O3"],
        ),
        CppExtension(
            name="atomrl.lumen.csrc._fp8_quant_dispatch",
            sources=[str(csrc / "fp8_quant_dispatch.cu")],
            extra_compile_args=["-O3"],
        ),
    ]
    return extensions, BuildExtension


ext_modules, build_ext = lumen_hip_extensions()
cmdclass = {"build_py": CleanBuildPy}
if build_ext is not None:
    cmdclass["build_ext"] = build_ext

setup(ext_modules=ext_modules, cmdclass=cmdclass)
