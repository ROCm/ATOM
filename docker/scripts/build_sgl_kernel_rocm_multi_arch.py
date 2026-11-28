#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""Build sgl_kernel once per AMDGPU arch and load the matching .so at runtime.

The SGLang release image used to build a single common_ops.so from the build
GPU (or from the last entry of GPU_ARCH). A gfx950-only .so then SIGSEGVs on
MI308 (gfx942). This script:

1. Patches the cloned sglang aot tree so setup_rocm.py honors AMDGPU_TARGET
   and load_utils.py selects gfx*/common_ops*.so from gcnArchName.
2. Builds every arch listed in GPU_ARCH with GPUs hidden.
3. Copies each installed common_ops*.so into sgl_kernel/<arch>/.

One image can then run on MI308 and MI355 without a per-machine rebuild.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


SETUP_MARKER = "# ATOM: honor explicit AMDGPU_TARGET for multi-arch image builds"
LOAD_MARKER = "# ATOM: ROCm multi-arch common_ops selection"


def _patch_setup_rocm(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if SETUP_MARKER in text:
        print(f"setup_rocm already patched: {path}")
        return

    old = '''default_target = "gfx942"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")
'''
    new = f'''default_target = "gfx942"
env_target = os.environ.get("AMDGPU_TARGET", "").strip()
amdgpu_target = env_target or default_target

{SETUP_MARKER}
# Detecting the build GPU would collapse a multi-arch image to one target.
if env_target:
    print(f"Using AMDGPU_TARGET from environment: {{amdgpu_target}}")
elif torch.cuda.is_available():
    try:
        amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        print(f"Detected AMDGPU_TARGET={{amdgpu_target}} from build GPU")
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {{e}}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {{amdgpu_target}}")
'''
    if old not in text:
        raise SystemExit(f"setup_rocm.py does not match the expected v0.5.20 block: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched setup_rocm.py: {path}")


def _patch_load_utils(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if LOAD_MARKER in text:
        print(f"load_utils already patched: {path}")
        return

    helper = f'''
def _get_amdgpu_arch():
    """Return the current HIP device arch (gfx942 / gfx950 / ...), or None."""
    {LOAD_MARKER}
    if getattr(torch.version, "hip", None) is None:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception as e:
        logger.debug(f"[sgl_kernel] Failed to read gcnArchName: {{e}}")
        return None


'''
    if "def _get_compute_capability():" not in text:
        raise SystemExit(f"load_utils.py missing _get_compute_capability: {path}")
    if "def _get_amdgpu_arch():" not in text:
        text = text.replace(
            "def _get_compute_capability():",
            helper + "def _get_compute_capability():",
            1,
        )

    old_select = '''    compute_capability = _get_compute_capability()
    logger.debug(
        f"[sgl_kernel] GPU Detection: compute_capability = {compute_capability}"
    )

    # Get the directory where sgl_kernel is installed
    sgl_kernel_dir = Path(__file__).parent
    logger.debug(f"[sgl_kernel] sgl_kernel directory: {sgl_kernel_dir}")

    # Determine which version to load based on GPU architecture
    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"
'''
    new_select = '''    compute_capability = _get_compute_capability()
    amdgpu_arch = _get_amdgpu_arch()
    logger.debug(
        f"[sgl_kernel] GPU Detection: compute_capability = {compute_capability}, "
        f"amdgpu_arch = {amdgpu_arch}"
    )

    # Get the directory where sgl_kernel is installed
    sgl_kernel_dir = Path(__file__).parent
    logger.debug(f"[sgl_kernel] sgl_kernel directory: {sgl_kernel_dir}")

    # Determine which version to load based on GPU architecture.
    # ROCm images ship one common_ops.so per gfx* subdirectory.
    if amdgpu_arch is not None:
        ops_subdir = amdgpu_arch
        variant_name = f"ROCm {amdgpu_arch}"
    elif compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"
'''
    if old_select not in text:
        raise SystemExit(f"load_utils.py does not match the expected select block: {path}")
    path.write_text(text.replace(old_select, new_select, 1), encoding="utf-8")
    print(f"patched load_utils.py: {path}")


def _parse_arches(gpu_arch: str) -> list[str]:
    arches = [part.strip() for part in gpu_arch.split(";") if part.strip()]
    if not arches:
        raise SystemExit(f"GPU_ARCH is empty: {gpu_arch!r}")
    allowed = {"gfx942", "gfx950", "gfx1250"}
    bad = [arch for arch in arches if arch not in allowed]
    if bad:
        raise SystemExit(f"unsupported AMDGPU arch(es) {bad}; allowed={sorted(allowed)}")
    # Preserve order, drop duplicates.
    out: list[str] = []
    for arch in arches:
        if arch not in out:
            out.append(arch)
    return out


def _installed_sgl_kernel_dir(python_bin: str) -> Path:
    code = (
        "import pathlib, sgl_kernel; "
        "print(pathlib.Path(sgl_kernel.__file__).resolve().parent)"
    )
    out = subprocess.check_output([python_bin, "-c", code], text=True).strip()
    return Path(out)


def _copy_common_ops(pkg_dir: Path, arch: str) -> list[Path]:
    matches = sorted(pkg_dir.glob("common_ops*.so"))
    if not matches:
        raise SystemExit(f"no common_ops*.so under {pkg_dir} after {arch} build")
    dest_dir = pkg_dir / arch
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in matches:
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
        print(f"installed {src.name} -> {dest}")
    return copied


def _assert_arch_in_so(so_path: Path, arch: str) -> None:
    # HIP embeds amdgcn-amd-amdhsa--gfxNNN in the code object.
    needle = f"amdgcn-amd-amdhsa--{arch}".encode()
    data = so_path.read_bytes()
    if needle not in data:
        raise SystemExit(f"{so_path} is missing code object for {arch}")
    print(f"verified {so_path.name} contains {needle.decode()}")


def build(aot_dir: Path, python_bin: str, gpu_arch: str) -> None:
    setup_rocm = aot_dir / "setup_rocm.py"
    load_utils = aot_dir / "python" / "sgl_kernel" / "load_utils.py"
    if not setup_rocm.is_file():
        raise SystemExit(f"missing {setup_rocm}")
    if not load_utils.is_file():
        raise SystemExit(f"missing {load_utils}")

    _patch_setup_rocm(setup_rocm)
    _patch_load_utils(load_utils)

    arches = _parse_arches(gpu_arch)
    print(f"building sgl_kernel for arches: {arches}")

    env = os.environ.copy()
    # Hide every GPU so setup_rocm / hipcc cannot collapse the target list.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""

    for arch in arches:
        build_dir = aot_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        cmd = [python_bin, "setup_rocm.py", "install"]
        print(f"==> {' '.join(cmd)} AMDGPU_TARGET={arch}")
        env["AMDGPU_TARGET"] = arch
        subprocess.check_call(cmd, cwd=str(aot_dir), env=env)
        pkg_dir = _installed_sgl_kernel_dir(python_bin)
        copied = _copy_common_ops(pkg_dir, arch)
        for so in copied:
            _assert_arch_in_so(so, arch)

    pkg_dir = _installed_sgl_kernel_dir(python_bin)
    for arch in arches:
        if not any((pkg_dir / arch).glob("common_ops*.so")):
            raise SystemExit(f"missing installed ops for {arch} under {pkg_dir / arch}")
    print(f"sgl_kernel multi-arch install complete under {pkg_dir}")
    for path in sorted(glob.glob(str(pkg_dir / "gfx*" / "common_ops*.so"))):
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aot-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--gpu-arch",
        default=os.environ.get("GPU_ARCH", "gfx942;gfx950"),
        help="Semicolon-separated AMDGPU targets, e.g. gfx942;gfx950",
    )
    args = parser.parse_args()
    build(args.aot_dir.resolve(), args.python, args.gpu_arch)


if __name__ == "__main__":
    main()
