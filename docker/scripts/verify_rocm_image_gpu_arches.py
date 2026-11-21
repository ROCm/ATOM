#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""Fail the image build if the serving stack cannot cover every required GPU arch.

ATOM ships one image for MI308 (gfx942) and MI355 (gfx950). The base image
builds Torch device packages, RCCL, and AITER with GPU_ARCH=gfx942;gfx950.
The SGLang layer must install one sgl_kernel common_ops.so per arch. This gate
scans installed HIP binaries so a single-arch regression cannot ship again.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import sysconfig
from collections import defaultdict
from pathlib import Path


ARCH_RE = re.compile(rb"amdgcn-amd-amdhsa--(gfx[0-9a-z]+)")

# Serving-critical components. Coverage is the UNION of arches across matching
# files (AITER keeps many per-arch objects). Missing an arch in the union fails
# the build.
COVERAGE_MARKERS = (
    ("aiter", ("/aiter", "amd_aiter", "amd-aiter")),
    ("mori", ("/mori", "amd_mori", "amd-mori")),
    ("atom", ("/atom/", "/atom-", "site-packages/atom")),
)


def _parse_arches(value: str) -> list[str]:
    arches = [part.strip() for part in value.split(";") if part.strip()]
    if not arches:
        raise SystemExit(f"empty arch list: {value!r}")
    return list(dict.fromkeys(arches))


def _looks_native(path: Path) -> bool:
    name = path.name
    if name.endswith((".py", ".pyc", ".pyi", ".txt", ".md", ".json", ".h", ".hpp", ".cuh", ".cmake")):
        return False
    if any(tok in name for tok in (".so", ".hsaco", ".co", ".hipfb")):
        return True
    return False


def _arches_in_file(path: Path) -> set[str]:
    try:
        size = path.stat().st_size
    except OSError:
        return set()
    if size > 400 * 1024 * 1024:
        return set()
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    return {m.group(1).decode() for m in ARCH_RE.finditer(data)}


def _norm(path: Path) -> str:
    return str(path).lower().replace("\\", "/")


def _collect(roots: list[Path]) -> dict[Path, set[str]]:
    found: dict[Path, set[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not _looks_native(path):
                continue
            arches = _arches_in_file(path)
            if arches:
                found[path] = arches
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--required-arches",
        default=os.environ.get("GPU_ARCH")
        or os.environ.get("GPU_ARCH_LIST")
        or "gfx942;gfx950",
    )
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument(
        "--require-sgl-kernel-subdirs",
        action="store_true",
        help="Require sgl_kernel/<arch>/common_ops*.so for every required arch",
    )
    parser.add_argument(
        "--strict-coverage",
        action="store_true",
        help="Fail when a coverage marker has HIP objects but is missing an arch",
    )
    args = parser.parse_args()
    required = _parse_arches(args.required_arches)

    purelib = Path(sysconfig.get_path("purelib"))
    roots = [purelib, Path("/app/aiter-test"), Path("/app/ATOM"), Path("/opt/venv")]
    roots.extend(Path(p) for p in args.root)

    print(f"required arches: {required}")
    print(f"scan roots: {[str(r) for r in roots if r.exists()]}")
    found = _collect([r for r in roots if r.exists()])
    print(f"native files with HIP code objects: {len(found)}")

    failures: list[str] = []
    warnings: list[str] = []

    # Per-component arch coverage (union).
    for name, markers in COVERAGE_MARKERS:
        union: set[str] = set()
        hits = 0
        for path, arches in found.items():
            n = _norm(path)
            if any(m in n for m in markers):
                union |= arches
                hits += 1
        print(f"coverage {name}: files={hits} arches={sorted(union)}")
        if hits == 0:
            warnings.append(f"{name}: no HIP objects found under scan roots")
            continue
        missing = [arch for arch in required if arch not in union]
        if missing:
            msg = f"{name}: HIP objects present but missing arches {missing} (have {sorted(union)})"
            if args.strict_coverage:
                failures.append(msg)
            else:
                warnings.append(msg)

    if args.require_sgl_kernel_subdirs:
        try:
            import sgl_kernel  # type: ignore

            pkg = Path(sgl_kernel.__file__).resolve().parent
        except Exception as exc:  # pragma: no cover
            failures.append(f"cannot import sgl_kernel: {exc}")
            pkg = None
        if pkg is not None:
            for arch in required:
                matches = sorted((pkg / arch).glob("common_ops*.so"))
                if not matches:
                    failures.append(f"missing {pkg / arch}/common_ops*.so")
                    continue
                for so in matches:
                    arches = _arches_in_file(so)
                    if arch not in arches:
                        failures.append(
                            f"{so}: expected code object for {arch}, found {sorted(arches)}"
                        )
                    else:
                        print(f"ok sgl_kernel/{arch}/{so.name}")

    if warnings:
        print(f"warnings: {len(warnings)}")
        for line in warnings:
            print(f"  warn: {line}")

    if failures:
        print("MULTI-ARCH GATE FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("multi-arch gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
