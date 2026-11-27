#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Merge gfx942 GEMM tune overlay into the shared runtime CSV.

AITER lookup key is (gfx, cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle).
MI308X reports cu_num=80 and MI300X reports cu_num=304 (both gfx942), so rows for
different cu_num values coexist in one CSV without conflict.

Only rows matching --target-cu in the overlay replace that cu_num slice; other cu_num
rows in the baseline are preserved unchanged.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

LUMEN = Path(__file__).resolve().parents[2]
AITER_ROOT = LUMEN / "third_party" / "aiter"
DEFAULT_RUNTIME = LUMEN / "examples/dsv4/configs/dsv4_bf16_tuned_gemm_mi308x.csv"
DEFAULT_OVERLAY = LUMEN / "examples/dsv4/.gemm_tune/dsv4_bf16_tuned_gfx942_mi308x.csv"

TAG_KEYS = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
]
TARGET_GFX = "gfx942"


def _tag(row):
    return tuple(row[k] for k in TAG_KEYS)


def _read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def merge(baseline, overlay, output, target_cu):
    target_cu = str(target_cu)
    base_rows = _read_rows(baseline)
    overlay_rows = [
        r
        for r in _read_rows(overlay)
        if r.get("gfx") == TARGET_GFX
        and str(r.get("cu_num")) == target_cu
        and r.get("libtype") != "torch"
    ]

    kept_other_cu = [
        r
        for r in base_rows
        if not (r.get("gfx") == TARGET_GFX and str(r.get("cu_num")) == target_cu)
    ]
    kept_target_cu = [
        r
        for r in base_rows
        if r.get("gfx") == TARGET_GFX
        and str(r.get("cu_num")) == target_cu
        and r.get("libtype") != "torch"
    ]

    out_map = {_tag(r): r for r in kept_other_cu + kept_target_cu}
    replaced = 0
    added = 0
    for r in overlay_rows:
        t = _tag(r)
        if t in out_map:
            replaced += 1
        else:
            added += 1
        out_map[t] = r

    merged = sorted(
        out_map.values(),
        key=lambda r: (int(r["cu_num"]), int(r["M"]), int(r["N"]), int(r["K"])),
    )
    _write_rows(output, merged)
    n_target = sum(1 for r in merged if str(r.get("cu_num")) == target_cu)
    n_other = len(merged) - n_target
    return len(merged), added + replaced, n_target, n_other


def _cu_from_overlay(overlay: Path):
    rows = _read_rows(overlay)
    counts = Counter(str(r.get("cu_num")) for r in rows if r.get("cu_num"))
    if not counts:
        return None
    return int(counts.most_common(1)[0][0])


def _cu_from_rocminfo():
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for block in re.split(r"Agent\s*\d+", out):
        if "Device Type" not in block or "GPU" not in block:
            continue
        m = re.search(r"Compute Unit\s*:\s*(\d+)", block)
        if m:
            return int(m.group(1))
    return None


def resolve_target_cu(explicit, overlay: Path):
    if explicit is not None:
        return explicit
    for key in ("CU_NUM", "TARGET_CU"):
        val = os.environ.get(key, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    if str(AITER_ROOT) not in sys.path:
        sys.path.insert(0, str(AITER_ROOT))
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return int(get_cu_num())
    except (RuntimeError, TypeError, ValueError):
        pass
    inferred = _cu_from_overlay(overlay)
    if inferred is not None:
        return inferred
    rocminfo_cu = _cu_from_rocminfo()
    if rocminfo_cu is not None:
        return rocminfo_cu
    raise SystemExit(
        "cannot detect cu_num for merge (set --target-cu, CU_NUM, or TARGET_CU)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--target-cu",
        type=int,
        default=None,
        help="Merge overlay rows for this cu_num (default: auto via get_cu_num/overlay)",
    )
    args = parser.parse_args()

    target_cu = resolve_target_cu(args.target_cu, args.overlay)
    total, n_overlay, n_target, n_other = merge(
        args.baseline, args.overlay, args.output, target_cu
    )
    print(
        f"wrote {args.output}: {total} rows "
        f"(cu={target_cu}: {n_target} rows from overlay {n_overlay} updates; "
        f"other cu_num preserved: {n_other} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
