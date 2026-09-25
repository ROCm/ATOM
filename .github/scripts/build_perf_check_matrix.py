#!/usr/bin/env python3
"""Compute the cell matrix for the PR perf check (``atom-perf-check.yaml``).

Sibling of ``build_benchmark_matrix.py``, which does the same job for the
nightly. Both read one catalog (``.github/benchmark/models.json``) through
``catalog.py``, so "what a model is" -- server args, env vars, concurrency
band -- has a single definition and cannot drift between the two lines.

Emits ``cells`` (a JSON list, one entry per GPU job) to ``$GITHUB_OUTPUT``.

Reads from the environment:
    PERF_MODELS            catalog prefixes, comma separated
    PERF_JUDGING_CONCS     levels the criterion reads
    PERF_REFERENCE_CONCS   levels measured and shown but never judged
    PERF_ISL / PERF_OSL / PERF_RATIO    the single scenario this line runs

This lives here rather than inline in the workflow for the reason
``build_benchmark_matrix.py`` gives for its own existence: workflow YAML is
not testable, and a matrix that silently builds too few cells is a failure
that only surfaces much later, at the judge, as `partial`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import build_cells, load_variants

CATALOG = ".github/benchmark/models.json"

# The check measures the BASE variant of each selected model, plus one paired
# variant: DeepSeek-V4-Pro MTP3 alongside its base. The pair is what separates
# a regression in the speculative-decode path from one in the model's general
# path -- only MTP3 moves, or both do.
#
# Every other catalog variant is deliberately absent: DPA, TBO, DPA TBO,
# DPA MTP3, DPA DSpark, DSpark, EAGLE3, MegaMoE. They are separate code paths
# and this line does not watch them; the nightly does. Adding one is a line
# here plus the GPU jobs it costs, not a change to the criterion.
PAIRED_VARIANTS = {"deepseek-v4-pro": "-mtp3"}


def selected_variants(prefixes: set[str]) -> set[tuple[str, str]]:
    """The (prefix, suffix) pairs this line measures, for the given models."""
    keep = {(p, "") for p in prefixes}
    for prefix, suffix in PAIRED_VARIANTS.items():
        if prefix in prefixes:
            keep.add((prefix, suffix))
    return keep


def check_against_catalog(
    path: str, prefixes: set[str], keep: set[tuple[str, str]]
) -> list[str]:
    """Name what the catalog does not have, before any GPU time is spent.

    A prefix that does not exist builds no cells at all, and a paired variant
    that has been renamed silently drops half of a pair -- the half whose
    absence the base/MTP3 comparison exists to detect.
    """
    variants = load_variants(path)
    known_prefixes = {v["prefix"] for v in variants}
    known_pairs = {(v["prefix"], v["suffix"]) for v in variants}
    problems = []
    for p in sorted(prefixes - known_prefixes):
        problems.append(f"PERF_MODELS names {p!r}, which is not a catalog prefix")
    for prefix, suffix in sorted(keep - known_pairs):
        if prefix in known_prefixes:
            problems.append(
                f"{prefix!r} has no variant with suffix {suffix!r} in the catalog"
            )
    return problems


def concurrency_levels() -> list[str]:
    """Judging levels first, then reference; order is display order."""
    raw = (
        os.environ["PERF_JUDGING_CONCS"]
        + ","
        + os.environ.get("PERF_REFERENCE_CONCS", "")
    )
    return [c for c in (x.strip() for x in raw.split(",")) if c]


def _emit(cells: list[dict]) -> None:
    payload = json.dumps(cells)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"cells={payload}\n")
    else:
        print(payload)


def main() -> int:
    concs = concurrency_levels()
    isl, osl = os.environ["PERF_ISL"], os.environ["PERF_OSL"]
    ratio = os.environ["PERF_RATIO"]
    param_lists = ";".join(f"{isl},{osl},{c},{ratio}" for c in concs)

    prefixes = {p.strip() for p in os.environ["PERF_MODELS"].split(",") if p.strip()}
    keep = selected_variants(prefixes)

    problems = check_against_catalog(CATALOG, prefixes, keep)
    if problems:
        for p in problems:
            print(f"::error::{p}", file=sys.stderr)
        return 1

    cells = build_cells(CATALOG, param_lists=param_lists, model_filter=prefixes)
    cells = [c for c in cells if (c["prefix"], c["suffix"]) in keep]

    # Fail here rather than warn. A variant whose concurrency band excludes one
    # of these levels contributes fewer cells; every GPU job still runs and the
    # shortfall only surfaces at the judge, as `partial`, which reads like a job
    # died rather than like a configuration that never could have been measured.
    expected = len(concs) * len(keep)
    if len(cells) != expected:
        built = {(c["prefix"] + c["suffix"], c["conc"]) for c in cells}
        missing = sorted(
            f"{prefix}{suffix} c={conc}"
            for prefix, suffix in keep
            for conc in concs
            if (prefix + suffix, int(conc)) not in built
        )
        print(
            f"::error::expected {expected} cells ({len(keep)} entries x "
            f"{len(concs)} levels), built {len(cells)}; missing: "
            f"{', '.join(missing) or 'none'} -- check the catalog's conc_min/"
            f"conc_max for these variants",
            file=sys.stderr,
        )
        return 1

    _emit(cells)
    print(
        f"{len(cells)} cells: {len(keep)} entries x {len(concs)} levels "
        f"({isl}/{osl}, judged at {os.environ['PERF_JUDGING_CONCS']})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
