# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which path a decode step of n tokens takes: a per-width table.

``mono``        one persistent kernel a layer (front half, router, MoE, shared
                expert, FFN all-reduce): n <= 16 as it always was, plus the
                wide builds (``WIDE_BUILDS``: 32, 48 and 64 rows, the 16-row group boundaries above 16) once accepted;
``hybrid``      mono front half + the production back half (``hybrid.py``): the
                stand-in while a wide build's MoE stages are being built, and a
                fallback entry for a width the full kernel does not beat;
``production``  the original model.

A step is never padded here: the engine pads a batch to its next captured size
(pad rows are not live, ``row_live``), so which n a wide build sees is the
capture sizes' (``--cudagraph-capture-sizes``) choice, made with the table.
Plain Python (no torch, no FlyDSL): testable on a CPU box. The n <= 16 entries are mono unless
the spec says otherwise, so a deployment that sets no switch behaves as before.
A width the table does not name is production.
"""

from __future__ import annotations

MONO, HYBRID, PRODUCTION = "mono", "hybrid", "production"
KINDS = (MONO, HYBRID, PRODUCTION)
SMALL_MAX_ROWS = 16  # the n <= 16 kernels' MFMA B operand
TABLE_MAX_ROWS = 64  # the table's reach; above it every step is production
# widths the wide full-mono kernels are built for (ATOM_MONO_WIDE gates them)
WIDE_BUILDS: tuple[int, ...] = (32, 48, 64)
# rows the hybrid front half serves
HYBRID_MAX_ROWS = 64

# What the table holds for 17..64 when a switch is on and the spec says nothing: the
# measured choice per width (TP4, native dtypes, q=1, ctx 2K, client step time at that
# token count, task-mono-hybrid-int/results/prof_*4/load.json, 2026-11-12, front64
# 49c460dbd + moe64 3c389daab; layer = rocprofv3 period of the layer's kernels):
#   production              32 rows 12.04 ms (200.3 us/layer)   64 rows 14.41 ms (242.4 us/layer)
#   hybrid (mono front + production back)   11.68 (-3%)         15.30 (+6%)   (64: front 71 us + eager back)
#   wide (whole layer, one kernel)           9.26 (-23%, 149.6 us/layer)   13.02 (-9.6%, 213.9 us/layer)
# so the wide build wins at both and the hybrid is never chosen. Re-measure when the
# kernels change; ATOM_MONO_WIDTHS overrides per run.
MEASURED_HYBRID: str | None = None
MEASURED_WIDE: str | None = "32:mono,48:mono,64:mono"


def parse_spec(spec: str | None) -> dict[int, str]:
    """``"17-64:hybrid,32:mono,33-64:production"`` -> {width: kind}; later items win."""
    table: dict[int, str] = {}
    for item in (spec or "").replace(" ", "").split(","):
        if not item:
            continue
        span, _, kind = item.partition(":")
        if kind not in KINDS:
            raise ValueError(f"width spec {item!r}: kind must be one of {KINDS}")
        lo, _, hi = span.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if not 1 <= lo <= hi <= TABLE_MAX_ROWS:
            raise ValueError(f"width spec {item!r}: widths are 1..{TABLE_MAX_ROWS}")
        table.update({n: kind for n in range(lo, hi + 1)})
    return table


class WidthPolicy:
    """``kind(n)`` for every n. ``hybrid``: ATOM_MONO_HYBRID; ``wide``:
    ATOM_MONO_WIDE (the wide mono builds); ``spec``: ATOM_MONO_WIDTHS, over the
    defaults."""

    def __init__(
        self,
        hybrid: bool = False,
        wide: bool = False,
        spec: str | None = None,
        wide_builds: tuple[int, ...] = WIDE_BUILDS,
        hybrid_max_rows: int = HYBRID_MAX_ROWS,
    ) -> None:
        self.hybrid, self.wide = hybrid, wide
        self.wide_builds, self.hybrid_max_rows = wide_builds, hybrid_max_rows
        table = {n: MONO for n in range(1, SMALL_MAX_ROWS + 1)}
        if hybrid:
            table.update(parse_spec(MEASURED_HYBRID))
        if wide:
            table.update(parse_spec(MEASURED_WIDE))
        table.update(parse_spec(spec))
        self._table = table

    def kind(self, n: int) -> str:
        want = self._table.get(n, PRODUCTION)
        if want == MONO and n > SMALL_MAX_ROWS:
            return MONO if self.wide and n in self.wide_builds else PRODUCTION
        if want == HYBRID:
            ok = self.hybrid and SMALL_MAX_ROWS < n <= self.hybrid_max_rows
            return HYBRID if ok else PRODUCTION
        return want

    def describe(self) -> str:
        """The table as runs of widths, for the startup log."""
        runs: list[list] = []
        for n in range(1, TABLE_MAX_ROWS + 1):
            k = self.kind(n)
            if runs and runs[-1][2] == k:
                runs[-1][1] = n
            else:
                runs.append([n, n, k])
        runs.append([TABLE_MAX_ROWS + 1, "up", PRODUCTION])
        return ", ".join(
            f"{lo}-{hi}:{k}" if hi != lo else f"{lo}:{k}" for lo, hi, k in runs
        )
