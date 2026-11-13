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

from typing import NamedTuple

from atom.models.minimax_m3.mono.config import CP_MAX_Q

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
# so the wide build wins at both and the hybrid is never chosen. The 48-row build (task-mono-nonexact
# results/s0_*, same method, rocprofv3 on, graph replay of B live rows; production = exact graph):
#   production   48 rows 13.71 ms (234.0 us/layer; 13.23 ms in the run above)
#   wide 48      11.31 ms (184.5 us/layer): -17.5% (-14.5% vs 13.23); padded into the 64-row graph the same
#                48 live rows cost 12.15 ms (196.7 us/layer): the 48-row build is -6.9% on top of padding.
# Dead rows cost nothing inside a live 16-row group and ~9 us/layer (single rank) to 12-17 us/layer
# (TP4, 33/40/48 live rows) per wholly dead group, so the builds sit on the group boundaries
# (32, 48, 64) and 24 / 40 / 56 are not built. Re-measure when the kernels change;
# ATOM_MONO_WIDTHS overrides per run.
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


# Indexer context parallelism in the wide builds (``ATOM_M3_INDEXER_CP``, the model-wide switch of ATOM's
# own CP: the index q projection is then full width on every rank). The kernel decides per step from the
# contexts (``index_score.cp_gate``): a long request (> LONG_FROM_BLOCKS blocks) of a graph whose batch is
# at least ``min_batch`` requests and whose long requests together hold at least ``min_blocks`` blocks.
# min_batch: ATOM's ``ATOM_M3_INDEXER_CP_MIN_BATCH`` (rows // max_query_len; 16 for NVFP4 / FP4 index caches,
# 1 for the others). min_blocks: a step takes the CP arm when its long requests hold >= min_blocks * q_len
# blocks in all (q_len: query rows a request). Measured on the merged scorer (TP4 nvfp4, 4 cards, front half a
# layer, CP off vs on, tables byte-equal, 32 and 64 rows, uniform contexts, per-request block counts 2K-215K):
# q1 loses below ~20K blocks in all (one live request +2.5..+3.8 us, 4-8K +6..+13, 16.4K +5.5..+6.2) and wins
# above (26.9K -3.5 / -4.3 us, 33K -8.4, 65K -36, 108K -68); q2 +9..+12 at 8-13K blocks, -6.8 at 33K, -26 at 54K;
# q4 +3.6..+12.5 at 8-20K, -1.9 at 27K, -6.2 at 31K; q8 (8 requests at most: < 16K blocks) +3..+16 everywhere. A
# step without a long request pays +0.2..+0.9 us for the gate whatever the threshold. The break-even grows
# with q_len, so the default 24576 blocks is per query row: q1 from 24.6K (past the ~20K crossing), q2 from 49K,
# q4 from 98K (above the 31K blocks 16 requests of 250K hold: never), q8 never. ATOM's 16384 (its own
# implementation, 16 requests x 128K) is a loss on mono (16.4K blocks: +5.5..+6.2 us).
DEFAULT_CP_MIN_BLOCKS = 24_576
NVFP4_CP_MIN_BATCH = 16


class IndexerCp(NamedTuple):
    """The CP arm of the wide builds: ``enabled`` (the model's full-width index q: ATOM_M3_INDEXER_CP), the
    graph's ``min_batch`` and the step's ``min_blocks``; plain Python."""

    enabled: bool = False
    min_batch: int = 1
    min_blocks: int = DEFAULT_CP_MIN_BLOCKS

    @staticmethod
    def parse(
        enabled: bool,
        nvfp4: bool,
        min_batch: int | None = None,
        min_blocks: str | int | None = None,
    ) -> IndexerCp:
        batch = (
            min_batch if min_batch is not None else (NVFP4_CP_MIN_BATCH if nvfp4 else 1)
        )
        blocks = DEFAULT_CP_MIN_BLOCKS if min_blocks in (None, "") else int(min_blocks)
        if batch < 1 or blocks < 0:
            raise ValueError(
                f"indexer CP: min batch {batch} (>= 1), min blocks {blocks} (>= 0)"
            )
        return IndexerCp(bool(enabled), batch, blocks)

    def options(self, n: int) -> dict:
        """The wide n-row build's options: CP in it, or nothing (the build of today). A graph of n rows holds
        at most n requests: below ``min_batch`` rows CP can never run, so the build has none of it.
        """
        if not self.enabled or n <= SMALL_MAX_ROWS or n < self.min_batch:
            return {}
        return {
            "index_cp": True,
            "cp_min_batch": self.min_batch,
            "cp_min_blocks": self.min_blocks,
        }

    def refuses(self, n: int, q_len: int) -> str | None:
        """Why a wide decode step of n rows of ``q_len`` rows a request cannot run in a CP build, or None:
        said, and the step takes the original path (whose CP arm serves it), never run without CP.
        """
        if self.options(n) and q_len > CP_MAX_Q:
            return (
                f"indexer CP (ATOM_M3_INDEXER_CP): {q_len} query rows a request, the wide builds'"
                f" CP scorer takes at most {CP_MAX_Q}"
            )
        return None
