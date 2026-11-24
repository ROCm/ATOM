# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Shared utilities for Lumen benchmarks.

Environment variables that control benchmark behaviour:

    LUMEN_BENCH_WARMUP   – override default warmup iterations  (default: per-call)
    LUMEN_BENCH_ITERS    – override default timing iterations   (default: per-call)
    LUMEN_BENCH_TRIM_PCT – percentage of outliers to trim from both tails
                           e.g. 10 trims the lowest 10 % and highest 10 %
                           (default: 0, no trimming)
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import statistics
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

import torch

TRACE_DIR = os.path.join(os.path.dirname(__file__), "traces")

# ---------------------------------------------------------------------------
# Hardware guards
# ---------------------------------------------------------------------------


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm GPU is required for this benchmark")


def require_aiter():
    try:
        import aiter  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("AITER is required for this benchmark") from exc


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Container for a single benchmark measurement."""

    name: str
    avg_ms: float
    min_ms: float = 0.0
    max_ms: float = 0.0
    std_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    cv_pct: float = 0.0
    iters: int = 0
    extra: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        s = f"{self.name:>45s}  avg={self.avg_ms:8.3f} ms"
        if self.min_ms > 0:
            s += f"  min={self.min_ms:8.3f} ms  max={self.max_ms:8.3f} ms"
        if self.std_ms > 0:
            s += f"  std={self.std_ms:7.3f} ms  cv={self.cv_pct:5.1f}%"
        for k, v in self.extra.items():
            if isinstance(v, float):
                s += f"  {k}={v:.4f}"
            else:
                s += f"  {k}={v}"
        return s


def _env_int(name: str, default: int | None) -> int | None:
    val = os.environ.get(name)
    if val is not None:
        return int(val)
    return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is not None:
        return float(val)
    return default


def _trim(times: list[float], pct: float) -> list[float]:
    """Remove *pct* % of samples from both tails."""
    if pct <= 0 or len(times) < 4:
        return times
    n = max(1, int(len(times) * pct / 100))
    s = sorted(times)
    return s[n : len(s) - n]


def _build_result(label: str, times: list[float], raw_iters: int) -> BenchResult:
    """Build a :class:`BenchResult` from a list of per-iteration times."""
    avg = statistics.mean(times)
    med = statistics.median(times)
    std = statistics.stdev(times) if len(times) >= 2 else 0.0
    cv = (std / avg * 100) if avg > 0 else 0.0
    s = sorted(times)
    p95_idx = min(math.ceil(0.95 * len(s)) - 1, len(s) - 1)
    return BenchResult(
        name=label or "unnamed",
        avg_ms=avg,
        min_ms=s[0],
        max_ms=s[-1],
        std_ms=std,
        median_ms=med,
        p95_ms=s[p95_idx],
        cv_pct=cv,
        iters=raw_iters,
    )


def cuda_timer(
    fn: Callable[[], object],
    warmup: int = 5,
    iters: int = 20,
    label: str = "",
    sync_before: bool = True,
    dist_barrier: bool = False,
    trim_pct: float | None = None,
) -> BenchResult:
    """Time *fn* on GPU using CUDA events.

    Args:
        warmup: Warmup iterations (overridden by ``LUMEN_BENCH_WARMUP``).
        iters: Timed iterations (overridden by ``LUMEN_BENCH_ITERS``).
        dist_barrier: If True, call ``dist.barrier()`` before each iteration
            to align all ranks.  Essential for benchmarks that include
            collective operations (allgather, reduce_scatter, etc.).
        trim_pct: Percentage of outliers to trim from both tails before
            computing statistics.  Overridden by ``LUMEN_BENCH_TRIM_PCT``.

    Returns a :class:`BenchResult` with avg / min / max / std / median / p95.
    """
    warmup = _env_int("LUMEN_BENCH_WARMUP", warmup) or warmup
    iters = _env_int("LUMEN_BENCH_ITERS", iters) or iters
    trim = _env_float("LUMEN_BENCH_TRIM_PCT", trim_pct if trim_pct is not None else 0.0)

    if dist_barrier:
        import torch.distributed as dist

    for _ in range(warmup):
        if dist_barrier:
            dist.barrier()
        fn()
    if sync_before:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        if dist_barrier:
            dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    trimmed = _trim(times, trim)
    return _build_result(label, trimmed, iters)


def cuda_timer_batch(
    fn: Callable[[], object],
    warmup: int = 5,
    iters: int = 20,
    label: str = "",
) -> BenchResult:
    """Time *fn* over *iters* invocations with a single pair of CUDA events.

    Useful when per-iteration event overhead matters (very fast kernels).
    """
    warmup = _env_int("LUMEN_BENCH_WARMUP", warmup) or warmup
    iters = _env_int("LUMEN_BENCH_ITERS", iters) or iters

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    avg = total_ms / iters
    return BenchResult(
        name=label or "unnamed",
        avg_ms=avg,
        iters=iters,
    )


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def track_cuda_memory(device: torch.device | None = None):
    """Context manager that yields a dict updated with peak memory stats."""
    dev = device or torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    mem_before = torch.cuda.memory_allocated(dev)
    info: dict[str, int] = {}
    yield info
    torch.cuda.synchronize(dev)
    info["peak_bytes"] = torch.cuda.max_memory_allocated(dev)
    info["allocated_before"] = mem_before
    info["allocated_after"] = torch.cuda.memory_allocated(dev)
    info["peak_delta"] = info["peak_bytes"] - mem_before


def format_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _stability_tag(cv: float) -> str:
    if cv <= 2.0:
        return ""
    if cv <= 5.0:
        return "  [~unstable]"
    return "  [!NOISY]"


def print_report(title: str, results: list[BenchResult]) -> None:
    """Pretty-print a table of benchmark results."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    for r in results:
        tag = _stability_tag(r.cv_pct) if r.cv_pct > 0 else ""
        print(f"  {r}{tag}")
    print(sep)
    print()


def print_report_with_table(title: str, results: list[BenchResult]) -> None:
    """Print both the detailed report and a summary table."""
    print_report(title, results)
    print_table(title, results)


def print_overlap_summary(
    *,
    t_compute: float,
    t_comm: float,
    t_seq: float,
    t_ovl: float,
    compute_label: str = "compute",
    comm_label: str = "comm",
) -> None:
    """Print a visual overlap analysis after print_report().

    Displays a bar chart showing how compute and comm contribute to total
    time, plus key metrics: hidden time, overlap efficiency, and speedup.
    """
    hidden_ms = t_seq - t_ovl
    speedup = t_seq / max(t_ovl, 1e-6)
    overlap_ratio = 1 - (t_ovl / max(t_compute + t_comm, 1e-6))
    pct_hidden = hidden_ms / max(t_compute, 1e-6) * 100

    # Visual bar: scale to 40 chars
    bar_width = 40
    scale = bar_width / max(t_seq, 1e-6)

    comp_bar = int(t_compute * scale)
    comm_bar = int(t_comm * scale)
    ovl_bar = int(t_ovl * scale)

    print(f"  ┌── Sequential ({'─' * (bar_width - 14)})┐")
    print(
        f"  │ {'█' * comp_bar}{'░' * (bar_width - comp_bar)} │ {compute_label}: {t_compute:.3f} ms"
    )
    print(
        f"  │ {'▓' * comm_bar}{'░' * (bar_width - comm_bar)} │ {comm_label}: {t_comm:.3f} ms"
    )
    print(f"  │ total = {t_seq:.3f} ms{' ' * max(bar_width - 20, 1)}│")
    print(f"  ├── Overlapped ({'─' * (bar_width - 14)})┤")
    print(f"  │ {'▓' * ovl_bar}{'░' * (bar_width - ovl_bar)} │ {t_ovl:.3f} ms")
    print(f"  └{'─' * bar_width}─┘")
    print()
    print(
        f"  Hidden {compute_label}: {hidden_ms:.3f} / {t_compute:.3f} ms "
        f"({pct_hidden:.0f}% of {compute_label} hidden)"
    )
    print(f"  Overlap ratio:  {overlap_ratio:.3f}")
    print(f"  Speedup:        {speedup:.2f}x")
    print()


def print_table(title: str, results: list[BenchResult]) -> None:
    """Print a summary table of all results, suitable for pasting into docs.

    Output looks like::

        ┌─────────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬────────┬──────────┐
        │ Name                        │ Avg (ms) │ Min (ms) │ Med (ms) │ P95 (ms) │ Max (ms) │ CV (%) │ Extras   │
        ├─────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┼──────────┤
        │ allgather alone             │    0.621 │    0.580 │    0.605 │    0.700 │    0.710 │    5.2 │          │
        └─────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┴──────────┘
    """
    if not results:
        return

    def _fmt_extras(r: BenchResult) -> str:
        parts = []
        for k, v in r.extra.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.2f}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    name_w = max(len(r.name) for r in results)
    name_w = max(name_w, 4) + 2
    extras = [_fmt_extras(r) for r in results]
    extra_w = max((len(e) for e in extras), default=0)
    extra_w = max(extra_w, 6) + 2

    num_w = 10

    def _col(s: str, w: int) -> str:
        return f" {s:<{w - 1}}"

    def _num(v: float, w: int = num_w) -> str:
        return f" {v:>{w - 2}.3f} "

    def _pct(v: float, w: int = 8) -> str:
        return f" {v:>{w - 2}.1f} "

    sep_parts = [
        "─" * name_w,
        "─" * num_w,
        "─" * num_w,
        "─" * num_w,
        "─" * num_w,
        "─" * num_w,
        "─" * 8,
        "─" * extra_w,
    ]

    top = "┌" + "┬".join(sep_parts) + "┐"
    mid = "├" + "┼".join(sep_parts) + "┤"
    bot = "└" + "┴".join(sep_parts) + "┘"

    header = (
        "│"
        + _col("Name", name_w)
        + "│"
        + _col("Avg(ms)", num_w)
        + "│"
        + _col("Min(ms)", num_w)
        + "│"
        + _col("Med(ms)", num_w)
        + "│"
        + _col("P95(ms)", num_w)
        + "│"
        + _col("Max(ms)", num_w)
        + "│"
        + _col("CV(%)", 8)
        + "│"
        + _col("Extras", extra_w)
        + "│"
    )

    print(f"\n  {title}")
    print(f"  {top}")
    print(f"  {header}")
    print(f"  {mid}")
    for r, ext in zip(results, extras):
        tag = " *" if r.cv_pct > 5 else " ~" if r.cv_pct > 2 else ""
        row = (
            "│"
            + _col(r.name, name_w)
            + "│"
            + _num(r.avg_ms)
            + "│"
            + _num(r.min_ms)
            + "│"
            + _num(r.median_ms)
            + "│"
            + _num(r.p95_ms)
            + "│"
            + _num(r.max_ms)
            + "│"
            + _pct(r.cv_pct)
            + "│"
            + _col(ext + tag, extra_w)
            + "│"
        )
        print(f"  {row}")
    print(f"  {bot}")
    print()


def dump_csv(results: list[BenchResult], path: str) -> None:
    """Write results to a CSV file for spreadsheet import."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        extra_keys: list[str] = []
        for r in results:
            for k in r.extra:
                if k not in extra_keys:
                    extra_keys.append(k)
        writer.writerow(
            [
                "name",
                "avg_ms",
                "min_ms",
                "median_ms",
                "p95_ms",
                "max_ms",
                "std_ms",
                "cv_pct",
                "iters",
            ]
            + extra_keys
        )
        for r in results:
            row = [
                r.name,
                f"{r.avg_ms:.4f}",
                f"{r.min_ms:.4f}",
                f"{r.median_ms:.4f}",
                f"{r.p95_ms:.4f}",
                f"{r.max_ms:.4f}",
                f"{r.std_ms:.4f}",
                f"{r.cv_pct:.2f}",
                str(r.iters),
            ]
            for k in extra_keys:
                v = r.extra.get(k, "")
                row.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            writer.writerow(row)


def dump_json(results: list[BenchResult], path: str) -> None:
    """Write results to a JSON file for CI integration."""
    data = []
    for r in results:
        d = {
            "name": r.name,
            "avg_ms": round(r.avg_ms, 4),
            "min_ms": round(r.min_ms, 4),
            "max_ms": round(r.max_ms, 4),
            "std_ms": round(r.std_ms, 4),
            "median_ms": round(r.median_ms, 4),
            "p95_ms": round(r.p95_ms, 4),
            "cv_pct": round(r.cv_pct, 2),
            "iters": r.iters,
        }
        d.update(r.extra)
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Tracing helpers (torch.profiler → Chrome / Perfetto JSON)
# ---------------------------------------------------------------------------


def _ensure_trace_dir(trace_path: str) -> None:
    d = os.path.dirname(trace_path)
    if d:
        os.makedirs(d, exist_ok=True)


def trace_fn(
    fn: Callable[[], object],
    trace_path: str,
    warmup: int = 3,
    active: int = 1,
    label: str = "",
    record_shapes: bool = True,
    with_stack: bool = False,
    with_modules: bool = False,
) -> str:
    """Run *fn* under ``torch.profiler`` and export a Chrome trace JSON.

    Args:
        fn: Zero-arg callable to profile.
        trace_path: Output ``.json`` path (directories created automatically).
        warmup: Profiler warmup iterations (not recorded).
        active: Iterations recorded in the trace.
        label: Optional label printed in the summary header.
        record_shapes: Record tensor shapes in the trace.
        with_stack: Capture Python call stacks (slower but more detail).
        with_modules: Record ``nn.Module`` hierarchy.

    Returns:
        The absolute path to the exported trace file.
    """
    _ensure_trace_dir(trace_path)

    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_modules=with_modules,
    ) as prof:
        for _ in range(active):
            fn()
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)

    header = label or os.path.basename(trace_path)
    print(f"\n--- Trace: {header} ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    size = os.path.getsize(trace_path)
    print(f"  Saved: {trace_path} ({format_bytes(size)})")
    return os.path.abspath(trace_path)


@contextlib.contextmanager
def trace_context(
    trace_path: str,
    record_shapes: bool = True,
    with_stack: bool = False,
    label: str = "",
) -> Generator[torch.profiler.profile, None, None]:
    """Context manager that wraps a code block in ``torch.profiler``.

    Use this instead of :func:`trace_fn` when the profiled code is not a
    single callable (e.g. multi-stream overlap benchmarks).

    .. code-block:: python

        with trace_context("traces/my_trace.json") as prof:
            # ... code to profile ...
        # trace JSON written automatically on exit

    Yields:
        The ``torch.profiler.profile`` object (rarely needed directly).
    """
    _ensure_trace_dir(trace_path)

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        with_stack=with_stack,
    )
    prof.__enter__()
    try:
        yield prof
    finally:
        prof.__exit__(None, None, None)
        prof.export_chrome_trace(trace_path)
        header = label or os.path.basename(trace_path)
        print(f"\n--- Trace: {header} ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        size = os.path.getsize(trace_path)
        print(f"  Saved: {trace_path} ({format_bytes(size)})")


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------


def compute_bandwidth_gb_s(bytes_transferred: int, time_ms: float) -> float:
    """Return effective bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    return bytes_transferred / (time_ms * 1e-3) / 1e9


def print_bandwidth_summary(
    *,
    label: str,
    bytes_transferred: int,
    time_ms: float,
) -> None:
    """Print bandwidth utilization for a communication operation."""
    bw = compute_bandwidth_gb_s(bytes_transferred, time_ms)
    print(
        f"  {label}: {bw:.1f} GB/s "
        f"({bytes_transferred / 1e6:.1f} MB in {time_ms:.3f} ms)"
    )


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def print_bench_warnings(
    *,
    result: BenchResult,
    overlap_ratio: float | None = None,
    speedup: float | None = None,
) -> None:
    """Print diagnostic warnings when benchmark results look suspicious."""
    warnings = []
    if result.cv_pct > 15:
        warnings.append(
            f"CV={result.cv_pct:.1f}% — results are noisy, "
            "consider more iterations or check for background load"
        )
    if result.p95_ms > 2 * result.median_ms and result.median_ms > 0:
        warnings.append(
            f"p95={result.p95_ms:.3f}ms >> median={result.median_ms:.3f}ms "
            "— high tail latency, possible GC or throttling"
        )
    if overlap_ratio is not None and overlap_ratio < 0:
        warnings.append(
            f"overlap ratio {overlap_ratio:.1%} is negative — "
            "fused path is slower than sequential, check for contention"
        )
    elif overlap_ratio is not None and overlap_ratio < 0.1:
        warnings.append(
            f"overlap ratio {overlap_ratio:.1%} < 10% — "
            "fusion is not hiding communication effectively"
        )
    if speedup is not None and speedup < 1.0:
        warnings.append(
            f"speedup {speedup:.2f}x < 1.0 — " "optimized path is slower than baseline"
        )
    for w in warnings:
        print(f"  \u26a0 WARNING: {w}")
