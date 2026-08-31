#!/usr/bin/env python3
"""Per-request interactivity metrics for ATOMesh agentic (AIPerf) runs.

Mirrors ``compute_p90_e2e_normalized_interactivity.py`` from
seungrokj/agentx_skills, extended to also produce the plain definition, so the
ATOMesh dashboard can offer the same two x-axis choices InferenceX publishes:

    itl_i = ITL_i                          # seconds per output token, decode only
    eff_i = ITL_i + TTFT_i / OSL_i         # effective seconds per output token

    Interactivity                = 1 / p90(itl)
    E2E Normalized Interactivity = 1 / p90(eff)

Percentile the latency first, then invert -- the same convention as
``p90_intvty = 1/p90_itl``, which is what InferenceX's own API returns. The
normalized variant amortizes prefill over the turn's output tokens; because eff
is a nonlinear function of three correlated per-request quantities, it cannot be
rebuilt from the aggregate columns of ``profile_export_aiperf.json``; it has to
be computed row by row from ``profile_export.jsonl``. Both percentiles are taken
over the same filtered record set, so the two numbers describe the same requests.

Stdlib only, on purpose: this is imported by ``process_result.py``, which runs on
``ubuntu-latest`` where numpy is not guaranteed to be installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Values for the ``interactivity_method`` field carried on every dashboard point,
# so a chart can tell which definition produced a given number.
# No constant for the plain 1/p90(ITL) metric: it is only ever written from
# per-request records, so the presence of ``interactivity_p90_itl`` on a point is
# its own definition and no point ever names it in ``interactivity_method``.
METHOD_P90_E2E = "p90_e2e_normalized"
METHOD_MEDIAN_TPOT = "median_tpot"

RECORDS_FILENAME = "profile_export.jsonl"
DEFAULT_PERCENTILE = 90.0

# AIPerf metric keys. inter_token_latency -- NOT
# full_response_inter_token_latency, which measures something else and yields a
# materially different number.
TPOT_KEY = "inter_token_latency"
TTFT_KEY = "time_to_first_token"
OSL_KEY = "output_sequence_length"

PROFILING_PHASE = "profiling"


def percentile_linear(values: Iterable[float], percentile: float) -> float:
    """numpy's default ``np.percentile(..., method="linear")``, without numpy.

    Interpolates between the two ranks straddling ``(n - 1) * percentile / 100``.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _metric(record: dict[str, Any], key: str) -> float | None:
    """Read ``record["metrics"][key]["value"]``, tolerating a bare number."""
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    entry = metrics.get(key)
    if isinstance(entry, dict):
        entry = entry.get("value")
    if isinstance(entry, bool) or not isinstance(entry, (int, float)):
        return None
    return float(entry)


def iter_records(path: Path) -> Iterator[tuple[dict[str, Any] | None, str]]:
    """Yield ``(record, raw_line)`` per non-blank line; record is None if unparseable.

    Streamed rather than read whole: a long agentic run writes one line per
    request and these files reach hundreds of MB.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield None, line
                continue
            yield (record if isinstance(record, dict) else None), line


def per_request_latencies(
    path: Path, include_warmup: bool = False
) -> tuple[list[tuple[float, float]], int]:
    """Per-request ``(itl_i, eff_i)`` in seconds, plus the count of unparseable lines.

    Keeps only successful profiling records with a positive output length -- the
    same filter the reference implementation applies, which drops AIPerf's warmup
    requests and anything cancelled during grace-period draining. One filter
    serves both metrics on purpose: they are plotted as alternative x-axes for
    the same point, so they have to describe the same set of requests.
    """
    latencies: list[tuple[float, float]] = []
    skipped_lines = 0
    for record, _raw in iter_records(path):
        if record is None:
            skipped_lines += 1
            continue
        if record.get("error"):
            continue
        if not include_warmup:
            metadata = record.get("metadata")
            phase = (
                metadata.get("benchmark_phase") if isinstance(metadata, dict) else None
            )
            if phase != PROFILING_PHASE:
                continue
        tpot_ms = _metric(record, TPOT_KEY)
        ttft_ms = _metric(record, TTFT_KEY)
        osl = _metric(record, OSL_KEY)
        if tpot_ms is None or ttft_ms is None or not osl or osl <= 0:
            continue
        itl_s = tpot_ms / 1000.0
        latencies.append((itl_s, itl_s + (ttft_ms / 1000.0) / osl))
    return latencies, skipped_lines


def agentic_interactivity(
    jsonl_path: Path | str,
    percentile: float = DEFAULT_PERCENTILE,
    include_warmup: bool = False,
) -> dict[str, Any]:
    """Compute both interactivity definitions for one ``profile_export.jsonl``.

    Raises ValueError when no record survives the filter, rather than returning a
    number derived from nothing.
    """
    path = Path(jsonl_path)
    latencies, skipped_lines = per_request_latencies(
        path, include_warmup=include_warmup
    )
    if not latencies:
        raise ValueError(f"no valid profiling records in {path}")
    itl_latency_s = percentile_linear([itl for itl, _eff in latencies], percentile)
    effective_latency_s = percentile_linear(
        [eff for _itl, eff in latencies], percentile
    )
    if itl_latency_s <= 0 or effective_latency_s <= 0:
        raise ValueError(f"non-positive p{percentile} latency in {path}")
    return {
        "path": str(path),
        # E2E normalized interactivity. Keyed "value" rather than something more
        # descriptive because it is the dashboard's primary axis and callers have
        # read it under that name since #1925.
        "value": 1.0 / effective_latency_s,
        "effective_latency_s": effective_latency_s,
        # Plain interactivity -- InferenceX's p90_intvty.
        "itl_value": 1.0 / itl_latency_s,
        "itl_latency_s": itl_latency_s,
        "n_requests": len(latencies),
        "percentile": percentile,
        "skipped_lines": skipped_lines,
    }


def locate_records(
    result_path: Path,
    model: str | None = None,
    topology: str | None = None,
    concurrency: int | None = None,
    artifact_dir: str | None = None,
) -> Path | None:
    """Find the ``profile_export.jsonl`` belonging to one ``pd-*.json`` result.

    ``pd_server_atom.sh:run_aiperf_agentic_benchmark`` writes the dashboard JSON
    to ``<results>/pd-...json`` and the AIPerf artifacts to the sibling
    ``<results>/aiperf-<model>-<topology>-c<conc>/``, so the lookup walks from
    most to least specific.
    """
    parent = result_path.parent
    candidates: list[Path] = []
    if artifact_dir:
        candidates.append(parent / Path(artifact_dir).name / RECORDS_FILENAME)
    if model and topology and concurrency is not None:
        candidates.append(
            parent / f"aiperf-{model}-{topology}-c{concurrency}" / RECORDS_FILENAME
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if concurrency is not None:
        matches = sorted(parent.rglob(f"aiperf-*-c{concurrency}/{RECORDS_FILENAME}"))
        if topology:
            narrowed = [p for p in matches if topology in p.parent.name]
            if narrowed:
                matches = narrowed
        if len(matches) == 1:
            return matches[0]
    return None


def _find_jsonl(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    hits = sorted(path.rglob(RECORDS_FILENAME))
    if not hits:
        sys.exit(f"ERROR: no {RECORDS_FILENAME} found under {path}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", help=f"{RECORDS_FILENAME} file, or a directory to search recursively"
    )
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = [
        agentic_interactivity(path, args.percentile, args.include_warmup)
        for path in _find_jsonl(Path(args.path))
    ]

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
        return

    label = (
        f"p{int(args.percentile) if args.percentile.is_integer() else args.percentile}"
    )
    print(
        f"| n | {label}_itl (s/token) | {label}_interactivity "
        f"| {label}_eff_latency (s/token) | {label}_e2e_normalized_interactivity "
        f"| source |"
    )
    print("|---|---|---|---|---|---|")
    for result in results:
        print(
            f"| {result['n_requests']} | {result['itl_latency_s']:.6f} "
            f"| {result['itl_value']:.3f} | {result['effective_latency_s']:.6f} "
            f"| {result['value']:.3f} | {result['path']} |"
        )


if __name__ == "__main__":
    main()
