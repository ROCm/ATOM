"""Export Prometheus latency data as a self-contained interactive HTML report.

Example: python .github/scripts/atomesh/observability/export_report.py --prometheus-url
http://127.0.0.1:9090 --start 2026-09-08T08:16:20Z --end 2026-09-08T08:17:28Z
--output results/latency-report.html
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import json
import math
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

STATISTICS = {"mean": None, "p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}


def timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise argparse.ArgumentTypeError("ISO timestamps must include a timezone")
        return parsed.timestamp()


def panels_for(deployment: str) -> list[dict]:
    api_ttft = "atom:time_to_first_token_seconds"
    itl = "atom:inter_token_latency_seconds"
    if deployment == "standalone":
        return [
            {
                "id": "api_ttft",
                "title": "Overall TTFT",
                "label": "API · STANDALONE",
                "detail": "API request arrival → first generated streaming output",
                "metric": api_ttft,
                "selector": 'job="atom",role="standalone",streaming="true"',
            },
            {
                "id": "decode_itl",
                "title": "Inter-token latency",
                "label": "API · STANDALONE",
                "detail": "Output intervals normalized and weighted by new token count",
                "metric": itl,
                "selector": 'job="atom",role="standalone"',
            },
        ]
    return [
        {
            "id": "mesh_ttft",
            "title": "Overall TTFT",
            "label": "MESH · INGRESS",
            "detail": "Mesh request arrival → first generated streaming output",
            "metric": "mesh_router_ttft_seconds",
            "selector": 'job="atom-mesh",router_type="http",backend_type="pd"',
        },
        {
            "id": "decode_itl",
            "title": "Inter-token latency",
            "label": "DECODE · OUTPUT",
            "detail": "Output intervals normalized and weighted by new token count",
            "metric": itl,
            "selector": 'job="atom",role="decode"',
        },
        {
            "id": "prefill_ttft",
            "title": "Prefill TTFT",
            "label": "PREFILL · LOCAL",
            "detail": "Prefill request arrival → first internal token delivery · non-streaming",
            "metric": api_ttft,
            "selector": 'job="atom",role="prefill",streaming="false"',
        },
        {
            "id": "decode_ttft",
            "title": "Decode TTFT",
            "label": "DECODE · LOCAL",
            "detail": "Decode request arrival → first generated output · streaming",
            "metric": api_ttft,
            "selector": 'job="atom",role="decode",streaming="true"',
        },
    ]


def query_for(panel: dict, statistic: str, window: int) -> str:
    metric, selector = panel["metric"], panel["selector"]

    def rate(suffix):
        return f"rate({metric}_{suffix}{{{selector}}}[{window}s])"

    if statistic == "mean":
        return f"1000 * sum({rate('sum')}) / sum({rate('count')})"
    return f"1000 * histogram_quantile({STATISTICS[statistic]}, sum by (le) ({rate('bucket')}))"


def fetch_series(url: str, query: str, start: float, end: float, step: int) -> list:
    args = urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "step": step}
    )
    request = urllib.request.Request(url.rstrip("/") + "/api/v1/query_range?" + args)
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("status") != "success":
        raise RuntimeError(result.get("error", "Prometheus query failed"))
    series = result["data"]["result"]
    if len(series) > 1:
        raise ValueError("Expected one aggregated series per panel statistic")
    points = series[0]["values"] if series else []
    return [
        [float(t), float(v) if math.isfinite(float(v)) else None] for t, v in points
    ]


def collect(args, *, diagnostics: list[str] | None = None) -> dict:
    panels = panels_for(args.deployment)
    errors = [] if diagnostics is None else diagnostics
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pending = {}
        for panel in panels:
            panel["series"], panel["queries"] = {}, {}
            for statistic in STATISTICS:
                query = query_for(panel, statistic, args.window)
                panel["queries"][statistic] = query
                future = pool.submit(
                    fetch_series,
                    args.prometheus_url,
                    query,
                    args.start,
                    args.end,
                    args.step,
                )
                pending[future] = panel, statistic
        for future in concurrent.futures.as_completed(pending):
            panel, statistic = pending[future]
            try:
                panel["series"][statistic] = future.result()
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                failed += 1
                panel["series"][statistic] = []
                errors.append(f"{panel['title']} / {statistic}: {exc}")
    if failed == len(pending):
        raise RuntimeError("All Prometheus queries failed: " + errors[-failed])
    return {
        "meta": {
            "title": args.title,
            "model": args.model,
            "start": args.start,
            "end": args.end,
            "step": args.step,
            "window": args.window,
            "source": args.prometheus_url,
            "kind": "prometheus",
            "exported_at": time.time(),
            "notes": errors if diagnostics is None else [],
        },
        "panels": panels,
    }


def demo_data() -> dict:
    """Explicitly labelled synthetic data for reviewing the report layout."""
    rng = random.Random(1729)
    start, step, count = 1788854400, 5, 121
    panels = panels_for("pd")
    for index, panel in enumerate(panels):
        base = [420, 6.8, 290, 76][index]
        points = []
        for i in range(count):
            wave = 1 + 0.12 * math.sin(i / 10 + index) + 0.04 * rng.random()
            bump = 0.36 * math.exp(-(((i - 76) / 7) ** 2))
            points.append(base * (wave + bump))
        panel["series"] = {
            stat: [
                [start + i * step, round(value * factor, 3)]
                for i, value in enumerate(points)
            ]
            for stat, factor in zip(STATISTICS, (1.0, 0.89, 1.16, 1.28, 1.53))
        }
    return {
        "meta": {
            "title": "Inference latency report",
            "model": "GLM-5.2 · CPP4 + DCP4",
            "start": start,
            "end": start + step * (count - 1),
            "step": step,
            "window": 60,
            "kind": "demo",
            "source": "Layout preview · synthetic data",
            "exported_at": time.time(),
            "notes": [],
        },
        "panels": panels,
    }


def validate_data(data: dict) -> None:
    """Validate the JSON interface: Unix seconds for time, milliseconds for values."""

    def number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        raise TypeError("Report data requires a meta object")
    meta = data["meta"]
    for field in ("start", "end", "step", "window"):
        if not number(meta.get(field)):
            raise ValueError(f"meta.{field} must be a finite number")
    if meta["end"] <= meta["start"] or meta["step"] <= 0 or meta["window"] <= 0:
        raise ValueError("Require start < end, step > 0 and window > 0")
    if not isinstance(data.get("panels"), list) or not data["panels"]:
        raise ValueError("Report data requires at least one panel")
    identifiers = set()
    for panel in data["panels"]:
        if not isinstance(panel, dict):
            raise TypeError("Each panel must be an object")
        for key in ("id", "title"):
            if not isinstance(panel.get(key), str) or not panel[key]:
                raise ValueError(f"Each panel requires a nonempty {key}")
        if panel["id"] in identifiers:
            raise ValueError("Panel identifiers must be unique")
        identifiers.add(panel["id"])
        if not isinstance(panel.get("series"), dict):
            raise TypeError("Each panel requires a series object")
        for statistic, points in panel["series"].items():
            if statistic not in STATISTICS or not isinstance(points, list):
                raise ValueError("Series must map mean/p50/p90/p95/p99 to arrays")
            previous = -math.inf
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(
                        "Each point must be [unix_seconds, milliseconds_or_null]"
                    )
                t, value = point
                if not number(t) or t <= previous:
                    raise ValueError(
                        "Point timestamps must be finite and strictly increasing"
                    )
                if value is not None and (not number(value) or value < 0):
                    raise ValueError("Latency must be nonnegative milliseconds or null")
                previous = t


def write_report(data: dict, output: str | Path) -> None:
    """Public JSON-to-HTML API. The report embeds data and needs no server or CDN.

    Required meta fields: start/end (Unix seconds), step/window (seconds).
    Each panel supplies id, title and series; series keys are mean/p50/p90/p95/p99,
    and values are arrays of [Unix timestamp, milliseconds or None] pairs.
    Set meta.kind='demo' only for explicitly synthetic preview data.
    """
    validate_data(data)
    data = copy.deepcopy(data)
    data["meta"].setdefault("kind", "recorded")
    for panel in data["panels"]:
        panel.setdefault("label", "RECORDED DATA")
        panel.setdefault("detail", "")
    output = Path(output)
    template = Path(__file__).with_name("report.html").read_text()
    payload = (
        json.dumps(data, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")


def collect_report(
    prometheus_url: str,
    start: float,
    end: float,
    *,
    deployment: str = "pd",
    step: int = 5,
    window: int = 60,
    title: str = "Inference latency report",
    model: str = "ATOM",
    diagnostics: list[str] | None = None,
) -> dict:
    """Fetch report data without publishing files.

    Callers managing run status can own query errors through ``diagnostics``;
    otherwise they are included in the returned report notes.
    """
    if (
        deployment not in {"pd", "standalone"}
        or start >= end
        or step <= 0
        or window <= 0
    ):
        raise ValueError("Invalid deployment or time range")
    return collect(
        SimpleNamespace(
            prometheus_url=prometheus_url,
            start=start,
            end=end,
            deployment=deployment,
            step=step,
            window=window,
            title=title,
            model=model,
        ),
        diagnostics=diagnostics,
    )


def generate_report(
    prometheus_url: str,
    start: float,
    end: float,
    output: str | Path,
    **options,
) -> dict:
    """Convenience API to collect and publish a standalone report once."""
    data = collect_report(prometheus_url, start, end, **options)
    write_report(data, output)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true")
    source.add_argument(
        "--input-json", type=Path, help="Previously exported report data"
    )
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--start", type=timestamp)
    parser.add_argument("--end", type=timestamp, default=time.time())
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--window", type=int, default=60, help="PromQL rate window, seconds"
    )
    parser.add_argument("--deployment", choices=("pd", "standalone"), default="pd")
    parser.add_argument("--title", default="Inference latency report")
    parser.add_argument("--model", default="ATOM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-data", type=Path)
    args = parser.parse_args()
    args.start = args.start if args.start is not None else args.end - 900
    if args.start >= args.end or args.step <= 0 or args.window <= 0:
        parser.error("Require start < end, step > 0 and window > 0")
    data = (
        demo_data()
        if args.demo
        else (
            json.loads(args.input_json.read_text())
            if args.input_json
            else collect(args)
        )
    )
    write_report(data, args.output)
    if args.save_data:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        args.save_data.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
