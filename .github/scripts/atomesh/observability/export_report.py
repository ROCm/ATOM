"""Export Prometheus inference metrics as a self-contained interactive HTML report.

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
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

STATISTICS = {"mean": None, "p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
GAUGE_SERIES = {
    "running",
    "waiting",
    "waiting_kv",
    "used",
    "evictable",
    "vacant",
    "hit",
    "reuse",
    "gpu",
    "lmcache",
}


def timestamp(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        # Prometheus uses nanosecond RFC3339 timestamps. Python 3.10 accepts
        # only 3 or 6 fractional digits, so normalize to microseconds first.
        value = re.sub(
            r"(\d{2}:\d{2}:\d{2})\.(\d+)",
            lambda m: m[1] + "." + m[2][:6].ljust(6, "0"),
            value,
            count=1,
        )
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise argparse.ArgumentTypeError("ISO timestamps must include a timezone")
        return parsed.timestamp()


def latency_panels_for(deployment: str) -> list[dict]:
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


def panels_for(deployment: str) -> list[dict]:
    panels = latency_panels_for(deployment)
    for panel in panels:
        panel["unit"] = "ms"
    roles = ("standalone",) if deployment == "standalone" else ("prefill", "decode")
    for role in roles:
        selector = f'job="atom",role="{role}"'
        common = {"selector": selector, "label": f"{role.upper()} · SCHEDULER"}
        panels.extend(
            [
                {
                    **common,
                    "id": f"{role}_queues",
                    "title": f"{role.title()} requests",
                    "detail": "Running, waiting for admission, and waiting for external KV · sampled state",
                    "metric": "atom:scheduler_requests",
                    "unit": "requests",
                    "kind": "queues",
                },
                {
                    **common,
                    "id": f"{role}_queue_time",
                    "title": f"{role.title()} queue time",
                    "detail": "Engine receipt → first forward dispatch · includes KV loading waits · one sample per request",
                    "metric": "atom:request_queue_time_seconds",
                    "unit": "ms",
                },
                {
                    **common,
                    "id": f"{role}_kv_blocks",
                    "title": f"{role.title()} KV block utilization",
                    "detail": "Used, evictable cached, and vacant blocks · capacity-weighted across pools",
                    "metric": "atom:scheduler_kv_cache_blocks",
                    "unit": "%",
                    "kind": "blocks",
                },
            ]
        )
    role = roles[-1]
    panels.append(
        {
            "id": "decode_batch_size",
            "title": "Actual decode batch size",
            "label": f"{role.upper()} · FORWARD",
            "detail": "Real decode request rows per forward · no dummy work, padding, or token weighting",
            "metric": "atom:decode_batch_size",
            "selector": f'job="atom",role="{role}"',
            "unit": "requests",
            "scale": 1,
        }
    )
    if deployment == "pd":
        panels.append(
            {
                "id": "pd_kv_transfer",
                "title": "PD KV transfer wait",
                "label": "DECODE · KV LOAD",
                "detail": "Enter remote KV wait → all workers complete · includes dispatch, handshake and notification",
                "metric": "atom:pd_kv_transfer_seconds",
                "selector": 'job="atom",role="decode"',
                "unit": "ms",
            }
        )
    for role in roles:
        common = {"selector": f'job="atom",role="{role}"', "role": role}
        panels.extend(
            [
                {
                    **common,
                    "id": f"{role}_cache_hit",
                    "title": f"{role.title()} cache reuse",
                    "label": f"{role.upper()} · PREFIX CACHE",
                    "kind": "cache",
                    "cache_breakdown": True,
                    "unit": "%",
                    "metric": "atom:prefix_cache_cached_tokens_total",
                    "detail": "Total reuse = GPU + LMCache · all three curves share the input-token denominator",
                },
                {
                    **common,
                    "id": f"{role}_gpu_forward",
                    "title": f"{role.title()} GPU forward",
                    "label": f"{role.upper()} · GPU WORKERS",
                    "unit": "ms",
                    "metric": "atom:gpu_forward_seconds",
                    "detail": "Per-worker target forward · includes GPU stream communication/waits · excludes input prep, sampling and drafting",
                },
            ]
        )
    role = roles[0]
    panels.append(
        {
            "id": f"{role}_request_gpu_forward",
            "role": role,
            "title": "Prefill request GPU forward",
            "label": f"{role.upper()} · GPU WORKERS",
            "detail": "Sum of participating batch times across initial prefill chunks · once per completed request per worker · shared batch time, not exclusive compute",
            "metric": "atom:prefill_request_gpu_forward_seconds",
            "selector": f'job="atom",role="{role}"',
            "unit": "ms",
            "overview": False,
        }
    )
    for suffix, title, detail in (
        (
            "request_tokens",
            "Uncached prompt tokens",
            "Prompt remainder at first local prefill dispatch · once per request",
        ),
        (
            "batch_tokens",
            "Actual prefill tokens",
            "Real scheduled prefill tokens per forward · each chunk counted · no cached prefix or padding",
        ),
    ):
        panels.append(
            {
                "id": f"{role}_{suffix}",
                "role": role,
                "title": title,
                "label": f"{role.upper()} · WORKLOAD",
                "detail": detail,
                "metric": f"atom:prefill_{suffix}",
                "selector": f'job="atom",role="{role}"',
                "unit": "tokens",
                "scale": 1,
            }
        )
    role = roles[-1]
    panels.append(
        {
            "id": "decode_context_tokens",
            "role": role,
            "title": "Decode batch context tokens",
            "label": f"{role.upper()} · WORKLOAD",
            "detail": "Sum of logical context lengths per real decode batch · no TP duplication or graph padding",
            "metric": "atom:decode_context_tokens",
            "selector": f'job="atom",role="{role}"',
            "unit": "tokens",
            "scale": 1,
        }
    )
    panels.append(
        {
            "id": "decode_request_context_tokens",
            "role": role,
            "title": "Decode request context tokens",
            "label": f"{role.upper()} · WORKLOAD",
            "detail": "Logical context per request on every decode forward · request-forward weighted · no padding",
            "metric": "atom:decode_request_context_tokens",
            "selector": f'job="atom",role="{role}"',
            "unit": "tokens",
            "scale": 1,
        }
    )
    for panel in panels:
        match = re.search(r'role="([^"]+)"', panel["selector"])
        panel.setdefault("role", match[1] if match else "overall")
        panel["category"] = (
            "cache"
            if panel.get("kind") in {"cache", "blocks"}
            else (
                "workload"
                if panel.get("kind") == "queues"
                or panel["unit"] in {"requests", "tokens"}
                else "latency"
            )
        )
        panel.setdefault(
            "overview",
            (
                any(
                    panel["id"].endswith(suffix)
                    for suffix in ("_ttft", "_queue_time", "_gpu_forward", "_cache_hit")
                )
                and panel["role"] != "overall"
            ),
        )
    return panels


def statistics_for(panel: dict):
    if panel.get("kind") == "cache":
        return ("reuse", "lmcache", "gpu") if panel.get("cache_breakdown") else ("hit",)
    if panel.get("kind") == "queues":
        return ("running", "waiting", "waiting_kv")
    if panel.get("kind") == "blocks":
        return ("used", "evictable", "vacant")
    return tuple(STATISTICS)


def block_count_query_for(panel: dict, state: str, *, by_instance=False) -> str:
    aggregate = "sum by (instance)" if by_instance else "sum"
    return f'{aggregate}({panel["metric"]}{{{panel["selector"]},state="{state}"}})'


def cache_count_query_for(panel, state, window, *, by_instance=False):
    aggregate = "sum by (instance)" if by_instance else "sum"
    count = "count by (instance)" if by_instance else "count"
    metrics = {
        "cached": "atom:prefix_cache_cached_tokens_total",
        "gpu": "atom:prefix_cache_cached_tokens_total",
        "lmcache": "atom:prefix_cache_offload_tokens_total",
        "prompt": "atom:prefix_cache_full_tokens_total",
    }
    if state == "reused":
        gpu = cache_count_query_for(panel, "gpu", window, by_instance=by_instance)
        offload = cache_count_query_for(
            panel, "lmcache", window, by_instance=by_instance
        )
        return f"({gpu} + {offload})"

    def increase(metric):
        return f'increase({metric}{{{panel["selector"]}}}[{window}s])'

    values = increase(metrics[state])
    result = f"{aggregate}({values})"
    if panel.get("cache_breakdown") and state in {"gpu", "lmcache"}:
        # A partial rollout/missing tier must not silently lower total reuse.
        # Zero counters from a service without LMCache remain valid samples.
        inputs = increase(metrics["prompt"])
        result = f"({result} and ({count}({values}) == {count}({inputs})))"
    return result


def query_for(panel: dict, statistic: str, window: int, *, by_instance=False) -> str:
    metric, selector = panel["metric"], panel["selector"]
    aggregate = "sum by (instance)" if by_instance else "sum"
    if panel.get("kind") == "cache":
        state = {
            "hit": "cached",
            "reuse": "reused",
            "gpu": "gpu",
            "lmcache": "lmcache",
        }[statistic]
        hit = cache_count_query_for(panel, state, window, by_instance=by_instance)
        total = cache_count_query_for(panel, "prompt", window, by_instance=by_instance)
        return f"100 * {hit} / {total}"
    if panel.get("kind") in {"queues", "blocks"}:
        numerator = f'{aggregate}({metric}{{{selector},state="{statistic}"}})'
        if panel["kind"] == "queues":
            return numerator
        return f'100 * {numerator} / {aggregate}({metric}{{{selector},state="total"}})'
    scale = panel.get("scale", 1000)

    def rate(suffix):
        return f"rate({metric}_{suffix}{{{selector}}}[{window}s])"

    if statistic == "mean":
        return f"{scale} * {aggregate}({rate('sum')}) / {aggregate}({rate('count')})"
    labels = "instance, le" if by_instance else "le"
    return f"{scale} * histogram_quantile({STATISTICS[statistic]}, sum by ({labels}) ({rate('bucket')}))"


def _fetch_vectors(url, query, start, end, step):
    args = urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "step": step}
    )
    request = urllib.request.Request(url.rstrip("/") + "/api/v1/query_range?" + args)
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("status") != "success":
        raise RuntimeError(result.get("error", "Prometheus query failed"))
    return result["data"]["result"]


def _points(vector):
    return [
        [float(t), float(v) if math.isfinite(float(v)) else None]
        for t, v in vector.get("values", [])
    ]


def fetch_series(url: str, query: str, start: float, end: float, step: int) -> list:
    series = _fetch_vectors(url, query, start, end, step)
    if len(series) > 1:
        raise ValueError("Expected one aggregated series per panel statistic")
    return _points(series[0]) if series else []


def fetch_instance_series(url, query, start, end, step):
    result = {}
    for vector in _fetch_vectors(url, query, start, end, step):
        instance = vector.get("metric", {}).get("instance")
        if not instance:
            continue
        if instance in result:
            raise ValueError("Expected one series per instance and statistic")
        result[instance] = _points(vector)
    return result


def panel_queries(panel, window, *, by_instance=False):
    for statistic in statistics_for(panel):
        yield "series", statistic, query_for(
            panel, statistic, window, by_instance=by_instance
        )
    if panel.get("kind") == "blocks":
        for state in ("used", "total"):
            yield "block_counts", state, block_count_query_for(
                panel, state, by_instance=by_instance
            )
    if panel.get("kind") == "cache":
        states = (
            ("reused", "prompt", "gpu", "lmcache")
            if panel.get("cache_breakdown")
            else ("cached", "prompt")
        )
        for state in states:
            yield "cache_counts", state, cache_count_query_for(
                panel, state, window, by_instance=by_instance
            )


def collect(args, *, diagnostics: list[str] | None = None) -> dict:
    panels = panels_for(args.deployment)
    errors = [] if diagnostics is None else diagnostics
    failed = 0
    aggregate_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pending = {}
        for panel in panels:
            panel["series"], panel["queries"], panel["instances"] = {}, {}, {}
            for grouped in (False, True):
                if grouped and panel["role"] == "overall":
                    continue
                for field, statistic, query in panel_queries(
                    panel, args.window, by_instance=grouped
                ):
                    panel["queries"][
                        f"{'instances' if grouped else 'all'}/{field}/{statistic}"
                    ] = query
                    if not grouped:
                        panel.setdefault(field, {})[statistic] = []
                        aggregate_total += 1
                    future = pool.submit(
                        fetch_instance_series if grouped else fetch_series,
                        args.prometheus_url,
                        query,
                        args.start,
                        args.end,
                        args.step,
                    )
                    pending[future] = panel, field, statistic, grouped
        for future in concurrent.futures.as_completed(pending):
            panel, field, statistic, grouped = pending[future]
            try:
                result = future.result()
                if grouped:
                    for instance, points in result.items():
                        scoped = panel["instances"].setdefault(instance, {"series": {}})
                        scoped.setdefault(field, {})[statistic] = points
                else:
                    panel[field][statistic] = result
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                if not grouped:
                    failed += 1
                errors.append(
                    f"{panel['title']} / {'instances' if grouped else 'all'} / {field} / {statistic}: {exc}"
                )
    if failed == aggregate_total:
        raise RuntimeError("All Prometheus queries failed: " + errors[-1])
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
            "instances": [
                {"role": role, "instance": instance}
                for role, instance in sorted(
                    {
                        (p["role"], instance)
                        for p in panels
                        for instance in p["instances"]
                    }
                )
            ],
        },
        "panels": panels,
    }


def demo_data() -> dict:
    """Explicitly labelled synthetic data for reviewing the report layout."""
    rng = random.Random(1729)
    start, step, count = 1788854400, 5, 121
    panels = panels_for("pd")
    for index, panel in enumerate(panels):
        base = [420, 6.8, 290, 76][index] if index < 4 else 12
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
            for stat, factor in zip(
                statistics_for(panel), (1.0, 0.89, 1.16, 1.28, 1.53)
            )
        }
        if panel["unit"] == "tokens":
            factor = 2000 if panel["id"] == "decode_context_tokens" else 150
            panel["series"] = {
                k: [[t, round(v * factor)] for t, v in points]
                for k, points in panel["series"].items()
            }
        if panel.get("kind") == "cache":
            panel["series"] = {
                k: [[start + i * step, v] for i in range(count)]
                for k, v in (("reuse", 84), ("gpu", 46), ("lmcache", 38))
            }
            panel["cache_counts"] = {
                k: [[start + i * step, v] for i in range(count)]
                for k, v in (
                    ("reused", 84000),
                    ("gpu", 46000),
                    ("lmcache", 38000),
                    ("prompt", 100000),
                )
            }
        if panel.get("kind") == "blocks":
            total = 10000 if panel["id"].startswith("prefill_") else 20000
            panel["block_counts"] = {
                key: [[start + i * step, value] for i in range(count)]
                for key, value in (("used", total * 0.4), ("total", total))
            }
            panel["series"] = {
                key: [[start + i * step, round(value, 3)] for i in range(count)]
                for key, value in (("used", 40), ("evictable", 35), ("vacant", 25))
            }
        if panel["role"] != "overall":
            panel["instances"] = {}
            for index, factor in enumerate((0.8, 1.2)):
                role = panel["role"]
                instance = f"{role}-{'a' if index == 0 else 'b'}:{8010 if role == 'prefill' else 8020}"
                scoped = {
                    "series": {
                        k: [[t, round(v * factor, 3)] for t, v in points]
                        for k, points in panel["series"].items()
                    }
                }
                if panel.get("kind") == "queues":
                    scoped["series"] = {
                        k: [[t, v * (0.4 if index == 0 else 0.6)] for t, v in points]
                        for k, points in panel["series"].items()
                    }
                if panel.get("kind") == "blocks":
                    share = 0.4 if index == 0 else 0.6
                    scoped["block_counts"] = {
                        "total": [
                            [t, v * share] for t, v in panel["block_counts"]["total"]
                        ],
                        "used": [
                            [t, v * (0.25 if index == 0 else 0.75)]
                            for t, v in panel["block_counts"]["used"]
                        ],
                    }
                    used = 25 if index == 0 else 50
                    scoped["series"] = {
                        k: [[start + i * step, v] for i in range(count)]
                        for k, v in (
                            ("used", used),
                            ("evictable", 35),
                            ("vacant", 65 - used),
                        )
                    }
                if panel.get("kind") == "cache":
                    gpu, lmcache, prompt = (
                        (10000, 20000, 40000) if index == 0 else (36000, 18000, 60000)
                    )
                    scoped["cache_counts"] = {
                        k: [[start + i * step, v] for i in range(count)]
                        for k, v in (
                            ("reused", gpu + lmcache),
                            ("gpu", gpu),
                            ("lmcache", lmcache),
                            ("prompt", prompt),
                        )
                    }
                    scoped["series"] = {
                        k: [[start + i * step, 100 * v / prompt] for i in range(count)]
                        for k, v in (
                            ("reuse", gpu + lmcache),
                            ("gpu", gpu),
                            ("lmcache", lmcache),
                        )
                    }
                panel["instances"][instance] = scoped
    return {
        "meta": {
            "title": "Agentic inference report",
            "model": "GLM-5.2 · CPP4 + DCP4",
            "start": start,
            "end": start + step * (count - 1),
            "step": step,
            "window": 60,
            "kind": "demo",
            "source": "Layout preview · synthetic data",
            "instances": [
                {"role": role, "instance": instance}
                for role, instance in sorted(
                    {
                        (p["role"], instance)
                        for p in panels
                        for instance in p.get("instances", {})
                    }
                )
            ],
            "exported_at": time.time(),
            "notes": [],
        },
        "panels": panels,
    }


def validate_data(data: dict) -> None:
    """Validate Unix timestamps and nonnegative values in each panel's unit."""

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
        if panel.get("unit", "ms") not in {"ms", "requests", "%", "tokens"}:
            raise ValueError("Panel unit must be ms, requests, tokens or %")
        instances = panel.get("instances", {})
        if not isinstance(instances, dict) or any(
            not isinstance(k, str) or not k for k in instances
        ):
            raise ValueError(
                "instances must map nonempty service addresses to series objects"
            )
        for bundle in [panel, *instances.values()]:
            if not isinstance(bundle, dict) or not isinstance(
                bundle.get("series"), dict
            ):
                raise TypeError("Each panel or instance requires a series object")
            fields = [
                ("series", STATISTICS.keys() | GAUGE_SERIES),
                ("block_counts", {"used", "total"}),
                ("cache_counts", {"cached", "prompt", "reused", "gpu", "lmcache"}),
            ]
            for field, allowed in fields:
                series = bundle.get(field, {})
                if not isinstance(series, dict) or series.keys() - allowed:
                    raise ValueError(
                        f"{field} must map supported statistics or states to arrays"
                    )
                for points in series.values():
                    if not isinstance(points, list):
                        raise TypeError(
                            "Series must map a supported statistic or state to arrays"
                        )
                    previous = -math.inf
                    for point in points:
                        if not isinstance(point, (list, tuple)) or len(point) != 2:
                            raise ValueError(
                                "Each point must be [unix_seconds, value_or_null]"
                            )
                        t, value = point
                        if not number(t) or t <= previous:
                            raise ValueError(
                                "Point timestamps must be finite and strictly increasing"
                            )
                        if value is not None and (not number(value) or value < 0):
                            raise ValueError(
                                "Metric values must be nonnegative numbers or null"
                            )
                        previous = t


def write_report(data: dict, output: str | Path) -> None:
    """Public JSON-to-HTML API. The report embeds data and needs no server or CDN.

    Required meta fields: start/end (Unix seconds), step/window (seconds).
    Each panel supplies id, title, unit (default ms), and series. Series contain
    statistics or scheduler states as [Unix timestamp, value or None] pairs.
    KV panels may also supply block_counts.used/total with raw block quantities.
    Set meta.kind='demo' only for explicitly synthetic preview data.
    """
    validate_data(data)
    data = copy.deepcopy(data)
    data["meta"].setdefault("kind", "recorded")
    for panel in data["panels"]:
        panel.setdefault("label", "RECORDED DATA")
        panel.setdefault("detail", "")
        panel.setdefault("unit", "ms")
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
    title: str = "Agentic inference report",
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
    parser.add_argument("--title", default="Agentic inference report")
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
