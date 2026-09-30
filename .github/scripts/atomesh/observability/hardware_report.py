"""Build hardware panels from actual Prometheus scrape samples, within the run."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from urllib.parse import urlencode
from urllib.request import urlopen

from hardware_exporter import METRICS


def quantile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[math.ceil(index)] - values[lower]) * (index - lower)


def summarize(points, expected, interval):
    values = [v for _, v in points if v is not None]
    result = {
        "samples": len(values),
        "coverage_percent": min(100, 100 * len(values) / max(1, expected)),
    }
    if values:
        result.update(
            mean=statistics.fmean(values),
            min=min(values),
            max=max(values),
            p5=quantile(values, 0.05),
            p95=quantile(values, 0.95),
            delta=values[-1] - values[0],
        )
        # Do not integrate across gaps longer than 1.5 scrape intervals.
        result["integral"] = sum(
            (v + w) / 2 * (u - t)
            for (t, v), (u, w) in pairwise(points)
            if 0 < u - t <= interval * 1.5
        )
        result["integrated_seconds"] = sum(
            u - t for (t, _), (u, _) in pairwise(points) if 0 < u - t <= interval * 1.5
        )
    return result


def bucket_series(points, start, end, step):
    buckets = defaultdict(list)
    for timestamp, value in points:
        if start < timestamp <= end and value is not None:
            buckets[max(0, math.ceil((timestamp - start) / step) - 1)].append(value)
    series = {key: [] for key in ("min", "mean", "max")}
    counts = []
    for index in range(math.ceil((end - start) / step)):
        timestamp = min(end, start + (index + 1) * step)
        values = buckets[index]
        for key, function in (("min", min), ("mean", statistics.fmean), ("max", max)):
            series[key].append([timestamp, function(values) if values else None])
        counts.append([timestamp, len(values)])
    return series, counts


def panels_from_vectors(vectors, start, end, *, step, interval, diagnostics):
    devices = {}
    samples = {}
    available = {}
    for vector in vectors:
        labels = vector["metric"]
        name = labels["__name__"]
        points = [
            [float(t), float(v)]
            for t, v in vector.get("values", [])
            if start < float(t) <= end and math.isfinite(float(v))
        ]
        if name == "atom_hardware_registration_errors" and any(
            v > 0 for _, v in points
        ):
            diagnostics.append(
                "Hardware device registration failed on "
                + labels.get("instance", "unknown node")
            )
        if name == "atom_hardware_selected_devices" and (
            not points or any(v == 0 for _, v in points)
        ):
            diagnostics.append(
                "Hardware exporter has no registered GPUs on "
                + labels.get("instance", "unknown node")
            )
        if "pci_bdf" not in labels:
            continue
        identity = (
            labels.get("hostname", labels.get("instance", "unknown")),
            labels["pci_bdf"],
        )
        devices[identity] = labels
        key = (identity, name)
        if key in samples and name != "atom_gpu_sensor_available":
            raise ValueError(f"Duplicate hardware source for {identity}: {name}")
        if name == "atom_gpu_sensor_available":
            available[identity, labels["sensor"]] = points
        else:
            samples[key] = points
    if not devices:
        diagnostics.append("No hardware GPU samples collected")
    panels = []
    expected = max(1, (end - start) / interval)
    for sensor, (metric, title, unit, _) in METRICS.items():
        panel = {
            "id": "hardware_" + sensor,
            "title": title,
            "label": "HARDWARE · GPU",
            "detail": "Per-GPU scrape samples; 5-second bins retain min/mean/max. All GPUs pools samples; select a GPU for its own curve.",
            "kind": "hardware",
            "category": "hardware",
            "role": "hardware",
            "overview": False,
            "unit": unit,
            "instances": {},
            "series": {},
            "metric": metric,
        }
        panel["detail"] = panel["detail"].replace("5-second", f"{step:g}-second")
        if sensor == "memory_busy":
            panel[
                "detail"
            ] += " Memory busy is activity, not measured HBM bandwidth utilization."
        pooled = []
        for identity, labels in sorted(devices.items()):
            points = samples.get((identity, metric), [])
            scale = 2**30 if unit == "GiB" else 1
            points = [[t, v / scale] for t, v in points]
            instance = (
                f"{identity[0]} / {identity[1]} / {labels.get('gpu_role', 'unknown')}"
            )
            series, counts = bucket_series(points, start, end, step)
            summary = summarize(points, expected, interval)
            if sensor == "power":
                summary["energy_joules"] = summary.get("integral")
            else:
                summary.pop("integral", None)
                summary.pop("integrated_seconds", None)
            health = available.get((identity, sensor), [])
            if any(v < 1 for _, v in health):
                diagnostics.append(
                    f"Hardware sensor unavailable: {instance} / {sensor}"
                )
            panel["instances"][instance] = {
                "series": series,
                "sample_counts": counts,
                "summary": summary,
                "device": {
                    "hostname": identity[0],
                    "pci_bdf": identity[1],
                    "role": labels.get("gpu_role"),
                    "card": labels.get("card"),
                },
            }
            pooled.extend(points)
        panel["series"], panel["sample_counts"] = bucket_series(
            sorted(pooled), start, end, step
        )
        panels.append(panel)
    return panels


def collect_hardware(url, start, end, *, step=5, interval=1, diagnostics):
    # An instant range-vector query returns original scrape timestamps, avoiding
    # lookback interpolation and duplicate query_range points in summaries.
    duration_ms = max(1, math.floor((end - start) * 1000))
    query = (
        '{job="atom-hardware",__name__=~"atom_gpu_.*|atom_hardware_.*"}['
        + str(duration_ms)
        + "ms]"
    )
    request = (
        url.rstrip("/") + "/api/v1/query?" + urlencode({"query": query, "time": end})
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("error", "Hardware query failed"))
    vectors = payload["data"]["result"]
    return (
        panels_from_vectors(
            vectors, start, end, step=step, interval=interval, diagnostics=diagnostics
        ),
        vectors,
    )
