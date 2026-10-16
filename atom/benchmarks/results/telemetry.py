# SPDX-License-Identifier: MIT
"""Allocated AMD GPU and Prometheus telemetry: collection and derived observations."""

import math
import re
import signal
import time
import urllib.request
from pathlib import Path

from .io import JsonlWriter, iter_jsonl, write_view

SAMPLE = re.compile(r"^(atom:[a-zA-Z0-9_:]+)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?$")


def _read(path, divisor=1):
    try:
        return int(path.read_text().strip()) / divisor
    except (OSError, ValueError):
        return None


def gpu_sample(device):
    root = Path(device["sysfs_path"])
    hwmons = list((root / "hwmon").glob("hwmon*"))
    power = None
    for hwmon in hwmons:
        for key in ("power1_average", "power1_input"):
            power = _read(hwmon / key, 1e6)
            if power is not None:
                break
        if power is not None:
            break
    return {
        "timestamp_ns": str(time.time_ns()),
        "pci_bus_id": device["pci_bus_id"],
        "uuid": device.get("uuid"),
        "power_w": power,
        "gpu_utilization_pct": _read(root / "gpu_busy_percent"),
        "vram_used_bytes": _read(root / "mem_info_vram_used"),
        "vram_total_bytes": _read(root / "mem_info_vram_total"),
    }


def collect(config, directory, interval=1.0, server_url=None):
    if interval < 0.1:
        raise ValueError("Telemetry interval must be >= 0.1 seconds")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    gpu = JsonlWriter(directory / "gpu_samples.jsonl")
    server = JsonlWriter(directory / "server_samples.jsonl")
    try:
        while running:
            start = time.monotonic()
            for device in config["hardware"].get("devices", []):
                gpu.write(gpu_sample(device))
            if server_url:
                row = {"timestamp_ns": str(time.time_ns()), "format": "prometheus_text"}
                try:
                    with urllib.request.urlopen(
                        server_url, timeout=min(interval, 2)
                    ) as response:
                        row["text"] = response.read(8 * 1024 * 1024).decode("utf-8")
                except (OSError, ValueError) as exc:
                    row["error"] = str(exc)
                server.write(row)
            time.sleep(max(0.01, interval - (time.monotonic() - start)))
    finally:
        for device in config["hardware"].get("devices", []):
            gpu.write(gpu_sample(device))
        gpu.close()
        server.close()


def integrate_power(path, devices, start_ns, end_ns, max_gap_s=3.0):
    """Clip trapezoids to the profiling window; gaps invalidate, never extrapolate."""
    if not devices or start_ns is None or end_ns is None:
        return {
            "valid": False,
            "reasons": ["missing device mapping or profiling window"],
        }
    start, end = int(start_ns), int(end_ns)
    energies = {d["pci_bus_id"]: 0.0 for d in devices}
    previous, first, reasons = {}, {}, set()
    for row in iter_jsonl(path):
        device, watts = row.get("pci_bus_id"), row.get("power_w")
        if device not in energies or type(watts) not in (int, float):
            continue
        if not math.isfinite(watts) or watts < 0:
            continue
        timestamp = int(row["timestamp_ns"])
        first.setdefault(device, timestamp)
        if device in previous:
            a, wa = previous[device]
            if timestamp <= a:
                reasons.add(f"{device}: non-increasing timestamps")
                continue
            if timestamp > start and a < end:
                if (timestamp - a) / 1e9 > max_gap_s:
                    reasons.add(f"{device}: sampling gap")
                else:
                    lo, hi = max(a, start), min(timestamp, end)
                    wlo = wa + (watts - wa) * (lo - a) / (timestamp - a)
                    whi = wa + (watts - wa) * (hi - a) / (timestamp - a)
                    energies[device] += (wlo + whi) / 2 * (hi - lo) / 1e9
        previous[device] = (timestamp, watts)
    for device in energies:
        if device not in first or first[device] > start or previous[device][0] < end:
            reasons.add(f"{device}: window not covered")
    if end <= start:
        reasons.add("nonpositive profiling window")
    if reasons:
        return {"valid": False, "reasons": sorted(reasons)}
    total = sum(energies.values())
    return {
        "valid": True,
        "energy_j": total,
        "per_device_energy_j": energies,
        "avg_total_gpu_power_w": total / ((end - start) / 1e9),
        "window_start_ns": str(start),
        "window_end_ns": str(end),
    }


def gpu_series(path, start_ns, end_ns, max_points=1800):
    """Bounded per-device min/max/mean buckets, in the request profiling window."""
    if start_ns is None or end_ns is None:
        return {"series": [], "reason": "missing profiling window"}
    start, end = int(start_ns), int(end_ns)
    width = max(10**9, math.ceil((end - start) / max_points))
    buckets = {}
    for row in iter_jsonl(path):
        t = int(row["timestamp_ns"])
        if not start <= t <= end:
            continue
        bucket = buckets.setdefault(((t - start) // width, row["pci_bus_id"]), {})
        for name in (
            "power_w",
            "gpu_utilization_pct",
            "vram_used_bytes",
            "vram_total_bytes",
        ):
            value = row.get(name)
            if type(value) not in (float, int) or not math.isfinite(value):
                continue
            stat = bucket.setdefault(
                name, {"min": value, "max": value, "sum": 0, "count": 0}
            )
            stat.update(
                min=min(stat["min"], value),
                max=max(stat["max"], value),
                sum=stat["sum"] + value,
                count=stat["count"] + 1,
            )
    result = []
    for (bucket, device), metrics in sorted(buckets.items()):
        for stat in metrics.values():
            stat["mean"] = stat.pop("sum") / stat["count"]
        result.append(
            {
                "start_ns": str(start + bucket * width),
                "window_seconds": width / 1e9,
                "pci_bus_id": device,
                "metrics": metrics,
            }
        )
    return {"series": result}


def attach_telemetry(summary, config, telemetry_dir, destination):
    """Derive optional telemetry without invalidating retained request evidence."""
    gpu_path = Path(telemetry_dir) / "gpu_samples.jsonl"
    if gpu_path.is_file():
        try:
            write_view(
                destination / "views" / "gpu.json",
                gpu_series(
                    gpu_path,
                    summary["window"]["start_ns"],
                    summary["window"]["end_ns"],
                ),
                summary,
            )
            summary["power"] = integrate_power(
                gpu_path,
                config["hardware"].get("devices", []),
                summary["window"]["start_ns"],
                summary["window"]["end_ns"],
            )
        except (ValueError, OSError, KeyError) as exc:
            summary["power"] = {
                "valid": False,
                "reasons": [f"telemetry parse failed: {exc}"],
            }
        summary["capabilities"]["power"] = summary["power"]["valid"]
        if summary["power"]["valid"]:
            energy = summary["power"]["energy_j"]
            tokens = summary["request_metrics"]["throughput"]["output"]["tokens"]
            completed = summary["accounting"]["counts"]["profiled_success"]
            summary["power"].update(
                joules_per_output_token=energy / tokens if tokens else None,
                joules_per_request=energy / completed,
            )
    server_path = Path(telemetry_dir) / "server_samples.jsonl"
    if server_path.is_file():
        try:
            server = derive_server_metrics(
                server_path,
                summary["window"]["start_ns"],
                summary["window"]["end_ns"],
            )
        except (ValueError, OSError, KeyError) as exc:
            server = {
                "available": False,
                "series": [],
                "reason": f"telemetry parse failed: {exc}",
            }
        write_view(destination / "views" / "server.json", server, summary)
        summary["capabilities"]["server_metrics"] = server["available"]
        if server["available"]:
            summary["server_metrics"] = {
                **server["summary"],
                "window": server["window"],
                "source": "atom_prometheus",
            }


def parse_prometheus(text):
    result = {}
    for line in text.splitlines():
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        name, labels, raw = match.groups()
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            result[name + (labels or "")] = value
    return result


def derive_server_metrics(path, start_ns, end_ns, max_points=1800):
    """Counters use observed in-window endpoints, without boundary extrapolation."""
    if start_ns is None or end_ns is None:
        return {"available": False, "reason": "missing profiling window", "series": []}
    start, end = int(start_ns), int(end_ns)
    width = max(1_000_000_000, math.ceil((end - start) / max_points))
    buckets = {}
    first, last, previous, resets = {}, {}, {}, set()
    errors = 0
    first_timestamp = last_timestamp = None
    for row in iter_jsonl(path):
        t = int(row["timestamp_ns"])
        if not start <= t <= end:
            continue
        if row.get("error"):
            errors += 1
            continue
        metrics = parse_prometheus(row.get("text", ""))
        if not metrics:
            continue
        first_timestamp = t if first_timestamp is None else min(first_timestamp, t)
        last_timestamp = t if last_timestamp is None else max(last_timestamp, t)
        bucket = buckets.setdefault((t - start) // width, {})
        for name, value in metrics.items():
            first.setdefault(name, (t, value))
            last[name] = (t, value)
            if (
                name.split("{")[0].endswith("_total")
                and name in previous
                and value < previous[name]
            ):
                resets.add(name)
            previous[name] = value
            stat = bucket.setdefault(
                name, {"min": value, "max": value, "sum": 0, "count": 0}
            )
            stat["min"] = min(stat["min"], value)
            stat["max"] = max(stat["max"], value)
            stat["sum"] += value
            stat["count"] += 1
            stat["last"] = value
    counters = {}
    for name, (a, va) in first.items():
        # Labeled counters preserve the original label set in their identity.
        if name.split("{")[0].endswith("_total"):
            b, vb = last[name]
            valid = b > a and name not in resets and vb >= va
            counters[name] = {
                "valid": valid,
                "start_ns": str(a),
                "end_ns": str(b),
                "delta": vb - va if valid else None,
            }
    series = [
        {
            "start_ns": str(start + index * width),
            "window_seconds": width / 1e9,
            "metrics": {
                k: {
                    "min": v["min"],
                    "max": v["max"],
                    "mean": v["sum"] / v["count"],
                    "last": v["last"],
                }
                for k, v in metrics.items()
            },
        }
        for index, metrics in sorted(buckets.items())
    ]
    summary = {"tokens": {}, "kv_cache": {}, "cache": {}}
    for source, target in (
        ("atom:prompt_tokens_total", "prompt_total"),
        ("atom:generation_tokens_total", "generation_total"),
        ("atom:requests_finished_total", "requests_completed"),
    ):
        if counters.get(source, {}).get("valid"):
            summary["tokens"][target] = counters[source]["delta"]
    if "atom:kv_cache_usage_ratio" in last:
        summary["kv_cache"]["gpu_usage_pct"] = (
            last["atom:kv_cache_usage_ratio"][1] * 100
        )
        summary["kv_cache"]["statistic"] = "last observed sample in profiling window"
    if "atom:prefix_cache_hit_ratio" in last:
        summary["cache"]["atom_lifetime_admitted_prefix_hit_ratio"] = last[
            "atom:prefix_cache_hit_ratio"
        ][1]
    return {
        "available": bool(series),
        "source": "atom_prometheus",
        "series": series,
        "summary": summary,
        "counter_deltas": counters,
        "counter_resets": sorted(resets),
        "scrape_errors": errors,
        "window": {
            "start_ns": str(first_timestamp) if first_timestamp is not None else None,
            "end_ns": str(last_timestamp) if last_timestamp is not None else None,
        },
        "cache_definition": "Preserve ATOM cache gauges/counters; do not equate them with other engines' cache denominators.",
    }
