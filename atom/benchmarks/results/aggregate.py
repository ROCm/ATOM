# SPDX-License-Identifier: MIT
"""Request metrics, derived data and InferenceX-compatible export."""

import gzip
import hashlib
import json
import math
import statistics
from array import array
from collections import Counter
from pathlib import Path

from . import AGGREGATION_VERSION
from .io import write_view
from .records import full_response_itl, number, timestamp

COMPATIBILITY_COMMIT = "8f9a30bf6351f435cecb28d8b2e4a57459bd76e2"

MODEL_KEYS = {
    "deepseek-ai/DeepSeek-V4.1-Flash": "dsv41flash",
    "deepseek-ai/DeepSeek-V4-Pro": "dsv4",
    "deepseek-ai/DeepSeek-R1-0528": "dsr1",
    "openai/gpt-oss-120b": "gptoss120b",
    "MiniMaxAI/MiniMax-M3": "minimaxm3",
    "moonshotai/Kimi-K3": "kimik3",
    "Qwen/Qwen3.5-397B-A17B": "qwen3.5",
}

PERCENTILES = {
    "p50": 0.5,
    "p75": 0.75,
    "p90": 0.9,
    "p95": 0.95,
    "p99": 0.99,
    "p99.9": 0.999,
}


def quantile(ordered, p):
    offset = (len(ordered) - 1) * p
    lo = math.floor(offset)
    hi = math.ceil(offset)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (offset - lo)


def stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": ordered[0],
        "max": ordered[-1],
        **{key: quantile(ordered, p) for key, p in PERCENTILES.items()},
    }


def inverse_stats(values, result):
    if not values:
        return result
    return {
        "count": len(values),
        **{k: 1 / result[k] for k in ("mean", *PERCENTILES)},
        "std": statistics.pstdev(1 / v for v in values),
        "definition": "reciprocal_of_matching_latency_statistic",
    }


def histogram(values, bins=32):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": n}
        for i, n in enumerate(counts)
    ]


def valid_request(row):
    reasons = []
    for name in ("input_tokens", "output_tokens"):
        value = number(row.get(name))
        if value is None or value < 0 or value != int(value):
            reasons.append(f"invalid_{name}")
    for name in ("ttft_s", "e2el_s"):
        value = number(row.get(name))
        if value is None or value <= 0:
            reasons.append(f"invalid_{name}")
    if not row.get("start_ns") or not row.get("end_ns"):
        reasons.append("missing_timestamps")
    else:
        try:
            if int(timestamp(row["end_ns"])) <= int(timestamp(row["start_ns"])):
                reasons.append("nonpositive_duration")
        except (ValueError, TypeError):
            reasons.append("invalid_timestamps")
    if (
        number(row.get("ttft_s")) is not None
        and number(row.get("e2el_s")) is not None
        and row["ttft_s"] > row["e2el_s"] + 1e-6
    ):
        reasons.append("ttft_exceeds_e2el")
    return reasons


def aggregate(records, config, destination):
    """Stream evidence and timeline; keep scalar samples for exact quantiles."""
    destination = Path(destination)
    (destination / "requests").mkdir(parents=True)
    (destination / "views" / "timeline").mkdir(parents=True)
    counts, dropped, phases = Counter(), Counter(), Counter()
    request_hash = hashlib.sha256()
    samples = {
        key: array("d")
        for key in (
            "ttft",
            "e2el",
            "native_itl",
            "tpot",
            "full_response_itl",
            "e2el_per_osl",
            "input",
            "output_actual",
            "output_expected",
        )
    }
    start_ns = end_ns = None
    totals = {"input": 0, "output": 0}
    timeline, parts, seen = [], [], set()
    end_seconds = Counter()
    missing_session = 0
    phase_windows = {}
    cache_hit = cache_total = 0
    cache_samples = 0

    def flush():
        if not timeline:
            return
        name = f"part-{len(parts):04}.json.gz"
        path = destination / "views" / "timeline" / name
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(timeline, f, allow_nan=False)
        parts.append({"path": name, "count": len(timeline)})
        timeline.clear()

    with gzip.open(
        destination / "requests" / "requests.jsonl.gz", "wt", encoding="utf-8"
    ) as evidence:
        for row in records:
            encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            evidence.write(encoded)
            request_hash.update(encoded.encode())
            counts["total"] += 1
            phases[row.get("phase", "unknown")] += 1
            phase = row.get("phase", "unknown")
            if row.get("start_ns") and row.get("end_ns"):
                try:
                    a, b = (
                        int(timestamp(row["start_ns"])),
                        int(timestamp(row["end_ns"])),
                    )
                    window = phase_windows.setdefault(phase, [a, b])
                    window[:] = [min(window[0], a), max(window[1], b)]
                except (TypeError, ValueError):
                    pass
            # Include failed/warmup/drain requests in diagnostic timelines.
            timeline.append(
                {
                    k: row[k]
                    for k in (
                        "request_id",
                        "session_id",
                        "parent_id",
                        "correlation_id",
                        "turn_index",
                        "start_ns",
                        "end_ns",
                        "phase",
                        "status",
                        "error",
                        "ttft_s",
                        "e2el_s",
                        "input_tokens",
                        "output_tokens",
                    )
                    if k in row
                }
            )
            if len(timeline) >= 1000:
                flush()
            identity = row.get("request_id")
            if not identity or identity in seen:
                dropped["duplicate_or_missing_request_id"] += 1
                counts["invalid"] += 1
                continue
            seen.add(identity)
            if row.get("phase") not in ("profiling", "warmup", "drain"):
                counts["invalid"] += 1
                dropped["unknown_phase"] += 1
                continue
            if row.get("status") not in ("success", "error", "cancelled"):
                counts["invalid"] += 1
                dropped["unknown_status"] += 1
                continue
            if row.get("status") != "success":
                counts[row.get("status", "error")] += 1
                continue
            if row.get("phase") != "profiling":
                counts["non_profiling"] += 1
                continue
            problems = valid_request(row)
            if problems:
                dropped.update(problems)
                counts["invalid"] += 1
                continue
            counts["profiled_success"] += 1
            missing_session += not bool(row.get("session_id"))
            a, b = int(row["start_ns"]), int(row["end_ns"])
            start_ns = a if start_ns is None else min(start_ns, a)
            end_ns = b if end_ns is None else max(end_ns, b)
            end_seconds[b // 10**9] += 1
            for name in ("ttft", "e2el", "native_itl"):
                value = number(row.get(f"{name}_s"))
                if value is not None and value > 0:
                    samples[name].append(value)
            if "native_itl_s" not in row:
                samples["native_itl"].extend(
                    v
                    for v in row.get("chunk_itls_s", [])
                    if number(v) is not None and v > 0
                )
            full_itl = full_response_itl(row)
            if full_itl is not None:
                samples["full_response_itl"].append(full_itl)
            for key, field in (
                ("input", "input_tokens"),
                ("output_actual", "output_tokens"),
                ("output_expected", "expected_output_tokens"),
            ):
                value = number(row.get(field))
                if value is not None and value >= 0:
                    samples[key].append(value)
            totals["input"] += row["input_tokens"]
            totals["output"] += row["output_tokens"]
            if row["output_tokens"] > 0:
                samples["e2el_per_osl"].append(row["e2el_s"] / row["output_tokens"])
            if row["output_tokens"] > 1:
                samples["tpot"].append(
                    (row["e2el_s"] - row["ttft_s"]) / (row["output_tokens"] - 1)
                )
            hit, total = (
                number(row.get("cached_input_tokens")),
                number(row.get("usage_input_tokens")),
            )
            if hit is not None and total is not None and 0 <= hit <= total:
                cache_hit += hit
                cache_total += total
                cache_samples += 1
    flush()
    duration = (end_ns - start_ns) / 1e9 if start_ns is not None else None
    gpu_count = config["hardware"]["gpu_count"]
    throughput = {
        "duration_seconds": duration,
        "window": "first_start_to_last_end_of_eligible_requests",
    }
    if duration and duration > 0:
        throughput.update(
            {
                name: {"tokens": n, "tokens_per_second": n / duration}
                for name, n in {**totals, "total": sum(totals.values())}.items()
            }
        )
        throughput["per_gpu"] = {
            f"{name}_tput_tps": throughput[name]["tokens_per_second"] / gpu_count
            for name in ("input", "output", "total")
        }
    sample_stats = {name: stats(values) for name, values in samples.items()}
    latency = {
        name: sample_stats[name]
        for name in ("ttft", "e2el", "native_itl", "tpot", "full_response_itl")
    }
    latency["itl"] = latency["full_response_itl"]
    latency["intvty"] = inverse_stats(
        samples["full_response_itl"], sample_stats["full_response_itl"]
    )
    latency["full_response_intvty"] = latency["intvty"]
    latency["e2e_norm_intvty"] = inverse_stats(
        samples["e2el_per_osl"], sample_stats["e2el_per_osl"]
    )
    summary = {
        "aggregation_version": AGGREGATION_VERSION,
        "requests_sha256": request_hash.hexdigest(),
        "accounting": {
            "counts": dict(counts),
            "phases": dict(phases),
            "invalid_reasons": dict(dropped),
        },
        "window": {
            "start_ns": str(start_ns) if start_ns is not None else None,
            "end_ns": str(end_ns) if end_ns is not None else None,
        },
        "phase_windows": {
            k: {"start_ns": str(v[0]), "end_ns": str(v[1])}
            for k, v in phase_windows.items()
        },
        "request_metrics": {
            "latency": latency,
            "throughput": throughput,
            "tokens": {
                k: sample_stats[k]
                for k in ("input", "output_actual", "output_expected")
            },
        },
        "capabilities": {
            "latency": bool(samples["e2el"]),
            "normalized_interactivity": bool(samples["e2el_per_osl"]),
            "full_response_interactivity": bool(samples["full_response_itl"]),
            "request_timeline": bool(parts),
            "session_timeline": bool(counts["profiled_success"])
            and not missing_session,
            "server_metrics": False,
            "power": False,
        },
    }
    summary["request_metrics"]["cache"] = {
        "samples": cache_samples,
        "hit_tokens": cache_hit,
        "prompt_tokens": cache_total,
        "client_usage_hit_rate": cache_hit / cache_total if cache_total else None,
    }
    write_view(
        destination / "views" / "timeline" / "index.json",
        {"parts": parts, "timestamp_unit": "unix_ns_string"},
        summary,
    )
    write_view(
        destination / "views" / "distributions.json",
        {
            k: {"stats": sample_stats[k], "histogram": histogram(v)}
            for k, v in samples.items()
        },
        summary,
    )
    # Adaptive buckets bound the view size for multi-hour runs and retain empty buckets.
    series = []
    if start_ns is not None:
        first, last = start_ns // 10**9, end_ns // 10**9
        width = max(1, math.ceil((last - first + 1) / 3600))
        buckets = Counter()
        for second, count in end_seconds.items():
            buckets[(second - first) // width] += count
        series = [
            {
                "start_ns": str((first + i * width) * 10**9),
                "window_seconds": width,
                "completed": buckets[i],
            }
            for i in range((last - first) // width + 1)
        ]
    write_view(
        destination / "views" / "series.json", {"requests_completed": series}, summary
    )
    return summary


def metadata_errors(config):
    errors = []
    if model_key(config) not in MODEL_KEYS.values():
        errors.append("unmapped InferenceX model key")
    if config["hardware"].get("key") not in ("mi300x", "mi325x", "mi350x", "mi355x"):
        errors.append("unmapped InferenceX hardware key")
    if config["model"].get("precision") not in ("fp4", "fp8", "bf16", "int4"):
        errors.append("unmapped InferenceX precision (use canonical fp4/fp8/bf16/int4)")
    return errors


def model_key(config):
    model = config["model"]
    return model.get("key") or MODEL_KEYS.get(model["name"].removeprefix("/models/"))


def inferencex_export(config, summary, recipe_id):
    model, hardware = config["model"], config["hardware"]
    if metadata_errors(config):
        return None
    p, workload = config["parallelism"], config["workload"]
    result = {
        "model": model["name"],
        "infmax_model_prefix": model_key(config),
        "precision": model["precision"],
        "hw": hardware["key"],
        "framework": "atom",
        "spec_decoding": config["recipe"].get("spec_method", "none"),
        "disagg": False,
        "is_multinode": False,
        "tp": p["tp"],
        "pp": p.get("pp", 1),
        "ep": p.get("ep", 1),
        "dp_attention": p.get("dp_attention", False),
        "num_gpus": hardware["gpu_count"],
        "conc": workload["concurrency"],
        "recipe_fingerprint": recipe_id,
        "image": config["software"].get("image"),
        "num_requests_successful": summary["accounting"]["counts"].get(
            "profiled_success", 0
        ),
        "num_requests_total": summary["accounting"]["phases"].get("profiling", 0),
        "request_metrics": summary["request_metrics"],
    }
    if workload["kind"] == "agentic":
        result.update(scenario_type="agentic-coding", users=workload["concurrency"])
    else:
        result.update(isl=workload.get("isl"), osl=workload.get("osl"))
    if "server_metrics" in summary:
        result["server_metrics"] = summary["server_metrics"]
    power = summary.get("power", {})
    if power.get("valid"):
        result.update(
            power_valid=1,
            power_metric_schema_version=2,
            avg_total_gpu_power_w=power["avg_total_gpu_power_w"],
            avg_power_w=power["avg_total_gpu_power_w"] / hardware["gpu_count"],
            total_gpu_energy_j=power["energy_j"],
        )
    return result
