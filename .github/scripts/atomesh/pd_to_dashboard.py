#!/usr/bin/env python3
"""Convert ATOMesh real P/D benchmark artifacts to dashboard input."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

RESULT_RE = re.compile(
    r"^pd-(?P<backend>[^-]+)-(?P<model>.+)-(?P<topology>[^-]+(?:-[^-]+)*)-"
    r"isl(?P<isl>\d+)-osl(?P<osl>\d+)-conc(?P<conc>\d+)-(?P<ratio>[0-9.]+)\.json$"
)
TOPOLOGY_RE = re.compile(r"(?P<p>\d+)p(?P<d>\d+)d", re.IGNORECASE)
TP_RE = re.compile(r"tp(?P<tp>\d+)", re.IGNORECASE)


def number(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def metric_entry(
    name: str, unit: str, value: float | None, extra: str | None
) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "name": name,
        "unit": unit,
        "value": round(float(value), 4),
        **({"extra": extra} if extra else {}),
    }


def string_value(*values: Any, default: str = "") -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return default


def int_value(*values: Any) -> int | None:
    parsed = number(*values)
    return int(parsed) if parsed is not None else None


def round_or_none(*values: Any, digits: int = 4) -> float | None:
    parsed = number(*values)
    return round(parsed, digits) if parsed is not None else None


def parse_payload_date(payload: dict[str, Any]) -> tuple[str | None, int | None]:
    raw = string_value(payload.get("date"), payload.get("created_at"))
    for fmt in ("%Y%m%d-%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw[:19] if "T" in fmt else raw, fmt)
            return dt.strftime("%Y-%m-%d"), int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None, None


def topology_resources(
    payload: dict[str, Any], fields: dict[str, Any]
) -> dict[str, int | None | bool]:
    text = " ".join(
        string_value(value)
        for value in (
            payload.get("display_topology"),
            payload.get("topology"),
            fields.get("topology"),
        )
    )
    topology = TOPOLOGY_RE.search(text)
    tp = TP_RE.search(text)
    prefill_workers = int_value(
        payload.get("prefill_workers"), payload.get("num_prefill_workers")
    )
    decode_workers = int_value(
        payload.get("decode_workers"), payload.get("num_decode_workers")
    )
    if topology:
        prefill_workers = prefill_workers or int(topology.group("p"))
        decode_workers = decode_workers or int(topology.group("d"))

    prefill_tp = int_value(
        payload.get("prefill_tp"), payload.get("prefill_tensor_parallel_size")
    )
    decode_tp = int_value(
        payload.get("decode_tp"), payload.get("decode_tensor_parallel_size")
    )
    if tp:
        prefill_tp = prefill_tp or int(tp.group("tp"))
        decode_tp = decode_tp or int(tp.group("tp"))

    num_prefill_gpu = int_value(payload.get("num_prefill_gpu"))
    num_decode_gpu = int_value(payload.get("num_decode_gpu"))
    if num_prefill_gpu is None and prefill_workers and prefill_tp:
        num_prefill_gpu = prefill_workers * prefill_tp
    if num_decode_gpu is None and decode_workers and decode_tp:
        num_decode_gpu = decode_workers * decode_tp
    total_gpu = int_value(payload.get("total_gpu"))
    if total_gpu is None and num_prefill_gpu is not None and num_decode_gpu is not None:
        total_gpu = num_prefill_gpu + num_decode_gpu

    lowered = text.lower()
    return {
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "prefill_tp": prefill_tp,
        "decode_tp": decode_tp,
        "num_prefill_gpu": num_prefill_gpu,
        "num_decode_gpu": num_decode_gpu,
        "total_gpu": total_gpu,
        "prefill_dpa": bool(
            payload.get("prefill_dpa")
            or payload.get("prefill_dp_attention")
            or "dpa" in lowered
        ),
        "decode_dpa": bool(
            payload.get("decode_dpa")
            or payload.get("decode_dp_attention")
            or "dpa" in lowered
        ),
    }


def extra_text(
    payload: dict[str, Any], run_url: str | None, slurm_job: str | None
) -> str:
    parts = []
    if run_url:
        parts.append(f"Run: {run_url}")
    if slurm_job:
        parts.append(f"slurm_job={slurm_job}")
    for key in (
        "gpu_name",
        "rocm_version",
        "docker_image",
        "precision",
        "display_topology",
        "random_range_ratio",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def perf_point_extra(base_extra: str, point: dict[str, Any]) -> str:
    encoded = quote(json.dumps(point, separators=(",", ":"), sort_keys=True), safe="")
    return " | ".join(part for part in (base_extra, f"perf_point={encoded}") if part)


def derive_fields(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    match = RESULT_RE.match(path.name)
    if match:
        fields = match.groupdict()
        fields["isl"] = int(fields["isl"])
        fields["osl"] = int(fields["osl"])
        fields["conc"] = int(fields["conc"])
        return fields

    model = payload.get("benchmark_model_name") or payload.get("model_id")
    if not model:
        return None
    return {
        "backend": payload.get("benchmark_backend") or payload.get("backend") or "atom",
        "model": str(model).split("/")[-1],
        "topology": payload.get("topology")
        or payload.get("display_topology")
        or "unknown",
        "isl": int(payload.get("random_input_len", 0)),
        "osl": int(payload.get("random_output_len", 0)),
        "conc": int(payload.get("max_concurrency", 0)),
        "ratio": str(payload.get("random_range_ratio", "")),
    }


def enrich_payload(
    path: Path, payload: dict[str, Any], fields: dict[str, Any]
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("benchmark_backend", "Atomesh")
    enriched.setdefault("dashboard_backend", "Atomesh")
    enriched.setdefault("benchmark_model_name", fields["model"])
    enriched.setdefault("topology", fields["topology"])
    enriched.setdefault("display_topology", fields["topology"])
    enriched.setdefault("random_input_len", fields["isl"])
    enriched.setdefault("random_output_len", fields["osl"])
    enriched.setdefault("max_concurrency", fields["conc"])
    enriched.setdefault("random_range_ratio", fields["ratio"])

    if "total_token_throughput" not in enriched:
        enriched["total_token_throughput"] = number(
            enriched.get("total_token_throughput"),
            enriched.get("total_throughput"),
        )
    if "input_throughput" not in enriched:
        total_input_tokens = number(enriched.get("total_input_tokens"))
        duration = number(
            enriched.get("benchmark_duration_s"), enriched.get("duration")
        )
        if total_input_tokens and duration:
            enriched["input_throughput"] = total_input_tokens / duration
    if "mean_e2el_ms" not in enriched:
        enriched["mean_e2el_ms"] = number(
            enriched.get("mean_e2el_ms"),
            enriched.get("mean_e2e_latency_ms"),
            enriched.get("mean_latency_ms"),
        )
    enriched.setdefault(
        "mean_tpot_ms",
        number(enriched.get("mean_tpot_ms"), enriched.get("mean_itl_ms")),
    )
    return enriched


def perf_point(
    path: Path,
    payload: dict[str, Any],
    fields: dict[str, Any],
    run_url: str | None,
    gsm8k: float | None,
) -> dict[str, Any]:
    resources = topology_resources(payload, fields)
    run_date, timestamp = parse_payload_date(payload)
    precision = string_value(
        payload.get("precision"), payload.get("dtype"), default="fp4"
    ).lower()
    hardware = string_value(
        payload.get("hardware"), payload.get("gpu_name"), default="mi355x"
    ).lower()
    if "mi355" in hardware:
        hardware = "mi355x"
    backend = string_value(
        payload.get("backend"), fields.get("backend"), default="atom"
    ).lower()
    display_backend = backend if backend.startswith("atomesh") else f"atomesh-{backend}"
    ratio = number(payload.get("random_range_ratio"), fields.get("ratio"))
    total_gpu = resources["total_gpu"]
    output_tput = number(payload.get("output_throughput"))
    total_tput = number(
        payload.get("total_token_throughput"), payload.get("total_throughput")
    )
    input_tput = number(payload.get("input_throughput"))
    tpot_ms = number(payload.get("mean_tpot_ms"), payload.get("mean_itl_ms"))
    interactivity = number(payload.get("interactivity"))
    if interactivity is None and tpot_ms:
        interactivity = 1000.0 / tpot_ms

    config_label = "_".join(
        part
        for part in (
            hardware,
            display_backend,
            precision,
            string_value(payload.get("display_topology"), fields.get("topology"))
            .lower()
            .replace("-", "_"),
        )
        if part
    )
    point = {
        "run_id": path.stem,
        "date": run_date,
        "timestamp": timestamp,
        "source": "ATOMesh",
        "client_bench": "inferencemax bench",
        "model": string_value(
            payload.get("benchmark_model_name"), fields.get("model"), default="unknown"
        ),
        "backend": display_backend,
        "config_label": config_label,
        "hardware": hardware,
        "precision": precision,
        "isl": int(payload["random_input_len"]),
        "osl": int(payload["random_output_len"]),
        "concurrency": int(payload["max_concurrency"]),
        "ratio": ratio,
        "ttft_ms": round_or_none(payload.get("mean_ttft_ms")),
        "ttft_p99": round_or_none(payload.get("p99_ttft_ms")),
        "tpot_ms": round_or_none(tpot_ms),
        "tpot_p99": round_or_none(
            payload.get("p99_tpot_ms"), payload.get("p99_itl_ms")
        ),
        "itl_ms": round_or_none(
            payload.get("mean_itl_ms"), payload.get("mean_tpot_ms")
        ),
        "e2el_ms": round_or_none(payload.get("mean_e2el_ms")),
        "e2el_p99": round_or_none(payload.get("p99_e2el_ms")),
        "median_ttft_ms": round_or_none(payload.get("median_ttft_ms")),
        "median_tpot_ms": round_or_none(
            payload.get("median_tpot_ms"), payload.get("median_itl_ms")
        ),
        "median_itl_ms": round_or_none(payload.get("median_itl_ms")),
        "median_e2el_ms": round_or_none(payload.get("median_e2el_ms")),
        "output_tput": round_or_none(output_tput),
        "input_tput": round_or_none(input_tput),
        "total_tput": round_or_none(total_tput),
        "req_tput": round_or_none(payload.get("request_throughput")),
        "completed": int_value(
            payload.get("completed"), payload.get("successful_requests")
        ),
        "duration": round_or_none(
            payload.get("benchmark_duration_s"), payload.get("duration")
        ),
        "num_prompts": int_value(payload.get("num_prompts")),
        "prefill_tp": resources["prefill_tp"],
        "decode_tp": resources["decode_tp"],
        "prefill_workers": resources["prefill_workers"],
        "decode_workers": resources["decode_workers"],
        "prefill_dpa": resources["prefill_dpa"],
        "decode_dpa": resources["decode_dpa"],
        "num_prefill_gpu": resources["num_prefill_gpu"],
        "num_decode_gpu": resources["num_decode_gpu"],
        "total_gpu": total_gpu,
        "interactivity": round_or_none(interactivity),
        "tput_per_gpu": round_or_none(
            total_tput / total_gpu if total_tput and total_gpu else None
        ),
        "input_tput_per_gpu": round_or_none(
            input_tput / total_gpu if input_tput and total_gpu else None
        ),
        "output_tput_per_gpu": round_or_none(
            output_tput / total_gpu if output_tput and total_gpu else None
        ),
        "run_url": run_url or "",
        "image": string_value(payload.get("docker_image"), payload.get("image")),
        "rocm": string_value(payload.get("rocm_version")),
        "slurm_job": string_value(payload.get("slurm_job_id")),
        "chart_group": "atomesh-model-performance",
        "chart_label": f"{hardware.upper()} ({display_backend} {precision.upper()})",
        "gsm8k": gsm8k,
    }
    return {key: value for key, value in point.items() if value is not None}


def dashboard_point_entry(point: dict[str, Any], extra: str) -> dict[str, Any] | None:
    point_label = (
        f"Atomesh::{point['model']} {point['config_label']} "
        f"{point['isl']}/{point['osl']} c={point['concurrency']} perf point"
    )
    point_value = number(
        point.get("output_tput_per_gpu"),
        point.get("tput_per_gpu"),
        point.get("output_tput"),
        point.get("total_tput"),
    )
    return metric_entry(
        point_label, "point", point_value, perf_point_extra(extra, point)
    )


def collect_dashboard_entries(
    paths: list[Path],
    run_url: str | None,
    gsm8k: float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        if path.name.endswith("-benchmark-action.json"):
            continue
        payload = read_json(path)
        if not payload:
            continue
        fields = derive_fields(path, payload)
        if not fields:
            continue
        payload = enrich_payload(path, payload, fields)
        extra = extra_text(payload, run_url, payload.get("slurm_job_id"))
        point = perf_point(path, payload, fields, run_url, gsm8k)
        point_entry = dashboard_point_entry(point, extra)
        if point_entry:
            entries.append(point_entry)
        rows.append(payload)
    return entries, rows


def find_eval_scores(root: Path) -> list[float]:
    scores = []
    for path in sorted(root.rglob("results*.json")):
        payload = read_json(path)
        if not payload:
            continue
        result = payload.get("results", {}).get("gsm8k", {})
        score = number(
            result.get("exact_match,flexible-extract"),
            result.get("exact_match,strict-match"),
            result.get("acc"),
        )
        if score is not None:
            scores.append(score)
    return scores


def write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    lines = [
        "### ATOMesh Real P/D Benchmark Summary",
        "",
        "| Model | Topology | ISL/OSL | Concurrency | Total tok/s | Input tok/s | Output tok/s | TTFT ms | TPOT ms | E2E ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {topology} | {isl}/{osl} | {conc} | {total} | {input_} | {output} | {ttft} | {tpot} | {e2e} |".format(
                model=row.get("benchmark_model_name", "--"),
                topology=row.get("display_topology") or row.get("topology", "--"),
                isl=row.get("random_input_len", "--"),
                osl=row.get("random_output_len", "--"),
                conc=row.get("max_concurrency", "--"),
                total=fmt(row.get("total_token_throughput")),
                input_=fmt(row.get("input_throughput")),
                output=fmt(row.get("output_throughput")),
                ttft=fmt(row.get("mean_ttft_ms")),
                tpot=fmt(row.get("mean_tpot_ms")),
                e2e=fmt(row.get("mean_e2el_ms")),
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    parsed = number(value)
    return "--" if parsed is None else f"{parsed:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", help="Directory containing benchmark artifacts")
    parser.add_argument(
        "--output", required=True, help="benchmark-action JSON output path"
    )
    parser.add_argument("--summary", default="benchmark-summary.md")
    parser.add_argument("--run-url", default=None)
    args = parser.parse_args()

    root = Path(args.result_dir)
    bench_paths = list(root.rglob("pd-*.json"))
    eval_scores = find_eval_scores(root)
    gsm8k = eval_scores[0] if eval_scores else None
    entries, rows = collect_dashboard_entries(bench_paths, args.run_url, gsm8k)
    Path(args.output).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    write_summary(rows, Path(args.summary))
    print(
        f"Generated {len(entries)} dashboard entries from {len(rows)} benchmark result(s)"
    )


if __name__ == "__main__":
    main()
