#!/usr/bin/env python3
"""Convert ATOMesh real P/D benchmark artifacts to dashboard input."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from interactivity import (
    METHOD_MEDIAN_TPOT,
    METHOD_P90_E2E,
    agentic_interactivity,
    locate_records,
)

AGENTIC_BENCHMARK_KIND = "aiperf_agentic"

# How each interactivity definition is spelled out in the markdown summary, so a
# reader never has to guess which formula produced the number in the column.
INTERACTIVITY_LABELS = {
    METHOD_P90_E2E: "P90 E2E Normalized",
    METHOD_MEDIAN_TPOT: "1 / median_tpot_s",
}

RESULT_RE = re.compile(
    r"^pd-(?P<backend>[^-]+)-(?P<model>.+)-(?P<topology>[^-]+(?:-[^-]+)*)-"
    r"isl(?P<isl>\d+)-osl(?P<osl>\d+)-conc(?P<conc>\d+)-(?P<ratio>[0-9.]+)\.json$"
)
TOPOLOGY_RE = re.compile(r"(?P<p>\d+)p(?P<d>\d+)d", re.IGNORECASE)
TP_RE = re.compile(r"tp(?P<tp>\d+)", re.IGNORECASE)
DCP_RE = re.compile(r"dcp(?P<dcp>\d+)", re.IGNORECASE)
EVAL_CONC_RE = re.compile(r"(?:^|[_-])c(?P<conc>\d+)(?:$|[_-])", re.IGNORECASE)
EVAL_TOPOLOGY_RE = re.compile(
    r"(?:^|[_-])(?P<topology>\d+p\d+d(?:[_-]dpa)?)(?:$|[_-])",
    re.IGNORECASE,
)


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


def read_env_file(path: Path) -> dict[str, str]:
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def slurm_job_env(path: Path) -> dict[str, str]:
    for parent in path.parents:
        env_path = parent / "docker.env"
        if env_path.is_file():
            return read_env_file(env_path)
        for env_path in sorted(parent.glob("docker-rank-*.env")):
            return read_env_file(env_path)
    return {}


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


def speculative_label(payload: dict[str, Any]) -> str | None:
    method = string_value(
        payload.get("speculative_method"), payload.get("spec_method")
    ).lower()
    tokens = int_value(
        payload.get("num_speculative_tokens"), payload.get("num_spec_tokens")
    )
    # Separated, because concatenating a method that already ends in a digit onto
    # its token count is unreadable: eagle3 with 2 tokens rendered as "eagle32".
    if method and tokens is not None:
        return f"{method}-{tokens}"
    return method or None


def pd_label(prefill: Any, decode: Any) -> str:
    """Render a prefill/decode pair as P8/D4, or as a bare value when they agree."""
    if prefill is not None and decode is not None:
        return str(prefill) if prefill == decode else f"P{prefill}/D{decode}"
    value = prefill if prefill is not None else decode
    return "--" if value is None else str(value)


def interactivity_value(payload: dict[str, Any]) -> float | None:
    # An already-resolved value wins: apply_p90_e2e_interactivity() writes the
    # p90 e2e normalized number here, and without this branch perf_point() would
    # silently re-derive the legacy median-TPOT value and overwrite it.
    explicit = number(payload.get("interactivity"))
    if explicit and explicit > 0:
        return explicit

    median_tpot = number(payload.get("median_tpot_ms"), payload.get("median_itl_ms"))
    if median_tpot and median_tpot > 0:
        return 1000.0 / median_tpot

    tpot = number(payload.get("mean_tpot_ms"), payload.get("mean_itl_ms"))
    if tpot and tpot > 0:
        return 1000.0 / tpot

    return None


def apply_agentic_interactivity(
    path: Path, payload: dict[str, Any], fields: dict[str, Any]
) -> None:
    """Set both interactivity definitions from the per-request AIPerf records.

    Agentic traces run a ~1M-token prefill per turn, so 1000/median_TPOT sees
    only the decode phase and hides the prefill cost entirely. The InferenceX
    definition amortizes TTFT over the turn's output tokens and takes p90 of the
    result -- see interactivity.py. It needs profile_export.jsonl, which only
    AIPerf writes, so standard ISL/OSL runs keep the legacy formula and are
    tagged as such.

    The same pass also yields the plain 1/p90(ITL) number InferenceX plots as
    "Interactivity", stored alongside as ``interactivity_p90_itl`` so the
    dashboard can offer both as x-axes for the same point.
    """
    if string_value(payload.get("benchmark_kind")) != AGENTIC_BENCHMARK_KIND:
        payload.setdefault("interactivity_method", METHOD_MEDIAN_TPOT)
        return

    records = locate_records(
        path,
        model=string_value(fields.get("model")) or None,
        topology=string_value(fields.get("topology")) or None,
        concurrency=int_value(fields.get("conc")),
        artifact_dir=string_value(payload.get("aiperf_artifact_dir")) or None,
    )
    if records is None:
        print(
            f"WARNING: {path.name} is an agentic result but no per-request "
            f"records were found next to it; falling back to "
            f"{METHOD_MEDIAN_TPOT} interactivity",
            file=sys.stderr,
        )
        payload["interactivity_method"] = METHOD_MEDIAN_TPOT
        return

    try:
        result = agentic_interactivity(records)
    except (OSError, ValueError) as exc:
        print(
            f"WARNING: cannot compute {METHOD_P90_E2E} interactivity from "
            f"{records}: {exc}; falling back to {METHOD_MEDIAN_TPOT}",
            file=sys.stderr,
        )
        payload["interactivity_method"] = METHOD_MEDIAN_TPOT
        return

    payload["interactivity"] = result["value"]
    payload["interactivity_method"] = METHOD_P90_E2E
    payload["interactivity_n_requests"] = result["n_requests"]
    payload["interactivity_p90_itl"] = result["itl_value"]


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

    prefill_dcp = int_value(
        payload.get("prefill_dcp"), payload.get("prefill_decode_context_parallel_size")
    )
    decode_dcp = int_value(
        payload.get("decode_dcp"), payload.get("decode_context_parallel_size")
    )
    dcp = DCP_RE.search(text)
    if dcp:
        prefill_dcp = prefill_dcp or int(dcp.group("dcp"))
        decode_dcp = decode_dcp or int(dcp.group("dcp"))

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
        "prefill_dcp": prefill_dcp,
        "decode_dcp": decode_dcp,
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


def topology_key(value: Any) -> str:
    text = string_value(value).lower().replace("-", "_")
    match = EVAL_TOPOLOGY_RE.search(text)
    return match.group("topology").replace("-", "_") if match else text


def model_key(value: Any) -> str:
    return string_value(value).strip().rstrip("/").split("/")[-1].lower()


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
    path: Path, payload: dict[str, Any], fields: dict[str, Any], hardware: str | None
) -> dict[str, Any]:
    env = slurm_job_env(path)
    enriched = dict(payload)
    enriched.setdefault("benchmark_backend", "Atomesh")
    enriched.setdefault("dashboard_backend", "Atomesh")
    enriched.setdefault("benchmark_model_name", fields["model"])
    enriched.setdefault("topology", fields["topology"])
    enriched.setdefault(
        "display_topology", env.get("DISPLAY_TOPOLOGY", fields["topology"])
    )
    enriched.setdefault("random_input_len", fields["isl"])
    enriched.setdefault("random_output_len", fields["osl"])
    enriched.setdefault("max_concurrency", fields["conc"])
    enriched.setdefault("random_range_ratio", fields["ratio"])
    enriched.setdefault("precision", env.get("PRECISION", ""))
    enriched.setdefault("docker_image", env.get("DOCKER_IMAGE", ""))
    enriched.setdefault("prefill_workers", env.get("PREFILL_WORKERS"))
    enriched.setdefault("decode_workers", env.get("DECODE_WORKERS"))
    enriched.setdefault("prefill_tp", env.get("PREFILL_TP"))
    enriched.setdefault("decode_tp", env.get("DECODE_TP"))
    runner = env.get("SLURM_SUBMIT_RUNNER", "")
    if hardware:
        enriched["hardware"] = hardware
    elif runner == "atomesh-cicd-mi350":
        enriched["hardware"] = "MI350X"
    elif runner == "atomesh-cicd":
        enriched["hardware"] = "MI355X"

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
    apply_agentic_interactivity(path, enriched, fields)
    enriched.setdefault("interactivity", interactivity_value(enriched))
    resources = topology_resources(enriched, fields)
    total_gpu = resources["total_gpu"]
    num_prefill_gpu = resources["num_prefill_gpu"]
    num_decode_gpu = resources["num_decode_gpu"]
    input_tput = number(enriched.get("input_throughput"))
    output_tput = number(enriched.get("output_throughput"))
    total_tput = number(
        enriched.get("total_token_throughput"), enriched.get("total_throughput")
    )
    enriched.setdefault(
        "tput_per_gpu", total_tput / total_gpu if total_tput and total_gpu else None
    )
    enriched.setdefault(
        "input_tput_per_gpu",
        input_tput / num_prefill_gpu if input_tput and num_prefill_gpu else None,
    )
    output_tput_denominator = num_decode_gpu or total_gpu
    enriched.setdefault(
        "output_tput_per_gpu",
        (
            output_tput / output_tput_denominator
            if output_tput and output_tput_denominator
            else None
        ),
    )
    return enriched


def perf_point(
    path: Path,
    payload: dict[str, Any],
    fields: dict[str, Any],
    run_url: str | None,
    accuracy: dict[str, Any] | None,
) -> dict[str, Any]:
    resources = topology_resources(payload, fields)
    run_date, timestamp = parse_payload_date(payload)
    precision = string_value(
        payload.get("precision"), payload.get("dtype"), default="fp4"
    ).lower()
    hardware = string_value(
        payload.get("hardware"), payload.get("gpu_name"), default="mi355x"
    ).lower()
    if "mi350" in hardware:
        hardware = "mi350x"
    elif "mi355" in hardware:
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
    interactivity = interactivity_value(payload)
    num_speculative_tokens = int_value(
        payload.get("num_speculative_tokens"), payload.get("num_spec_tokens")
    )

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
        "benchmark_kind": string_value(payload.get("benchmark_kind")) or None,
        "scenario": string_value(payload.get("scenario")) or None,
        "public_dataset": string_value(payload.get("public_dataset")) or None,
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
        "ttft_p90": round_or_none(payload.get("p90_ttft_ms")),
        "ttft_p99": round_or_none(payload.get("p99_ttft_ms")),
        "tpot_ms": round_or_none(tpot_ms),
        "tpot_p90": round_or_none(
            payload.get("p90_tpot_ms"), payload.get("p90_itl_ms")
        ),
        "tpot_p99": round_or_none(
            payload.get("p99_tpot_ms"), payload.get("p99_itl_ms")
        ),
        "itl_ms": round_or_none(
            payload.get("mean_itl_ms"), payload.get("mean_tpot_ms")
        ),
        "itl_p90": round_or_none(
            payload.get("p90_itl_ms"), payload.get("p90_tpot_ms")
        ),
        "e2el_ms": round_or_none(payload.get("mean_e2el_ms")),
        "e2el_p90": round_or_none(payload.get("p90_e2el_ms")),
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
        "prefill_dcp": resources["prefill_dcp"],
        "decode_dcp": resources["decode_dcp"],
        "speculative_method": speculative_label(payload),
        "num_speculative_tokens": num_speculative_tokens,
        "prefill_workers": resources["prefill_workers"],
        "decode_workers": resources["decode_workers"],
        "prefill_dpa": resources["prefill_dpa"],
        "decode_dpa": resources["decode_dpa"],
        "num_prefill_gpu": resources["num_prefill_gpu"],
        "num_decode_gpu": resources["num_decode_gpu"],
        "total_gpu": total_gpu,
        "interactivity": round_or_none(interactivity),
        # Which definition produced the value above -- agentic points computed
        # from per-request records report p90_e2e_normalized, everything else
        # reports median_tpot. The dashboard labels the two differently.
        "interactivity_method": string_value(payload.get("interactivity_method"))
        or METHOD_MEDIAN_TPOT,
        "interactivity_n_requests": int_value(payload.get("interactivity_n_requests")),
        # 1/p90(ITL) -- what InferenceX plots as plain "Interactivity", as opposed
        # to the E2E-normalized number in "interactivity" above. Only points
        # computed from per-request records carry it, so its presence is its
        # definition and no companion _method field is needed.
        "interactivity_p90_itl": round_or_none(payload.get("interactivity_p90_itl")),
        # Prefill prefix-cache token hit rate as a 0-1 fraction. Absent unless the
        # case enables prefix caching and the run was long enough for the engine
        # to print a "[Cache Stats]" line.
        "cache_hit_rate": round_or_none(payload.get("cache_hit_rate")),
        "cache_hit_tokens": int_value(payload.get("cache_hit_tokens")),
        "cache_total_tokens": int_value(payload.get("cache_total_tokens")),
        "tput_per_gpu": round_or_none(
            total_tput / total_gpu if total_tput and total_gpu else None
        ),
        "input_tput_per_gpu": round_or_none(
            input_tput / resources["num_prefill_gpu"]
            if input_tput and resources["num_prefill_gpu"]
            else None
        ),
        "output_tput_per_gpu": round_or_none(
            output_tput / (resources["num_decode_gpu"] or total_gpu)
            if output_tput and (resources["num_decode_gpu"] or total_gpu)
            else None
        ),
        "run_url": run_url or "",
        "image": string_value(payload.get("docker_image"), payload.get("image")),
        "rocm": string_value(payload.get("rocm_version")),
        "slurm_job": string_value(payload.get("slurm_job_id")),
        "chart_group": "atomesh-model-performance",
        "chart_label": f"{hardware.upper()} ({display_backend} {precision.upper()})",
        "accuracy_task": accuracy.get("task") if accuracy else None,
        "accuracy_metric": accuracy.get("metric") if accuracy else None,
        "accuracy_score": (
            round_or_none(accuracy.get("value"), digits=4) if accuracy else None
        ),
        "accuracy_score_raw": accuracy.get("raw") if accuracy else None,
        "accuracy_strict": (
            round_or_none(accuracy.get("strict"), digits=4) if accuracy else None
        ),
        "accuracy_resolved": (
            int_value(accuracy.get("resolved")) if accuracy else None
        ),
        "accuracy_total": int_value(accuracy.get("total")) if accuracy else None,
        "accuracy_threshold": (
            round_or_none(accuracy.get("threshold"), digits=4) if accuracy else None
        ),
        "accuracy_fewshot": (int_value(accuracy.get("fewshot")) if accuracy else None),
    }
    if accuracy and accuracy.get("task") == "gsm8k":
        point["gsm8k"] = round_or_none(accuracy.get("value"), digits=4)
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
    accuracy_scores: dict[tuple[str, str, int], dict[str, Any]],
    hardware: str | None,
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
        payload = enrich_payload(path, payload, fields, hardware)
        conc = int(payload.get("max_concurrency", fields["conc"]))
        model = model_key(payload.get("benchmark_model_name") or fields["model"])
        topology = topology_key(fields["topology"])
        accuracy = accuracy_scores.get((model, topology, conc))
        if accuracy is None:
            accuracy = accuracy_scores.get(("", topology, conc))
        if accuracy is None:
            accuracy = accuracy_scores.get((model, "", conc))
        if accuracy is None:
            accuracy = accuracy_scores.get(("", "", conc))
        if accuracy is not None:
            payload["accuracy_task"] = accuracy.get("task")
            payload["accuracy_metric"] = accuracy.get("metric")
            payload["accuracy_score"] = accuracy.get("value")
            payload["accuracy_score_raw"] = accuracy.get("raw")
            payload["accuracy_strict"] = accuracy.get("strict")
            payload["accuracy_resolved"] = accuracy.get("resolved")
            payload["accuracy_total"] = accuracy.get("total")
            payload["accuracy_threshold"] = accuracy.get("threshold")
            payload["accuracy_fewshot"] = accuracy.get("fewshot")
            if accuracy.get("task") == "gsm8k":
                payload["gsm8k"] = accuracy.get("value")
                payload["gsm8k_raw"] = accuracy.get("raw")
        extra = extra_text(payload, run_url, payload.get("slurm_job_id"))
        point = perf_point(path, payload, fields, run_url, accuracy)
        point_entry = dashboard_point_entry(point, extra)
        if point_entry:
            entries.append(point_entry)
        rows.append(payload)
    return entries, rows


def eval_concurrency(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = EVAL_CONC_RE.search(part)
        if match:
            return int(match.group("conc"))
    return None


def eval_topology(path: Path) -> str:
    for part in reversed(path.parts):
        match = EVAL_TOPOLOGY_RE.search(part)
        if match:
            return match.group("topology").replace("-", "_").lower()
    return ""


def find_eval_scores(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    scores = {}
    for path in sorted(root.rglob("results*.json")):
        payload = read_json(path)
        if not payload:
            continue
        conc = eval_concurrency(path)
        if conc is None:
            continue
        results = payload.get("results", {})
        task = ""
        metric = ""
        score_raw = None
        strict = None
        resolved = None
        total = None

        if "swebench_lite" in results:
            task = "swebench_lite"
            metric = "resolved"
            result = results[task]
            score_raw = result.get("exact_match,resolved")
            details = payload.get("swebench", {})
            resolved = details.get("resolved")
            total = details.get("total")
        elif "gsm8k" in results:
            task = "gsm8k"
            metric = "flexible-extract"
            result = results[task]
            score_raw = next(
                (
                    value
                    for value in (
                        result.get("exact_match,flexible-extract"),
                        result.get("exact_match,strict-match"),
                        result.get("acc"),
                    )
                    if value not in (None, "")
                ),
                None,
            )
            strict = number(result.get("exact_match,strict-match"))
        else:
            continue

        score = number(score_raw)
        if score is not None:
            env = slurm_job_env(path)
            task_config = payload.get("configs", {}).get(task, {})
            global_config = payload.get("config", {})
            fewshot = task_config.get("num_fewshot")
            if fewshot is None:
                fewshot = global_config.get("num_fewshot")
            scores[(model_key(env.get("MODEL_NAME")), eval_topology(path), conc)] = {
                "task": task,
                "metric": metric,
                "value": round(score, 4),
                "raw": f"{score:.4f}",
                "strict": round(strict, 4) if strict is not None else None,
                "resolved": int_value(resolved),
                "total": int_value(total),
                "threshold": number(env.get("EVAL_THRESHOLD")),
                "fewshot": int_value(fewshot),
            }
    return scores


INTERACTIVITY_HEADER = "Interactivity"

# (header, right-aligned). The interactivity header is rewritten per table by
# summary_headers() so it can name the definition behind its numbers.
SUMMARY_LAYOUT: list[tuple[str, bool]] = [
    ("Hardware", False),
    ("Model", False),
    ("Topology", False),
    ("ISL/OSL", False),
    ("Concurrency", True),
    (INTERACTIVITY_HEADER, True),
    ("TP", False),
    ("DCP", False),
    ("Spec", False),
    ("Total tok/s", True),
    ("Input tok/s", True),
    ("Output tok/s", True),
    ("Total tok/s/GPU", True),
    ("Input tok/s/GPU", True),
    ("Output tok/s/GPU", True),
    ("TTFT mean ms", True),
    ("TTFT p90 ms", True),
    ("TTFT p99 ms", True),
    ("TPOT mean ms", True),
    ("TPOT p90 ms", True),
    ("TPOT p99 ms", True),
    ("E2E mean ms", True),
    ("E2E p90 ms", True),
    ("E2E p99 ms", True),
    ("Cache Hit", True),
    ("Accuracy Task", False),
    ("Accuracy", True),
]
INTERACTIVITY_COLUMN = SUMMARY_LAYOUT.index((INTERACTIVITY_HEADER, True))


def interactivity_method(row: dict[str, Any]) -> str:
    return string_value(row.get("interactivity_method")) or METHOD_MEDIAN_TPOT


def summary_headers(rows: list[dict[str, Any]]) -> list[str]:
    """Column headers, with the interactivity definition named in its own header.

    Only possible when every row shares one definition. The aggregate summary
    merges agentic and fixed ISL/OSL cases, so a mixed table is normal there and
    keeps the plain header; summary_note() carries the definitions instead.
    """
    methods = {interactivity_method(row) for row in rows}
    label = INTERACTIVITY_LABELS.get(methods.pop()) if len(methods) == 1 else None
    headers = [header for header, _ in SUMMARY_LAYOUT]
    if label:
        headers[INTERACTIVITY_COLUMN] = f"{INTERACTIVITY_HEADER} ({label})"
    return headers


def summary_note(rows: list[dict[str, Any]]) -> str:
    """Spell out the interactivity definitions when the header cannot.

    Which one applies to a row is decided entirely by whether that row is an
    agentic trace, so naming both is a complete specification -- no per-row
    column needed.
    """
    if len({interactivity_method(row) for row in rows}) < 2:
        return ""
    return (
        f"Interactivity definition varies by row: "
        f"{INTERACTIVITY_LABELS[METHOD_P90_E2E]} for agentic traces, "
        f"{INTERACTIVITY_LABELS[METHOD_MEDIAN_TPOT]} for fixed ISL/OSL cases."
    )


def summary_cells(row: dict[str, Any]) -> list[str]:
    resources = topology_resources(row, {})
    return [
        string_value(row.get("hardware"), default="--"),
        string_value(row.get("benchmark_model_name"), default="--"),
        string_value(row.get("display_topology"), row.get("topology"), default="--"),
        "{}/{}".format(
            row.get("random_input_len", "--"), row.get("random_output_len", "--")
        ),
        string_value(row.get("max_concurrency"), default="--"),
        fmt(row.get("interactivity")),
        pd_label(resources["prefill_tp"], resources["decode_tp"]),
        pd_label(resources["prefill_dcp"], resources["decode_dcp"]),
        speculative_label(row) or "--",
        fmt(row.get("total_token_throughput")),
        fmt(row.get("input_throughput")),
        fmt(row.get("output_throughput")),
        fmt(row.get("tput_per_gpu")),
        fmt(row.get("input_tput_per_gpu")),
        fmt(row.get("output_tput_per_gpu")),
        fmt(row.get("mean_ttft_ms")),
        fmt(row.get("p90_ttft_ms")),
        fmt(row.get("p99_ttft_ms")),
        fmt(number(row.get("mean_tpot_ms"), row.get("mean_itl_ms"))),
        fmt(number(row.get("p90_tpot_ms"), row.get("p90_itl_ms"))),
        fmt(number(row.get("p99_tpot_ms"), row.get("p99_itl_ms"))),
        fmt(row.get("mean_e2el_ms")),
        fmt(row.get("p90_e2el_ms")),
        fmt(row.get("p99_e2el_ms")),
        fmt_pct(row.get("cache_hit_rate")),
        string_value(row.get("accuracy_task"), default="--"),
        fmt(row.get("accuracy_score"), digits=4),
    ]


def write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    headers = summary_headers(rows)
    lines = [
        "### ATOMesh Model Performance Benchmark Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if right else "---" for _, right in SUMMARY_LAYOUT)
        + " |",
    ]
    lines.extend("| " + " | ".join(summary_cells(row)) + " |" for row in rows)
    note = summary_note(rows)
    if note:
        lines.extend(["", note])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    return "--" if parsed is None else f"{parsed:.{digits}f}"


def fmt_pct(value: Any, digits: int = 2) -> str:
    """Render a 0-1 fraction as a percentage; "--" when the metric is missing."""
    parsed = number(value)
    return "--" if parsed is None else f"{parsed * 100:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", help="Directory containing benchmark artifacts")
    parser.add_argument(
        "--output", required=True, help="benchmark-action JSON output path"
    )
    parser.add_argument("--summary", default="benchmark-summary.md")
    parser.add_argument("--run-url", default=None)
    parser.add_argument("--hardware", default=None)
    args = parser.parse_args()

    root = Path(args.result_dir)
    bench_paths = list(root.rglob("pd-*.json"))
    accuracy_scores = find_eval_scores(root)
    entries, rows = collect_dashboard_entries(
        bench_paths, args.run_url, accuracy_scores, args.hardware
    )
    Path(args.output).write_text(json.dumps(entries, indent=2), encoding="utf-8")
    write_summary(rows, Path(args.summary))
    print(
        f"Generated {len(entries)} dashboard entries from {len(rows)} benchmark result(s) "
        f"and {len(accuracy_scores)} accuracy score(s)"
    )


if __name__ == "__main__":
    main()
