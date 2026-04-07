#!/usr/bin/env python3
"""Convert OOT benchmark JSON results to github-action-benchmark input."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VARIANT_RE = re.compile(r"-(mtp\d*)-")
DEFAULT_BACKEND = "ATOM-vLLM"


def derive_model_name(result_path: Path, payload: dict) -> str:
    display_name = payload.get("benchmark_model_name")
    if display_name:
        return str(display_name)

    model = str(payload.get("model_id", "")).split("/")[-1]
    if not model:
        model = result_path.stem

    match = VARIANT_RE.search(result_path.stem)
    if match:
        model = f"{model}-{match.group(1)}"

    return model


def append_metric(
    entries: list[dict],
    *,
    label_prefix: str,
    metric_label: str,
    unit: str,
    value: object,
    extra: str | None = None,
) -> None:
    if value is None:
        return

    entry = {
        "name": f"{label_prefix} {metric_label}",
        "unit": unit,
        "value": round(float(value), 2),
    }
    if extra:
        entry["extra"] = extra
    entries.append(entry)


def is_dashboard_publish_allowed(payload: dict) -> bool:
    publish_flag = payload.get("dashboard_publish_allowed")
    if publish_flag is None:
        return True
    if isinstance(publish_flag, bool):
        return publish_flag
    return str(publish_flag).strip().lower() not in {"0", "false", "no"}


def build_entries(result_dir: Path, run_url: str | None) -> list[dict]:
    entries: list[dict] = []

    for result_path in sorted(result_dir.glob("*.json")):
        if result_path.name == "regression_report.json":
            continue

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        if not is_dashboard_publish_allowed(payload):
            continue

        if "output_throughput" not in payload:
            continue

        model = derive_model_name(result_path, payload)
        isl = int(payload.get("random_input_len", 0))
        osl = int(payload.get("random_output_len", 0))
        conc = int(payload.get("max_concurrency", 0))
        label_prefix = f"{DEFAULT_BACKEND}::{model} {isl}/{osl} c={conc}"
        extra = f"Run: {run_url}" if run_url else ""
        gpu_name = payload.get("gpu_name", "")
        gpu_vram = payload.get("gpu_vram_gb", 0)
        rocm_ver = payload.get("rocm_version", "")
        oot_image_tag = payload.get("oot_image_tag", "")
        if gpu_name:
            extra += f" | GPU: {gpu_name}"
        if gpu_vram:
            extra += f" | VRAM: {gpu_vram}GB"
        if rocm_ver:
            extra += f" | ROCm: {rocm_ver}"
        if oot_image_tag:
            extra += f" | Docker: {oot_image_tag}"
        extra = extra or None

        append_metric(
            entries,
            label_prefix=label_prefix,
            metric_label="throughput (tok/s)",
            unit="tok/s",
            value=payload.get("output_throughput"),
            extra=extra,
        )
        append_metric(
            entries,
            label_prefix=label_prefix,
            metric_label="Total Tput (tok/s)",
            unit="tok/s",
            value=payload.get("total_token_throughput"),
            extra=extra,
        )
        append_metric(
            entries,
            label_prefix=label_prefix,
            metric_label="TTFT (ms)",
            unit="ms",
            value=payload.get("mean_ttft_ms"),
            extra=extra,
        )
        append_metric(
            entries,
            label_prefix=label_prefix,
            metric_label="TPOT (ms)",
            unit="ms",
            value=payload.get("mean_tpot_ms"),
            extra=extra,
        )

        gpu_count = payload.get("tensor_parallel_size")
        if gpu_count is not None:
            entries.append(
                {
                    "name": f"{label_prefix} _gpu_count",
                    "unit": "",
                    "value": int(gpu_count),
                }
            )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OOT benchmark JSON files to github-action-benchmark input"
    )
    parser.add_argument(
        "result_dir", help="Directory containing OOT benchmark JSON files"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--run-url",
        default=None,
        help="Optional GitHub Actions run URL added to each metric as extra metadata",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    entries = build_entries(result_dir, args.run_url)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Generated {len(entries)} OOT entries at {output_path}")


if __name__ == "__main__":
    main()
