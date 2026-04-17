#!/usr/bin/env python3
"""Convert accuracy test JSON results to github-action-benchmark input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_model_configs(models_path: Path) -> dict[str, dict]:
    """Load models_accuracy.json and index by model_name."""
    models = json.loads(models_path.read_text(encoding="utf-8"))
    return {m["model_name"]: m for m in models}


def build_entries(
    result_dir: Path,
    run_url: str | None,
    model_configs: dict[str, dict],
    backend: str = "ATOM",
) -> list[dict]:
    entries: list[dict] = []

    for artifact_dir in sorted(result_dir.iterdir()):
        if not artifact_dir.is_dir():
            continue

        # Artifact name format: "accuracy-ModelName"
        model_name = artifact_dir.name
        if model_name.startswith("accuracy-"):
            model_name = model_name[len("accuracy-") :]

        # Find the latest JSON result file
        json_files = sorted(artifact_dir.glob("*.json"), reverse=True)
        if not json_files:
            continue

        result_file = json_files[0]
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        # Extract accuracy scores
        results = data.get("results", {})
        gsm8k = results.get("gsm8k", {})
        score = gsm8k.get("exact_match,flexible-extract")
        if score is None:
            continue
        strict_score = gsm8k.get("exact_match,strict-match")

        # Lookup model config for threshold and baseline
        cfg = model_configs.get(model_name, {})

        # Build extra metadata
        extra_parts = []
        if run_url:
            extra_parts.append(f"Run: {run_url}")

        threshold = cfg.get("accuracy_threshold")
        if threshold is not None:
            extra_parts.append(f"Threshold: {threshold}")

        baseline = cfg.get("accuracy_baseline")
        if baseline is not None:
            extra_parts.append(f"Baseline: {baseline}")

        baseline_model = cfg.get("accuracy_baseline_model")
        if baseline_model:
            extra_parts.append(f"BaselineModel: {baseline_model}")

        baseline_note = cfg.get("_baseline_note")
        if baseline_note:
            extra_parts.append(f"BaselineNote: {baseline_note}")

        ci_metadata = data.get("atom_ci_metadata", {})
        docker_image = ci_metadata.get("docker_image")
        if docker_image:
            extra_parts.append(f"Docker: {docker_image}")
        gpu_name = ci_metadata.get("gpu_name")
        if gpu_name:
            extra_parts.append(f"GPU: {gpu_name}")
        gpu_vram_gb = ci_metadata.get("gpu_vram_gb")
        if gpu_vram_gb not in (None, ""):
            try:
                gpu_vram_val = float(gpu_vram_gb)
            except (TypeError, ValueError):
                gpu_vram_val = None
            if gpu_vram_val is not None:
                if gpu_vram_val.is_integer():
                    extra_parts.append(f"VRAM: {int(gpu_vram_val)}GB")
                else:
                    extra_parts.append(f"VRAM: {gpu_vram_val:g}GB")
        rocm_version = ci_metadata.get("rocm_version")
        if rocm_version:
            extra_parts.append(f"ROCm: {rocm_version}")

        try:
            if strict_score is not None:
                extra_parts.append(f"strict-match: {round(float(strict_score), 4)}")
        except (TypeError, ValueError):
            pass

        # Include num_fewshot: check configs.gsm8k first, then top-level config
        lm_config = data.get("config", {})
        task_configs = data.get("configs", {})
        num_fewshot = task_configs.get("gsm8k", {}).get("num_fewshot") or lm_config.get(
            "num_fewshot"
        )
        if num_fewshot is not None:
            extra_parts.append(f"fewshot: {num_fewshot}")

        model_args = lm_config.get("model_args", "")
        if isinstance(model_args, str) and model_args:
            for arg in model_args.split(","):
                if arg.startswith("model="):
                    extra_parts.append(f"Model: {arg[6:]}")
                    break
        elif isinstance(model_args, dict) and "model" in model_args:
            extra_parts.append(f"Model: {model_args['model']}")

        extra = " | ".join(extra_parts) if extra_parts else None

        try:
            score_val = round(float(score), 4)
        except (TypeError, ValueError):
            continue

        entry = {
            "name": f"{backend}::{model_name} accuracy (GSM8K)",
            "unit": "score",
            "value": score_val,
        }
        if extra:
            entry["extra"] = extra
        entries.append(entry)

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert accuracy test results to github-action-benchmark input"
    )
    parser.add_argument(
        "result_dir", help="Directory containing downloaded accuracy artifacts"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--run-url", default=None, help="GitHub Actions run URL")
    parser.add_argument("--backend", default="ATOM", help="Backend label")
    parser.add_argument(
        "--models",
        required=True,
        help="Path to models_accuracy.json (contains threshold, baseline, baseline_model)",
    )
    args = parser.parse_args()

    model_configs = _load_model_configs(Path(args.models))

    result_dir = Path(args.result_dir)
    entries = build_entries(result_dir, args.run_url, model_configs, args.backend)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Generated {len(entries)} accuracy entries at {output_path}")


if __name__ == "__main__":
    main()
