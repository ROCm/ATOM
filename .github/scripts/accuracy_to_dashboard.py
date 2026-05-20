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

        # Lookup model config for threshold and baseline. We need the config
        # before extracting scores because the accuracy task can be overridden
        # per-model in models_accuracy.json (e.g. Gemma 4 uses gsm8k_cot).
        cfg = model_configs.get(model_name, {})
        accuracy_task = cfg.get("accuracy_task", "gsm8k")

        # Extract accuracy scores. lm_eval keys results by task name, and the
        # primary metric depends on the task: atom-test.yaml uses
        # exact_match,strict-match for gsm8k_cot and exact_match,flexible-extract
        # for plain gsm8k (see RESULT_METRIC selection around line 562). The
        # dashboard MUST publish the same metric the CI threshold / baseline
        # refer to, otherwise reviewers see a dashboard score that doesn't
        # match the threshold check in the PR run.
        if accuracy_task == "gsm8k_cot":
            primary_metric = "exact_match,strict-match"
            secondary_metric = "exact_match,flexible-extract"
        else:
            primary_metric = "exact_match,flexible-extract"
            secondary_metric = "exact_match,strict-match"

        results = data.get("results", {})
        task_results = results.get(accuracy_task, {})
        score = task_results.get(primary_metric)
        if score is None:
            continue
        # Keep the non-primary metric around for the "extra" metadata field
        # so reviewers still see both numbers, just labelled clearly.
        strict_score = task_results.get(secondary_metric)

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
                # Label the secondary metric by what it actually is, not the
                # hard-coded "strict-match" string. For gsm8k the secondary
                # is strict-match; for gsm8k_cot it's flexible-extract.
                secondary_label = secondary_metric.split(",", 1)[-1]
                extra_parts.append(f"{secondary_label}: {round(float(strict_score), 4)}")
        except (TypeError, ValueError):
            pass

        # Include num_fewshot: check configs.<task> first, then top-level config
        lm_config = data.get("config", {})
        task_configs = data.get("configs", {})
        num_fewshot = task_configs.get(accuracy_task, {}).get("num_fewshot") or lm_config.get(
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

        # MTP acceptance rate (extracted from server log during accuracy test)
        mtp_rate = ci_metadata.get("mtp_acceptance_rate")
        if mtp_rate is not None:
            mtp_entry = {
                "name": f"{backend}::{model_name} MTP acceptance (%)",
                "unit": "%",
                "value": round(float(mtp_rate), 2),
            }
            if extra:
                mtp_entry["extra"] = extra
            entries.append(mtp_entry)

            avg_toks = ci_metadata.get("avg_tokens_per_forward")
            if avg_toks is not None:
                entries.append(
                    {
                        "name": f"{backend}::{model_name} avg toks/fwd (tok/fwd)",
                        "unit": "tok/fwd",
                        "value": round(float(avg_toks), 2),
                    }
                )

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
