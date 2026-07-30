#!/usr/bin/env python3
"""Run the official SWE-bench harness and publish ATOMesh accuracy JSON."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"
DEFAULT_TASK = "swebench_lite"


def load_predictions(path: Path, model_name: str) -> list[dict[str, str]]:
    """Load mini-swe-agent JSON/JSONL and normalize it for SWE-bench."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if isinstance(payload, dict):
        rows = list(payload.values())
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must contain a JSON object, array, or JSONL")

    predictions: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prediction {index} is not a JSON object")
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"prediction {index} has no valid instance_id")
        patch = row.get("model_patch")
        if patch is None:
            patch = ""
        if not isinstance(patch, str):
            raise ValueError(
                f"prediction {instance_id} has a non-string model_patch"
            )
        predictions[instance_id] = {
            "instance_id": instance_id,
            "model_name_or_path": model_name,
            "model_patch": patch,
        }

    if not predictions:
        raise ValueError(f"{path} contains no predictions")
    return list(predictions.values())


def write_predictions(predictions: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def run_harness(
    predictions_path: Path,
    *,
    dataset_name: str,
    run_id: str,
    work_dir: Path,
    max_workers: int,
    instance_timeout: int,
    namespace: str,
) -> None:
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        "test",
        "--predictions_path",
        str(predictions_path),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(instance_timeout),
        "--namespace",
        namespace,
    ]
    print(f"[swebench] running official harness: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=work_dir, check=True)


def find_report(work_dir: Path, model_name: str, run_id: str) -> Path:
    sanitized_model = model_name.replace("/", "__")
    preferred = work_dir / f"{sanitized_model}.{run_id}.json"
    if preferred.is_file():
        return preferred

    for path in sorted(work_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and {
            "submitted_instances",
            "resolved_instances",
        }.issubset(payload):
            return path
    raise FileNotFoundError(
        f"official SWE-bench report was not found under {work_dir}"
    )


def parse_report(report: dict[str, Any]) -> tuple[int, int]:
    resolved = report.get("resolved_instances")
    submitted = report.get("submitted_instances")
    if (
        isinstance(resolved, bool)
        or not isinstance(resolved, int)
        or isinstance(submitted, bool)
        or not isinstance(submitted, int)
        or submitted <= 0
        or resolved < 0
        or resolved > submitted
    ):
        raise ValueError(
            "invalid official report counts: "
            f"resolved_instances={resolved!r}, "
            f"submitted_instances={submitted!r}"
        )
    return resolved, submitted


def build_results(
    *,
    task_name: str,
    dataset_name: str,
    model_name: str,
    harness_version: str,
    resolved: int,
    submitted: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    rate = resolved / submitted
    stderr = math.sqrt(rate * (1.0 - rate) / submitted)
    return {
        "model_name": model_name,
        "swebench_version": harness_version,
        "results": {
            task_name: {
                "alias": task_name,
                "exact_match,resolved": rate,
                "exact_match_stderr,resolved": stderr,
            }
        },
        "configs": {
            task_name: {
                "dataset_path": dataset_name,
                "dataset_name": None,
                "test_split": "test",
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "resolved"}],
            }
        },
        "n-samples": {
            task_name: {
                "effective": submitted,
                "original": submitted,
            }
        },
        "swebench": {
            "resolved": resolved,
            "total": submitted,
            "resolved_rate": rate,
            "report": report,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score mini-swe-agent predictions with local Docker"
    )
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--instance-timeout", type=int, default=900)
    parser.add_argument("--namespace", default="swebench")
    parser.add_argument("--expected-instances", type=int)
    parser.add_argument("--harness-version", default="4.1.0")
    parser.add_argument(
        "--report",
        help="Use an existing official report instead of running Docker",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_workers <= 0 or args.instance_timeout <= 0:
        raise ValueError("worker count and instance timeout must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(
        Path(args.predictions_file),
        args.model_name,
    )
    if (
        args.expected_instances is not None
        and len(predictions) != args.expected_instances
    ):
        raise ValueError(
            f"expected {args.expected_instances} predictions, "
            f"found {len(predictions)}"
        )

    predictions_path = out_dir / "predictions.jsonl"
    write_predictions(predictions, predictions_path)
    print(
        f"[swebench] staged {len(predictions)} predictions at "
        f"{predictions_path}",
        flush=True,
    )

    if args.report:
        report_path = Path(args.report)
    else:
        run_harness(
            predictions_path,
            dataset_name=args.dataset_name,
            run_id=args.run_id,
            work_dir=out_dir,
            max_workers=args.max_workers,
            instance_timeout=args.instance_timeout,
            namespace=args.namespace,
        )
        report_path = find_report(out_dir, args.model_name, args.run_id)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved, submitted = parse_report(report)
    if (
        args.expected_instances is not None
        and submitted != args.expected_instances
    ):
        raise ValueError(
            f"official report submitted {submitted} instances; "
            f"expected {args.expected_instances}"
        )

    stable_report = out_dir / f"swebench_report_{args.task_name}.json"
    stable_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = build_results(
        task_name=args.task_name,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        harness_version=args.harness_version,
        resolved=resolved,
        submitted=submitted,
        report=report,
    )
    results_path = out_dir / f"results_{args.task_name}.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[swebench] {args.task_name}: resolved "
        f"{resolved}/{submitted} = {resolved / submitted:.4f}",
        flush=True,
    )
    print(f"[swebench] result: {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
