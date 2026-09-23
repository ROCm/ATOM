# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare full GSM8K lm-eval runs without inventing acceptance tolerances.

Both directories must contain one results JSON and one GSM8K samples JSONL
written by ``lm_eval --log_samples``. Prompts, targets, task settings and seeds
must agree. Report both recipe filters and retain per-question evidence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

FILTERS = ("strict-match", "flexible-extract")


def _one_file(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} in {directory}, found {paths}")
    return paths[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(directory: Path) -> dict:
    result_path = _one_file(directory, "results*.json")
    samples_path = _one_file(directory, "samples_gsm8k*.jsonl")
    result = json.loads(result_path.read_text())
    if result["n-samples"]["gsm8k"] != {"original": 1319, "effective": 1319}:
        raise ValueError(f"Not a complete GSM8K test run: {result_path}")
    samples = {}
    # JSON strings can legally contain Unicode line separators. JSONL records
    # are separated by LF; str.splitlines() would also split inside answers.
    for line in samples_path.read_text().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        key = (row["doc_id"], row["filter"])
        if key in samples:
            raise ValueError(f"Duplicate sample {key} in {samples_path}")
        if row["exact_match"] not in (0, 1):
            raise ValueError(f"Invalid exact_match in {samples_path}: {key}")
        row["exact_match"] = int(row["exact_match"])
        samples[key] = row
    expected = {(i, f) for i in range(1319) for f in FILTERS}
    if samples.keys() != expected:
        raise ValueError(f"Missing or unexpected samples in {samples_path}")
    for name in FILTERS:
        score = sum(samples[i, name]["exact_match"] for i in range(1319)) / 1319
        reported = result["results"]["gsm8k"][f"exact_match,{name}"]
        if not math.isclose(score, reported, abs_tol=1e-12):
            raise ValueError(f"Sample/aggregate score mismatch for {name}")
    return {
        "result": result,
        "samples": samples,
        "files": {
            "results": {"path": str(result_path), "sha256": _sha256(result_path)},
            "samples": {"path": str(samples_path), "sha256": _sha256(samples_path)},
        },
    }


def compare(reference_dir: Path, candidate_dir: Path) -> dict:
    reference, candidate = _load(reference_dir), _load(candidate_dir)
    for key in (
        "configs",
        "versions",
        "n-shot",
        "higher_is_better",
        "n-samples",
        "lm_eval_version",
        "system_instruction",
        "fewshot_as_multiturn",
        "chat_template",
    ):
        if reference["result"][key] != candidate["result"][key]:
            raise ValueError(f"Evaluation settings differ: {key}")
    # These determine task order/few-shot selection and generation defaults.
    for key in (
        "model",
        "model_args",
        "gen_kwargs",
        "random_seed",
        "numpy_seed",
        "torch_seed",
        "fewshot_seed",
    ):
        if reference["result"]["config"].get(key) != candidate["result"]["config"].get(
            key
        ):
            raise ValueError(f"Evaluation config differs: {key}")
    rows = []
    for doc_id in range(1319):
        row = {"doc_id": doc_id, "filters": {}}
        for name in FILTERS:
            lhs = reference["samples"][doc_id, name]
            rhs = candidate["samples"][doc_id, name]
            for key in ("doc_hash", "prompt_hash", "target_hash", "arguments"):
                if lhs[key] != rhs[key]:
                    raise ValueError(f"Sample {doc_id}/{name} differs: {key}")
            row["prompt_hash"] = lhs["prompt_hash"]
            row["doc_hash"] = lhs["doc_hash"]
            row["target_hash"] = lhs["target_hash"]
            row["identical_response"] = lhs["resps"] == rhs["resps"]
            row["filters"][name] = {
                "reference": lhs["exact_match"],
                "candidate": rhs["exact_match"],
                "reference_answer": lhs["filtered_resps"],
                "candidate_answer": rhs["filtered_resps"],
            }
        rows.append(row)
    metrics = {}
    for name in FILTERS:
        cells = [row["filters"][name] for row in rows]
        ref = sum(cell["reference"] for cell in cells)
        cand = sum(cell["candidate"] for cell in cells)
        metrics[name] = {
            "reference_correct": ref,
            "candidate_correct": cand,
            "reference_score": ref / 1319,
            "candidate_score": cand / 1319,
            "delta_percentage_points": (cand - ref) / 1319 * 100,
            "reference_only_correct": sum(
                cell["reference"] and not cell["candidate"] for cell in cells
            ),
            "candidate_only_correct": sum(
                cell["candidate"] and not cell["reference"] for cell in cells
            ),
            "both_correct": sum(
                cell["reference"] and cell["candidate"] for cell in cells
            ),
            "both_incorrect": sum(
                not cell["reference"] and not cell["candidate"] for cell in cells
            ),
        }
    return {
        "task": "gsm8k",
        "samples": 1319,
        "matching_prompts_and_targets": True,
        "identical_responses": sum(row["identical_response"] for row in rows),
        "reference": {
            "files": reference["files"],
            "official_result": reference["result"],
        },
        "candidate": {
            "files": candidate["files"],
            "official_result": candidate["result"],
        },
        "metrics": metrics,
        "per_question": rows,
        "acceptance_threshold": None,
        "note": "Measured paired differences; no statistical equivalence or "
        "accuracy threshold is inferred from the observed scores.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--result-file", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference_dir, args.candidate_dir)
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    args.result_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
