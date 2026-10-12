#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Report an lm-eval GSM8K score and enforce an optional accuracy threshold."""

import argparse
import json
import math
from pathlib import Path


def probability(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number in [0, 1]: {value!r}") from exc
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be a finite number in [0, 1]: {value!r}")
    return result


def check_result(result_dir, concurrency, threshold=None):
    # lm-eval also writes sample/config files; select a summary by its payload.
    summaries = []
    for path in result_dir.rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            if "gsm8k" in data["results"]:
                summaries.append((path, data))
    if not summaries:
        raise ValueError(f"no GSM8K result JSON found in {result_dir}")
    path, data = max(summaries, key=lambda item: item[0].stat().st_mtime_ns)
    metrics = data["results"]["gsm8k"]
    if not isinstance(metrics, dict):
        raise ValueError("GSM8K metrics must be an object")
    score = probability(metrics.get("exact_match,flexible-extract"), "GSM8K score")
    print(f"[eval] result={path}")
    print(f"[eval] concurrent={concurrency} exact_match,flexible-extract = {score}")
    print(json.dumps(data["results"], indent=2))
    if threshold is not None:
        threshold = probability(threshold, "GSM8K threshold")
        if score < threshold:
            raise ValueError(
                f"GSM8K score {score:.4f} is below threshold {threshold:.4f}"
            )
        print(f"[eval] GSM8K threshold passed: {score:.4f} >= {threshold:.4f}")
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("concurrency", type=int)
    parser.add_argument("threshold", nargs="?", default="")
    args = parser.parse_args()
    try:
        check_result(args.result_dir, args.concurrency, args.threshold or None)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"[eval][FAIL] {exc}\n")


if __name__ == "__main__":
    main()
