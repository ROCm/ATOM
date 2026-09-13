# SPDX-License-Identifier: MIT
"""Paired summaries of lm-evaluation-harness artifacts, with prompt identity checks."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def _check_finite(values):
    if isinstance(values, float) and not math.isfinite(values):
        raise ValueError("Non-finite model response in quality artifact")
    if isinstance(values, (list, tuple)):
        for value in values:
            _check_finite(value)


def paired_summary(target, reference):
    if target["reference_manifest"] != reference["reference_manifest"]:
        raise ValueError("Reference revisions differ")
    if set(target["samples"]) != set(reference["samples"]):
        raise ValueError("Task sets differ")
    rng = np.random.default_rng(1234)
    output = {}
    for task, samples in target["samples"].items():
        expected = reference["samples"][task]
        identity = lambda row: (row["doc_id"], row["filter"])
        left, right = {identity(row): row for row in samples}, {
            identity(row): row for row in expected
        }
        if (
            left.keys() != right.keys()
            or len(left) != len(samples)
            or len(right) != len(expected)
        ):
            raise ValueError(f"Unpaired or repeated documents in {task}")
        for row in (*samples, *expected):
            _check_finite(row.get("resps", []))
            _check_finite(row.get("filtered_resps", []))
        for key in left:
            for field in ("doc_hash", "prompt_hash", "target_hash"):
                if left[key][field] != right[key][field]:
                    raise ValueError(f"{task} {key}: different {field}")
        metrics = {}
        for name in target["results"][task]:
            if "," not in name or "stderr" in name:
                continue
            metric, filter_name = name.rsplit(",", 1)
            keys = [key for key in left if key[1] == filter_name]
            if not keys or metric not in left[keys[0]]:
                continue
            a = np.array([left[key][metric] for key in keys], dtype=np.float64)
            b = np.array([right[key][metric] for key in keys], dtype=np.float64)
            if not (np.isin(a, (0, 1)).all() and np.isin(b, (0, 1)).all()):
                raise ValueError(
                    f"Only binary task metrics are supported: {task}/{name}"
                )
            differences = a - b
            counts = np.array([(differences == value).sum() for value in (-1, 0, 1)])
            bootstrap = rng.multinomial(len(keys), counts / len(keys), size=10000)
            means = (bootstrap[:, 2] - bootstrap[:, 0]) / len(keys)
            losses, _, gains = counts.tolist()
            metrics[name] = {
                "documents": len(keys),
                "target": float(a.mean()),
                "reference": float(b.mean()),
                "delta": float(differences.mean()),
                "paired_bootstrap_95_ci": np.quantile(means, (0.025, 0.975)).tolist(),
                "gains": gains,
                "losses": losses,
                "mcnemar_exact_p": (
                    binomtest(gains, gains + losses, p=0.5).pvalue
                    if gains + losses
                    else 1.0
                ),
            }
        output[task] = metrics
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired_summary(
        json.loads(args.target.read_text()), json.loads(args.reference.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
