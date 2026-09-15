# SPDX-License-Identifier: MIT
"""Fit DSpark STS on labeled requests and report a disjoint holdout cohort.

This offline diagnostic produces a candidate profile, not a serving default.
Request cohorts must be assigned before collecting any verification outcomes.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def observations(report, cohort):
    batches = report["workload"]["batches"]
    rows = [
        observation
        for case in report["cases"]
        if batches[case["case"]]["cohort"] == cohort
        for observation in case["confidence_observations"]
    ]
    if not rows:
        raise ValueError(f"No observations for {cohort}")
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    offered = np.asarray([row["verified_drafts"] for row in rows])
    accepted = np.asarray([row["accepted_drafts"] for row in rows])
    if (
        confidence.ndim != 2
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
        or np.any((accepted < 0) | (accepted > offered))
        or np.any((offered < 0) | (offered > confidence.shape[1]))
    ):
        raise ValueError("Invalid confidence/verification observations")
    position = np.arange(1, confidence.shape[1] + 1)
    # Unverified suffixes are censored, never labeled as rejections.
    return confidence, accepted[:, None] >= position, offered[:, None] >= position


def scaled_survival(confidence, temperatures):
    c = np.clip(confidence, 1e-6, 1 - 1e-6)
    logits = np.log(c) - np.log1p(-c)
    return np.cumprod(1 / (1 + np.exp(-logits / temperatures)), axis=1)


def ece(probability, labels, bins=15):
    index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    count = np.bincount(index, minlength=bins)
    predicted = np.bincount(index, weights=probability, minlength=bins)
    observed = np.bincount(index, weights=labels, minlength=bins)
    return float(np.abs(predicted - observed).sum() / count.sum())


def fit_temperatures(confidence, labels, observed):
    temperatures = np.ones(confidence.shape[1], dtype=np.float64)
    grid = np.geomspace(0.25, 4, 161)
    for position in range(confidence.shape[1]):
        valid = observed[:, position]
        if not valid.any():
            raise ValueError(f"No verified observations at position {position + 1}")
        candidates = []
        for temperature in grid:
            temperatures[position] = temperature
            probability = scaled_survival(confidence, temperatures)[valid, position]
            candidates.append(
                (
                    ece(probability, labels[valid, position]),
                    abs(np.log(temperature)),
                    temperature,
                )
            )
        temperatures[position] = min(candidates)[2]
    return temperatures


def metrics(confidence, labels, observed, temperatures):
    survival = scaled_survival(confidence, temperatures)
    result = []
    for position in range(confidence.shape[1]):
        valid = observed[:, position]
        probability, actual = survival[valid, position], labels[valid, position]
        if not len(probability):
            raise ValueError(f"No evaluation observations at position {position + 1}")
        clipped = np.clip(probability, 1e-12, 1 - 1e-12)
        result.append(
            {
                "position": position + 1,
                "observations": int(valid.sum()),
                "predicted_survival": float(probability.mean()),
                "observed_survival": float(actual.mean()),
                "ece": ece(probability, actual),
                "brier": float(np.square(probability - actual).mean()),
                "binary_nll": float(
                    -(
                        actual * np.log(clipped) + (1 - actual) * np.log1p(-clipped)
                    ).mean()
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    report = json.loads(payload)
    if not report["completed"]:
        raise ValueError("Confidence collection must complete before fitting")
    fit_rows, holdout_rows = set(), set()
    for batch in report["workload"]["batches"]:
        target = fit_rows if batch["cohort"] == "fit" else holdout_rows
        target.update(batch["dataset_rows"])
    if not fit_rows or not holdout_rows or fit_rows & holdout_rows:
        raise ValueError("Fit and holdout must contain disjoint dataset requests")
    train = observations(report, "fit")
    temperatures = fit_temperatures(*train)
    output = {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "workload": report["workload"],
        "tp": report["tp"],
        "cache_dtype": report["cache_dtype"],
        "graph": report["graph"],
        "sts_temperatures": temperatures.tolist(),
        "method": "sequential grid search minimizing cumulative-survival ECE (15 fixed bins)",
        "deployment_status": "candidate_only; hardware SPS and held-out runtime gate still required",
        "cohorts": {},
    }
    for cohort in ("fit", "holdout"):
        values = observations(report, cohort)
        output["cohorts"][cohort] = {
            "verification_blocks": len(values[0]),
            "raw": metrics(*values, np.ones_like(temperatures)),
            "calibrated": metrics(*values, temperatures),
        }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
