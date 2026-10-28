"""Fit a DSpark deployment profile from numeric GPU/acceptance observations."""

import argparse
import json
from pathlib import Path

import numpy as np

from atom.spec_decode.calibration import validate_calibration
from atom.spec_decode.startup_sps import sps_table_from_points


def survival(confidence, temperatures):
    c = np.clip(np.asarray(confidence, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(c) - np.log1p(-c)
    conditional = 1 / (1 + np.exp(-logits / np.asarray(temperatures)))
    return np.cumprod(conditional, axis=1)


def cumulative_ece(confidence, accepted, temperatures, bins=15):
    predicted = survival(confidence, temperatures)
    actual = np.asarray(accepted)[:, None] >= np.arange(1, predicted.shape[1] + 1)
    error = 0.0
    for j in range(predicted.shape[1]):
        groups = np.minimum((predicted[:, j] * bins).astype(int), bins - 1)
        for k in range(bins):
            take = groups == k
            if take.any():
                error += take.mean() * abs(
                    predicted[take, j].mean() - actual[take, j].mean()
                )
    return error / predicted.shape[1]


def fit_sts(records, width=5, min_blocks=100):
    if any(
        len(r["confidence"]) != width or not 0 <= r["accepted"] <= width
        for r in records
    ):
        raise ValueError("Invalid confidence/acceptance record")
    if any(
        not np.isfinite(r["confidence"]).all()
        or min(r["confidence"]) < 0
        or max(r["confidence"]) > 1
        for r in records
    ):
        raise ValueError("Invalid confidence values")
    train = [r for r in records if int(r["request"][:8], 16) % 5 != 0]
    valid = [r for r in records if int(r["request"][:8], 16) % 5 == 0]
    if min(len(train), len(valid)) < min_blocks:
        raise ValueError("Insufficient request-disjoint training/held-out blocks")
    c = np.asarray([r["confidence"] for r in train])
    a = np.asarray([r["accepted"] for r in train])
    temperatures = np.ones(width)
    grid = [0.25, 0.35, 0.5, 0.7, 0.85, 1, 1.2, 1.5, 2, 3, 4]
    for _ in range(3):
        for j in range(width):
            candidates = []
            for value in grid:
                trial = temperatures.copy()
                trial[j] = value
                candidates.append((cumulative_ece(c, a, trial), abs(value - 1), value))
            temperatures[j] = min(candidates)[2]
    stats = {
        "training_blocks": len(train),
        "held_out_blocks": len(valid),
        "split": "request hash modulo 5; held-out requests never fit temperatures",
    }
    for name, rows in [("training", train), ("held_out", valid)]:
        c = np.asarray([r["confidence"] for r in rows])
        a = np.asarray([r["accepted"] for r in rows])
        stats[name + "_raw_ece"] = cumulative_ece(c, a, np.ones(width))
        stats[name + "_fitted_ece"] = cumulative_ece(c, a, temperatures)
    return temperatures.tolist(), stats


def fit_sps(points, max_rows):
    """Strict: the measurements must cover the deployment's row limit."""
    return sps_table_from_points(points, max_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # The startup sweep's cache file (ATOM_DSPARK_SPS_CACHE_DIR) of the same
    # deployment the confidence records came from.
    p.add_argument("--sps-measurements", type=Path, required=True)
    p.add_argument("--confidence", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    measured = json.loads(args.sps_measurements.read_text(encoding="utf-8"))
    records = []
    for path in args.confidence:
        identity = json.loads(
            (path.parent / "identity.json").read_text(encoding="utf-8")
        )
        if identity != measured["identity"]:
            raise ValueError("Confidence and SPS came from different deployments")
        records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    temperatures, stats = fit_sts(records)
    maximum = measured["max_num_seqs"]
    data = {
        "version": 1,
        "identity": measured["identity"],
        "draft_width": 5,
        "max_num_seqs": maximum,
        "sts_temperatures": temperatures,
        # A DP graph cap leaves the largest row counts unrecorded; they extend
        # along the last measured slope, as at startup.
        "sps_table": sps_table_from_points(
            measured["points"], maximum * 6, extrapolate=True
        ),
        "fit": stats,
        "sps_scope": measured["scope"],
    }
    validate_calibration(data, data["identity"], width=5, max_batch=maximum)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
