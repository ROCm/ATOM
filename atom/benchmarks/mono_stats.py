# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail-closed V4 Mono execution evidence, outside workload timing windows."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from atom.benchmarks.mono_env import configured_policy, configured_profile

COUNTERS = (
    "decode_attempts",
    "selected_steps",
    "graph_replays",
    "eager_forwards",
    "fallback_steps",
)


def snapshot(base_url: str) -> dict:
    with urlopen(base_url.rstrip("/") + "/debug/mono_stats", timeout=35) as response:
        return json.load(response)


def _ranks(snapshot: dict) -> dict:
    rows = snapshot.get("ranks", [])
    if snapshot.get("version") != 1 or len(rows) != 8:
        raise ValueError("V4 Mono evidence requires all eight TP rank replies")
    result = {}
    for row in rows:
        rank = row.get("rank")
        if row.get("error") or not isinstance(row.get("stats"), dict):
            raise ValueError(f"Mono snapshot failed on rank {rank}: {row.get('error')}")
        if type(rank) is not int or rank in result or rank not in range(8):
            raise ValueError(f"invalid or duplicate Mono rank {rank}")
        result[rank] = row["stats"]
    return result


def execution_delta(
    before: dict,
    after: dict,
    profile: str,
    policy: str,
    *,
    min_hit_rate: float = 0.0,
) -> dict:
    """Validate *measured* execution, not graph capture or launch warmup."""
    if profile not in ("ordinary", "mtp3_target") or policy not in (
        "auto",
        "off",
        "force",
    ):
        raise ValueError("invalid V4 Mono qualification profile/policy")
    if not 0 <= min_hit_rate <= 1:
        raise ValueError("minimum Mono hit rate must be between zero and one")
    initial, final = _ranks(before), _ranks(after)
    deltas = []
    for rank, end in sorted(final.items()):
        start = initial[rank]
        if start.get("policy") != policy or end.get("policy") != policy:
            raise ValueError(f"Mono policy mismatch on rank {rank}")
        if policy == "off":
            if start.get("installed") or end.get("installed"):
                raise ValueError(f"off control installed Mono on rank {rank}")
            delta = dict.fromkeys(COUNTERS, 0)
        else:
            for stats in (start, end):
                if not stats.get("enabled") or stats.get("profile") != profile:
                    raise ValueError(
                        f"Mono profile disabled or mismatched on rank {rank}"
                    )
            delta = {}
            for key in COUNTERS:
                values = start.get(key), end.get(key)
                if any(type(value) is not int or value < 0 for value in values):
                    raise ValueError(f"invalid {key} on rank {rank}")
                delta[key] = values[1] - values[0]
                if delta[key] < 0:
                    raise ValueError(f"Mono counters reset on rank {rank}")
            executed = delta["graph_replays"] + delta["eager_forwards"]
            if executed <= 0 or executed != delta["selected_steps"]:
                raise ValueError(f"no completed measured Mono execution on rank {rank}")
            if delta["decode_attempts"] != executed + delta["fallback_steps"]:
                raise ValueError(f"unaccounted decode attempts on rank {rank}")
        reasons = {}
        prior, current = start.get("fallback_reasons", {}), end.get(
            "fallback_reasons", {}
        )
        for key in prior.keys() | current.keys():
            values = prior.get(key, 0), current.get(key, 0)
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"invalid fallback reason count on rank {rank}")
            reasons[key] = values[1] - values[0]
            if reasons[key] < 0:
                raise ValueError(f"Mono fallback reason counters reset on rank {rank}")
        if sum(reasons.values()) != delta["fallback_steps"]:
            raise ValueError(f"unaccounted fallback reasons on rank {rank}")
        deltas.append({"rank": rank, **delta, "fallback_reasons": reasons})
    if any(
        tuple(row[key] for key in COUNTERS) != tuple(deltas[0][key] for key in COUNTERS)
        or row["fallback_reasons"] != deltas[0]["fallback_reasons"]
        for row in deltas[1:]
    ):
        raise ValueError("TP ranks disagree on measured Mono execution counters")
    reference = deltas[0]
    attempts = reference["decode_attempts"]
    hit_rate = reference["selected_steps"] / attempts if attempts else 0.0
    if policy != "off" and hit_rate < min_hit_rate:
        raise ValueError(
            f"measured Mono hit rate {hit_rate:.4f} is below {min_hit_rate:.4f}"
        )
    return {
        "version": 1,
        "profile": profile,
        "policy": policy,
        "measurement_valid": True,
        "mono_hit_rate": hit_rate,
        "ranks": deltas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("snapshot", "verify"))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument(
        "--after", type=Path, help="Verify an already saved post-drain snapshot"
    )
    parser.add_argument("--result", type=Path)
    parser.add_argument("--min-samples", type=int, default=0)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args()
    data = json.loads(args.after.read_text()) if args.after else snapshot(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    _ranks(data)  # Fail before launching a workload when any worker is unavailable.
    if args.action == "verify":
        if not args.before or not args.result:
            parser.error("verify requires --before and --result")
        result = json.loads(args.result.read_text())
        if args.min_samples:
            effective = result.get("n-samples", {}).get("gsm8k", {}).get("effective", 0)
            if effective != args.min_samples:
                raise ValueError(
                    f"expected {args.min_samples} GSM8K samples, got {effective}"
                )
        evidence = execution_delta(
            json.loads(args.before.read_text()),
            data,
            configured_profile(),
            configured_policy(),
            min_hit_rate=args.min_hit_rate,
        )
        result.setdefault("atom_ci_metadata", {})["mono_execution"] = evidence
        args.result.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
