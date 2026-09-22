#!/usr/bin/env python3
"""Check the image's AIPerf and compare completed 1P1D C / 2P2D 2C runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def preflight(binary, output):
    import aiperf

    help_text = subprocess.check_output([binary, "profile", "--help"], text=True)
    required = (
        "--scenario",
        "--public-dataset",
        "--trajectory-start-min-ratio",
        "--trajectory-start-max-ratio",
        "--warmup-requests-per-lane",
        "--trace-idle-gap-cap-seconds",
        "--warmup-grace-period",
    )
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise ValueError(f"Image AIPerf lacks agentic options: {missing}")
    source = Path(aiperf.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    payload = {
        "version": subprocess.check_output([binary, "--version"], text=True).strip(),
        "source_sha256": digest.hexdigest(),
        "binary": binary,
        "requested_catalog_commit": os.environ.get("AIPERF_COMMIT", ""),
        "mode": "preinstalled",
        "image": os.environ.get("DOCKER_IMAGE", ""),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def only_json(root, pattern):
    paths = list(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern}, found {len(paths)}")
    return read_json(paths[0])


def read_run(cell, results):
    root = results / cell["id"]
    submitted = read_json(results / f"{cell['id']}.cell.json")
    for key in (
        "model",
        "image",
        "benchmark",
        "server_args",
        "service",
        "concurrency",
        "scaling",
    ):
        if submitted[key] != cell[key]:
            raise ValueError(f"submitted {key} differs from the experiment matrix")
    status = only_json(root, "job-result.json")
    if (
        status["result"].get("state") != "COMPLETED"
        or status["result"].get("return_code") != 0
    ):
        raise ValueError("Slurm/workload did not complete successfully")
    metrics = only_json(root, "pd-*.json")
    if metrics.get("max_concurrency") != cell["concurrency"][0]:
        raise ValueError("result concurrency differs from matrix")
    required = ["request_throughput", "output_throughput", "successful_requests"]
    required += [
        f"p{p}_{metric}_ms" for metric in ("ttft", "itl", "e2el") for p in (95, 99)
    ]
    for key in required:
        if not finite(metrics.get(key)) or metrics[key] < 0:
            raise ValueError(f"missing or invalid {key}")
    if any(metrics[key] <= 0 for key in required[:3]):
        raise ValueError("no successful measured throughput")
    failure = metrics.get("request_error_rate_pct")
    if finite(failure):
        failure /= 100
    else:
        failed = metrics.get("failed_requests")
        if not finite(failed) or failed < 0:
            raise ValueError("missing failure count/rate; cannot assume zero failures")
        failure = failed / (failed + metrics["successful_requests"])
    if not 0 <= failure <= 1:
        raise ValueError("invalid failure rate")
    build = only_json(root, "mesh-build.json")
    if build.get("profile") != "release" or build.get("source_dirty") not in (
        False,
        "false",
    ):
        raise ValueError("missing clean release Mesh build provenance")
    if any(
        not build.get(key)
        for key in ("commit", "binary_sha256", "lockfile_sha256", "cargo")
    ):
        raise ValueError("missing Mesh source/binary/dependency provenance")
    version = only_json(root, "aiperf-version.json")
    if not version.get("source_sha256") or version.get("mode") != "preinstalled":
        raise ValueError("missing preinstalled AIPerf provenance")
    if "@sha256:" not in cell["image"] or version.get("image") != cell["image"]:
        raise ValueError("run image is not the matrix's pinned digest")
    routing = None
    if cell["service"]["router"]["policy"] == "kv_cache_aware":
        preflight = only_json(root, "cache-routing-preflight.json")
        routing = only_json(root, "routing-decisions.json")
        if (
            preflight.get("policy") != "kv_cache_aware"
            or routing.get("policy") != "kv_cache_aware"
            or not finite(routing.get("selected"))
            or routing["selected"] <= 0
            or not finite(routing.get("fallback"))
            or routing["fallback"] < 0
        ):
            raise ValueError("no verified calibrated kv_cache_aware selections")
        routing["selected_fraction"] = routing["selected"] / (
            routing["selected"] + routing["fallback"]
        )
    return {
        "metrics": metrics,
        "failure_rate": failure,
        "mesh": build,
        "aiperf": version,
        "active_gpus": cell["scaling"]["active_gpus"],
        "allocated_gpus": cell["scaling"]["allocated_gpus"],
        "settings": comparable_config(submitted),
        "routing": routing,
    }


def comparable_config(cell):
    return {
        **{
            key: cell[key]
            for key in (
                "model",
                "model_path",
                "image",
                "server_args",
                "benchmark",
                "env",
            )
        },
        "service": {
            role: {k: v for k, v in cfg.items() if k != "workers"}
            for role, cfg in cell["service"].items()
            if role != "router"
        },
    }


def compare(matrix, results):
    groups = {}
    for cell in matrix["include"]:
        meta = cell["scaling"]
        group = groups.setdefault(meta["baseline_case"], {})
        if meta["scale"] in group:
            raise ValueError("duplicate scale in matrix")
        group[meta["scale"]] = cell
    pairs = []
    for name, cells in groups.items():
        pair = {"baseline_case": name, "status": "incomplete"}
        try:
            if set(cells) != {1, 2}:
                raise ValueError("missing a member of the C/2C pair")
            baseline, scaled = cells[1], cells[2]
            pair["concurrency"] = [baseline["concurrency"][0], scaled["concurrency"][0]]
            if pair["concurrency"][1] != 2 * pair["concurrency"][0]:
                raise ValueError("expected exactly C versus 2C")
            if comparable_config(baseline) != comparable_config(scaled):
                raise ValueError("paired workload, image or per-worker settings differ")
            runs = [read_run(cell, results) for cell in (baseline, scaled)]
            pair["runs"] = runs
            a, b = runs
            if a["settings"] != b["settings"]:
                raise ValueError("submitted worker settings/model paths differ")
            if a["mesh"]["commit"] != b["mesh"]["commit"]:
                raise ValueError("paired Mesh source commits differ")
            if any(
                a["mesh"][key] != b["mesh"][key] for key in ("lockfile_sha256", "cargo")
            ):
                raise ValueError("paired Mesh dependency locks/toolchains differ")
            if a["aiperf"]["source_sha256"] != b["aiperf"]["source_sha256"]:
                raise ValueError("paired AIPerf source differs")
            if any(
                r["failure_rate"] > baseline["benchmark"]["failed_request_threshold"]
                for r in runs
            ):
                raise ValueError("failure rate exceeds the configured threshold")
            ratio = b["active_gpus"] / a["active_gpus"]
            pair["throughput"] = {}
            for key in ("request_throughput", "output_throughput"):
                speedup = b["metrics"][key] / a["metrics"][key]
                pair["throughput"][key] = {
                    "speedup": speedup,
                    "per_gpu_efficiency": speedup / ratio,
                    "per_active_gpu": [
                        r["metrics"][key] / r["active_gpus"] for r in runs
                    ],
                    "per_allocated_gpu": [
                        r["metrics"][key] / r["allocated_gpus"] for r in runs
                    ],
                }
            pair["status"] = "measured"
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            pair["reason"] = str(exc)
        pairs.append(pair)
    return {
        "pairs": pairs,
        "complete": bool(pairs) and all(p["status"] == "measured" for p in pairs),
    }


def markdown(report):
    lines = [
        "# GLM agentic scaling: 1P1D C → 2P2D 2C",
        "",
        "Each node runs PP4/TP1 prefill and TP4/DCP4 decode.",
        "1P1D retains its baseline policy. 2P2D uses kv_cache_aware for BOTH P and D.",
        "The existing fixed MTP acceptance rate is retained. This is trace replay performance,",
        "not agent task correctness. Ratios include both scaling and policy changes.",
        "Calibrated selection coverage includes warmup and retry attempts; any fallback is reported.",
        "",
        "Per-GPU efficiency is throughput speedup / GPU-count ratio (1.0 = linear scaling).",
        "Latency percentiles describe successful profiling requests; failure rates are shown separately.",
        "Cache-hit rate is API-reported prompt reuse, not a CPU-only hit rate.",
        "",
        "| C → 2C | Status | Request/s speedup | Output tok/s speedup | Output/GPU efficiency |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in report["pairs"]:
        conc = (
            " → ".join(str(c) for c in pair.get("concurrency", []))
            or pair["baseline_case"]
        )
        if pair["status"] != "measured":
            lines.append(f"| {conc} | INCOMPLETE: {pair['reason']} | — | — | — |")
            continue
        req = pair["throughput"]["request_throughput"]
        out = pair["throughput"]["output_throughput"]
        lines.append(
            f"| {conc} | measured | {req['speedup']:.3f}× | {out['speedup']:.3f}× | {out['per_gpu_efficiency']:.3f} |"
        )
    for pair in report["pairs"]:
        if "runs" not in pair:
            continue
        lines += [
            "",
            f"## {pair['baseline_case']}",
            "",
            "| Metric | 1P1D | 2P2D |",
            "|---|---:|---:|",
        ]
        a, b = pair["runs"]
        if b.get("routing"):
            decisions = b["routing"]
            lines.append(
                f"| Calibrated selection fraction | N/A (one pair) | {decisions['selected_fraction']:.2%} |"
            )
            lines.append(f"| Fallback attempts | N/A | {decisions['fallback']:.0f} |")
        for key in ("active_gpus", "allocated_gpus", "failure_rate"):
            lines.append(f"| {key} | {a[key]:.4g} | {b[key]:.4g} |")
        for key in (
            "request_throughput",
            "output_throughput",
            "cache_hit_rate",
            "benchmark_duration_s",
        ) + tuple(
            f"p{p}_{metric}_ms" for metric in ("ttft", "itl", "e2el") for p in (95, 99)
        ):
            values = [
                (
                    f"{r['metrics'][key]:.4g}"
                    if finite(r["metrics"].get(key))
                    else "unavailable"
                )
                for r in (a, b)
            ]
            lines.append(f"| {key} | {' | '.join(values)} |")
        lines += [
            "",
            f"Mesh source: `{a['mesh']['commit']}`. AIPerf: `{a['aiperf']['version']}`.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--aiperf", required=True)
    check.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--matrix", type=Path, required=True)
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args.aiperf, args.output)
        return 0
    result = compare(read_json(args.matrix), args.results)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.summary.write_text(markdown(result))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
