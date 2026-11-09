#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan an ATOM sweep and collect its measured points into agg_bmk.json."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import unquote


def cluster_env(hardware):
    """Use the same TW / Spur submission settings as atomesh-benchmark.yaml."""
    if hardware not in {"mi355x", "mi350x"}:
        raise ValueError("hardware must be mi355x (TW) or mi350x (Spur)")
    spur = hardware == "mi350x"
    nodes = (
        ",".join(
            f"ml-ai-ubuntu-gpu-mi350x8-2304gb-fabric-{index}"
            for index in (6, 7, 13, 14, 16)
        )
        if spur
        else ""
    )
    return {
        "ATOMESH_SLURM_ACCOUNT": "" if spur else "amd-atom",
        "ATOMESH_SLURM_PARTITION": "" if spur else "amd-spur",
        "ATOMESH_SLURM_SUBMIT_RUNNER": "atomesh-cicd-mi350" if spur else "atomesh-cicd",
        "ATOMESH_MODEL_ROOT": "/data/junyyang/models" if spur else "/mnt/models",
        "ATOMESH_LOG_ROOT": (
            "/data/${USER}/logs/ATOMESH_LOG/" if spur else "/it-share/ATOMESH_LOG/"
        ),
        "ATOMESH_PD_RANK_MAPPING_POLICY": "idx2idx" if spur else "none",
        "ATOMESH_SINGLE_NODE": "auto",
        "ATOMESH_1P1D_NODES": nodes,
        "ATOMESH_2P1D_NODES": nodes,
    }


def recipe_fingerprint(cell):
    recipe = {
        key: value
        for key, value in cell.items()
        if key not in {"id", "name", "concurrency", "concurrency_x", "suite"}
    }
    # The catalog defaults evaluation concurrency to benchmark concurrency.
    recipe["accuracy"] = {
        key: value for key, value in cell["accuracy"].items() if key != "concurrency"
    }
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def plan_sweep(manifest, catalog, *, repository, head_sha, pr_number):
    from pd_matrix import build_cells

    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Sweep configuration must declare version: 1")
    allowed = {
        "version",
        "model",
        "hardware",
        "mode",
        "image",
        "suite",
        "cases",
        "concurrency",
    }
    if set(manifest) - allowed:
        raise ValueError(f"Unknown sweep settings: {sorted(set(manifest) - allowed)}")
    model, mode = manifest.get("model"), manifest.get("mode")
    if model not in catalog.get("models", {}):
        raise ValueError("model must name an existing models_atomesh.yaml model")
    if mode not in {"aggregated", "pd"}:
        raise ValueError(
            "mode must be aggregated or pd; one sweep describes one serving mode"
        )
    image = manifest.get("image", "")
    if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be pinned with @sha256:<64 hex characters>")
    names = manifest.get("cases")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("cases must be a nonempty list of existing recipe names")
    if len(set(names)) != len(names):
        raise ValueError("cases must not contain duplicates")
    concurrency = manifest.get("concurrency")
    if concurrency is not None and (
        not isinstance(concurrency, list)
        or not concurrency
        or any(type(value) is not int or value < 1 for value in concurrency)
        or len(set(concurrency)) != len(concurrency)
    ):
        raise ValueError("concurrency must contain distinct positive integers")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("head_sha must be the exact checked-out commit")
    hardware = manifest.get("hardware")
    environment = cluster_env(hardware)
    previous = {key: os.environ.get(key) for key in environment}
    try:
        os.environ.update(environment)
        cells = build_cells(
            catalog,
            suite=manifest.get("suite", ""),
            model_filter={model},
            case_filter=set(names),
            benchmark_kind_filter={"aiperf_agentic"},
            override_image=image,
            override_benchmark_concurrency=concurrency,
            override_eval_concurrency=None,
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if {cell["name"] for cell in cells} != set(names):
        raise ValueError(
            "Every case must exist in the selected suite and use aiperf_agentic"
        )
    if any(
        cell["disaggregated"] != (mode == "pd") or cell["eval_only"] for cell in cells
    ):
        raise ValueError(
            "All selected recipes must benchmark the declared serving mode"
        )
    workloads = {
        (
            cell["precision"],
            json.dumps(cell["benchmark"], sort_keys=True),
            tuple(cell["isl"]),
            cell["osl"],
        )
        for cell in cells
    }
    if len(workloads) != 1:
        raise ValueError("A curve must use one precision and workload")
    # GitHub can retry a failed concurrency job without re-running its siblings.
    expanded = []
    identities = set()
    for cell in cells:
        for conc in cell["concurrency"]:
            identity = (recipe_fingerprint(cell), conc)
            if identity in identities:
                raise ValueError(
                    "The same recipe/concurrency is selected more than once"
                )
            identities.add(identity)
            point_cell = copy.deepcopy(cell)
            # Short, deterministic IDs also keep artifact paths and Slurm names bounded.
            point_cell["id"] = f"sweep-{identity[0][:16]}-c{conc}"
            point_cell["name"] = re.sub(r"-c\d+$", "", cell["name"]) + f"-c{conc}"
            point_cell["concurrency"] = [conc]
            point_cell["concurrency_x"] = str(conc)
            expanded.append(point_cell)
    cells = expanded
    if len(cells) > 256:
        raise ValueError("A sweep supports at most 256 concurrency jobs")
    groups = {}
    for cell in cells:
        name = re.sub(r"-c\d+$", "", cell["name"])
        groups.setdefault(name, []).append(cell)
    return {
        "version": 1,
        "repository": repository,
        "head_sha": head_sha,
        "pr_number": pr_number,
        "hardware": hardware,
        "mode": mode,
        "model": model,
        "model_root": environment["ATOMESH_MODEL_ROOT"],
        "configuration": manifest,
        "cells": cells,
        "groups": [
            {
                "name": name,
                "matrix_json": json.dumps({"include": group}, separators=(",", ":")),
            }
            for name, group in groups.items()
        ],
    }


def latest_cell_job(jobs, cell, run_attempt):
    # The reusable workflow's caller adds a "group / " prefix to the job name.
    name = f"sweep-point:{cell['id']}"
    candidates = [
        job
        for job in jobs
        if job.get("name", "").split(" / ")[-1] == name
        and 1 <= job.get("run_attempt", 0) <= run_attempt
    ]
    return max(candidates, key=lambda job: job["run_attempt"], default=None)


def cell_artifact(plan, directory, cell, run_id, attempt):
    path = directory / f"atomesh-model-benchmark-{cell['id']}-attempt-{attempt}"
    provenance_path = path / f"{cell['id']}-sweep-provenance.json"
    if not provenance_path.is_file():
        return None
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("run_id") != run_id or provenance.get("run_attempt") != attempt:
        raise ValueError("Artifact provenance belongs to a different run/attempt")
    if provenance.get("head_sha") != plan["head_sha"]:
        raise ValueError("A retry cannot reuse measurements from another commit")
    previous_cell = provenance["cell"]
    if previous_cell != cell:
        raise ValueError("A retry cannot reuse measurements from a different recipe")
    return path


def measured_points(summary):
    entries = (
        json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else []
    )
    points = {}
    for entry in entries:
        for part in entry.get("extra", "").split(" | "):
            if not part.startswith("perf_point="):
                continue
            point = json.loads(unquote(part[len("perf_point=") :]))
            conc = point.get("concurrency")
            if conc in points and points[conc] != point:
                raise ValueError(
                    f"Ambiguous results in {summary.name} at concurrency {conc}"
                )
            points[conc] = point
    return points


def valid_point(point, cell, hardware, run_url):
    if (
        not point
        or not all(
            isinstance(point.get(key), (int, float))
            and not isinstance(point[key], bool)
            and math.isfinite(point[key])
            and point[key] > 0
            for key in ("interactivity", "interactivity_p90_itl", "total_tput")
        )
        or point.get("interactivity_method") != "p90_e2e_normalized"
    ):
        return False
    if (
        point.get("hardware") != hardware
        or point.get("disaggregated") != cell["disaggregated"]
    ):
        raise ValueError("Measured hardware/serving mode conflicts with the sweep plan")
    if (
        point.get("model") != cell["model"]
        or point.get("precision", "").lower() != cell["precision"].lower()
    ):
        raise ValueError("Measured model/precision conflicts with the sweep plan")
    if point.get("run_url", "").rstrip("/") not in {
        run_url,
        run_url.split("/attempts/")[0],
    }:
        raise ValueError("Measured point belongs to a different run/attempt")
    return True


def collect_sweep(plan, directory, *, run_id, run_attempt, benchmark_result, jobs):
    run_url = f"https://github.com/{plan['repository']}/actions/runs/{run_id}/attempts/{run_attempt}"
    records, missing = [], []
    expected = 0
    for cell in plan["cells"]:
        expected += len(cell["concurrency"])
        job = latest_cell_job(jobs, cell, run_attempt)
        if (
            not job
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            missing.extend(
                {
                    "case": cell["name"],
                    "concurrency": conc,
                    "reason": (
                        f"Latest point job: {job.get('conclusion') or job.get('status')}"
                        if job
                        else "Point job not found"
                    ),
                    "run_attempt": job["run_attempt"] if job else None,
                }
                for conc in cell["concurrency"]
            )
            continue
        point_attempt = job["run_attempt"]
        artifact = cell_artifact(plan, directory, cell, run_id, point_attempt)
        summary = artifact / f"{cell['id']}-benchmark-action.json" if artifact else None
        points = measured_points(summary) if summary else {}
        fingerprint = recipe_fingerprint(cell)
        point_run_url = f"https://github.com/{plan['repository']}/actions/runs/{run_id}/attempts/{point_attempt}"
        for conc in cell["concurrency"]:
            point = points.get(conc)
            if not valid_point(point, cell, plan["hardware"], point_run_url):
                missing.append(
                    {
                        "case": cell["name"],
                        "concurrency": conc,
                        "reason": "Missing result or required P90 metric",
                    }
                )
                continue
            records.append(
                {
                    "model": point["model"],
                    "hw": plan["hardware"],
                    "conc": conc,
                    "framework": "atom-disagg" if cell["disaggregated"] else "atom",
                    "disagg": cell["disaggregated"],
                    "is_multinode": cell["num_nodes"] > 1,
                    "num_nodes": cell["num_nodes"],
                    "num_gpus": point.get("total_gpu"),
                    "precision": point["precision"],
                    "image": cell["image"],
                    "recipe_fingerprint": fingerprint,
                    "scenario_type": point.get("scenario", "agentic-coding"),
                    "dataset": {"loader": point.get("public_dataset", "")},
                    "prefill_num_workers": point.get("prefill_workers"),
                    "decode_num_workers": point.get("decode_workers"),
                    "request_metrics": {
                        "latency": {
                            "intvty": {"p90": point["interactivity_p90_itl"]},
                            "e2e_norm_intvty": {"p90": point["interactivity"]},
                        },
                        "throughput": {
                            "total": {"tokens_per_second": point["total_tput"]},
                            "input": {"tokens_per_second": point.get("input_tput")},
                            "output": {"tokens_per_second": point.get("output_tput")},
                        },
                    },
                    "pr_number": plan["pr_number"],
                    "head_sha": plan["head_sha"],
                    "run_url": point_run_url,
                    "run_attempt": point_attempt,
                    "sweep_run_attempt": run_attempt,
                }
            )
    complete = (
        benchmark_result == "success" and not missing and len(records) == expected
    )
    manifest = {
        key: plan[key]
        for key in (
            "version",
            "repository",
            "pr_number",
            "head_sha",
            "hardware",
            "mode",
            "model",
            "configuration",
        )
    }
    manifest.update(
        run_id=run_id,
        run_attempt=run_attempt,
        run_url=run_url,
        expected_points=expected,
        measured_points=len(records),
        reused_points=sum(record["run_attempt"] < run_attempt for record in records),
        point_attempts=[
            {
                "concurrency": record["conc"],
                "recipe_fingerprint": record["recipe_fingerprint"],
                "run_attempt": record["run_attempt"],
                "run_url": record["run_url"],
            }
            for record in records
        ],
        missing=missing,
        status="complete" if complete else "partial",
    )
    for record in records:
        record["sweep_status"] = manifest["status"]
    return records, manifest


def write_output(values):
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.writelines(f"{key}={value}\n" for key, value in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("manifest", type=Path)
    plan_parser.add_argument(
        "--catalog", type=Path, default=Path(".github/benchmark/models_atomesh.yaml")
    )
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("directory", type=Path)
    point_parser = commands.add_parser("validate-point")
    point_parser.add_argument("provenance", type=Path)
    point_parser.add_argument("--hardware", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        import yaml

        plan = plan_sweep(
            yaml.safe_load(args.manifest.read_text(encoding="utf-8")),
            yaml.safe_load(args.catalog.read_text(encoding="utf-8")),
            repository=os.environ["GITHUB_REPOSITORY"],
            head_sha=os.environ["SWEEP_HEAD_SHA"],
            pr_number=int(os.environ.get("SWEEP_PR_NUMBER") or 0),
        )
        Path("sweep-plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        write_output(
            {
                "groups_json": json.dumps(plan["groups"], separators=(",", ":")),
                "model_root": plan["model_root"],
                "hardware": plan["hardware"],
                "plan_artifact": f"sweep-plan-attempt-{os.environ['GITHUB_RUN_ATTEMPT']}",
            }
        )
    elif args.command == "validate-point":
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        cell = provenance["cell"]
        summary = args.provenance.parent / f"{cell['id']}-benchmark-action.json"
        points = measured_points(summary)
        url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{provenance['run_id']}/attempts/{provenance['run_attempt']}"
        if not all(
            valid_point(points.get(conc), cell, args.hardware, url)
            for conc in cell["concurrency"]
        ):
            raise ValueError(
                "Point is missing a required P90 metric; retry this job after resolving the failure"
            )
    else:
        plan = json.loads(Path("sweep-plan.json").read_text(encoding="utf-8"))
        records, manifest = collect_sweep(
            plan,
            args.directory,
            run_id=int(os.environ["GITHUB_RUN_ID"]),
            run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
            benchmark_result=os.environ["SWEEP_BENCHMARK_RESULT"],
            jobs=json.loads(Path("sweep-jobs.json").read_text(encoding="utf-8")),
        )
        Path("agg_bmk.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        Path("sweep_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        write_output({"complete": str(manifest["status"] == "complete").lower()})
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            origin = f"PR #{plan['pr_number']}" if plan["pr_number"] else "Manual"
            with Path(summary).open("a", encoding="utf-8") as output:
                output.write(
                    f"### {origin} · {plan['mode']} sweep\n\n"
                    f"Commit: `{plan['head_sha']}` · [CI run]({manifest['run_url']})\n\n"
                    f"**{manifest['status']}**: {len(records)}/{manifest['expected_points']} measured points.\n\n"
                    f"Retained {manifest['reused_points']} successful points from earlier attempts of this run.\n\n"
                    f"Download `results_bmk-attempt-{manifest['run_attempt']}` to import `agg_bmk.json` into the dashboard.\n\n"
                    "For failures, resolve the cause and use **Re-run failed jobs**. A changed commit requires a new run.\n"
                )


if __name__ == "__main__":
    main()
