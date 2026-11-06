#!/usr/bin/env python3
"""Materialize the measured V9 runtime outside the checkout, without starting GPUs."""

import argparse
import ast
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(spec, aiter, output):
    source = aiter / spec["source_path"]
    if sha(source) != spec["source_sha256"]:
        raise ValueError(f"Pinned AITER source changed: {source}")
    original = source.read_text().splitlines(keepends=True)
    for name, record in spec["outputs"].items():
        lines = list(original)
        for edit in reversed(record["edits"]):
            lines[edit["start"] : edit["end"]] = edit["lines"]
        path = output / name
        path.write_text("".join(lines))
        if sha(path) != record["sha256"]:
            raise ValueError(
                f"Reconstructed source differs from measured source: {name}"
            )


def prepare(reference, aiter, output):
    lock = json.loads((HERE / "LOCK.json").read_text())
    head = subprocess.check_output(
        ["git", "-C", str(aiter), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != lock["aiter_sha"]:
        raise ValueError(
            "Use the measured AITER SHA; dependency migration is unqualified"
        )
    for name, expected in lock["production_file_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Measured model/owner source changed: {name}")
    for name, record in lock["source_files"].items():
        path = HERE / name
        digest = hashlib.sha256(
            ast.dump(ast.parse(path.read_text()), include_attributes=False).encode()
        ).hexdigest()
        if digest != record["original_ast_sha256"]:
            raise ValueError(
                f"Runtime differs beyond formatting from measured code: {name}"
            )
    if sha(reference / "qualification.json") != lock["qualification_sha256"]:
        raise ValueError("The reference qualification receipt changed")
    for name, expected in lock["external_artifact_sha256"].items():
        if sha(reference / "candidate_source" / name) != expected:
            raise ValueError(f"Measured binary/calibration changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "candidate_source"
    shutil.copytree(
        HERE / "runtime", runtime, ignore=shutil.ignore_patterns("__pycache__")
    )
    reconstruct(
        json.loads((HERE / "kernels/flydsl_builders.json").read_text()), aiter, runtime
    )
    for name in lock["external_artifact_sha256"]:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / "candidate_source" / name, target)
    shutil.copytree(
        HERE / "bootstrap",
        output / "bootstrap",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("formal_client.py", "preflight.py", "fp8_binary_manifest.json"):
        shutil.copy2(reference / name, output / name)
    qualification = copy.deepcopy(
        json.loads((reference / "qualification.json").read_text())
    )
    # Numerical evidence stays immutable and external. Runtime wrappers retain the
    # measured AST, while the generated FlyDSL builders and binaries retain bytes.
    qualification["candidate_source_sha256"] = {
        str(path.relative_to(runtime)): sha(path)
        for path in runtime.rglob("*")
        if path.is_file() and path.suffix in (".py", ".so", ".pt")
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n"
    )
    environment = dict(
        line.split("=", 1)
        for line in (reference / "environment.list").read_text().splitlines()
    )
    for directory in ("counters", "owner", "communication", "prometheus"):
        (output / directory).mkdir()
    (output / "owner/capacity.txt").write_text("full\n")
    (output / "owner/phase.txt").write_text("warmup\n")
    environment.update(
        WORKSPACE_ROOT=str(REPO.parent),
        DSV41_SERVICE_RUN=str(output),
        DSV41_COMBINED_FROZEN_SOURCE=str(runtime),
        DSV41_COMBINED_QUALIFICATION=str(output / "qualification.json"),
        DSV41_COMBINED_REPORT=str(output / "communication"),
        DSV41_OWNER_SERVING_OUTPUT=str(output / "owner"),
        DSV41_OWNER_SERVING_CONTROL=str(output / "owner/capacity.txt"),
        DSV41_OWNER_GRAPH_PHASE_FILE=str(output / "owner/phase.txt"),
        P0_COUNTER_DIR=str(output / "counters"),
        PROMETHEUS_MULTIPROC_DIR=str(output / "prometheus"),
        PYTHONPATH=":".join(
            map(
                str,
                [
                    output / "bootstrap",
                    runtime,
                    output,
                    REPO / "tools/dsv41",
                    REPO,
                    aiter,
                ],
            )
        ),
    )
    (output / "environment.list").write_text(
        "".join(f"{key}={value}\n" for key, value in environment.items())
    )
    manifest = copy.deepcopy(lock["profile"])
    manifest["client"][1] = str(output / "formal_client.py")
    manifest["client"][manifest["client"].index("--result-dir") + 1] = str(output)
    manifest["source_package"] = str(HERE)
    manifest["reference_run"] = str(reference)
    manifest["qualification_limits"] = [
        "Measured gfx950/TP4 deployment and calibration only",
        "Task accuracy and observer transfer unqualified",
        "Preparing this runtime is not a new GPU/performance result",
    ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "output": str(output),
        "runtime_files": len(qualification["candidate_source_sha256"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--aiter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.reference_run.resolve(),
                args.aiter.resolve(),
                args.output.resolve(),
            )
        )
    )
