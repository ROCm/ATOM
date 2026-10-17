#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Read an existing TW MiniMax C32 run from a CI submitter; never submit work."""

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

CASE = "minimax-m3-mxfp4-minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32-workspace_validation"
CASES = (
    "minimax-m3-mxfp4-minimax-m3-paged-prefill-kernel-tuning-workspace_validation",
    CASE,
    "minimax-m3-mxfp4-minimax-m3-fp4-1p1d-tp4-eagle3-gsm8k-20shot-full-c32-workspace_validation",
)
ROOTS = (
    Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG"),
    Path("/it-share/ATOMESH_LOG"),
)
PATTERNS = (
    "slurm_job-*/benchmark_results/paged-prefill-tuning/*.log",
    "slurm_job-*/benchmark_results/paged-prefill-tuning/*.jsonl",
    "slurm_job-*/benchmark_results/paged-prefill-tuning/rejected-configs.json",
    "slurm_job-*/benchmark_results/paged-prefill-tuning/sweep-summary.json",
    "slurm-*.out",
    "slurm-*.err",
    "slurm_job-*/rank-*/*.log",
    "slurm_job-*/logs/*.log",
    "slurm_job-*/logs/*/*.log",
    "slurm_job-*/benchmark_results/pd-atom-*.json",
    "slurm_job-*/benchmark_results/*/profile_export_aiperf.json",
    "slurm_job-*/benchmark_results/*/aiperf.log",
    "slurm_job-*/benchmark_results/*/*/aiperf.log",
    "slurm_job-*/benchmark_results/lmcache-host-memory/*.log",
    "slurm_job-*/benchmark_results/lmcache-host-memory/summary.json",
    "slurm_job-*/benchmark_results/lmcache-host-memory/host/hugetlb.json",
    "slurm_job-*/benchmark_results/runtime-diagnostics/*",
    "slurm_job-*/benchmark_results/runtime-diagnostics/*/*",
    "slurm_job-*/benchmark_results/*/benchmark-guard.json",
    "slurm_job-*/eval_results/*/results*.json",
    "slurm_job-*/eval_results/*/*/results*.json",
    "slurm_job-*/job-result.json",
    "slurm_job-*/rank-rc-*",
)
MAX_FILE_BYTES = 8 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_DIAGNOSTIC_BYTES = 2 << 20


def run_id(value):
    if not re.fullmatch(r"[0-9]{5,20}", value):
        raise argparse.ArgumentTypeError("expected a numeric GitHub run ID")
    return value


def scheduler_state(target):
    command = ["squeue"]
    if os.environ.get("USES_SPUR_CONTROLLER") == "1" and os.environ.get(
        "SPUR_CONTROLLER_ADDR"
    ):
        command += ["--controller", os.environ["SPUR_CONTROLLER_ADDR"]]
    base_command = command.copy()
    command += ["--noheader", "--format=%i|%j|%T|%M|%R|%N"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=False
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 6 and any(
                fields[1].strip().startswith(f"{case}-{target}-") for case in CASES
            ):
                rows.append(line)
        state = {
            "returncode": result.returncode,
            "rows": rows,
            "stderr": result.stderr[-4096:],
        }
        pending = [
            line.split("|")[0].strip()
            for line in rows
            if line.split("|")[2].strip() == "PENDING"
        ]
        pending = [job for job in pending if re.fullmatch(r"[0-9]+", job)]
        if pending:
            try:
                estimate = subprocess.run(
                    base_command
                    + [
                        "--start",
                        "--noheader",
                        "--jobs=" + ",".join(pending),
                        "--format=%i|%S|%R|%N",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                state["start_estimate"] = {
                    "returncode": estimate.returncode,
                    "rows": [
                        line
                        for line in estimate.stdout.splitlines()
                        if line.split("|")[0].strip() in pending
                    ],
                    "stderr": estimate.stderr[-4096:],
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                state["start_estimate"] = {"error": str(exc)}
        return state
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def summarize_eval_samples(path):
    """Compact per-question evidence without exporting repeated few-shot prompts."""
    filters, inputs = {}, {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = row["doc_id"]
            score = row["exact_match"]
            values = filters.setdefault(row["filter"], {})
            if doc_id in values:
                raise ValueError(f"duplicate evaluation sample {doc_id}")
            values[doc_id] = {
                "score": score,
                "answer": str(row.get("filtered_resps", ""))[:200],
                "target": str(row["doc"]["answer"]).rsplit("####", 1)[-1].strip(),
            }
            inputs[doc_id] = [row["doc_hash"], row["prompt_hash"], row["target_hash"]]
    return {
        "input_digest": hashlib.sha256(
            json.dumps(sorted(inputs.items()), ensure_ascii=True).encode()
        ).hexdigest(),
        "filters": {
            name: {
                "samples": len(rows),
                "correct": sum(row["score"] for row in rows.values()),
                "wrong": {
                    str(doc_id): row
                    for doc_id, row in rows.items()
                    if row["score"] != 1
                },
            }
            for name, rows in filters.items()
        },
    }


def collect(target, output, *, roots=ROOTS):
    run_id(target)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": target,
        "host": socket.gethostname(),
        "unix_time": time.time(),
        "scheduler": scheduler_state(target),
        "roots": [],
        "files": [],
        "errors": [],
    }
    used = 0
    seen = set()
    for root in roots:
        row = {"root": str(root), "exists": root.is_dir(), "runs": []}
        manifest["roots"].append(row)
        for source in sorted(
            path for case in CASES for path in root.glob(f"{case}-{target}-*")
        ):
            source = source.resolve()
            if not source.is_relative_to(root.resolve()) or not source.is_dir():
                continue
            row["runs"].append(str(source))
            for sample in source.glob(
                "slurm_job-*/eval_results/*/*/samples_gsm8k_*.jsonl"
            ):
                if not sample.resolve().is_relative_to(source):
                    continue
                try:
                    scores = summarize_eval_samples(sample)
                    destination = (
                        output / source.name / sample.relative_to(source)
                    ).with_suffix(".scores.json")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(scores) + "\n")
                except (OSError, ValueError, KeyError) as exc:
                    manifest["errors"].append({"path": str(sample), "error": str(exc)})
            for pattern in PATTERNS:
                # Rotated history must not crowd out another rank's live evidence.
                for path in sorted(
                    source.glob(pattern),
                    key=lambda path: (".previous." in path.name, path),
                ):
                    resolved = path.resolve()
                    if (
                        resolved in seen
                        or not resolved.is_relative_to(source)
                        or not resolved.is_file()
                    ):
                        continue
                    seen.add(resolved)
                    if used >= MAX_TOTAL_BYTES or len(manifest["files"]) >= 512:
                        manifest["limit_reached"] = True
                        break
                    relative = path.relative_to(source)
                    destination = output / source.name / relative
                    try:
                        with resolved.open("rb") as stream:
                            size = os.fstat(stream.fileno()).st_size
                            file_limit = MAX_FILE_BYTES
                            if "runtime-diagnostics" in relative.parts:
                                file_limit = min(file_limit, MAX_DIAGNOSTIC_BYTES)
                            count = min(size, file_limit, MAX_TOTAL_BYTES - used)
                            stream.seek(max(0, size - count))
                            data = stream.read(count)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(data)
                        used += len(data)
                        manifest["files"].append(
                            {
                                "source": str(path),
                                "artifact": str(destination.relative_to(output)),
                                "original_bytes": size,
                                "copied_bytes": len(data),
                                "tail_only": size > len(data),
                            }
                        )
                    except OSError as exc:
                        manifest["errors"].append(
                            {"path": str(path), "error": str(exc)}
                        )
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def emit_log_annotations(output):
    """Provide a bounded API-readable fallback when blob downloads are unavailable."""
    manifest = output / "snapshot.json"
    if manifest.is_file():
        state = json.loads(manifest.read_text())
        state.pop("files", None)
        (output / "runtime-state.json").write_text(json.dumps(state) + "\n")
    selected = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        if path.name in (
            "container.log",
            "container-eval.log",
            "runtime-state.json",
            "sweep-summary.json",
            "profile_export_aiperf.json",
            "prefill-rank-0.log",
            "decode-rank-0.log",
            "aiperf.log",
        ) or (
            path.suffix == ".json"
            and path.name.startswith(("pd-atom-", "results_", "samples_gsm8k_"))
        ):
            selected.append(path)
        elif (path.name.startswith("slurm-") and path.suffix in (".out", ".err")) or (
            "lmcache-host-memory" in path.parts and path.suffix in (".log", ".json")
        ):
            selected.append(path)
        elif "paged-prefill-tuning" in path.parts and path.suffix == ".log":
            tail = path.read_text(errors="replace")[-16000:]
            if any(
                marker in tail
                for marker in (
                    "Traceback",
                    "Memory access fault",
                    '"accuracy": "failed"',
                )
            ):
                selected.append(path)
    selected.sort(
        key=lambda path: (
            path.name not in ("sweep-summary.json", "container.log"),
            str(path),
        )
    )
    for log_limit in (16000, 4000, 1000):
        files, tails = {}, []
        for path in selected[:40]:
            relative = str(path.relative_to(output))
            raw = path.read_text(errors="replace")
            limit = len(raw) if path.suffix == ".json" else log_limit
            files[relative] = raw[-limit:]
            if len(raw) > limit:
                tails.append(relative)
        payload = json.dumps(
            {"format": "atomesh-log-snapshot-v1", "files": files, "tail_only": tails}
        )
        if len(payload) <= 400000:
            break
    encoded = base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode()
    # Check annotations are truncated at 4 KiB by the API. Compress the
    # snapshot and keep each notice below that limit and the ten-notice quota.
    while len(encoded) > 35000:
        longest = max(files, key=lambda name: len(files[name]))
        files[longest] = files[longest][len(files[longest]) // 2 :]
        if longest not in tails:
            tails.append(longest)
        payload = json.dumps(
            {"format": "atomesh-log-snapshot-v1", "files": files, "tail_only": tails}
        )
        encoded = base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode()
    parts = [encoded[i : i + 3500] for i in range(0, len(encoded), 3500)]
    for index, part in enumerate(parts):
        escaped = part.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(
            f"::notice title=ATOMESH log snapshot {index + 1}/{len(parts)}::{escaped}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=run_id)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-annotations", action="store_true")
    args = parser.parse_args()
    collect(args.run_id, args.output)
    if args.github_annotations:
        emit_log_annotations(args.output)


if __name__ == "__main__":
    main()
