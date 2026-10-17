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
import urllib.error
import urllib.request
from collections import Counter
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
    "slurm_job-*/cell-metadata.json",
    "slurm_job-*/metadata-rank-*.json",
    "slurm_job-*/rank-rc-*",
    "slurm_job-*/rank-workload-*",
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


def github_failure_log(target, output):
    """Fetch submitter failures inside CI without forwarding auth to blob storage."""
    token = os.environ.get("GH_TOKEN")
    if not token:
        return

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def api(path):
        request = urllib.request.Request(
            "https://api.github.com/repos/ROCm/ATOM/" + path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/vnd.github+json",
            },
        )
        try:
            with opener.open(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 302:
                raise
            location = exc.headers["Location"]
            if not location.startswith("https://"):
                raise ValueError("non-HTTPS artifact redirect")
            # A fresh request intentionally omits the GitHub Authorization header.
            with urllib.request.urlopen(location, timeout=60) as response:
                return response.read()

    result = {}
    try:
        state = json.loads(api(f"actions/runs/{target}"))
        if state["status"] != "completed" or state["conclusion"] == "success":
            return
        jobs = json.loads(api(f"actions/runs/{target}/jobs?per_page=100"))["jobs"]
        for job in jobs:
            if job["conclusion"] not in ("failure", "timed_out", "cancelled"):
                continue
            lines = (
                api(f"actions/jobs/{job['id']}/logs")
                .decode(errors="replace")
                .splitlines()
            )
            markers = re.compile(
                r"scancel|cancel.*Slurm|exiting rc|received TERM|Traceback|Error:|ERROR:|No such file|exit code|command not found|Permission denied|bad substitution",
                re.IGNORECASE,
            )
            indices = set()
            for i, line in enumerate(lines):
                if "[spur:" not in line and markers.search(line):
                    indices.update(range(max(0, i - 5), min(len(lines), i + 6)))
            result[str(job["id"])] = {
                "name": job["name"],
                "excerpts": "\n".join(lines[i][:1200] for i in sorted(indices))[
                    -24000:
                ],
            }
    except (OSError, ValueError) as exc:
        # Exception text may contain signed URLs; emit only the type/status.
        result["error"] = type(exc).__name__
        result["http_status"] = getattr(exc, "code", None)
    (output / "workflow-failure.json").write_text(json.dumps(result) + "\n")


def job_diagnostics(source):
    """Read final scheduler accounting and small lifecycle files, without execution."""
    result = {"files": {}, "scheduler": {}}
    for job in source.glob("slurm_job-*"):
        job_id = job.name.removeprefix("slurm_job-")
        if not re.fullmatch(r"[0-9]+", job_id):
            continue
        for path in sorted(job.glob("*")):
            if path.is_file():
                result["files"][str(path.relative_to(source))] = {
                    "bytes": path.stat().st_size,
                    "body": (
                        path.read_text(errors="replace")
                        if path.name.startswith(
                            ("rank-rc-", "rank-status-", "workload-status-")
                        )
                        and path.stat().st_size < 4096
                        else None
                    ),
                }
        command = ["scontrol"]
        if os.environ.get("USES_SPUR_CONTROLLER") == "1" and os.environ.get(
            "SPUR_CONTROLLER_ADDR"
        ):
            command += ["--controller", os.environ["SPUR_CONTROLLER_ADDR"]]
        try:
            query = subprocess.run(
                command + ["show", "job", job_id],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            result["scheduler"][job_id] = {
                "returncode": query.returncode,
                "stdout": query.stdout[-16000:],
                "stderr": query.stderr[-2000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["scheduler"][job_id] = {"error": str(exc)}
    return result


def summarize_prefill_shapes(path):
    """Summarize scheduler metadata from the complete log, not the copied tail.

    These are scheduled target batches; they are not kernel timings or an
    assertion that every draft invocation uses exactly the same query shape.
    """
    pattern = re.compile(
        r"Scheduled prefill batch: (\d+) reqs, (\d+) new tokens "
        r"\(done: (\[[0-9, ]*\]), new: (\[[0-9, ]*\])\)"
    )
    batches, shapes = [], Counter()
    first, last, malformed = None, None, 0
    with path.open(errors="replace") as stream:
        for line in stream:
            if "Scheduled prefill batch:" not in line:
                continue
            match = pattern.search(line)
            if match is None:
                malformed += 1
                continue
            count, total = map(int, match.group(1, 2))
            cached, new = map(json.loads, match.group(3, 4))
            if len(cached) != count or len(new) != count or sum(new) != total:
                malformed += 1
                continue
            timestamp = line[: match.start()].strip()
            first = first or timestamp
            last = timestamp
            contexts = [c + q for c, q in zip(cached, new)]
            batches.append(
                (count, total, max(contexts), sum(contexts), bool(sum(cached)))
            )
            shapes[tuple(zip(new, contexts))] += 1

    def quantiles(values):
        if not values:
            return {}
        values = sorted(values)
        return {
            label: values[round((len(values) - 1) * percentile)]
            for label, percentile in (
                ("p50", 0.5),
                ("p90", 0.9),
                ("p99", 0.99),
                ("max", 1),
            )
        }

    def describe(shape, count):
        return {
            "q_lens": [v[0] for v in shape],
            "kv_lens": [v[1] for v in shape],
            "count": count,
        }

    return {
        "source": str(path),
        "first": first,
        "last": last,
        "scope": "complete available scheduler log; may include warmup",
        "batches": len(batches),
        "malformed_lines": malformed,
        "cached_batches": sum(row[4] for row in batches),
        "batch_size": quantiles([row[0] for row in batches]),
        "query_tokens": quantiles([row[1] for row in batches]),
        "max_context": quantiles([row[2] for row in batches]),
        "sum_context": quantiles([row[3] for row in batches]),
        "frequent_shapes": [
            describe(shape, count) for shape, count in shapes.most_common(12)
        ],
        "largest_query_batches": [
            describe(shape, count)
            for shape, count in sorted(
                shapes.items(),
                key=lambda item: sum(v[0] for v in item[0]),
                reverse=True,
            )[:12]
        ],
        "largest_context_batches": [
            describe(shape, count)
            for shape, count in sorted(
                shapes.items(),
                key=lambda item: sum(v[1] for v in item[0]),
                reverse=True,
            )[:12]
        ],
    }


def summarize_prefill_tail_windows(path, requests):
    """Read scheduler activity around the 20 slowest requests, without GPU work.

    Log clocks are interpreted as UTC, matching the benchmark containers.
    Scheduled QK pairs describe workload, not kernel duration or queue time.
    """
    seconds = set()
    windows = []
    for request in requests.get("top", [])[:20]:
        start_ns = request.get("request_start_ns")
        ttft = request.get("ttft_ms")
        if start_ns is None or ttft is None:
            continue
        start = int(start_ns // 1_000_000_000)
        # Bound malformed records rather than scanning an unbounded time span.
        duration = min(600, max(0, int(ttft / 1000) + 1))
        seconds.update(s % 86400 for s in range(start - 1, start + duration + 2))
        windows.append(
            {
                "request_id": request.get("x_request_id"),
                "start_ns": start_ns,
                "ttft_ms": ttft,
            }
        )
    clock = re.compile(r"\[atom (\d{2}):(\d{2}):(\d{2})\]")
    shape = re.compile(
        r"Scheduled prefill batch: (\d+) reqs, (\d+) new tokens "
        r"\(done: (\[[0-9, ]*\]), new: (\[[0-9, ]*\])\)"
    )
    timeline = {}
    compile_events, compile_count = [], 0
    with path.open(errors="replace") as stream:
        for line in stream:
            stamp = clock.search(line)
            if stamp is None:
                continue
            h, m, sec = map(int, stamp.groups())
            second = 3600 * h + 60 * m + sec
            if second not in seconds:
                continue
            if re.search(r"compil|\bJIT\b", line, re.IGNORECASE):
                compile_count += 1
                if len(compile_events) < 30:
                    compile_events.append(line.rstrip()[:500])
            match = shape.search(line)
            if match is None:
                continue
            count, total = map(int, match.group(1, 2))
            cached, new = map(json.loads, match.group(3, 4))
            if len(cached) != count or len(new) != count or sum(new) != total:
                continue
            key = f"{h:02}:{m:02}:{sec:02}"
            row = timeline.setdefault(
                key, {"batches": 0, "queries": 0, "causal_qk_pairs": 0, "max_kv": 0}
            )
            row["batches"] += 1
            row["queries"] += total
            row["causal_qk_pairs"] += sum(
                c * q + q * (q + 1) // 2 for c, q in zip(cached, new)
            )
            row["max_kv"] = max(row["max_kv"], max(c + q for c, q in zip(cached, new)))
    return {
        "clock": "UTC assumed for server log",
        "scope": "20 slowest TTFT windows, at most 600 seconds each",
        "windows": windows,
        "scheduler_by_second": timeline,
        "compilation_log_matches": compile_count,
        "compilation_events": compile_events,
        "limitation": "Scheduler activity is not per-kernel timing; absent compilation messages do not rule out JIT.",
    }


def summarize_aiperf_tail(path):
    """Stream completed profiling requests into compact tail-latency evidence."""
    from interactivity import agentic_throughput

    def metric(record, name):
        value = record.get("metrics", {}).get(name)
        if isinstance(value, dict):
            value = value.get("value")
        return value if isinstance(value, (int, float)) else None

    def percentile(values, p):
        values = sorted(values)
        if not values:
            return None
        rank = (len(values) - 1) * p / 100
        low = int(rank)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (rank - low)

    rows, errors, malformed, cancelled = [], 0, 0, 0
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "profiling":
                continue
            if record.get("error"):
                errors += 1
                continue
            if metadata.get("was_cancelled"):
                cancelled += 1
                continue
            ttft = metric(record, "time_to_first_token")
            if ttft is None:
                continue
            row = {
                key: metadata.get(key)
                for key in (
                    "x_request_id",
                    "source_trace_id",
                    "source_outer_idx",
                    "turn_index",
                    "source_kind",
                    "request_start_ns",
                    "request_end_ns",
                )
            }
            row.update(
                ttft_ms=ttft,
                isl=metric(record, "input_sequence_length"),
                cached=metric(record, "usage_prompt_cache_read_tokens"),
                osl=metric(record, "output_sequence_length"),
                latency_ms=metric(record, "request_latency"),
            )
            row["uncached"] = (
                row["isl"] - row["cached"]
                if row["isl"] is not None and row["cached"] is not None
                else None
            )
            rows.append(row)
    groups = []
    for field in ("isl", "uncached"):
        low = 0
        for high in (1024, 8192, 32768, 131072, 524288, 1048577):
            population = [
                r["ttft_ms"]
                for r in rows
                if r[field] is not None and low <= r[field] < high
            ]
            groups.append(
                {
                    "field": field,
                    "low": low,
                    "high": high,
                    "count": len(population),
                    "p50_ms": percentile(population, 50),
                    "p90_ms": percentile(population, 90),
                    "p99_ms": percentile(population, 99),
                }
            )
            low = high
    return {
        "throughput": agentic_throughput(path) if rows else None,
        "successful": len(rows),
        "errors": errors,
        "malformed": malformed,
        "cancelled": cancelled,
        "buckets": groups,
        "ttft_ms": {
            str(p): percentile([r["ttft_ms"] for r in rows], p) for p in (50, 90, 99)
        },
        "top": sorted(rows, key=lambda r: r["ttft_ms"], reverse=True)[:100],
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
            diagnostics = output / source.name / "job-diagnostics.json"
            diagnostics.parent.mkdir(parents=True, exist_ok=True)
            diagnostics.write_text(json.dumps(job_diagnostics(source)) + "\n")
            for prefill in source.glob("slurm_job-*/logs/prefill-rank-0.log"):
                if not prefill.resolve().is_relative_to(source):
                    continue
                try:
                    shapes = summarize_prefill_shapes(prefill)
                    destination = (
                        output / source.name / prefill.relative_to(source)
                    ).with_name("prefill-shapes.json")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(shapes) + "\n")
                except (OSError, ValueError) as exc:
                    manifest["errors"].append({"path": str(prefill), "error": str(exc)})
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
            for records in source.glob(
                "slurm_job-*/benchmark_results/*/profile_export.jsonl"
            ):
                if not records.resolve().is_relative_to(source):
                    continue
                try:
                    summary = summarize_aiperf_tail(records)
                    destination = (
                        output / source.name / records.relative_to(source)
                    ).with_name("aiperf-tail-summary.json")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(summary) + "\n")
                    prefill = records.parents[2] / "logs/prefill-rank-0.log"
                    if prefill.is_file() and prefill.resolve().is_relative_to(source):
                        windows = summarize_prefill_tail_windows(prefill, summary)
                        destination.with_name("prefill-tail-windows.json").write_text(
                            json.dumps(windows) + "\n"
                        )
                except (OSError, ValueError) as exc:
                    manifest["errors"].append({"path": str(records), "error": str(exc)})
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
        state["log_sizes"] = [
            {
                key: item[key]
                for key in ("artifact", "original_bytes", "copied_bytes", "tail_only")
            }
            for item in state.pop("files", [])
            if Path(item["artifact"]).name
            in (
                "container.log",
                "prefill-rank-0.log",
                "decode-rank-0.log",
                "router.log",
                "aiperf.log",
            )
        ]
        (output / "runtime-state.json").write_text(json.dumps(state) + "\n")
    selected = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        if (
            path.name
            in (
                "container.log",
                "container-eval.log",
                "router.log",
                "job-result.json",
                "job-diagnostics.json",
                "workflow-failure.json",
                "cell-metadata.json",
                "metadata-rank-0.json",
                "runtime-state.json",
                "prefill-shapes.json",
                "prefill-tail-windows.json",
                "aiperf-tail-summary.json",
                "sweep-summary.json",
                "profile_export_aiperf.json",
                "prefill-rank-0.log",
                "decode-rank-0.log",
                "aiperf.log",
            )
            or (
                path.suffix == ".json"
                and path.name.startswith(("pd-atom-", "results_", "samples_gsm8k_"))
            )
            or path.name.startswith(("rank-rc-", "rank-workload-"))
            or (path.name.startswith("slurm-") and path.suffix in (".out", ".err"))
            or (
                "lmcache-host-memory" in path.parts and path.suffix in (".log", ".json")
            )
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
        reducible = [
            name
            for name in files
            if Path(name).name
            not in (
                "aiperf-tail-summary.json",
                "prefill-tail-windows.json",
                "sweep-summary.json",
                "job-result.json",
            )
            and not Path(name).name.startswith(("pd-atom-", "results_"))
            and len(files[name]) > 1000
        ]
        if not reducible:
            raise RuntimeError("Required JSON summaries exceed annotation budget")
        longest = max(reducible, key=lambda name: len(files[name]))
        if longest.endswith(".json"):
            files[longest] = json.dumps(
                {
                    "omitted_from_annotations": True,
                    "original_chars": len(files[longest]),
                }
            )
        else:
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
    github_failure_log(args.run_id, args.output)
    if args.github_annotations:
        emit_log_annotations(args.output)


if __name__ == "__main__":
    main()
