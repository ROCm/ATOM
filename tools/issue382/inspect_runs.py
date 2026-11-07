"""Read bounded snapshots of existing experiment logs; never allocate GPUs."""

import argparse
import datetime
import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path

REPLACED_JOBS = {
    "41654896595": ("11878", "issue382-a1-c140-mooncake_store"),
    "41654904191": ("11879", "issue382-a2-c140-mooncake_store"),
    "41654913083": ("11880", "issue382-a1-correctness64-accuracy"),
    "41654920836": ("11881", "issue382-a2-correctness64-accuracy"),
}

GLM_PENDING_JOBS = {
    "42337229149": ("13343", "295e2f40933709084774fb87c25a395a5f8457ae"),
    "42337231792": ("13342", "b46fe0923b87017fa1d4d00b7b8e07a12db1aadf"),
}


def shorten_pending_glm_jobs(ids):
    """Lower only these two pending accuracy allocations from eight to two hours.

    Keep their job IDs, submission times, priority, account and placement. A
    running or mismatched allocation is never updated. This is opt-in and
    independent of inspection and the issue382 retirement action.
    """
    if any(rid not in GLM_PENDING_JOBS for rid in ids):
        raise ValueError("time-limit update is restricted to the two GLM runs")
    results = []
    for rid in ids:
        job_id, head = GLM_PENDING_JOBS[rid]
        before = command(["scontrol", "show", "job", job_id])
        row = {"run_id": rid, "job_id": job_id, "before": before}
        results.append(row)
        fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", before.get("stdout", "")))
        expected = (
            "glm-5.2-mxfp4-glm-52-mxfp4-1p1d-cpp4-dcp4-gsm8k-20shot-"
            f"landing-c32-accuracy-{rid}-1"
        )
        if before.get("returncode") != 0 or any(
            fields.get(key) != value
            for key, value in {
                "JobId": job_id,
                "JobName": expected,
                "Account": "amd-atom",
                "Partition": "amd-spur",
            }.items()
        ):
            row["error"] = "job identity does not match the recorded experiment"
            continue
        if fields.get("JobState") != "PENDING":
            row["skipped_state"] = fields.get("JobState")
            continue
        if fields.get("TimeLimit") == "02:00:00":
            row["already_shortened"] = True
            continue
        workdir = fields.get("WorkDir", "")
        if fields.get("TimeLimit") != "08:00:00" or not workdir.startswith(
            "/share_nfs/ATOMESH_RUNNER/"
        ):
            row["error"] = "unexpected original limit or source directory"
            continue
        source = command(["git", "-C", workdir, "rev-parse", "HEAD"])
        row["source_head"] = source
        if source.get("returncode") != 0 or source.get("stdout", "").strip() != head:
            row["error"] = "source head changed"
            continue
        row["update"] = command(
            ["scontrol", "update", f"JobId={job_id}", "TimeLimit=02:00:00"]
        )
        row["after"] = command(["scontrol", "show", "job", job_id])
    return results


def retire_replaced_jobs(ids):
    """Cancel only the four recorded jobs with confirmed source drift.

    This is an explicit recovery action, disabled during normal inspection.
    Match the full original job name before cancelling a numeric job ID.
    """
    if any(rid not in REPLACED_JOBS for rid in ids):
        raise ValueError("retirement is restricted to the four replaced runs")
    results = []
    for rid in ids:
        job_id, case = REPLACED_JOBS[rid]
        before = command(["scontrol", "show", "job", job_id])
        stdout = before.get("stdout", "")
        expected = f"minimax-m3.1-issue382-round1-{case}-{rid}-1"
        fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", stdout))
        if before.get("returncode") != 0:
            raise RuntimeError(f"cannot verify original job {job_id}: {before}")
        if fields.get("JobId") != job_id or fields.get("JobName") != expected:
            raise RuntimeError(f"refusing to cancel mismatched job {job_id}")
        result = {"run_id": rid, "job_id": job_id, "before": before}
        if fields.get("JobState") in {"PENDING", "RUNNING", "CONFIGURING"}:
            result["cancel"] = command(["scancel", job_id])
            if result["cancel"].get("returncode") != 0:
                raise RuntimeError(f"cancellation failed: {result}")
        else:
            result["skipped_state"] = fields.get("JobState")
        result["after"] = command(["scontrol", "show", "job", job_id])
        results.append(result)
        print(f"retirement={json.dumps(result)}", flush=True)
    return results


def collect_completed_results(source, destination, max_file_bytes=1 << 30):
    """Recover full records after the single-node worker finishes successfully.

    Raw records are never truncated. Missing or oversized files are reported
    so validation can reject an incomplete recovery.
    """
    recovered = []
    for job_dir in source.glob("slurm_job-*"):
        if job_dir.is_symlink() or not job_dir.is_dir():
            continue
        try:
            proof = json.loads((job_dir / "rank-workload-0.json").read_text())
            rc = (job_dir / "rank-rc-0").read_text().strip()
        except (OSError, ValueError):
            continue
        if (
            proof.get("job_id") != job_dir.name.removeprefix("slurm_job-")
            or proof.get("num_ranks") != 1
            or proof.get("rank") != 0
            or proof.get("status") != "completed"
            or not proof.get("run_token")
            or rc != "0"
        ):
            continue
        for directory, dirs, files in os.walk(job_dir, followlinks=False):
            dirs[:] = [
                d
                for d in dirs
                if not (Path(directory) / d).is_symlink()
                and d not in {".cache", ".aiter", "torch_traces", "mesh-build"}
            ]
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                relative = path.relative_to(job_dir)
                result_file = relative.parts[0] in {
                    "benchmark_results",
                    "eval_results",
                } and path.suffix in {
                    ".json",
                    ".jsonl",
                    ".gz",
                    ".csv",
                    ".txt",
                    ".md",
                    ".log",
                }
                if not (
                    result_file
                    or name == "issue382-metrics.jsonl.gz"
                    or name
                    in {"rank-workload-0.json", "rank-rc-0", "job-result.json"}
                ):
                    continue
                record = {"path": str(path.relative_to(source))}
                try:
                    before = path.stat()
                    record["bytes"] = before.st_size
                    if before.st_size > max_file_bytes:
                        record["error"] = (
                            "exceeds full-result copy limit; not copied"
                        )
                    else:
                        target = destination / path.relative_to(source)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                        after = path.stat()
                        if (before.st_size, before.st_mtime_ns) != (
                            after.st_size,
                            after.st_mtime_ns,
                        ):
                            record["error"] = "source changed during recovery"
                        else:
                            record["copied_full"] = True
                except OSError as exc:
                    record["error"] = str(exc)
                recovered.append(record)
    return recovered


def command(args):
    try:
        result = subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=20
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def snapshot(source, destination):
    """Copy at most 256 KiB per selected log, preserving line boundaries."""
    rows = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if d not in {".cache", ".aiter", "torch_traces", "mesh-build"}
        ]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            inventory_only = name in {
                "server_metrics_export.json",
                "profile_export.json",
            }
            if inventory_only or path.suffix not in {
                ".log",
                ".out",
                ".err",
                ".json",
            }:
                if inventory_only or path.suffix in {".jsonl", ".gz", ".csv"}:
                    try:
                        info = path.stat()
                        rows.append(
                            {
                                "path": str(path.relative_to(source)),
                                "bytes": info.st_size,
                                "mtime": info.st_mtime,
                                "inventory_only": True,
                            }
                        )
                    except OSError as exc:
                        rows.append(
                            {
                                "path": str(path.relative_to(source)),
                                "error": str(exc),
                            }
                        )
                continue
            relative = path.relative_to(source)
            try:
                info = path.stat()
                if path.suffix == ".json" and info.st_size > 128 * 1024:
                    rows.append(
                        {
                            "path": str(relative),
                            "bytes": info.st_size,
                            "mtime": info.st_mtime,
                            "inventory_only": True,
                        }
                    )
                    continue
                with path.open("rb") as stream:
                    offset = max(0, info.st_size - 256 * 1024)
                    stream.seek(offset)
                    data = stream.read(256 * 1024)
                if offset:
                    data = data.partition(b"\n")[2]
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if offset:
                    with path.open("rb") as stream:
                        target.with_name(target.name + ".head").write_bytes(
                            stream.read(64 * 1024)
                        )
                rows.append(
                    {
                        "path": str(relative),
                        "bytes": info.st_size,
                        "mtime": info.st_mtime,
                        "copied_bytes": len(data),
                    }
                )
            except OSError as exc:
                rows.append({"path": str(relative), "error": str(exc)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--job-map", type=json.loads, default={})
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retire-replaced-runs", action="store_true")
    parser.add_argument("--shorten-pending-glm-jobs", action="store_true")
    args = parser.parse_args()
    ids = args.run_ids.split(",")
    if not 1 <= len(ids) <= 4 or any(
        not re.fullmatch(r"[0-9]+", rid) for rid in ids
    ):
        parser.error("provide one to four numeric GitHub run IDs")
    if args.retire_replaced_runs and any(
        rid not in REPLACED_JOBS for rid in ids
    ):
        parser.error("retirement is restricted to the four replaced runs")
    if not isinstance(args.job_map, dict) or any(
        rid not in ids
        or not isinstance(jobs, list)
        or len(jobs) > 4
        or any(not re.fullmatch(r"[0-9]+", str(job)) for job in jobs)
        for rid, jobs in args.job_map.items()
    ):
        parser.error(
            "job-map must map inspected run IDs to up to four numeric job IDs"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = "/usr/local/slurm-24.05.5.1/bin:" + os.environ.get(
        "PATH", ""
    )
    queue = command(["squeue", "--noheader", "--format=%i %T %D %j %N %R"])
    lines = queue.get("stdout", "").splitlines()
    queue["stdout"] = "\n".join(
        line for line in lines if any(f"-{rid}-" in line for rid in ids)
    )
    print(f"queue_result={json.dumps(queue)}", flush=True)
    report = {
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "collector_host": socket.gethostname(),
        "sbatch_help": command(["sbatch", "--help"]),
        "scontrol_help": command(["scontrol", "--help"]),
        "scontrol_update_help": command(["scontrol", "update", "--help"]),
        "queue": queue,
        "runs": [],
    }
    for rid in ids:
        row = {"run_id": rid, "jobs": [], "log_roots": []}
        job_ids = {str(job) for job in args.job_map.get(rid, [])}
        for line in lines:
            if f"-{rid}-" not in line:
                continue
            job_id = line.split()[0]
            if job_id.isdigit():
                job_ids.add(job_id)
        for base in (
            Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG"),
            Path("/it-share/ATOMESH_LOG"),
        ):
            if not base.is_dir():
                continue
            for source in base.glob(f"*-{rid}-*"):
                if source.is_dir() and not source.is_symlink():
                    target = args.output / rid / source.name
                    row["log_roots"].append(
                        {
                            "source": str(source),
                            "files": snapshot(source, target),
                            "completed_results": collect_completed_results(
                                source, target
                            ),
                        }
                    )
                    # Preserve workload progress when runner failures prevent
                    # artifact upload. Full raw records still require recovery.
                    for log in sorted(target.rglob("aiperf.log")):
                        if log.parent.name == "logs":
                            continue
                        print(
                            f"run={rid} progress={json.dumps(log.read_text(errors='replace').splitlines()[-12:])}",
                            flush=True,
                        )
                    for job_dir in source.glob("slurm_job-*"):
                        job_id = job_dir.name.removeprefix("slurm_job-")
                        if job_id.isdigit():
                            job_ids.add(job_id)
        for job_id in sorted(job_ids):
            job = {"id": job_id, **command(["scontrol", "show", "job", job_id])}
            workdir = re.search(r"WorkDir=(\S+)", job.get("stdout", ""))
            if workdir and workdir[1].startswith("/share_nfs/ATOMESH_RUNNER/"):
                job["source_head"] = command(
                    ["git", "-C", workdir[1], "rev-parse", "HEAD"]
                )
                job["source_status"] = command(
                    [
                        "git",
                        "-C",
                        workdir[1],
                        "status",
                        "--short",
                        "--untracked-files=no",
                    ]
                )
            row["jobs"].append(job)
            # Preserve controller evidence even if the artifact uploader fails.
            print(
                f"run={rid} job={job_id} controller={json.dumps(job)}",
                flush=True,
            )
        report["runs"].append(row)
        print(
            f"run={rid} jobs={len(row['jobs'])} log_roots={len(row['log_roots'])}",
            flush=True,
        )
    if args.retire_replaced_runs:
        report["retirement"] = retire_replaced_jobs(ids)
    if args.shorten_pending_glm_jobs:
        report["time_limit_updates"] = shorten_pending_glm_jobs(ids)
    (args.output / "inspection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
