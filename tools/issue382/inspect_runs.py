"""Read bounded snapshots of existing experiment logs; never allocate GPUs."""

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
from pathlib import Path


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = args.run_ids.split(",")
    if not 1 <= len(ids) <= 4 or any(
        not re.fullmatch(r"[0-9]+", rid) for rid in ids
    ):
        parser.error("provide one to four numeric GitHub run IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = "/usr/local/slurm-24.05.5.1/bin:" + os.environ.get(
        "PATH", ""
    )
    queue = command(["squeue", "--noheader", "--format=%i %T %D %j %N %R"])
    lines = queue.get("stdout", "").splitlines()
    queue["stdout"] = "\n".join(
        line for line in lines if any(f"-{rid}-" in line for rid in ids)
    )
    report = {
        "collected_at": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "collector_host": socket.gethostname(),
        "sbatch_help": command(["sbatch", "--help"]),
        "queue": queue,
        "runs": [],
    }
    for rid in ids:
        row = {"run_id": rid, "jobs": [], "log_roots": []}
        job_ids = set()
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
                        }
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
    (args.output / "inspection.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
