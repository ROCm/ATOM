"""Read only the status and logs belonging to one existing PP experiment."""
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import shutil
import subprocess

RUN = os.environ.get("OBSERVE_RUN_ID", "41430932776")
assert RUN.isdigit(), "An existing numeric GitHub run ID is required"
CASE = "m31-stack-mxfp4-pp4-index-freq4-equal-gsm8k20"
print("Observed at", datetime.now(timezone.utc).isoformat(), flush=True)


def query(args):
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=20)
        if result.returncode:
            print("Query error", args[0], result.stderr[:1000])
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        print("Query unavailable", args[0], type(exc).__name__)
        return ""


queue = query(["squeue", "--noheader", "--format=%A|%256j|%T|%M|%R"])
matches = [line for line in queue.splitlines() if RUN in line and CASE in line]
print("Matching queue entries:", *matches, sep="\n", flush=True)

roots = [Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG"), Path("/it-share/ATOMESH_LOG")]
seen = set()
for root in roots:
    for case_dir in sorted(root.glob(f"*{CASE}*-{RUN}-*")):
        resolved = case_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        print("Case directory", case_dir, flush=True)
        selected = list(case_dir.glob("slurm-*.out")) + list(case_dir.glob("slurm-*.err"))
        for job in case_dir.glob("slurm_job-*"):
            job_id = job.name.removeprefix("slurm_job-")
            if job_id.isdigit():
                info = query(["scontrol", "show", "job", job_id])
                print("Controller job", job_id)
                print("\n".join(line for line in info.splitlines() if any(
                    key in line for key in ("JobId=", "JobState=", "RunTime=", "StartTime=", "NodeList="))))
            selected += [job / "job-result.json", job / "rank-workload-0.json"]
            selected += list((job / "logs/eval").glob("*.log"))
            selected += list((job / "rank-0").glob("container-eval.log"))
        for path in selected:
            if not path.is_file():
                continue
            saved = Path("observed-logs") / path.relative_to(case_dir)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, saved)
            lines = path.read_text(errors="replace").splitlines()
            print("\nLOG", path, "lines", len(lines), "mtime",
                  datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
            error_starts = [i for i, line in enumerate(lines) if re.search(
                r"Traceback|AssertionError|RuntimeError|KeyError|cache miss|ERROR|TimeoutError", line)]
            for i in error_starts[:6]:
                print("Error context at", i + 1)
                print("\n".join(lines[max(0, i-3):i+20]))
            print("Tail:")
            print("\n".join(lines[-45:]))
        print(flush=True)
