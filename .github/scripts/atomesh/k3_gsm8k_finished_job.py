"""Inspect and release only the completed K3 GSM8K allocation 5141."""

import json
import os
import pwd
import re
import subprocess
import time
from pathlib import Path

from pd_job_result import workload_completed

JOB = "5141"
TOKEN = "b4ce150ddc46409da1a224bbcaa3f276"
CELL = "kimi-k3-mxfp4-vllm-dspark-kimi-k3-gsm8k-without-59164-dspark3-tp8-dcp8-eager-vllm"
NAME = f"{CELL}-36832052267-1"
ROOT = Path(
    f"/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG/{CELL}"
    "-36832052267-20261001074510/slurm_job-5141"
)
CONTROLLERS = [f"http://pit2-vm-amd-large-{n:02}:6817" for n in (4, 2, 3)]
OUT = Path("inspection/finished-job-5141")


def command(args, name):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        result = {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        result = {"error": str(exc)}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def finished(root):
    if not workload_completed(root, JOB, TOKEN, 2):
        return False
    for phase in ("eval",):
        if (root / f"phase-{phase}.complete").read_text().strip() != f"{TOKEN}:{phase}":
            return False
    for rank in range(2):
        expected = {
            "job_id": JOB,
            "run_token": TOKEN,
            "rank": rank,
            "query_rc": 0,
            "return_code": 0,
            "finished": True,
        }
        if json.loads((root / f"cleanup-complete-{rank}.json").read_text()) != expected:
            return False
        if (root / f"cleanup-containers-{rank}.txt").read_text().strip():
            return False
    return True


def queue_row(result, owner):
    if result.get("rc") != 0 or result.get("stderr", "").strip():
        raise ValueError("Queue query failed or emitted a diagnostic")
    rows = []
    for line in result["stdout"].splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) != 4 or not re.fullmatch(r"[0-9]+", fields[0]):
            raise ValueError("Malformed queue response")
        if fields[0] == JOB:
            rows.append(fields)
    if not rows:
        return None
    if len(rows) != 1 or rows[0][1:3] != [NAME, owner]:
        raise ValueError("Job identity mismatch")
    return rows[0]


def queue(controller, name, owner):
    return queue_row(
        command(
            ["squeue", "--controller", controller, "--noheader", "--format=%i|%j|%u|%T"],
            name,
        ),
        owner,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    owner = pwd.getpwuid(os.getuid()).pw_name
    for index, controller in enumerate(CONTROLLERS):
        command(["sdiag", "--controller", controller], f"{index}-sdiag")
        command(
            ["scontrol", "--controller", controller, "show", "job", JOB],
            f"{index}-job-before",
        )
    if not finished(ROOT):
        raise RuntimeError("Completed workload and successful cleanup are required")
    rejection = f"scancel: error cancelling job {JOB}: not the Raft leader"
    for index, controller in enumerate(CONTROLLERS):
        row = queue(controller, f"{index}-queue-before", owner)
        if row is None:
            print(f"{controller}: job absent; no cancellation")
            continue
        if row[3] != "RUNNING":
            raise RuntimeError(f"Unexpected allocation state: {row[3]}")
        if not finished(ROOT):
            raise RuntimeError("Completion evidence changed")
        result = command(
            ["scancel", "--controller", controller, JOB], f"{index}-cancel"
        )
        # Spur may exit zero even when its per-job cancellation is rejected.
        lines = [
            line.strip()
            for key in ("stdout", "stderr")
            for line in result.get(key, "").splitlines()
            if line.strip()
        ]
        if result.get("rc") in (0, 1) and lines == [rejection]:
            print(f"{controller}: explicit follower rejection; trying next controller")
            continue
        if result.get("rc") != 0:
            raise RuntimeError("Cancellation outcome unknown; no further attempt")
        for attempt in range(6):
            time.sleep(2)
            command(
                ["scontrol", "--controller", controller, "show", "job", JOB],
                f"{index}-job-after-{attempt}",
            )
            if queue(controller, f"{index}-queue-after-{attempt}", owner) is None:
                print(f"{controller}: allocation {JOB} absent after cancellation")
                return
        raise RuntimeError("Cancellation not confirmed; no further attempt")
    raise RuntimeError("No controller confirmed allocation release")


if __name__ == "__main__":
    main()
