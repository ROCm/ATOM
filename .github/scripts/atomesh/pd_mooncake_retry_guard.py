"""Reconcile the previous Mooncake job before the one authorized retry."""
import json
import os
from pathlib import Path
import subprocess
import time

JOB_ID = "8455"
JOB_NAME = "kimi-k3-mooncake-c48-kimi-k3-mooncake-inferencex-c48-vllm-38677494938-1"
ACCOUNT = "amd-frameworks"
OWNER = "junyan.yang@amd.com"
CONTROLLER = os.environ.get("SPUR_CONTROLLER_ADDR") or (
    "http://pit2-vm-amd-large-04:6817,http://pit2-vm-amd-large-02:6817,"
    "http://pit2-vm-amd-large-03:6817"
)
TERMINAL = {"COMPLETE", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT",
            "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def query_job():
    response = subprocess.run(
        ["squeue", "--controller", CONTROLLER, "--noheader",
         "--format", "%A|%j|%a|%u|%T"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [line.split("|") for line in response.stdout.splitlines()
            if line.split("|", 1)[0].strip() == JOB_ID]
    if not rows:
        return None
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("Cannot identify the previous job unambiguously")
    job_id, name, account, owner, state = [field.strip() for field in rows[0]]
    if (name, account, owner) != (JOB_NAME, ACCOUNT, OWNER):
        raise RuntimeError("Previous job identity mismatch; refusing cancellation")
    return dict(job_id=job_id, name=name, account=account, owner=owner, state=state)


def reconcile(report):
    previous = query_job()
    report["before"] = previous
    if previous is None or previous["state"].split("+")[0] in TERMINAL:
        report["result"] = "already_absent_or_terminal"
        report["after"] = previous
        return
    print(f"[prior-job] cancelling verified Mooncake job {JOB_ID}", flush=True)
    report["cancel_attempted"] = True
    subprocess.run(["scancel", "--controller", CONTROLLER, JOB_ID],
                   check=True, timeout=20)
    deadline = time.monotonic() + 90
    while True:
        current = query_job()
        report["after"] = current
        if current is None or current["state"].split("+")[0] in TERMINAL:
            report["result"] = "cancelled_and_absent_or_terminal"
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Previous job remains active; refusing a second allocation")
        time.sleep(5)


def main():
    report = {"prior_run_id": 38677494938, "job_id": JOB_ID,
              "cancel_attempted": False, "result": "unverified"}
    path = Path("atomesh-results/prior-slurm-8455.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reconcile(report)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("[prior-job] " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
