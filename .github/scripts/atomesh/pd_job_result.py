#!/usr/bin/env python3
"""Keep the scheduler outcome and the completed ATOMesh workload outcome."""

import argparse
import json
from pathlib import Path


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def publish(run_dir, job_id, run_token, rank, num_ranks, status):
    write_json(
        run_dir / f"rank-workload-{rank}.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "run_token": run_token,
            "rank": rank,
            "num_ranks": num_ranks,
            "status": status,
        },
    )


def workload_completed(run_dir, job_id, run_token, num_ranks):
    if not run_token or num_ranks < 1:
        return False
    for rank in range(num_ranks):
        expected = {
            "schema_version": 1,
            "job_id": job_id,
            "run_token": run_token,
            "rank": rank,
            "num_ranks": num_ranks,
            "status": "completed",
        }
        try:
            actual = json.loads((run_dir / f"rank-workload-{rank}.json").read_text())
            rc = (run_dir / f"rank-rc-{rank}").read_text().strip()
        except (OSError, ValueError):
            return False
        if actual != expected or rc != "0":
            return False
    return True


def failed_rank(run_dir, job_id, run_token, num_ranks):
    if not run_token or num_ranks < 1:
        return None
    for rank in range(num_ranks):
        try:
            record = json.loads((run_dir / f"rank-workload-{rank}.json").read_text())
            rc = int((run_dir / f"rank-rc-{rank}").read_text().strip())
        except (OSError, ValueError):
            continue
        expected = {
            "schema_version": 1,
            "job_id": job_id,
            "run_token": run_token,
            "rank": rank,
            "num_ranks": num_ranks,
        }
        if (
            isinstance(record, dict)
            and all(record.get(key) == value for key, value in expected.items())
            and record.get("status") in ("running", "completed")
            and 0 < rc <= 255
        ):
            return {**record, "return_code": rc}
    return None


def resolve(
    run_dir,
    job_id,
    run_token,
    num_ranks,
    state,
    exit_code,
    rc,
    spur,
    workload_check_rc=0,
):
    completed = workload_completed(run_dir, job_id, run_token, num_ranks)
    failure = failed_rank(run_dir, job_id, run_token, num_ranks)
    failure_rc = failure["return_code"] if failure else workload_check_rc
    # Only the generic Spur failure may be reconciled. Explicit cancellation,
    # signals, timeouts, node failures and OOM retain the scheduler outcome.
    override = (
        spur
        and completed
        and not failure_rc
        and (state, exit_code, rc) == ("FAILED", "1:0", 1)
    )
    reject_success = bool(failure_rc and rc == 0)
    effective = {"state": state, "return_code": rc, "source": "scheduler"}
    if reject_success:
        effective = {
            "state": "FAILED",
            "return_code": failure_rc,
            "source": "workload" if failure else "workload_check",
        }
    elif override:
        effective = {"state": "COMPLETED", "return_code": 0, "source": "workload"}
    workload = {"state": "COMPLETED" if completed else "UNVERIFIED"}
    if failure:
        workload = {"state": "FAILED", "return_code": failure_rc}
    elif workload_check_rc:
        workload = {"state": "CHECK_FAILED"}
    if workload_check_rc:
        workload["check_return_code"] = workload_check_rc
    return {
        "schema_version": 1,
        "job_id": job_id,
        "scheduler": {"state": state, "exit_code": exit_code, "return_code": rc},
        "workload": workload,
        "result": effective,
        "scheduler_workload_mismatch": bool(override or reject_success),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "resolve", "check-failed"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--status", choices=("running", "completed"))
    parser.add_argument("--scheduler-state")
    parser.add_argument("--scheduler-exit-code")
    parser.add_argument("--scheduler-rc", type=int)
    parser.add_argument("--workload-check-rc", type=int, default=0)
    parser.add_argument("--scheduler-error-path", type=Path)
    parser.add_argument("--spur", choices=("0", "1"), default="0")
    args = parser.parse_args()
    if not 0 <= args.workload_check_rc <= 255:
        parser.error("workload-check-rc must be between 0 and 255")
    if args.action == "check-failed":
        failure = failed_rank(args.run_dir, args.job_id, args.run_token, args.num_ranks)
        if failure is None and args.scheduler_error_path:
            try:
                errors = args.scheduler_error_path.read_text().splitlines()
            except OSError:
                errors = []
            if "Error: RunStep dispatch failed" in errors and not workload_completed(
                args.run_dir, args.job_id, args.run_token, args.num_ranks
            ):
                failure = {
                    "job_id": args.job_id,
                    "run_token": args.run_token,
                    "source": "spur_dispatch",
                    "error": "RunStep dispatch failed",
                    "return_code": 1,
                }
        if failure is None:
            return 0
        write_json(args.run_dir / "workload-failure.json", failure)
        detail = failure.get("error") or f"rank {failure['rank']} exited"
        print(
            f"ERROR: Slurm job {args.job_id} {detail} "
            f"rc={failure['return_code']}; cancelling the remaining workload."
        )
        return 1
    if args.action == "publish":
        if args.rank is None or not 0 <= args.rank < args.num_ranks or not args.status:
            parser.error("publish requires a valid --rank and --status")
        publish(
            args.run_dir,
            args.job_id,
            args.run_token,
            args.rank,
            args.num_ranks,
            args.status,
        )
        return 0
    if (
        not args.scheduler_state
        or not args.scheduler_exit_code
        or args.scheduler_rc is None
    ):
        parser.error("resolve requires the scheduler state, exit code and return code")
    result = resolve(
        args.run_dir,
        args.job_id,
        args.run_token,
        args.num_ranks,
        args.scheduler_state,
        args.scheduler_exit_code,
        args.scheduler_rc,
        args.spur == "1",
        args.workload_check_rc,
    )
    write_json(args.run_dir / "job-result.json", result)
    if result["scheduler_workload_mismatch"] and result["result"]["return_code"] == 0:
        print(
            "WARNING: Spur reported FAILED/1:0, but every worker completed all workload phases. "
            "Using the workload result for CI; "
            "the scheduler failure is retained in job-result.json."
        )
    elif result["scheduler_workload_mismatch"]:
        print(
            "WARNING: Scheduler reported success, but the workload or its failure check failed. "
            "Using the nonzero result for CI; "
            "the scheduler result is retained in job-result.json."
        )
    print(f"workload_state={result['workload']['state']}")
    print(f"result_state={result['result']['state']}")
    print(f"result_exit_code={result['result']['return_code']}")
    return result["result"]["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
