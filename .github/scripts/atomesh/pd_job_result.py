#!/usr/bin/env python3
"""Keep the scheduler outcome and the completed ATOMesh workload outcome."""

import argparse
import json
from pathlib import Path


def write_json(path, payload):
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


def resolve(run_dir, job_id, run_token, num_ranks, state, exit_code, rc, spur):
    completed = workload_completed(run_dir, job_id, run_token, num_ranks)
    # Only the generic Spur failure may be reconciled. Explicit cancellation,
    # signals, timeouts, node failures and OOM retain the scheduler outcome.
    override = spur and completed and (state, exit_code, rc) == ("FAILED", "1:0", 1)
    return {
        "schema_version": 1,
        "job_id": job_id,
        "scheduler": {"state": state, "exit_code": exit_code, "return_code": rc},
        "workload": {"state": "COMPLETED" if completed else "UNVERIFIED"},
        "result": {
            "state": "COMPLETED" if override else state,
            "return_code": 0 if override else rc,
            "source": "workload" if override else "scheduler",
        },
        "scheduler_workload_mismatch": bool(override),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "resolve"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--status", choices=("running", "completed"))
    parser.add_argument("--scheduler-state")
    parser.add_argument("--scheduler-exit-code")
    parser.add_argument("--scheduler-rc", type=int)
    parser.add_argument("--spur", choices=("0", "1"), default="0")
    args = parser.parse_args()
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
    )
    write_json(args.run_dir / "job-result.json", result)
    if result["scheduler_workload_mismatch"]:
        print(
            "WARNING: Spur reported FAILED/1:0, but every worker completed all workload phases. "
            "Using the workload result for CI; "
            "the scheduler failure is retained in job-result.json."
        )
    print(f"workload_state={result['workload']['state']}")
    print(f"result_state={result['result']['state']}")
    print(f"result_exit_code={result['result']['return_code']}")
    return result["result"]["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
