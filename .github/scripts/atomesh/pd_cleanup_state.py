"""Read only this run's peer failure and cleanup acknowledgements."""

import argparse
import json
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def matches(value, job_id, token):
    return (
        isinstance(value, dict)
        and value.get("job_id") == job_id
        and value.get("run_token") == token
    )


def condition(action, root, job_id, token, ranks):
    if action == "failed":
        value = load(root / "workload-failure.json")
        return (
            matches(value, job_id, token)
            and type(value.get("return_code")) is int
            and 0 < value["return_code"] <= 255
            and (
                value.get("num_ranks") == ranks
                or value.get("source") == "spur_dispatch"
            )
        )
    prefix = "gpu-preflight" if action == "ready" else "cleanup-complete"
    for rank in range(ranks):
        value = load(root / f"{prefix}-{rank}.json")
        if (
            not matches(value, job_id, token)
            or type(value.get("rank")) is not int
            or value.get("rank") != rank
        ):
            return False
        if action == "ready" and value.get("passed") is not True:
            return False
        if action == "cleaned" and value.get("finished") is not True:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("failed", "ready", "cleaned"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--num-ranks", type=int, required=True)
    args = parser.parse_args()
    if args.num_ranks < 1 or not args.run_token:
        parser.error("A positive rank count and run token are required")
    try:
        return (
            0
            if condition(
                args.action, args.run_dir, args.job_id, args.run_token, args.num_ranks
            )
            else 1
        )
    except (OSError, ValueError, TypeError) as error:
        print(f"Cannot validate peer cleanup state: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
