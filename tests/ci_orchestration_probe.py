#!/usr/bin/env python3
"""Small CPU-only payload for the disposable CI orchestration test PR."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--seconds", type=int, default=0)
    parser.add_argument("--fail", default="0")
    args = parser.parse_args()

    payload = {
        "workflow": args.workflow,
        "case": args.case,
        "event_name": os.environ.get("GITHUB_EVENT_NAME", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
        "head_ref": os.environ.get("GITHUB_HEAD_REF", ""),
        "seconds": args.seconds,
        "will_fail": _truthy(args.fail),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    started = time.monotonic()
    while time.monotonic() - started < args.seconds:
        elapsed = int(time.monotonic() - started)
        remaining = max(args.seconds - elapsed, 0)
        print(
            f"probe {args.case}: {elapsed}s elapsed, {remaining}s remaining",
            flush=True,
        )
        time.sleep(min(15, remaining or 1))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write(f"## CI orchestration probe: {args.case}\n\n")
            summary.write("| Field | Value |\n| --- | --- |\n")
            summary.writelines(
                f"| {key} | `{value}` |\n" for key, value in payload.items()
            )

    if _truthy(args.fail):
        print(f"probe {args.case}: configured failure", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
