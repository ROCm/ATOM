#!/usr/bin/env python3
"""Validate whether a downloaded run can serve as a dashboard baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_dashboard_publish_allowed(payload: dict) -> bool:
    publish_flag = payload.get("dashboard_publish_allowed")
    if publish_flag is None:
        return True
    if isinstance(publish_flag, bool):
        return publish_flag
    return str(publish_flag).strip().lower() not in {"0", "false", "no"}


def validate_result_dir(result_dir: Path) -> bool:
    has_valid_result = False
    for path in result_dir.rglob("*.json"):
        if path.name == "regression_report.json" or path.name.endswith(
            "_benchmark_summary.json"
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if "output_throughput" not in payload:
            continue
        has_valid_result = True
        if not is_dashboard_publish_allowed(payload):
            return False
    return has_valid_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate dashboard-eligible benchmark artifacts"
    )
    parser.add_argument("result_dir", help="Directory containing downloaded artifacts")
    args = parser.parse_args()

    return 0 if validate_result_dir(Path(args.result_dir)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
