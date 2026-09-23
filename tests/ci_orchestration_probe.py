"""Disposable CPU payloads for live testing PR #2374; never merge this harness."""

import json
import os
import sys
import time
from pathlib import Path


def main():
    config = json.loads(Path("tests/ci_orchestration_probe.json").read_text())
    kind = sys.argv[1]
    case = os.environ.get("PROBE_CASE", "")
    print(
        json.dumps(
            {
                "kind": kind,
                "case": case,
                "revision": config["revision"],
                "event": os.environ.get("GITHUB_EVENT_NAME"),
                "run": os.environ.get("GITHUB_RUN_ID"),
                "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            }
        ),
        flush=True,
    )
    if kind == "models":
        models = [
            {"runner": "ubuntu-latest", "model_name": name}
            for name in ("success-probe", "failure-probe")
        ]
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"models_json={json.dumps(models)}\n")
        return 0
    if kind == "smoke":
        time.sleep(config["smoke_delay_seconds"])
        return config["smoke_exit_code"]
    if kind == "accuracy":
        time.sleep(config["matrix_delay_seconds"])
        return int(case == config["matrix_fail_case"])
    if kind in {"wheel", "dashboard"}:
        return 0
    raise ValueError(f"Unknown probe: {kind}")


if __name__ == "__main__":
    sys.exit(main())
