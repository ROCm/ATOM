"""Exercise the exact child launcher and real benchmark CLI before loading."""

import hashlib
import json
import os
import sys


def main():
    import run_owner_serving as serving

    before = dict(os.environ)
    candidate = before["DSV41_OWNER_SERVING"] == "1"
    assert before["DSV41_REBASE_SERVICE_SERVER"] == "0"
    assert before["DSV41_STEP12_OWNER_EP4"] == str(int(candidate))
    assert getattr(serving.client_environment, "_step12_ep_client", False)
    child_env = serving.client_environment()
    assert child_env["DSV41_STEP12_OWNER_EP4"] == "0"
    assert child_env["DSV41_OWNER_SERVING"] == "0"
    assert child_env["DSV41_REBASE_SERVICE_SERVER"] == "0"
    assert dict(os.environ) == before
    label = "client_entry_" + os.environ["DSV41_STEP12_PREFLIGHT_LABEL"]
    argv = [sys.executable, "-m", "atom.benchmarks.benchmark_serving", "--help"]
    serving.child(argv, label, timeout=120)
    log = serving.OUTPUT / (label + ".log")
    text = log.read_text()
    assert "--num-prompts" in text and "--max-concurrency" in text
    assert "Traceback" not in text and "AssertionError" not in text
    result = {
        "status": "actual_client_environment_and_benchmark_module_startup_PASS",
        "candidate": candidate,
        "command": argv,
        "parent_EP": before["DSV41_STEP12_OWNER_EP4"],
        "child_EP": child_env["DSV41_STEP12_OWNER_EP4"],
        "child_owner": child_env["DSV41_OWNER_SERVING"],
        "child_server": child_env["DSV41_REBASE_SERVICE_SERVER"],
        "parent_unchanged": dict(os.environ) == before,
        "log": str(log),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "scope": "Real benchmark --help through production child/env/Python entry; no weights or HTTP requests.",
    }
    (serving.OUTPUT / (label + ".json")).open("x").write(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
