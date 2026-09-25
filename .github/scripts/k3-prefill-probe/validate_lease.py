"""Accept only the intended missing-ACK failure after a successful delayed ACK."""

import argparse
import datetime
import json
import math
import re
from pathlib import Path


def log_time(line, year):
    timestamp = re.search(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    assert timestamp, line
    return (
        datetime.datetime.strptime(f"{year}-{timestamp.group(1)}", "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
    )


def kv_usage(path):
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in path.read_text().splitlines()
        if line.startswith("vllm:kv_cache_usage_perc{")
    ]
    assert values and all(math.isfinite(v) and 0 <= v <= 1 for v in values), path
    return sum(values)


def validate(root):
    jobs = sorted((root / "logs").glob("slurm_job-*"))
    assert len(jobs) == 1, f"Expected one collected job, got {jobs}"
    job = jobs[0]
    probe = job / "prefill-probe"
    delayed = json.loads((probe / "delayed-ack-complete.json").read_text())
    armed = json.loads((probe / "expiry-armed.json").read_text())
    exports = [
        json.loads((probe / name).read_text())
        for name in ("delayed-export.json", "expiry-export.json")
    ]
    json.dumps([delayed, armed, exports], allow_nan=False)
    assert exports[0]["transfer_id"] != exports[1]["transfer_id"]
    for value in exports:
        assert value["input_tokens"] == 65537
        assert value["response"]["id"] == f"cmpl-{value['request_id']}"
        assert value["response"]["usage"]["completion_tokens"] == 1
        assert (
            value["response"]["kv_transfer_params"]["transfer_id"]
            == value["transfer_id"]
        )
    assert exports[0]["transfer_id"] == delayed["transfer_id"]
    assert exports[1]["transfer_id"] == armed["transfer_id"]
    assert delayed["held_seconds"] >= 70
    assert delayed["kv_usage_before_ack"] > 0
    assert delayed["kv_usage_after_ack"] == 0
    assert kv_usage(probe / "held-before-ack.metrics") == delayed["kv_usage_before_ack"]
    assert kv_usage(probe / "released-after-ack.metrics") == 0
    assert delayed["mock_ack_ranks"] == 8
    assert armed["lease_seconds"] == 180 and armed["ack_sent"] is False
    lines = (job / "rank-0/container.log").read_text(errors="replace").splitlines()
    workload = (probe / "workload.log").read_text(errors="replace")
    assert "Missing-ACK lease did not terminate" not in workload
    assert "Traceback (most recent call last)" not in workload
    holds = []
    for value in exports:
        matches = [
            line
            for line in lines
            if "PD_LEASE hold " in line
            and f"rid={value['response']['id']}-0 " in line
            and f"transfer_id={value['transfer_id']} " in line
        ]
        assert len(matches) == 1 and "timeout=180.0" in matches[0], matches
        holds.append(matches[0])
    ack = [
        line
        for line in lines
        if "PD_LEASE release_ack" in line
        and f"rid={exports[0]['response']['id']}-0 " in line
        and f"transfer_id={delayed['transfer_id']} " in line
    ]
    assert len(ack) == 1, f"Missing or repeated producer ACK: {ack}"
    held = float(re.search(r"held_s=([0-9.]+)", ack[0]).group(1))
    assert 70 <= held < 180, held
    assert not any(
        "PD_LEASE release_ack" in line
        and f"transfer_id={armed['transfer_id']} " in line
        for line in lines
    )
    fatal = [
        line
        for line in lines
        if "RuntimeError: MoRI-IO READ source lease expired before release ACK" in line
        and f"{[exports[1]['response']['id'] + '-0']}" in line
    ]
    assert fatal, "The armed request did not fail with the expected source lease error"
    year = datetime.datetime.fromtimestamp(
        armed["armed_at"], datetime.timezone.utc
    ).year
    fatal_time = log_time(fatal[0], year)
    hold_elapsed = fatal_time - log_time(holds[1], year)
    assert 179 <= hold_elapsed <= 240, hold_elapsed
    elapsed = fatal_time - armed["armed_at"]
    assert 175 <= elapsed <= 240, elapsed
    supervisor = [
        line
        for line in lines
        if re.search(
            r"\[workload\]\[FAIL\] serving process \d+ exited during \S+ rc=[1-9]\d*",
            line,
        )
    ]
    assert len(supervisor) == 1, supervisor
    assert lines.index(supervisor[0]) > lines.index(fatal[0])
    assert int((root / "driver.rc").read_text()) != 0
    result = json.loads((job / "job-result.json").read_text())
    assert result["job_id"] == job.name.removeprefix("slurm_job-")
    assert result["scheduler"]["state"] in ("FAILED", "CANCELLED")
    assert type(result["result"]["return_code"]) is int
    assert result["result"]["return_code"] != 0
    assert int((job / "rank-rc-0").read_text()) != 0
    assert not list(job.glob("phase-*.complete")), "Negative phase was marked complete"
    assert (job / "cleanup-query-0.rc").read_text().strip() == "0"
    assert not (job / "cleanup-containers-0.txt").read_text().strip()
    return {
        "expected_fault_pass": True,
        "job": job.name,
        "delayed_ack_held_seconds": held,
        "fatal_after_armed_seconds": elapsed,
        "fatal_after_hold_seconds": hold_elapsed,
        "fatal_log": fatal[0],
        "workload_return_code": result["result"]["return_code"],
        "cleanup_query_rc": 0,
        "remaining_containers": [],
        "scope": "P-only mock ACK lifecycle and expected fatal expiry; no RDMA READ or accuracy test",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = validate(args.result_dir)
    (args.result_dir / "lease-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
