"""Choose one fixed P/D pair before GPU submission using read-only host metrics.

This runs inside the performance workflow. It creates no Slurm job, pulls no
image and removes no data. The allocated-host preflight remains authoritative.
"""

import argparse
import http.client
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

REQUIRED_BYTES = 200 * 1024**3  # Assume the pinned image is not cached.
REQUIRED_INODES = 100000
STORAGE_PATHS = ("/var/lib/docker", "/var/lib/containerd", "/tmp")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
SAMPLE = re.compile(r"(\w+)(?:\{(.*)\})? ([^ ]+)(?: .*)?$")


def storage_report(metrics, now):
    mounts = {}
    timestamp = None
    collector_ok = False
    for line in metrics.splitlines():
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        if not (
            name.startswith("node_filesystem_")
            or name in ("node_time_seconds", "node_scrape_collector_success")
        ):
            continue
        labels = {
            key: json.loads('"' + value + '"')
            for key, value in LABEL.findall(raw_labels or "")
        }
        value = float(raw_value)
        if name == "node_time_seconds":
            timestamp = value
        elif name == "node_scrape_collector_success":
            if labels.get("collector") == "filesystem":
                collector_ok = value == 1
        elif labels.get("mountpoint", "").startswith("/"):
            key = (labels["mountpoint"], labels.get("device"), labels.get("fstype"))
            mounts.setdefault(key, {})[name.removeprefix("node_filesystem_")] = value

    if timestamp is None or not math.isfinite(timestamp) or abs(now - timestamp) > 120:
        raise ValueError("Missing or stale node exporter timestamp")
    if not collector_ok:
        raise ValueError("Filesystem collector did not succeed")

    checks = []
    for path in STORAGE_PATHS:
        ancestors = [
            key
            for key in mounts
            if path == key[0] or path.startswith(key[0].rstrip("/") + "/")
        ]
        if not ancestors:
            raise ValueError(f"No filesystem evidence for {path}")
        depth = max(len(key[0]) for key in ancestors)
        deepest = [key for key in ancestors if len(key[0]) == depth]
        if len(deepest) != 1:
            raise ValueError(f"Ambiguous filesystem evidence for {path}")
        key = deepest[0]
        fields = mounts[key]
        required = ("avail_bytes", "files_free", "files", "readonly", "device_error")
        if any(
            field not in fields or not math.isfinite(fields[field])
            for field in required
        ):
            raise ValueError(f"Incomplete filesystem evidence for {path}")
        passed = (
            fields["avail_bytes"] >= REQUIRED_BYTES
            and fields["files"] > 0
            and fields["files_free"] >= REQUIRED_INODES
            and fields["readonly"] == fields["device_error"] == 0
            and key[2] not in ("tmpfs", "devtmpfs", "overlay", "squashfs")
        )
        checks.append(
            {
                "path": path,
                "mountpoint": key[0],
                "device": key[1],
                "fstype": key[2],
                "free_bytes": int(fields["avail_bytes"]),
                "free_inodes": int(fields["files_free"]),
                "passed": passed,
            }
        )
    return {"passed": all(row["passed"] for row in checks), "filesystems": checks}


def fetch_metrics(address):
    if not re.fullmatch(r"[A-Za-z0-9.-]+", address):
        raise ValueError("Invalid node address")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{address}:9100/metrics", timeout=3) as response:
        data = response.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Node metrics exceeded size limit")
    return data.decode()


def select_pair(candidates, node_records, fetch=fetch_metrics):
    reports = []
    healthy = []
    for node in candidates:
        record = node_records.get(node, {})
        state = record.get("State", "UNKNOWN")
        row = {"node": node, "state": state, "passed": False}
        reports.append(row)
        if not re.fullmatch(r"(?:IDLE|ALLOCATED|MIXED|COMPLETING)(?:\+CLOUD)?", state):
            row["reason"] = "Node is not schedulable"
            continue
        try:
            row.update(storage_report(fetch(record.get("NodeAddr", node)), time.time()))
        except (OSError, ValueError, http.client.HTTPException) as error:
            row["reason"] = f"{type(error).__name__}: {error}"
        if row["passed"]:
            healthy.append((state != "IDLE", candidates.index(node), node))
    selected = [item[2] for item in sorted(healthy)[:2]]
    return {"selected": selected if len(selected) == 2 else [], "nodes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = args.candidates.split(",")
    if len(set(candidates)) != len(candidates) or any(
        not re.fullmatch(r"pit2-p03-g[0-9]{2}", node) for node in candidates
    ):
        parser.error("Expected unique nodes from the configured TW pool")
    report = {
        "checked_at": time.time(),
        "candidates": candidates,
        "required_free_bytes": REQUIRED_BYTES,
        "required_free_inodes": REQUIRED_INODES,
        "image_assumed_cached": False,
        "selected": [],
    }
    try:
        output = subprocess.run(
            ["scontrol", "show", "nodes", "--oneliner"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout
        records = {}
        for line in output.splitlines():
            record = dict(
                re.findall(r"(?:^|\s)(NodeName|NodeAddr|State)=([^\s]+)", line)
            )
            if "NodeName" in record:
                records[record["NodeName"]] = record
        report.update(select_pair(candidates, records))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if len(report["selected"]) != 2:
        raise SystemExit("No storage-qualified P/D pair; no GPU job submitted")
    if output_path := os.environ.get("GITHUB_ENV"):
        with Path(output_path).open("a") as output:
            nodes = ",".join(report["selected"])
            output.write(f"ATOMESH_1P1D_NODES={nodes}\nATOMESH_NODE_POOL={nodes}\n")


if __name__ == "__main__":
    main()
