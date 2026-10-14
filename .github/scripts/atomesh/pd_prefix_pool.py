"""Read-only pre-sbatch eligible POOL, never a selected P/D pair.

Adapted from pd_select_storage_nodes.py filesystem parsing. Node-exporter
mount coverage is not proof of directory existence, Docker root configuration,
or job cgroup headroom: allocated-host checks remain authoritative.
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

REQUIRED_BYTES = 200 * 1024**3  # Unknown image cache always means uncached.
REQUIRED_MEMORY = 1799000000000 + 64 * 1024**3
REQUIRED_INODES = 100000
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')
SAMPLE = re.compile(r"(\w+)(?:\{(.*)\})? ([^ ]+)(?: .*)?$")


def resource_report(metrics, now, paths):
    mounts = {}
    timestamp = available = None
    collector_ok = False
    for line in metrics.splitlines():
        match = SAMPLE.fullmatch(line)
        if not match:
            continue
        name, raw_labels, raw_value = match.groups()
        if not (
            name.startswith("node_filesystem_")
            or name
            in (
                "node_time_seconds",
                "node_scrape_collector_success",
                "node_memory_MemAvailable_bytes",
            )
        ):
            continue
        labels = {
            key: json.loads('"' + value + '"')
            for key, value in LABEL.findall(raw_labels or "")
        }
        value = float(raw_value)
        if name == "node_time_seconds":
            timestamp = value
        elif name == "node_memory_MemAvailable_bytes":
            available = value
        elif name == "node_scrape_collector_success":
            if labels.get("collector") == "filesystem":
                collector_ok = value == 1
        elif labels.get("mountpoint", "").startswith("/"):
            key = (labels["mountpoint"], labels.get("device"), labels.get("fstype"))
            fields = mounts.setdefault(key, {})
            field = name.removeprefix("node_filesystem_")
            if field in fields:
                raise ValueError("Duplicate filesystem metric")
            fields[field] = value
    if timestamp is None or not math.isfinite(timestamp) or abs(now - timestamp) > 120:
        raise ValueError("Missing or stale node exporter timestamp")
    if not collector_ok:
        raise ValueError("Filesystem collector did not succeed")
    if available is None or not math.isfinite(available) or available < REQUIRED_MEMORY:
        raise ValueError(
            "Insufficient or unknown host MemAvailable for either P/D role"
        )
    checks = []
    for path in paths:
        if not Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Storage paths must be absolute without traversal")
        ancestors = [
            key
            for key in mounts
            if path == key[0] or path.startswith(key[0].rstrip("/") + "/")
        ]
        if not ancestors:
            raise ValueError(f"No filesystem coverage for {path}")
        depth = max(len(key[0]) for key in ancestors)
        deepest = [key for key in ancestors if len(key[0]) == depth]
        if len(deepest) != 1:
            raise ValueError(f"Ambiguous filesystem coverage for {path}")
        key = deepest[0]
        fields = mounts[key]
        required = ("avail_bytes", "files_free", "files", "readonly", "device_error")
        if any(
            field not in fields or not math.isfinite(fields[field])
            for field in required
        ):
            raise ValueError(f"Incomplete filesystem evidence for {path}")
        if not (
            fields["avail_bytes"] >= REQUIRED_BYTES
            and fields["files"] > 0
            and REQUIRED_INODES <= fields["files_free"] <= fields["files"]
            and fields["readonly"] == fields["device_error"] == 0
            and key[2] not in (None, "tmpfs", "devtmpfs", "overlay", "squashfs")
        ):
            raise ValueError(f"Insufficient/invalid filesystem for {path}: {fields}")
        checks.append(
            {
                "path": path,
                "mountpoint": key[0],
                "device": key[1],
                "fstype": key[2],
                **fields,
            }
        )
    return {
        "filesystems": checks,
        "host_available_bytes": int(available),
        "metrics_timestamp": timestamp,
    }


def fetch_metrics(address):
    if not re.fullmatch(r"[A-Za-z0-9.-]+", address):
        raise ValueError("Invalid node address")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://{address}:9100/metrics", timeout=3) as response:
        data = response.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Node metrics exceeded size limit")
    return data.decode()


def parse_nodes(output):
    records = {}
    for section in re.split(r"(?=\bNodeName=)", output):
        record = dict(
            re.findall(
                r"(?:^|\s)(NodeName|NodeAddr|State|Partitions|Gres)=([^\s]+)", section
            )
        )
        if "NodeName" in record:
            name = record["NodeName"]
            if name in records or not re.fullmatch(r"[A-Za-z0-9.-]+", name):
                raise ValueError("Duplicate/invalid node inventory")
            records[name] = record
    if not records:
        raise ValueError("Empty node inventory")
    return records


def qualify(records, partition, paths, fetch=fetch_metrics, now=time.time):
    reports, eligible = [], []
    if not partition:
        raise ValueError("Unknown target partition")
    for node, record in records.items():
        row = {"node": node, "scheduler": record, "passed": False}
        reports.append(row)
        try:
            if partition not in record.get("Partitions", "").split(","):
                raise ValueError("Not in target partition")
            if not re.fullmatch(
                r"(?:IDLE|ALLOCATED|MIXED|COMPLETING)(?:\+CLOUD)?",
                record.get("State", ""),
            ):
                raise ValueError("Node is not schedulable")
            gpu_entries = re.findall(
                r"(?:^|,)gpu:mi355x:([0-9]+)(?:\([^)]*\))?(?=,|$)",
                record.get("Gres", "").lower(),
            )
            if sum(map(int, gpu_entries)) != 8:
                raise ValueError("Expected explicit inventory of eight MI355X GPUs")
            row.update(
                resource_report(fetch(record.get("NodeAddr", node)), now(), paths)
            )
            row["passed"] = True
            eligible.append(node)
        except (OSError, ValueError, http.client.HTTPException) as error:
            row["reason"] = f"{type(error).__name__}: {error}"
    return {
        "status": "QUALIFIED_POOL" if len(eligible) >= 2 else "BLOCKED",
        "eligible": eligible,
        "num_nodes": 2,
        "nodes": reports,
        "partition": partition,
    }


def update_cell(cell, report):
    if cell["name"] != "kimi-k3-vllm-c48-prefix-read-fixed" or cell["num_nodes"] != 2:
        raise ValueError("Pool selector is restricted to the one 2-node case")
    if report["status"] != "QUALIFIED_POOL" or len(set(report["eligible"])) < 2:
        raise ValueError("No qualified pool; no sbatch permitted")
    return dict(cell, nodes=report["eligible"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", default="")
    parser.add_argument("--controller", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--docker-root", default="/var/lib/docker")
    parser.add_argument("--containerd-root", default="/var/lib/containerd")
    parser.add_argument("--tmp-root", default="/tmp")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell-output", required=True, type=Path)
    args = parser.parse_args()
    command = ["scontrol"] + (
        ["--controller", args.controller] if args.controller else []
    )
    report = {
        "status": "BLOCKED",
        "checked_at": time.time(),
        "eligible": [],
        "required_free_bytes": REQUIRED_BYTES,
        "required_free_inodes": REQUIRED_INODES,
        "required_host_available_bytes": REQUIRED_MEMORY,
        "image_cache": "unknown_assume_uncached",
        "authority": "mount coverage only; directory existence, actual Docker root and cgroup limits require allocated-host validation",
    }
    try:

        def query(*words):
            return subprocess.check_output(command + list(words), text=True, timeout=20)

        partition = args.partition
        if not partition:
            partitions = []
            for section in re.split(
                r"(?=\bPartitionName=)", query("show", "partition")
            ):
                fields = dict(
                    re.findall(r"(?:^|\s)(PartitionName|Default)=([^\s]+)", section)
                )
                if fields.get("Default") == "YES":
                    partitions.append(fields["PartitionName"])
            if len(partitions) != 1:
                raise ValueError("Cannot identify one default partition")
            partition = partitions[0]
        raw = query("show", "nodes")
        report["node_inventory_raw"] = raw
        records = parse_nodes(raw)
        cell = json.loads(os.environ["CELL_JSON"])
        if cell["nodes"]:
            unknown = set(cell["nodes"]) - records.keys()
            if unknown:
                raise ValueError(f"Unknown requested candidates: {sorted(unknown)}")
            records = {node: records[node] for node in cell["nodes"]}
        paths = [args.docker_root, args.containerd_root, args.tmp_root, args.run_dir]
        report["storage_paths"] = paths
        report.update(qualify(records, partition, paths))
        # A long inventory scan must not hand stale early samples to sbatch.
        finished = time.time()
        for row in report["nodes"]:
            if row["passed"] and abs(finished - row["metrics_timestamp"]) > 120:
                row.update(passed=False, reason="Metrics expired during pool scan")
        report["eligible"] = [row["node"] for row in report["nodes"] if row["passed"]]
        report["status"] = (
            "QUALIFIED_POOL" if len(report["eligible"]) >= 2 else "BLOCKED"
        )
        report["finished_at"] = finished
        updated = update_cell(cell, report)
        args.cell_output.write_text(json.dumps(updated, indent=2) + "\n")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        report["status"] = "BLOCKED"
        report["error"] = f"{type(error).__name__}: {error}"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "QUALIFIED_POOL":
        raise SystemExit("No storage-qualified pool; no GPU job submitted")


if __name__ == "__main__":
    main()
