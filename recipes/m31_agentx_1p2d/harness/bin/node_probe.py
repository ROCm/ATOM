#!/usr/bin/env python3
"""node_probe.py [--image IMG] [--drop DIR ...]: one node's state as JSON (runs ON the node, host python3).

Reports, per NUMA node: MemTotal/MemFree/FilePages (GiB) and the free memory in
>= 2 MiB buddy blocks (what THP owners can fault without compaction); the RDMA
devices with NUMA node and port state; GPU VRAM in use (amd-smi); running
containers (name, image, owner job if atomesh-*); whether the image is present.
--drop DIR evicts the clean page cache of DIR's files first (posix_fadvise, no
root needed), as the launcher does before the Store owners pin memory.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

GIB = 1 << 30


def meminfo(node):
    info = {}
    for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
        f = line.split()
        if len(f) >= 4 and f[2].endswith(":") and f[3].isdigit():
            info[f[2][:-1]] = int(f[3]) * (1024 if f[-1] == "kB" else 1)
    return info


def buddy_big(node):
    """Bytes free in blocks of order >= 9 (2 MiB with 4 KiB pages) on a NUMA node."""
    total = 0
    for line in Path("/proc/buddyinfo").read_text().splitlines():
        f = line.split()
        # Node 0, zone   Normal  c0 c1 ... c10
        if len(f) < 5 or f[0] != "Node" or f[1].rstrip(",") != str(node):
            continue
        counts = [int(x) for x in f[4:]]
        for order, count in enumerate(counts):
            if order >= 9:
                total += count * (4096 << order)
    return total


def drop_cache(paths):
    dropped = 0
    for root in paths:
        for dirpath, _, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                try:
                    fd = os.open(p, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    dropped += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return dropped


def rdma_devices():
    out = {}
    root = Path("/sys/class/infiniband")
    if not root.exists():
        return out
    for dev in sorted(root.iterdir()):
        try:
            numa = int((dev / "device/numa_node").read_text())
        except (OSError, ValueError):
            numa = -1
        states = []
        for port in sorted((dev / "ports").glob("*")):
            try:
                states.append((port / "state").read_text().strip())
            except OSError:
                pass
        out[dev.name] = {"numa": numa, "ports": states}
    return out


def gpus():
    try:
        text = subprocess.run(["amd-smi", "monitor", "-v"], capture_output=True, text=True, timeout=60, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in text.splitlines()[1:]:
        f = line.split()
        if f and f[0].isdigit():
            used = f[2] if len(f) > 2 else ""
            m = re.match(r"([0-9.]+)", used)
            rows.append({"gpu": int(f[0]), "vram_used_mb": float(m.group(1)) if m else None, "raw": line.strip()})
    return rows


def containers():
    try:
        text = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    rows = []
    for line in text.splitlines():
        name, image, status = (line.split("\t") + ["", "", ""])[:3]
        rows.append({"name": name, "image": image, "status": status})
    return rows


def listening_ports():
    try:
        text = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    ports = set()
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 4 and f[3].rsplit(":", 1)[-1].isdigit():
            ports.add(int(f[3].rsplit(":", 1)[-1]))
    return sorted(ports)


def image_present(image):
    if not image:
        return None
    rc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=120, check=False).returncode
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()
    dropped = drop_cache(args.drop) if args.drop else 0
    nodes = sorted(int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*"))
    numa = {}
    for n in nodes:
        mi = meminfo(n)
        numa[str(n)] = {
            "total_gib": round(mi.get("MemTotal", 0) / GIB, 1),
            "free_gib": round(mi.get("MemFree", 0) / GIB, 1),
            "file_gib": round(mi.get("FilePages", 0) / GIB, 1),
            "shmem_gib": round(mi.get("Shmem", 0) / GIB, 1),
            "anon_huge_gib": round(mi.get("AnonHugePages", 0) / GIB, 1),
            "big_free_gib": round(buddy_big(n) / GIB, 1),
        }
    print(json.dumps({
        "host": os.uname().nodename,
        "page_cache_dropped_files": dropped,
        "numa": numa,
        "rdma": rdma_devices(),
        "gpus": gpus(),
        "containers": containers(),
        "image_present": image_present(args.image),
        "listening": listening_ports(),
    }))


if __name__ == "__main__":
    main()
