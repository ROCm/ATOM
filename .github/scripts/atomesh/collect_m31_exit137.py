#!/usr/bin/env python3
"""Read existing run logs and host exit evidence; never allocate GPUs or stop jobs."""
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

RUN_ID = "40483597163"
ROOT = Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
manifest = {"run_id": RUN_ID, "collector_host": socket.gethostname(), "files": [], "errors": []}
nodes = {"pit2-p03-g09"}
jobs = set()
used = 0
for run in sorted(ROOT.glob(f"minimax-m3.1-flash-*-{RUN_ID}-*")):
    for pattern in ("slurm-*.out", "slurm-*.err", "slurm_job-*/*.json", "slurm_job-*/rank-rc-*", "slurm_job-*/rank-*/*.log"):
        for src in sorted(run.glob(pattern)):
            if not src.is_file() or not src.resolve().is_relative_to(ROOT.resolve()):
                continue
            if used >= 64 << 20:
                break
            try:
                with src.open("rb") as f:
                    size = os.fstat(f.fileno()).st_size
                    f.seek(max(0, size - (4 << 20)))
                    data = f.read(4 << 20)
                dst = OUT / src.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                used += len(data)
                manifest["files"].append({"source": str(src), "bytes": size, "copied": len(data)})
                if src.name == "cell-metadata.json":
                    meta = json.loads(data)
                    jobs.add(str(meta["slurm_job_id"]))
                    nodes.update(meta["nodes"].split(","))
            except (OSError, ValueError, KeyError) as e:
                manifest["errors"].append({"path": str(src), "error": str(e)})

def collect_command(name, command, timeout=45):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        (OUT / name).write_bytes(p.stdout[-(8 << 20):])
        return p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        (OUT / name).write_text(str(e))
        return 1

for job in sorted(jobs):
    if not re.fullmatch(r"[0-9]+", job):
        continue
    script = "source .github/scripts/slurm_submit_helpers.sh; detect_slurm_backend; query_slurm_controller_job " + job + "; query_slurm_accounting_job " + job
    collect_command(f"job-{job}-accounting.txt", ["bash", "-c", script])

# Inspect only state fields and bounded exit events. Do not dump container env.
remote = r"""
hostname
date -u
free -h
journalctl -k --since '2026-10-27 09:49:00 UTC' --no-pager 2>&1 | grep -Ei 'oom|out of memory|killed process|memory cgroup|permission|denied' | tail -100
docker events --since '2026-10-27T09:49:00Z' --until "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --filter type=container --format '{{json .}}' 2>&1 | grep -E 'atomesh-|permission|denied' | tail -200
docker ps -a --filter name=atomesh-minimax-m3.1 --format '{{.ID}} {{.Names}} {{.Status}}'
for cid in $(docker ps -aq --filter name=atomesh-minimax-m3.1); do docker inspect --format '{{.Name}} {{json .State}}' "$cid"; done
"""
for node in sorted(nodes):
    if not re.fullmatch(r"pit2-p[0-9]+-g[0-9]+", node):
        continue
    collect_command(f"node-{node}.txt", ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={OUT / 'known_hosts'}", node, remote])
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"files": len(manifest["files"]), "jobs": sorted(jobs), "nodes": sorted(nodes), "errors": manifest["errors"]}))
