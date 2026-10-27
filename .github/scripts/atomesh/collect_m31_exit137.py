#!/usr/bin/env python3
"""Read existing run logs and host exit evidence; never allocate GPUs or stop jobs."""
import json
import os
import re
import socket
import shlex
import subprocess
import sys
from pathlib import Path

RUN_ID = "40486874680"
ROOT = Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
manifest = {"run_id": RUN_ID, "collector_host": socket.gethostname(), "files": [], "errors": []}
containers_by_node = {}
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
                    for rank, node in enumerate(meta["nodes"].split(",")):
                        name = f"atomesh-{meta['cell_id']}-{meta['slurm_job_id']}-{rank}"
                        containers_by_node.setdefault(node, set()).add(name)
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

# Query exact containers discovered in this run's metadata. No host kernel
# journal, broad container listing, environment, or other-job events are read.
for node, containers in sorted(containers_by_node.items()):
    if not re.fullmatch(r"pit2-p[0-9]+-g[0-9]+", node):
        continue
    commands = []
    for name in sorted(containers):
        if not re.fullmatch(r"atomesh-minimax-m3[.1a-zA-Z0-9_-]+", name):
            continue
        q = shlex.quote(name)
        commands.append("docker inspect --format '{{.Name}} {{json .State}}' " + q)
        commands.append("docker events --since '2026-10-27T10:15:00Z' --until "
                        + '\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"'
                        + " --filter type=container --filter container=" + q
                        + " --format '{{.Time}} {{.Action}} {{.Actor.ID}} {{index .Actor.Attributes \"name\"}} exit={{index .Actor.Attributes \"exitCode\"}} signal={{index .Actor.Attributes \"signal\"}}'")
    collect_command(f"node-{node}-own-containers.txt", ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={OUT / 'known_hosts'}", node, "\n".join(commands)])
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"files": len(manifest["files"]), "jobs": sorted(jobs), "nodes": sorted(containers_by_node), "errors": manifest["errors"]}))
