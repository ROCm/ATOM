#!/usr/bin/env python3
"""Copy bounded logs of one native M3 vLLM run from shared NFS; no GPU job."""
import json
import os
import shutil
import subprocess
from pathlib import Path

run_id = os.environ['ATOMESH_ENV_ATOMESH_INSPECT_RUN_ID']
if run_id != 'latest' and not run_id.isdecimal():
    raise ValueError('Inspection requires a numeric Actions run ID')
root = Path(os.environ['SLURM_LOG_ROOT'])
out = Path(os.environ['RESULT_DIR']) / 'inspection'
out.mkdir(parents=True, exist_ok=True)
runs = sorted(root.glob('*m31-vllm-1p1d-tp4-mcstore-*-mooncake_store-*' if run_id == 'latest' else f'*m31-vllm-*-{run_id}-*'))
if run_id == 'latest' and runs:
    runs = [max(runs, key=lambda p: p.stat().st_mtime)]
if not runs:
    raise RuntimeError(f'No native M3 vLLM run {run_id} in {root}')
index = []
for run in runs:
    for p in sorted(run.rglob('*')):
        if not p.is_file() or p.suffix not in {'.log', '.out', '.err', '.json', '.prom', '.rc'}:
            continue
        # Never copy environment/credential files or unbounded binary artifacts.
        if 'env' in p.name or 'token' in p.name or 'secret' in p.name:
            continue
        dest = out / run.name / p.relative_to(run)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with p.open('rb') as source:
            source.seek(max(0, p.stat().st_size - 131072))
            dest.write_bytes(source.read(131072))
        index.append({'path': str(p), 'size': p.stat().st_size, 'copied': str(dest)})
    ids = [p.name.removeprefix('slurm_job-') for p in run.glob('slurm_job-*')
           if p.name.removeprefix('slurm_job-').isdecimal()]
    if ids and shutil.which('squeue'):
        result = subprocess.run(['squeue', '-j', ','.join(ids), '-o', '%i %j %T %R'],
                                capture_output=True, text=True, timeout=20)
        (out / 'squeue.txt').write_text(result.stdout + result.stderr)
(out / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
print(f'READ-ONLY INSPECTION: Actions {run_id}, {len(runs)} directories, {len(index)} files; no sbatch')
for row in index:
    if row['path'].endswith(('prefill.log', 'decode.log', 'container.log', '.err')):
        print(row['path'])
        print(Path(row['copied']).read_text(errors='replace')[-6000:])
