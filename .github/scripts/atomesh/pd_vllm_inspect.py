#!/usr/bin/env python3
"""Publish only whitelisted numeric/boolean status for one known experiment.

Never copies logs, environment, scheduler text, arbitrary JSON, or file contents.
No Slurm submission, SSH, container action, or shared-filesystem write occurs.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

run_id = os.environ['ATOMESH_ENV_ATOMESH_INSPECT_RUN_ID']
if not run_id.isdecimal():
    raise ValueError('Status inspection requires a numeric Actions run ID')
root = Path(os.environ['SLURM_LOG_ROOT']).resolve()
out = Path(os.environ['RESULT_DIR']) / 'inspection'
out.mkdir(parents=True, exist_ok=True)
# Exact task namespace and Actions ID; only its timestamp suffix is variable.
prefix = f'minimax-m3.1-flash-vllm-m31-vllm-1p1d-tp4-mcstore-smoke-mooncake_store-{run_id}-'
runs = [p for p in root.glob(prefix + '*')
        if p.parent.resolve() == root and p.name[len(prefix):].isdecimal()]
rows = []
for run in runs:
    for job in run.glob('slurm_job-*'):
        job_id = job.name.removeprefix('slurm_job-')
        if not job_id.isdecimal() or job.resolve().parent != run.resolve():
            continue
        row = {'actions_run_id': int(run_id), 'slurm_job_id': int(job_id),
               'image_preflight_complete': (job / 'image-hardware.json').is_file(),
               'model_snapshot_resolved': (job / 'resolved-experiment.json').is_file()}
        for role in ('prefill', 'decode', 'proxy', 'mooncake-master'):
            log = job / 'logs' / f'{role}.log'
            row[role + '_log_exists'] = log.is_file()
            if log.is_file():
                row[role + '_log_bytes'] = log.stat().st_size
                with log.open('rb') as source:
                    source.seek(max(0, log.stat().st_size - 131072))
                    tail = source.read(131072).decode(errors='replace')
                for name, marker in {'traceback': 'Traceback (most recent call last)',
                    'engine_failure': 'Engine core initialization failed',
                    'ready': 'Application startup complete',
                    'out_of_memory': 'out of memory',
                    'mr_failure': 'ibv_reg_mr failed'}.items():
                    row[f'{role}_{name}'] = marker in tail
        memory_file = job / 'memory-before-download.json'
        if memory_file.is_file():
            memory = json.loads(memory_file.read_text())
            for key in ('memory.max', 'memory.current', 'memory.peak'):
                value = str(memory.get(key, '')).strip()
                if value.isdecimal():
                    row[key] = int(value)
                elif value == 'max':
                    row[key] = 'max'
            for line in str(memory.get('memory.events', '')).splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] in {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill'} and parts[1].isdecimal():
                    row['memory.events.' + parts[0]] = int(parts[1])
        state_file = job / 'rank-0/docker-state.json'
        if state_file.is_file():
            state = json.loads(state_file.read_text())
            for key in ('OOMKilled', 'Running', 'ExitCode'):
                if type(state.get(key)) in (int, bool):
                    row['container_' + key] = state[key]
        container_log = job / 'rank-0/container.log'
        if container_log.is_file():
            with container_log.open('rb') as source:
                source.seek(max(0, container_log.stat().st_size - 131072))
                tail = source.read(131072).decode(errors='replace')
            progress = re.findall(r'Fetching\s+\d+ files:[^\r\n]*?\|\s*(\d+)/(\d+)', tail)
            if progress:
                row['download_files_done'], row['download_files_total'] = map(int, progress[-1])
        rows.append(row)
result = {'kind': 'experiment_status_only', 'actions_run_id': int(run_id), 'jobs': rows}
if shutil.which('squeue'):
    try:
        queue = subprocess.run(['squeue', '--name', prefix[:-1] + '-1', '--noheader',
                                '--format', '%i|%T|%R'], capture_output=True, text=True, timeout=20)
        result['scheduler_query_rc'] = queue.returncode
        result['scheduler'] = []
        for line in queue.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) != 3 or not fields[0].isdecimal():
                continue
            if fields[1] not in {'PENDING', 'RUNNING', 'COMPLETING', 'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'}:
                continue
            status = {'job_id': int(fields[0]), 'state': fields[1]}
            if re.fullmatch(r'pit2-p03-g[0-9]+', fields[2]):
                status['node'] = fields[2]
            elif fields[2].strip('()') in {'Resources', 'Priority', 'ReqNodeNotAvail', 'None'}:
                status['reason'] = fields[2].strip('()')
            result['scheduler'].append(status)
    except subprocess.TimeoutExpired:
        result['scheduler_query_timed_out'] = True

(out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
