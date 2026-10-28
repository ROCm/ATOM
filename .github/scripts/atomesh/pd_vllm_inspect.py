#!/usr/bin/env python3
"""Inspect one known experiment; optionally recover bounded service log tails.

Recovery reads only named logs and worker failures within this task namespace.
No Slurm submission, SSH, container action, or shared-filesystem write occurs.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

requested_run_id = os.environ['ATOMESH_ENV_ATOMESH_INSPECT_RUN_ID']
if not requested_run_id.isdecimal() and requested_run_id != 'latest':
    raise ValueError('Status inspection requires a numeric Actions run ID or latest')
root = Path(os.environ['SLURM_LOG_ROOT']).resolve()
out = Path(os.environ['RESULT_DIR']) / 'inspection'
out.mkdir(parents=True, exist_ok=True)
# Exact task namespace and Actions ID; only its timestamp suffix is variable.
case = os.environ.get('ATOMESH_ENV_ATOMESH_INSPECT_CASE', 'smoke')
if case not in {'smoke', 'c32', 'validated-sweep', 'triton-probe'}:
    raise ValueError('Unknown native experiment case')
task_prefix = f'minimax-m3.1-flash-vllm-m31-vllm-1p1d-tp4-mcstore-{case}-mooncake_store-'
if case == 'triton-probe':
    task_prefix = 'minimax-m3.1-flash-vllm-m31-vllm-triton-probe-mooncake_store-'
run_id = requested_run_id
if run_id == 'latest':
    # Resolve only this experiment's exact case namespace; never inspect arbitrary jobs.
    candidates = []
    for path in root.glob(task_prefix + '*'):
        match = re.fullmatch(re.escape(task_prefix) + r'(\d+)-(\d+)', path.name)
        if match and path.parent.resolve() == root:
            candidates.append(int(match[1]))
    run_id = str(max(candidates)) if candidates else '0'
prefix = task_prefix + run_id + '-'
runs = [p for p in root.glob(prefix + '*')
        if p.parent.resolve() == root and p.name[len(prefix):].isdecimal()]
rows = []
for run in runs:
    for job in run.glob('slurm_job-*'):
        job_id = job.name.removeprefix('slurm_job-')
        if not job_id.isdecimal() or job.resolve().parent != run.resolve():
            continue
        current = job
        phases = []
        suite_file = job / 'suite-status.json'
        if suite_file.is_file():
            try:
                suite = json.loads(suite_file.read_text())
            except json.JSONDecodeError:
                suite = []
            for phase in suite:
                if (phase.get('phase') in {'smoke', 'c32', 'c48', 'c64'}
                        and phase.get('status') in {'running', 'success', 'failed'}
                        and type(phase.get('concurrency')) is int
                        and phase['concurrency'] in {1, 32, 48, 64}):
                    phases.append({k: phase[k] for k in ('phase', 'status', 'concurrency')})
            if phases:
                current = job / phases[-1]['phase']
        row = {'actions_run_id': int(run_id), 'slurm_job_id': int(job_id),
               'suite_phases': phases,
               'image_preflight_complete': (current / 'image-hardware.json').is_file(),
               'model_snapshot_resolved': (current / 'resolved-experiment.json').is_file()}
        if case == 'triton-probe':
            row['triton_cases'] = []
            summary = job / 'triton-probe/summary.json'
            if summary.is_file():
                try:
                    cases = json.loads(summary.read_text())
                except json.JSONDecodeError:
                    cases = []
                for probe in cases:
                    if probe.get('case') not in {'default-prefill', 'default-decode',
                                                  'fresh-prefill', 'fresh-decode'}:
                        continue
                    if probe.get('status') not in {'running', 'success', 'failed'}:
                        continue
                    entry = {'case': probe['case'], 'status': probe['status']}
                    result = job / 'triton-probe' / probe['case'] / 'results.json'
                    if result.is_file():
                        try:
                            kernels = json.loads(result.read_text())
                        except json.JSONDecodeError:
                            kernels = []
                        entry['completed_kernels'] = sum(k.get('status') == 'success' for k in kernels)
                        entry['failed_kernels'] = sum(k.get('status') == 'failed' for k in kernels)
                    row['triton_cases'].append(entry)
        for role in ('prefill', 'decode', 'proxy', 'mooncake-master'):
            log = current / 'logs' / f'{role}.log'
            row[role + '_log_exists'] = log.is_file()
            if log.is_file():
                row[role + '_log_bytes'] = log.stat().st_size
                row[role + '_log_mtime'] = int(log.stat().st_mtime)
                with log.open('rb') as source:
                    source.seek(max(0, log.stat().st_size - 131072))
                    tail = source.read(131072).decode(errors='replace')
                for name, marker in {'traceback': 'Traceback (most recent call last)',
                    'engine_failure': 'Engine core initialization failed',
                    'ready': ('service_ready=true' if role == 'mooncake-master' else
                              'All prefiller instances are ready.' if role == 'proxy' else
                              'Application startup complete'),
                    'out_of_memory': 'out of memory',
                    'hip209': 'no kernel image is available for execution on the device',
                    'model_loaded': 'Model loading took',
                    'prefetch_finished': 'Prefetching checkpoint files into page cache finished',
                    'rocm_nvfp4_attention': 'MiniMax M3 sparse attention selected ROCm NVFP4',
                    'mr_failure': 'ibv_reg_mr failed'}.items():
                    row[f'{role}_{name}'] = marker in tail
        if os.environ.get('ATOMESH_ENV_ATOMESH_INSPECT_RECOVER_LOGS') == '1':
            recovery = out / 'recovered' / job.name
            sources = [current / 'logs' / f'{role}.log'
                       for role in ('prefill', 'decode', 'proxy', 'mooncake-master')]
            sources += [current / 'runtime-evidence' / f'{role}-rank-{rank}' / 'failure.txt'
                        for role in ('prefill', 'decode') for rank in range(4)]
            for source in sources:
                if not source.is_file() or not source.resolve().is_relative_to(job.resolve()):
                    continue
                target = recovery / source.relative_to(current)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as stream:
                    stream.seek(max(0, source.stat().st_size - 2 * 1024 * 1024))
                    target.write_bytes(stream.read(2 * 1024 * 1024))
        row['runtime_stages'] = {}
        for role in ('prefill', 'decode'):
            row['runtime_stages'][role] = {
                stage: sum((current / 'runtime-evidence' / f'{role}-rank-{rank}' / (stage + '.json')).is_file()
                           for rank in range(4))
                for stage in ('before-kv-init', 'after-kv-init', 'before-warmup', 'after-warmup', 'at-failure')}
        row['store_memory_gate_finished'] = (current / 'store-memory-gate/result.json').is_file()
        row['smoke_results_exist'] = (current / 'smoke.json').is_file()
        memory_file = current / 'memory-before-download.json'
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
        events = job / 'rank-0/container-events.log'
        if events.is_file():
            row['container_events'] = []
            for line in events.read_text().splitlines():
                match = re.fullmatch(r'(\d+) (create|start|kill|die|destroy) signal=(\d*)', line)
                if match:
                    event = {'time': int(match[1]), 'action': match[2]}
                    if match[3]:
                        event['signal'] = int(match[3])
                    row['container_events'].append(event)
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

result['accounting'] = []
helper = Path(__file__).resolve().parents[1] / 'slurm_submit_helpers.sh'
for job_id in sorted({str(row['slurm_job_id']) for row in rows}):
    if not shutil.which('scontrol'):
        break
    try:
        # Use the same tested controller/accounting fallback as real submissions.
        # Spur rejects native sacct -X/-P; the shared helper selects its syntax.
        query = subprocess.run(['bash', '-c',
            'source "$1"; detect_slurm_backend; query_slurm_result "$2"',
            'inspect-slurm', str(helper), job_id], capture_output=True, text=True, timeout=45)
        entry = {'job_id': int(job_id), 'query_rc': query.returncode,
                 'reported_status': query.stdout.strip()[:200], 'query_error': query.stderr.strip()[-500:]}
        for line in query.stdout.splitlines():
            match = re.fullmatch(r'([A-Z_]+)\+?\|(\d+:\d+)?', line.strip())
            if match and match[1] in {'PENDING', 'RUNNING', 'COMPLETING', 'COMPLETE', 'COMPLETED',
                    'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
                    'BOOT_FAIL', 'SUSPENDED', 'CONFIGURING', 'REQUEUED', 'DEADLINE'}:
                entry['state'] = match[1]
                if match[2]:
                    entry['exit_code'] = match[2]
        result['accounting'].append(entry)
    except subprocess.TimeoutExpired:
        result['accounting'].append({'job_id': int(job_id), 'query_timed_out': True})

(out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
