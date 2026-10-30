"""Bound C48 startup to a confirmed Spur allocation and observable workers."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

TERMINAL = {'CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'COMPLETING', 'PREEMPTED'}
FIELDS = {'JobId', 'JobState', 'Restarts', 'NumNodes', 'NodeList', 'StartTime'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def event(args, phase, **fields):
    row = dict(observed_at=datetime.now(timezone.utc).isoformat(),
               job_id=args.job_id, phase=phase, **fields)
    print('[pd.launch] ' + json.dumps(row), file=sys.stderr, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_suffix('.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
    return row


def command(argv, timeout=10):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} query failed rc={result.returncode}: {result.stderr[-1200:]}')
    return result.stdout


def query_job(args):
    raw = command(['scontrol', '--controller', args.controller,
                   'show', 'job', args.job_id])
    blocks = re.split(r'(?=\bJobId=)', raw)
    rows = []
    for block in blocks:
        fields = {key: value for key, value in re.findall(r'(\w+)=(\S+)', block)
                  if key in FIELDS}
        if fields.get('JobId') == args.job_id:
            rows.append(fields)
    if len(rows) != 1:
        raise RuntimeError('Controller did not return exactly the requested job')
    row = rows[0]
    for key in ('Restarts', 'NumNodes'):
        if not row.get(key, '').isdecimal():
            raise RuntimeError(f'Missing controller {key}')
        row[key] = int(row[key])
    row['JobState'] = row.get('JobState', '').upper().split('+')[0]
    return row


def validate_context(args):
    ids = [os.environ[key] for key in ('SPUR_JOB_ID', 'SLURM_JOB_ID') if os.environ.get(key)]
    if not ids or any(job != args.job_id for job in ids):
        raise RuntimeError('Startup gate requires the current allocation identity')


def wait_ready(args):
    validate_context(args)
    deadline = time.monotonic() + args.timeout_seconds
    restart = None
    last = None
    while True:
        row = query_job(args)
        if row != last:
            event(args, 'allocation-observed', controller=row)
            last = row
        if restart is None:
            restart = row['Restarts']
        elif row['Restarts'] != restart:
            raise RuntimeError('Allocation restarted while this batch was waiting; refusing old batch dispatch')
        state = row['JobState']
        if state in TERMINAL:
            raise RuntimeError(f'Allocation is {state}; refusing worker dispatch')
        if state == 'RUNNING':
            if row['NumNodes'] != args.num_ranks:
                raise RuntimeError('Confirmed allocation has an unexpected node count')
            proof = event(args, 'allocation-ready', controller=row)
            save(args.output, proof)
            # Shell exports this generation to the workers before its one exec.
            print(restart, flush=True)
            return 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Timed out waiting for RUNNING allocation')
        time.sleep(min(args.poll_seconds, remaining))


def worker_entry(args):
    validate_context(args)
    rank = int(os.environ['SPUR_TASK_OFFSET'])
    restart = int(os.environ['ATOMESH_SPUR_DISPATCH_RESTARTS'])
    if not 0 <= rank < args.num_ranks or restart < 0:
        raise RuntimeError('Invalid worker rank or allocation generation')
    save(args.run_dir / f'worker-entry-{rank}.json', dict(
        job_id=args.job_id, run_token=args.run_token, rank=rank,
        restarts=restart, node=socket.gethostname(),
        observed_at=datetime.now(timezone.utc).isoformat()))
    return 0


def started_workers(args, restart):
    result = []
    for rank in range(args.num_ranks):
        path = args.run_dir / f'worker-entry-{rank}.json'
        try:
            row = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            continue
        if (str(row.get('job_id')) == args.job_id and row.get('run_token') == args.run_token
                and row.get('rank') == rank and row.get('restarts') == restart):
            result.append(rank)
    return result


def watch_startup(args, clock=time.monotonic):
    """Only cancel a confirmed allocation with no worker entry in its generation."""
    path = args.run_dir / 'startup-watch.json'
    if args.state != 'RUNNING':
        save(path, dict(job_id=args.job_id, run_token=args.run_token,
                        phase='waiting-allocation', scheduler_state=args.state))
        return 0
    row = query_job(args)
    if row['JobState'] != 'RUNNING':
        save(path, dict(job_id=args.job_id, run_token=args.run_token,
                        phase='waiting-allocation', controller=row))
        event(args, 'allocation-view-changed', controller=row, queue_state=args.state)
        return 0
    current = clock()
    try:
        previous = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        previous = {}
    same = (previous.get('job_id') == args.job_id
            and previous.get('run_token') == args.run_token
            and previous.get('restarts') == row['Restarts'])
    since = previous.get('running_since', current) if same else current
    since = min(since, current)
    workers = started_workers(args, row['Restarts'])
    elapsed = current - since
    phase = 'worker-observed' if workers else 'waiting-workers'
    rc = 0
    if not workers and elapsed >= args.timeout_seconds:
        phase, rc = 'worker-entry-timeout', 124
    report = dict(job_id=args.job_id, run_token=args.run_token,
                  restarts=row['Restarts'], controller=row, phase=phase,
                  running_since=since, elapsed_seconds=elapsed, worker_ranks=workers,
                  observed_at=datetime.now(timezone.utc).isoformat())
    save(path, report)
    if previous.get('phase') != phase or rc:
        event(args, phase, controller=row, worker_ranks=workers, elapsed_seconds=elapsed)
    return rc


def check_previous(args):
    manifest = json.loads(args.previous_manifest.read_text())
    jobs = manifest['jobs']
    if not jobs or any(
        not (job.get('job_id') or job.get('job_name'))
        or ('job_id' in job and not str(job['job_id']).isdecimal())
        or ('job_name' in job and not re.fullmatch(r'[A-Za-z0-9_.-]{1,255}', job['job_name']))
        for job in jobs
    ):
        raise RuntimeError('Previous-job manifest must name exact job IDs or full job names')
    ids = {str(job['job_id']) for job in jobs if 'job_id' in job}
    names = {job['job_name'] for job in jobs if 'job_name' in job}
    output_format = '%i|%.256j' if names else '%i'
    rows = command(['squeue', '--controller', args.controller,
                    '--noheader', f'--format={output_format}']).splitlines()
    active = set()
    ambiguous = []
    for row in rows:
        job_id, separator, name = row.strip().partition('|')
        job_id = job_id.strip()
        name = name.strip()
        if not job_id.isdecimal() or (names and (
            not separator or not name or '%' in name or len(name) >= 256
            or name in {'-', '(null)', 'N/A', 'Unknown', 'unknown', '<unknown>'}
        )):
            raise RuntimeError('Invalid or incomplete active-job inventory')
        if job_id in ids or name in names:
            active.add(job_id)
        elif names and any(expected.startswith(name.rstrip('+')) for expected in names):
            # A scheduler that truncates job names cannot establish absence.
            ambiguous.append({'job_id': job_id, 'reported_name': name})
    active = sorted(active)
    save(args.output, dict(checked_at=datetime.now(timezone.utc).isoformat(),
                          jobs=jobs, still_active=active, ambiguous_names=ambiguous,
                          verified_inactive=not active and not ambiguous,
                          evidence_scope='Fresh scheduler absence; does not prove container cleanup'))
    if active or ambiguous:
        raise RuntimeError(f'Previous task jobs remain active or names are incomplete: '
                           f'{active or ambiguous}; refusing a duplicate workload')
    print('[C48] Exact previous jobs are absent from the active queue', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['wait', 'worker-entry', 'watch', 'check-previous'])
    parser.add_argument('--job-id')
    parser.add_argument('--controller')
    parser.add_argument('--num-ranks', type=int, default=2)
    parser.add_argument('--run-token')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--previous-manifest', type=Path)
    parser.add_argument('--state', default='')
    parser.add_argument('--timeout-seconds', type=float, default=180)
    parser.add_argument('--poll-seconds', type=float, default=2)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.timeout_seconds, args.poll_seconds)):
        parser.error('Timeout and polling interval must be positive and finite')
    if args.mode != 'check-previous' and (not args.job_id or not args.job_id.isdecimal()):
        parser.error('Numeric allocation ID required')
    if args.mode in {'wait', 'watch', 'check-previous'} and not args.controller:
        parser.error('Explicit controller required')
    if args.mode in {'worker-entry', 'watch'} and (
            not args.run_dir or not re.fullmatch(r'[0-9a-f]{32}', args.run_token or '')):
        parser.error('Run directory and token required')
    if args.mode in {'wait', 'check-previous'} and not args.output:
        parser.error('Evidence output required')
    if args.mode == 'check-previous' and not args.previous_manifest:
        parser.error('Exact previous-job manifest required')
    try:
        return {'wait': wait_ready, 'worker-entry': worker_entry,
                'watch': watch_startup, 'check-previous': check_previous}[args.mode](args)
    except Exception as error:
        event(args, 'allocation-check-failed', error=f'{type(error).__name__}: {error}')
        return 75


if __name__ == '__main__':
    sys.exit(main())
