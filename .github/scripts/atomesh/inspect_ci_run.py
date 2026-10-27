"""Report only the requested CI run's Slurm status; never publish log contents."""
import json
import os
from pathlib import Path
import re
import subprocess

run_id = os.environ['INSPECT_RUN_ID']
if not re.fullmatch(r'[0-9]+', run_id):
    raise SystemExit('INSPECT_RUN_ID must contain only digits')
os.environ['PATH'] = '/usr/local/slurm-24.05.5.1/bin:/usr/local/sbin:/usr/local/bin:' + os.environ.get('PATH', '')
job_ids = set()
for root in [Path('/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG'), Path('/it-share/ATOMESH_LOG')]:
    if root.is_dir():
        for directory in root.glob(f'*-{run_id}-*'):
            for job_dir in directory.glob('slurm_job-*'):
                job_id = job_dir.name.removeprefix('slurm_job-')
                if job_id.isdigit():
                    job_ids.add(job_id)
report = {'run_id': run_id, 'jobs': []}
for job_id in sorted(job_ids):
    record = {'slurm_job_id': job_id}
    try:
        result = subprocess.run(['squeue', '--jobs', job_id, '--noheader', '--format=%i|%T|%M|%R'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        record['query_exit_code'] = result.returncode
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) != 4 or fields[0].strip() != job_id:
                continue
            _, state, elapsed, reason = [f.strip() for f in fields]
            record['state'] = state if re.fullmatch(r'[A-Z_]+', state) else 'unknown'
            record['elapsed'] = elapsed if re.fullmatch(r'[0-9:-]+', elapsed) else 'unknown'
            reason = reason.strip('()')
            record['pending_reason_category'] = reason if reason in {'Resources', 'Priority', 'ReqNodeNotAvail', 'Dependency', 'None'} else 'other'
    except (OSError, subprocess.TimeoutExpired):
        record['query_error'] = True
    report['jobs'].append(record)
print(json.dumps(report, indent=2), flush=True)
