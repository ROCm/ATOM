"""Read-only snapshot of one ATOMesh run's shared logs and Slurm state."""
import json
import os
from pathlib import Path
import re
import subprocess

run_id = os.environ['INSPECT_RUN_ID']
if not re.fullmatch(r'[0-9]+', run_id):
    raise SystemExit('INSPECT_RUN_ID must contain only digits')
out = Path('ci-run-inspection')
out.mkdir(exist_ok=True)
os.environ['PATH'] = '/usr/local/slurm-24.05.5.1/bin:/usr/local/sbin:/usr/local/bin:' + os.environ.get('PATH', '')
summary = {'run_id': run_id, 'logs': [], 'commands': {}}


def command(label, args, keep=None):
    try:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        output = result.stdout
        if keep and result.returncode == 0:
            output = '\n'.join(line for line in output.splitlines() if keep(line)) + '\n'
        summary['commands'][label] = {'returncode': result.returncode, 'output': output}
    except (OSError, subprocess.TimeoutExpired) as exc:
        summary['commands'][label] = {'error': str(exc)}
    print(label, json.dumps(summary['commands'][label]), flush=True)


command('queue_for_run', ['squeue', '--noheader', '--format=%i|%.220j|%T|%R|%N|%M'], lambda line: run_id in line)
command('target_node', ['scontrol', 'show', 'node', 'pit2-p03-g10'])
for root in [Path('/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG'), Path('/it-share/ATOMESH_LOG')]:
    if not root.is_dir():
        continue
    for directory in root.glob(f'*-{run_id}-*'):
        job_dirs = list(directory.glob('slurm_job-*'))
        for job_dir in job_dirs:
            job_id = job_dir.name.removeprefix('slurm_job-')
            if job_id.isdigit():
                command('accounting_' + job_id, ['sacct', '--jobs', job_id, '--parsable2', '--format=JobID,State,Elapsed,NodeList,ExitCode'])
                command('job_' + job_id, ['scontrol', 'show', 'job', job_id])
        candidates = list(directory.glob('*.out')) + list(directory.glob('*.err'))
        for job_dir in job_dirs:
            candidates.extend(job_dir.glob('*.json'))
            candidates.extend(job_dir.glob('logs/*-rank-*.log'))
            candidates.extend(job_dir.glob('rank-*/container.log'))
            candidates.extend(job_dir.glob('benchmark_results/*/torch_profile.log'))
        for source in candidates:
            if not source.is_file():
                continue
            relative = source.relative_to(directory)
            target = out / directory.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            with source.open('rb') as f:
                f.seek(max(0, size - 4 * 1024 * 1024))
                target.write_bytes(f.read())
            summary['logs'].append({'source': str(source), 'size': size, 'copied_tail_bytes': target.stat().st_size})
print(json.dumps(summary['logs'], indent=2), flush=True)
(out/'inspection.json').write_text(json.dumps(summary, indent=2) + '\n')
