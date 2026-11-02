"""Read only this validation's queue records and bounded shared-log excerpts."""
import datetime
import json
import subprocess
from pathlib import Path

RUNS = (41415572765, 41423516489, 41425998224)
PREFIX = 'minimax-m3.1-flash-quark-mxfp4-minimax-m31-flash-mxfp4-1p1d-pp4-tp4-dspark7-nvfp4-mcstore-agentx'
LOG_ROOT = Path('/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG')
OUT = Path('m31-review-inspection')
OUT.mkdir(exist_ok=True)
CONTROLLER = ','.join(f'http://pit2-vm-amd-large-{n}:6817' for n in ('04', '02', '03'))
summary = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'runs': {}, 'queue': []}
try:
    proc = subprocess.run(['squeue', '--controller', CONTROLLER, '--noheader',
                           '--format=%A|%j|%T|%N|%R|%M'],
                          capture_output=True, text=True, timeout=30)
    summary['queue_returncode'] = proc.returncode
    summary['queue_stderr'] = proc.stderr[-2000:]
    summary['queue'] = [line for line in proc.stdout.splitlines()
                        if PREFIX in line and any(f'-{run}-' in line for run in RUNS)]
except Exception as exc:
    summary['queue_error'] = str(exc)

for run in RUNS:
    entries = []
    for root in sorted(LOG_ROOT.glob(f'{PREFIX}*-{run}-*')):
        if not root.resolve().is_relative_to(LOG_ROOT.resolve()):
            continue
        files = sorted({*root.glob('slurm-*.out'), *root.glob('slurm-*.err'),
                        *root.glob('slurm_job-*/cell-metadata.json'),
                        *root.glob('slurm_job-*/job-result.json'),
                        *root.glob('slurm_job-*/rank-0/container.log'),
                        *root.glob('slurm_job-*/logs/prefill-rank-0.log'),
                        *root.glob('slurm_job-*/logs/decode-rank-0.log')})
        entry = {'root': str(root), 'files': []}
        for path in files:
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            stat = path.stat()
            with path.open('rb') as source:
                head = source.read(8192)
                source.seek(max(0, stat.st_size - 32768))
                tail = source.read(32768)
            relative = path.relative_to(root)
            target = OUT / str(run) / root.name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'=== FIRST 8 KiB ===\n' + head + b'\n=== LAST 32 KiB ===\n' + tail)
            entry['files'].append({'path': str(relative), 'bytes': stat.st_size,
                                   'mtime_utc': datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat()})
        entries.append(entry)
    summary['runs'][str(run)] = entries

(OUT / 'status.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
