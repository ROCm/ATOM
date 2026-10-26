#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read only the two PR2425 benchmark runs; never submit or cancel GPU work."""
import argparse
import json
import os
import re
from pathlib import Path
import socket
import time

RUNS = {
    'c32': ('40264271709', '10051', '20261026033043'),
    'c48': ('40264276316', '10052', '20261026033045'),
}
BASE = Path('/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG')
CASE = 'glm-5.2-mxfp4-glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-chunked-1m-c32-nightly'
PER_FILE = 8 << 20
PER_RUN = 64 << 20
PATTERNS = (
    'slurm_job-{job}/benchmark_results/**/benchmark-guard.json',
    'slurm_job-{job}/benchmark_results/**/aiperf.log',
    'slurm_job-{job}/benchmark_results/**/profile_export_aiperf.json',
    'slurm_job-{job}/benchmark_results/**/profile_export.json',
    'slurm_job-{job}/benchmark_results/**/profile_export_console.txt',
    'slurm_job-{job}/benchmark_results/**/summary.json',
    'slurm_job-{job}/benchmark_results/pd-*.json',
    'slurm_job-{job}/*.json',
    'slurm_job-{job}/rank-rc-*',
    'slurm_job-{job}/logs/*.log',
    'slurm_job-{job}/logs/*/*.log',
    'slurm-{job}.err',
    'slurm-{job}.out',
    'slurm_job-{job}/benchmark_results/**/profile_export.jsonl',
    'slurm_job-{job}/rank-*/*.log',
)


def collect_runner_diagnostics(output):
    records = {}
    for label, runner, target in (
        ('c32', 'atomesh-runner-03', '40264271709'),
        ('c48', 'atomesh-runner-02', '40264276316'),
    ):
        root = Path('/share_nfs/ATOMESH_RUNNER') / runner / '_diag'
        record = {'root': str(root), 'files': []}
        records[label] = record
        # These two jobs started at 03:30 UTC. Inspect only their worker files,
        # then require the expected GitHub run ID before exporting any lines.
        for path in sorted(root.glob('Worker_20261026-0330*.log')):
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            selected, identified, last_timestamp = [], False, None
            scanned = 0
            with path.open('rb') as stream:
                for raw in stream:
                    scanned += len(raw)
                    if scanned > 64 << 20:
                        break
                    line = raw.decode(errors='replace').rstrip()
                    identified |= target in line
                    match = re.search(r'2026-10-26 (\d{2}:\d{2}:\d{2})', line)
                    if not match:
                        continue
                    last_timestamp = match[1]
                    if not '05:04:00' <= match[1] <= '05:19:00':
                        continue
                    if re.search(r'token|authorization|secret|password|sig=|x-amz', line, re.I):
                        continue
                    if re.search(r'error|exception|fail|exit|kill|cancel|finished|completed|disconnect|connection|shutdown|retry|lost', line, re.I):
                        selected.append(line[:2000])
            info = {'source': str(path), 'identified_expected_run': identified,
                    'scanned_bytes': scanned, 'last_timestamp': last_timestamp,
                    'scan_limited': scanned > 64 << 20}
            record['files'].append(info)
            if identified:
                destination = output / label / ('runner-' + path.name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text('\n'.join(selected[-250:]) + '\n')
                info.update(artifact=str(destination.relative_to(output)), selected_lines=len(selected))
    return records


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'host': socket.gethostname(), 'unix_time': time.time(), 'runs': {}}
    for label, (run, job, stamp) in RUNS.items():
        source = BASE / f'{CASE}-{run}-{stamp}'
        record = {'run_id': run, 'job_id': job, 'source': str(source), 'exists': source.is_dir(), 'files': [], 'errors': []}
        manifest['runs'][label] = record
        if not source.is_dir():
            continue
        if not source.resolve().is_relative_to(BASE.resolve()):
            raise ValueError('Run directory escapes the fixed log root')
        used, seen = 0, set()
        for pattern in PATTERNS:
            for path in sorted(source.glob(pattern.format(job=job))):
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file() or not resolved.is_relative_to(source.resolve()):
                    continue
                seen.add(resolved)
                if used >= PER_RUN:
                    record['limit_reached'] = True
                    break
                relative = path.relative_to(source)
                destination = output / label / relative
                try:
                    with resolved.open('rb') as stream:
                        size = os.fstat(stream.fileno()).st_size
                        count = min(size, PER_FILE, PER_RUN - used)
                        stream.seek(max(0, size - count))
                        data = stream.read(count)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    used += len(data)
                    record['files'].append({'source': str(path), 'artifact': str(destination.relative_to(output)), 'original_bytes': size, 'copied_bytes': len(data), 'tail_only': size > len(data)})
                except OSError as exc:
                    record['errors'].append({'path': str(path), 'error': str(exc)})
    manifest["runner_diagnostics"] = collect_runner_diagnostics(output)
    (output / 'snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    collect(args.output)
