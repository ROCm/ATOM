"""Verify this cohort's previous jobs have stopped before another GPU submission."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

PRIOR = {
    '7732': ('v0', '20261012181354'),
}
TERMINAL = {'CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}


def fields(text, job_id, case):
    values = dict(re.findall(r'(\w+)=([^\s]+)', text))
    expected = f'kimi-k3-c48-gap-kimi-k3-vllm-c48-gap-{case}-vllm-38289351181-1'
    assert values.get('JobId') == job_id and values.get('JobName') == expected, values
    assert values.get('Account') == 'amd-frameworks', values
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = {'purpose': 'Admission check before the replacement C48 GPU performance run', 'jobs': {}}
    try:
        for job_id, (case, stamp) in PRIOR.items():
            record = evidence['jobs'][job_id] = {'queries': []}
            def query():
                result = subprocess.run(['scontrol', '--controller', args.controller, 'show', 'job', job_id], capture_output=True, text=True, timeout=20)
                record['queries'].append({'rc': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
                assert result.returncode == 0, f'Cannot verify previous job {job_id}'
                return fields(result.stdout, job_id, case)
            state = query()
            if state.get('JobState') not in TERMINAL:
                # Only the exact old job/run/account verified above can be cancelled.
                result = subprocess.run(['scancel', '--controller', args.controller, job_id], capture_output=True, text=True, timeout=20)
                record['cancel'] = {'rc': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
                deadline = time.monotonic() + 60
                while state.get('JobState') not in TERMINAL and time.monotonic() < deadline:
                    time.sleep(5)
                    state = query()
            assert state.get('JobState') in TERMINAL, f'Previous job {job_id} has not stopped'
            unallocated = state.get('NodeList') == '(null)' and state.get('StartTime') == 'N/A'
            record['never_allocated'] = unallocated
            if not unallocated:
                cell = f'kimi-k3-c48-gap-kimi-k3-vllm-c48-gap-{case}-vllm'
                directory = Path(f'/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG/{cell}-38289351181-{stamp}/slurm_job-{job_id}')
                record['cleanup'] = {}
                for rank in (0, 1):
                    proof = json.loads((directory / f'cleanup-complete-{rank}.json').read_text())
                    assert str(proof['job_id']) == job_id and proof['finished'] and proof['query_rc'] == 0, proof
                    assert not (directory / f'cleanup-containers-{rank}.txt').read_text().strip(), 'Previous containers remain'
                    record['cleanup'][str(rank)] = proof
            record['verified'] = True
        evidence['verified'] = True
    finally:
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('[C48] Previous Slurm jobs and container cleanup verified', flush=True)


if __name__ == '__main__':
    main()
