"""Verify durable closure for this cohort's known, already-finished prior job."""
import argparse
import json
from pathlib import Path

PRIOR = {
    '7788': {
        'case': 'v0',
        'run_id': '38296153235',
        'stamp': '20261012191433',
        'run_token': '5640dc8d41c9427bb7603a04ec2baa50',
    },
}
TERMINAL = {'CANCELLED', 'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'}


def verify_closed_job(job_id, identity):
    cell = f'kimi-k3-c48-gap-kimi-k3-vllm-c48-gap-{identity["case"]}-vllm'
    log_root = (
        f'/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG/{cell}-'
        f'{identity["run_id"]}-{identity["stamp"]}/slurm_job-{job_id}'
    )
    directory = Path(log_root)
    metadata = json.loads((directory / 'cell-metadata.json').read_text())
    assert metadata['cell_id'] == cell and str(metadata['slurm_job_id']) == job_id, metadata
    assert metadata['backend'] == 'vllm' and metadata['log_root'] == log_root, metadata
    result = json.loads((directory / 'job-result.json').read_text())
    assert str(result['job_id']) == job_id, result
    assert result['scheduler']['state'] in TERMINAL, result
    assert result['workload']['state'] in {'COMPLETED', 'FAILED'}, result
    cleanup = {}
    for rank in (0, 1):
        proof = json.loads((directory / f'cleanup-complete-{rank}.json').read_text())
        assert str(proof['job_id']) == job_id and proof['rank'] == rank, proof
        assert proof['run_token'] == identity['run_token'], proof
        assert proof['finished'] is True and proof['query_rc'] == 0, proof
        assert int((directory / f'cleanup-query-{rank}.rc').read_text()) == 0
        assert proof['return_code'] == int((directory / f'rank-rc-{rank}').read_text()), proof
        assert not (directory / f'cleanup-containers-{rank}.txt').read_text().strip(), 'Previous containers remain'
        cleanup[str(rank)] = proof
    return {
        'source': 'durable_terminal_and_cleanup_markers',
        'log_root': log_root,
        'run_id': identity['run_id'],
        'run_token': identity['run_token'],
        'scheduler': result['scheduler'],
        'workload': result['workload'],
        'cleanup': cleanup,
        'verified': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', help='Retained for caller compatibility; no historical query is required')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = {
        'purpose': 'Admission check before the replacement C48 GPU performance run',
        'controller_query_required': False,
        'reason': 'The exact prior run is already closed; require its complete durable terminal and cleanup evidence.',
        'jobs': {},
    }
    try:
        for job_id, identity in PRIOR.items():
            evidence['jobs'][job_id] = verify_closed_job(job_id, identity)
        evidence['verified'] = True
    finally:
        args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print('[C48] Prior terminal workload and both container-cleanup records verified', flush=True)


if __name__ == '__main__':
    main()
