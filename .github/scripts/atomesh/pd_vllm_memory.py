#!/usr/bin/env python3
"""Prepare native Store THP pages; recover only this allocation's memory failures."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
GIB = 1 << 30
RETRY_RC = 86
REASONS = {'rdma_registration_enomem'}
MARKER = 'vllm-memory-retry.json'


class StoreMemoryUnavailable(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def memory_failure(reason, out):
    if reason not in REASONS:
        raise ValueError(reason)
    root = Path(os.environ.get('ATOMESH_ALLOCATION_RUN_DIR', os.environ['RUN_DIR']))
    write_json(root / MARKER, {
        'schema': 1, 'job_id': os.environ.get('SLURM_JOB_ID', ''),
        'run_token': os.environ.get('ATOMESH_RUN_TOKEN', ''),
        'node': os.environ.get('ATOMESH_ALLOCATED_NODE', ''), 'reason': reason,
        'phase': Path(out).name, 'retry_rc': RETRY_RC,
    })
    raise StoreMemoryUnavailable(f'{reason}: Store memory unavailable on this allocation')


def valid_marker(root, job_id, token):
    try:
        data = json.loads((Path(root) / MARKER).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(data, dict) or data.get('schema') != 1 or not token
            or data.get('job_id') != job_id or data.get('run_token') != token
            or data.get('reason') not in REASONS or data.get('retry_rc') != RETRY_RC
            or not isinstance(data.get('node'), str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', data['node']) is None):
        return None
    return data


def cpu_numa_nodes(sysfs=Path('/sys')):
    return [int(path.name.removeprefix('node'))
            for path in sorted((sysfs / 'devices/system/node').glob('node[0-9]*'))
            if (path / 'cpulist').read_text().strip()]


def prepare(store, model_path, out, timeout=600):
    """Successful CI 40376309047: best-effort THP, then try native Store setup.

    Preparation does not reserve pages or prove that MR registration will work.
    Only the actual Store registration result can trigger memory recovery.
    """
    from numa_memory_budget import node_meminfo
    out = Path(out)
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'nodes': [], 'warnings': [], 'reserve_gib': 64,
             'stage': 'after_model_load_before_store_init', 'started_at': time.time()}
    summary = out / 'memory-preparation.json'
    segments = store['global_segment_size'] * 8
    if segments <= 0:
        raise ValueError('Expected positive Store global_segment_size')

    def warn(reason, **details):
        state['warnings'].append({'reason': reason, **details})
        print(f'[store-memory][WARN] {reason}: {details}; continuing toward Store registration', flush=True)

    write_json(summary, state)
    print('[store-memory] dropping clean model page cache before NUMA THP preparation', flush=True)
    try:
        # Exact snapshot only. This helper uses fadvise and never deletes files.
        result = subprocess.run([sys.executable, str(HERE / 'drop_page_cache.py'), str(model_path)],
                                check=False, timeout=120)
        if result.returncode:
            warn('cache_eviction_failed', rc=result.returncode)
    except subprocess.TimeoutExpired:
        warn('cache_eviction_timeout', timeout_seconds=120)
    try:
        nodes = cpu_numa_nodes()
    except OSError as exc:
        nodes = []
        warn('numa_unavailable', error=str(exc))
    if not nodes:
        warn('no_cpu_numa_nodes')
    for node in nodes:
        row = {'node': node, 'desired_bytes': int(segments / len(nodes))}
        state['nodes'].append(row)
        try:
            info = node_meminfo(Path('/sys'), node)
            reclaimable = info['MemFree'] + info['Active(file)'] + info['Inactive(file)']
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            row['status'] = 'numa_memory_unavailable'
            warn(row['status'], node=node, error=str(exc))
            write_json(summary, state)
            continue
        # Match the successful worker: global segments only, capped by current
        # free + reclaimable file pages with 64 GiB left on this NUMA node.
        gib = min(segments / len(nodes), reclaimable - 64 * GIB) / GIB
        row['reclaimable_bytes'] = reclaimable
        row['target_gib'] = max(0, gib)
        if gib * GIB < row['desired_bytes']:
            warn('thp_target_capped', node=node, target_gib=max(0, gib))
        if gib < 1:
            row['status'] = 'skipped_low_memory'
            write_json(summary, state)
            continue
        result_path = out / f'thp-node-{node}.json'
        # Integer GiB rounding matches the successful CI's helper invocation.
        row['requested_bytes'] = int(f'{gib:.0f}') * GIB
        print(f'[store-memory] NUMA {node}: compacting {gib:.0f} of {row["desired_bytes"] / GIB:g} GiB', flush=True)
        began = time.monotonic()
        try:
            with (logs / f'thp-node-{node}.log').open('w') as log:
                result = subprocess.run([
                    sys.executable, str(HERE / 'numa_exec.py'), str(node),
                    sys.executable, str(HERE / 'pd_vllm_memory.py'), 'probe',
                    '--gib', f'{gib:.0f}', '--output', str(result_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            row['rc'] = result.returncode
            row['status'] = 'probe_failed' if result.returncode else 'completed'
        except subprocess.TimeoutExpired:
            row.update(status='thp_timeout', rc=None, timeout_seconds=timeout)
        row['elapsed_seconds'] = time.monotonic() - began
        try:
            measured = json.loads(result_path.read_text())
            huge = measured['huge_bytes']
            if not isinstance(huge, int) or huge < 0:
                raise ValueError('Invalid huge_bytes')
            row['huge_bytes'] = huge
        except (OSError, ValueError, KeyError, TypeError):
            row['huge_bytes'] = None
            if row['status'] == 'completed':
                row['status'] = 'probe_result_unavailable'
        if row['status'] == 'completed' and row['huge_bytes'] < row['requested_bytes']:
            row['status'] = 'thp_shortfall'
        if row['status'] != 'completed':
            warn(row['status'], node=node, rc=row['rc'], huge_bytes=row['huge_bytes'],
                 requested_bytes=row['requested_bytes'])
        write_json(summary, state)
    state['status'] = 'completed_with_warnings' if state['warnings'] else 'completed'
    state['finished_at'] = time.time()
    write_json(summary, state)
    print(f'[store-memory] best-effort THP preparation {state["status"]}; attempting Store registration', flush=True)
    return state


def prepare_after_load(store, model_path, out, role, rank, timeout=7200):
    """All eight workers reach KV initialization before one prepares host memory.

    This must run before Worker.initialize_from_config constructs the Store.
    Waiting before taking the lock lets both TP groups arrive without deadlock.
    A terminal result releases peers on success or failure; phase directories
    are fresh for each run and are archived before any local memory retry.
    """
    if role not in ('prefill', 'decode') or rank not in range(4):
        raise ValueError(f'Expected 1P1D TP4+TP4, got {role=} {rank=}')
    out = Path(out)
    gate = out / 'store-memory-gate'
    gate.mkdir(parents=True, exist_ok=True)
    participant = f'{role}-{rank}'
    write_json(gate / f'{participant}.ready.json', {
        'role': role, 'rank': rank, 'pid': os.getpid(),
        'stage': 'after_model_load_before_store_init', 'arrived_at': time.time(),
    })
    expected = [gate / f'{r}-{i}.ready.json'
                for r in ('prefill', 'decode') for i in range(4)]
    terminal = gate / 'result.json'
    deadline = time.monotonic() + timeout
    print(f'[store-memory] {participant}: model loaded; waiting for all eight workers', flush=True)
    with (gate / 'prepare.lock').open('a') as lock:
        while True:
            if terminal.exists():
                result = json.loads(terminal.read_text())
                if result['status'] != 'completed':
                    raise RuntimeError(f"Store memory preparation failed: {result.get('error')}")
                print(f'[store-memory] {participant}: preparation complete; initializing Store', flush=True)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for all workers and Store memory preparation')
            if all(path.exists() for path in expected):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    try:
                        # Another worker may have completed between our check and lock.
                        if terminal.exists():
                            continue
                        print('[store-memory] all eight workers loaded; preparing once before Store initialization', flush=True)
                        try:
                            preparation = prepare(store, model_path, out)
                        except Exception as exc:
                            write_json(terminal, {'status': 'failed', 'coordinator': participant,
                                                  'error': str(exc), 'finished_at': time.time()})
                            raise
                        write_json(terminal, {'status': 'completed', 'coordinator': participant,
                                              'preparation_status': preparation['status'],
                                              'workers': [p.stem for p in expected],
                                              'finished_at': time.time()})
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    continue
            time.sleep(0.2)


def clean_host():
    """Only use permissions already available; never grant new container caps."""
    paths = [Path('/proc/sys/vm/drop_caches'), Path('/proc/sys/vm/compact_memory')]
    if not all(os.access(path, os.W_OK) for path in paths):
        return False
    try:
        os.sync()
        paths[0].write_text('3\n')
        paths[1].write_text('1\n')
    except OSError:
        return False
    return True


def archive_phases(root, name):
    # Keep old evidence inside a tarball so recursive result collectors cannot
    # select any partial measurement from an abandoned native attempt.
    root = Path(root)
    paths = [root / part for part in
             ('smoke', 'c32', 'c48', 'c64', 'logs', 'benchmark_results',
              'suite-status.json', 'memory-preparation.json')]
    with tarfile.open(root / name, 'w:gz') as archive:
        for path in paths:
            if path.exists():
                archive.add(path, arcname=path.name)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def recover_host(root, job_id, token, node):
    """After the failed container is stopped, clean once if host rights allow."""
    root = Path(root)
    marker = valid_marker(root, job_id, token)
    if not marker or marker['node'] != node or (root / 'host-memory-recovery.json').exists():
        return RETRY_RC
    began = time.monotonic()
    status = 'unavailable'
    # The child bounds sync/compaction as well as sudo; sudo is non-interactive.
    command = [sys.executable, str(HERE / 'pd_vllm_memory.py'), 'clean-host']
    paths = [Path('/proc/sys/vm/drop_caches'), Path('/proc/sys/vm/compact_memory')]
    if not all(os.access(path, os.W_OK) for path in paths):
        sudo = shutil.which('sudo')
        command = [sudo, '-n', *command] if sudo else []
    if command:
        try:
            result = subprocess.run(command, timeout=180, check=False)
            status = 'cleaned' if result.returncode == 0 else 'unavailable'
        except subprocess.TimeoutExpired:
            # Do not start another allocation while host cleanup could still run.
            status = 'timeout'
    write_json(root / 'host-memory-recovery.json', {
        'job_id': job_id, 'node': node, 'reason': marker['reason'],
        'status': status, 'elapsed_seconds': time.monotonic() - began})
    print(f'[store-memory] host cleanup {status} on {node}', flush=True)
    if status == 'cleaned':
        archive_phases(root, 'memory-before-local-retry.tar.gz')
        (root / MARKER).unlink()
        return 0
    return RETRY_RC if status == 'unavailable' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('probe')
    probe.add_argument('--gib', type=float, required=True)
    probe.add_argument('--output', type=Path, required=True)
    sub.add_parser('clean-host')
    for name in ('recover-host', 'retry-node', 'archive-failed'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--root', type=Path, required=True)
        cmd.add_argument('--job-id', required=True)
        cmd.add_argument('--token', required=True)
        if name == 'recover-host':
            cmd.add_argument('--node', required=True)
    args = parser.parse_args()
    if args.command == 'probe':
        from compact_huge_pages import fault_huge_pages
        size, huge, elapsed = fault_huge_pages(args.gib)
        write_json(args.output, {'requested_bytes': size, 'huge_bytes': huge, 'elapsed_seconds': elapsed})
        tag = '[WARN] ' if huge < size else ''
        print(f'[store-memory] {tag}{huge / GIB:g} of {size / GIB:g} GiB came as THP in {elapsed:g} s', flush=True)
        return 0
    if args.command == 'clean-host':
        return 0 if clean_host() else RETRY_RC
    if args.command == 'recover-host':
        return recover_host(args.root, args.job_id, args.token, args.node)
    data = valid_marker(args.root, args.job_id, args.token)
    try:
        rc = (args.root / 'rank-rc-0').read_text().strip()
    except OSError:
        rc = ''
    if data and rc == str(RETRY_RC):
        if args.command == 'archive-failed':
            archive_phases(args.root, 'memory-before-resubmit.tar.gz')
            return 0
        print(data['node'])
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
