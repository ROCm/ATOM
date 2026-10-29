#!/usr/bin/env python3
"""Prepare native Store THP pages; recover only this allocation's memory failures."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
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


def bind_thread_memory(nodes):
    """MPOL_BIND this thread, and threads it creates afterwards, to ``nodes``.

    ``None`` restores MPOL_DEFAULT. A policy is per thread, so the Store's
    registration threads inherit it only if they are created while it is set.
    """
    syscall_nr = {'x86_64': 238, 'aarch64': 237}[platform.machine()]
    libc = ctypes.CDLL(None, use_errno=True)
    if nodes is None:
        rc = libc.syscall(syscall_nr, 0, None, ctypes.c_ulong(0))
    else:
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * (max(nodes) // bits + 1))()
        for node in nodes:
            mask[node // bits] |= 1 << (node % bits)
        # The kernel reads maxnode - 1 bits.
        rc = libc.syscall(syscall_nr, 2, mask, ctypes.c_ulong(len(mask) * bits + 1))
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, f'set_mempolicy({nodes}) failed: {os.strerror(err)}')


def prepare(store, model_path, out, timeout=1800, first_timeout=600):
    """Successful CI 40376309047: best-effort THP, then try native Store setup.

    Preparation does not reserve pages or prove that MR registration will work.
    Only the actual Store registration result can trigger memory recovery.
    """
    from numa_memory_budget import node_meminfo
    out = Path(out)
    logs = out / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'nodes': [], 'warnings': [], 'reserve_gib': 64, 'store_ranks': 4,
             'stage': 'after_model_load_before_store_init', 'started_at': time.time()}
    summary = out / 'memory-preparation.json'
    segments = store['global_segment_size'] * state['store_ranks']
    if segments <= 0:
        raise ValueError('Expected positive Store global_segment_size')

    def warn(reason, **details):
        state['warnings'].append({'reason': reason, **details})
        print(f'[store-memory][WARN] {reason}: {details}', flush=True)

    write_json(summary, state)
    print('[store-memory] dropping clean model page cache (this and other models) before NUMA THP preparation',
          flush=True)
    # Earlier allocations leave other models' page cache on this node; on
    # pit2-p03-g12 it filled about 1 TiB and left no free 2 MiB block on NUMA 0.
    # This helper uses fadvise and never deletes files.
    roots = [str(model_path)] + [root for root in os.environ.get(
        'ATOMESH_VLLM_PAGE_CACHE_ROOTS', '/share_nfs/models').split(':') if root and Path(root).is_dir()]
    state['page_cache_roots'] = roots
    began = time.monotonic()
    try:
        result = subprocess.run([sys.executable, str(HERE / 'drop_page_cache.py'), *roots],
                                check=False, timeout=900)
        if result.returncode:
            warn('cache_eviction_failed', rc=result.returncode)
    except subprocess.TimeoutExpired:
        warn('cache_eviction_timeout', timeout_seconds=900)
    state['page_cache_seconds'] = time.monotonic() - began
    try:
        nodes = cpu_numa_nodes()
    except OSError as exc:
        nodes = []
        warn('numa_unavailable', error=str(exc))
    if not nodes:
        warn('no_cpu_numa_nodes')
    # Spur runs the container unprivileged, so this is usually 'unavailable'.
    state['host_clean'] = drop_and_compact_host(nodes)
    write_json(summary, state)

    def probe_node(node, desired, limit, attempt):
        row = {'node': node, 'desired_bytes': int(desired), 'attempt': attempt}
        try:
            info = node_meminfo(Path('/sys'), node)
            reclaimable = info['MemFree'] + info['Active(file)'] + info['Inactive(file)']
        except (OSError, ValueError, KeyError, SystemExit) as exc:
            row['status'] = 'numa_memory_unavailable'
            warn(row['status'], node=node, error=str(exc))
            return row
        # Global segments only, capped by current free + reclaimable file pages
        # with 64 GiB left on this NUMA node.
        gib = min(desired, reclaimable - 64 * GIB) / GIB
        row['reclaimable_bytes'] = reclaimable
        row['target_gib'] = max(0, gib)
        row['huge_blocks_before'] = node_free_huge_blocks(node)
        if gib * GIB < row['desired_bytes']:
            warn('thp_target_capped', node=node, target_gib=max(0, gib))
        if gib < 1:
            row['status'] = 'skipped_low_memory'
            return row
        result_path = out / f'thp-node-{node}-{attempt}.json'
        # Integer GiB rounding matches the successful CI's helper invocation.
        row['requested_bytes'] = int(f'{gib:.0f}') * GIB
        print(f'[store-memory] NUMA {node} ({attempt}): compacting {gib:.0f} of '
              f'{row["desired_bytes"] / GIB:g} GiB, limit {limit} s', flush=True)
        began = time.monotonic()
        try:
            with (logs / f'thp-node-{node}-{attempt}.log').open('w') as log:
                result = subprocess.run([
                    sys.executable, str(HERE / 'numa_exec.py'), str(node),
                    sys.executable, str(HERE / 'pd_vllm_memory.py'), 'probe',
                    '--gib', f'{gib:.0f}', '--output', str(result_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=limit, check=False)
            row['rc'] = result.returncode
            row['status'] = 'probe_failed' if result.returncode else 'completed'
        except subprocess.TimeoutExpired:
            row.update(status='thp_timeout', rc=None, timeout_seconds=limit)
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
        return row

    # A node whose memory is pinned by page cache other tenants keep mapped
    # (pit2-p03-g12 NUMA 0: 833 GiB active file, no free 2 MiB block) never
    # compacts. Give up on it early and place the whole pool on the nodes that
    # did compact; the Prefill workers bind their Store allocation there.
    for node in nodes:
        state['nodes'].append(probe_node(node, segments / len(nodes), min(timeout, first_timeout), 'share'))
        write_json(summary, state)
    usable = [row['node'] for row in state['nodes'] if row.get('status') == 'completed']
    failed = [row['node'] for row in state['nodes']
              if row.get('status') not in ('completed', 'skipped_low_memory')]
    state['store_nodes'] = list(nodes) if not failed else []
    if failed and usable:
        rows = [probe_node(node, segments / len(usable), timeout, 'pool') for node in usable]
        state['nodes'].extend(rows)
        if all(row.get('status') == 'completed' for row in rows):
            state['store_nodes'] = usable
    state['status'] = 'completed_with_warnings' if state['warnings'] else 'completed'
    state['finished_at'] = time.time()
    write_json(summary, state)
    print(f'[store-memory] THP preparation {state["status"]}; Store nodes {state["store_nodes"]}', flush=True)
    return state


def prepare_after_load(store, model_path, out, role, rank, timeout=7200):
    """All eight workers reach KV initialization before one prepares host memory.

    Only the four Prefill ranks own Store segments. All eight workers wait so
    Decode's model loading cannot refill the shared node's cache during cleanup.
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
                return result
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
                        if not preparation['store_nodes']:
                            detail = ', '.join(
                                f"node {row.get('node')} ({row.get('attempt')}) {row.get('status')}"
                                for row in preparation['nodes'] if row.get('status') != 'completed')
                            error = ('hugepage compact did not finish before Store initialization'
                                     + (': ' + detail if detail else ': no NUMA node was compacted'))
                            write_json(terminal, {'status': 'failed', 'coordinator': participant,
                                                  'error': error,
                                                  'preparation_status': preparation['status'],
                                                  'finished_at': time.time()})
                            raise RuntimeError(error)
                        write_json(terminal, {'status': 'completed', 'coordinator': participant,
                                              'preparation_status': preparation['status'],
                                              'store_nodes': preparation['store_nodes'],
                                              'workers': [p.stem for p in expected],
                                              'finished_at': time.time()})
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    continue
            time.sleep(0.2)


def node_free_huge_blocks(node):
    """Free order>=9 (2 MiB) blocks of a node's Normal zone, from /proc/buddyinfo."""
    for line in Path('/proc/buddyinfo').read_text().splitlines():
        fields = line.replace(',', ' ').split()
        if fields[1] == str(node) and fields[3] == 'Normal':
            return sum(int(count) << (order - 9) for order, count in enumerate(fields[4:]) if order >= 9)
    return None


def drop_and_compact_host(nodes):
    """Drop the host page cache and compact each node before faulting THP.

    Page cache left by earlier allocations fragments a node badly enough that a
    single-threaded THP fault can stall in direct compaction for over 30 min
    (job 10979, NUMA 1). The native container is privileged, so writing these
    sysctls uses permissions it already has; without them this only reports.
    """
    from numa_memory_budget import node_meminfo
    drop = Path('/proc/sys/vm/drop_caches')
    record = {'status': 'unavailable', 'nodes': {}}
    if not os.access(drop, os.W_OK):
        return record
    began = time.monotonic()
    try:
        before = {node: (node_meminfo(Path('/sys'), node)['MemFree'], node_free_huge_blocks(node))
                  for node in nodes}
        os.sync()
        drop.write_text('3\n')
        record['drop_seconds'] = time.monotonic() - began
        for node in nodes:
            started = time.monotonic()
            path = Path(f'/sys/devices/system/node/node{node}/compact')
            (path if os.access(path, os.W_OK) else Path('/proc/sys/vm/compact_memory')).write_text('1\n')
            record['nodes'][str(node)] = {
                'free_bytes_before': before[node][0], 'huge_blocks_before': before[node][1],
                'free_bytes_after': node_meminfo(Path('/sys'), node)['MemFree'],
                'huge_blocks_after': node_free_huge_blocks(node),
                'compact_seconds': time.monotonic() - started}
    except (OSError, ValueError, KeyError, IndexError, SystemExit) as exc:
        record.update(status='failed', error=str(exc))
        return record
    record.update(status='completed', elapsed_seconds=time.monotonic() - began)
    print(f'[store-memory] host page cache dropped and nodes compacted: {json.dumps(record)}', flush=True)
    return record


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
