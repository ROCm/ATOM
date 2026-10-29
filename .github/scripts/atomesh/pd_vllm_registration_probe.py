#!/usr/bin/env python3
"""Measure unprofiled Mooncake GPU memory costs with four ranks, no model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def barrier(root, stage, rank, timeout=240):
    dump(root / 'barriers' / stage / f'{rank}.json', {'rank': rank})
    deadline = time.monotonic() + timeout
    while not all((root / 'barriers' / stage / f'{i}.json').exists() for i in range(4)):
        if list(root.glob('rank-*/failure.json')):
            raise RuntimeError(f'Peer failed before barrier {stage}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Four-rank barrier {stage} timed out')
        time.sleep(0.1)


def worker(root, rank, kv_gib):
    import torch
    from pd_vllm_stage_probe import probe
    out = root / f'rank-{rank}'
    out.mkdir(parents=True, exist_ok=True)
    timeline = []
    try:
        torch.cuda.set_device(rank)
        assert torch.cuda.device_count() == 4
        assert 'MI355' in torch.cuda.get_device_name(rank)
        def observe(stage, tag=None):
            free, total = torch.cuda.mem_get_info()
            row = {'stage': stage, 'free_bytes': free, 'total_bytes': total,
                   'torch_allocated_bytes': torch.cuda.memory_allocated(),
                   'torch_reserved_bytes': torch.cuda.memory_reserved(),
                   'device': torch.cuda.current_device(), 'time': time.time()}
            timeline.append(row)
            dump(out / 'memory.json', timeline)
            print('[registration-memory] ' + json.dumps(row), flush=True)
            if tag is not None:
                probe(out, stage, tag)
        observe('initial')
        # Match the full D worker's ~64 GiB non-KV footprint; no checkpoint is read.
        weights = torch.empty(64 * GIB, dtype=torch.uint8, device='cuda')
        weights.fill_(1)
        torch.cuda.synchronize()
        barrier(root, 'weights', rank)
        observe('before-engine', 10)
        from mooncake.engine import TransferEngine
        import socket
        engine = TransferEngine()
        rc = engine.initialize(socket.gethostbyname(socket.gethostname()), 'P2PHANDSHAKE', 'rdma', '')
        dump(out / 'engine.json', {'initialize_rc': rc, 'rank': rank,
             'MC_TE_FILTERS': os.environ.get('MC_TE_FILTERS'),
             'MC_HIP_NUM_STREAMS': os.environ.get('MC_HIP_NUM_STREAMS'),
             'MC_HIP_NUM_EVENTS': os.environ.get('MC_HIP_NUM_EVENTS')})
        assert rc == 0, f'Engine initialization failed: {rc}'
        barrier(root, 'engine-init', rank)
        observe('after-engine-init', 11)
        kv = torch.empty(kv_gib * GIB, dtype=torch.uint8, device='cuda')
        kv.fill_(2)
        torch.cuda.synchronize()
        barrier(root, 'kv-allocation', rank)
        observe('after-kv-allocation', 12)
        rc = engine.batch_register_memory([kv.data_ptr()], [kv.numel()])
        dump(out / 'registration.json', {'rc': rc, 'bytes': kv.numel()})
        assert rc == 0, f'GPU registration failed: {rc}'
        barrier(root, 'kv-registration', rank)
        observe('after-kv-registration', 13)
        # Exercise the installed original kernel with an exact CPU check.
        from vllm.v1.worker.utils import _zero_kv_blocks_kernel
        segment = torch.full((4, 1536), 7, dtype=torch.int32, device='cuda')
        tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device='cuda')
        _zero_kv_blocks_kernel[(2, 1, 2)](
            tensor([segment.data_ptr()], torch.uint64), tensor([1536], torch.int64),
            tensor([1300], torch.int64), tensor([0, 2], torch.int64), BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        expected = torch.full((4, 1536), 7, dtype=torch.int32)
        expected[[0, 2], :1300] = 0
        torch.testing.assert_close(segment.cpu(), expected, rtol=0, atol=0)
        stages = [json.loads((out / f'stage-probe-{s}' / 'result.json').read_text())
                  for s in ('before-engine', 'after-engine-init', 'after-kv-allocation', 'after-kv-registration')]
        assert all(s['status'] == 'success' for s in stages), 'One or more stage probes failed'
        dump(out / 'result.json', {'status': 'success', 'native_zero_verified': True})
        # Keep all four large allocations alive until every rank completes checks.
        barrier(root, 'verification', rank)
        rc = engine.batch_unregister_memory([kv.data_ptr()])
        assert rc == 0, f'GPU unregistration failed: {rc}'
        return 0
    except Exception:
        failure = {'status': 'failed', 'traceback': traceback.format_exc()}
        dump(out / 'failure.json', failure)
        dump(out / 'result.json', failure)
        print(failure['traceback'], flush=True)
        return 1


def run_suite(out):
    from pd_vllm_m31 import role_env
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE.parents[1] / 'benchmark/vllm_m31/experiment.json').read_text())
    base = dict(os.environ)
    base.update(plan['common_env'])
    base.update(VLLM_PLUGINS='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    rows = []
    cases = [('default-193g', 193, {}),
             ('one-stream-193g', 193, {'MC_HIP_NUM_STREAMS': '1', 'MC_HIP_NUM_EVENTS': '1'}),
             ('default-128g', 128, {})]
    for name, kv_gib, overrides in cases:
        root = out / name
        root.mkdir()
        children = []
        logs = []
        row = {'case': name, 'status': 'running', 'weights_gib_per_rank': 64,
               'kv_gib_per_rank': kv_gib, 'ranks': 4, 'overrides': overrides}
        rows.append(row)
        dump(out / 'summary.json', rows)
        try:
            for rank in range(4):
                env = role_env(base, 'decode', '127.0.0.1', plan['decode']['env'])
                for key in ('MC_HIP_NUM_STREAMS', 'MC_HIP_NUM_EVENTS', 'MC_DISABLE_HIP'):
                    env.pop(key, None)
                env.update(overrides)
                env['MC_TE_FILTERS'] = f'rdma{rank + 4}'
                env['TRITON_CACHE_DIR'] = str(Path(os.environ['XDG_CACHE_HOME']) / name / f'rank-{rank}')
                (root / f'rank-{rank}').mkdir()
                log = (root / f'rank-{rank}/worker.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    str(root), str(rank), str(kv_gib)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, cwd='/tmp'))
            deadline = time.monotonic() + 360
            while any(p.poll() is None for p in children) and time.monotonic() < deadline:
                time.sleep(0.5)
            if any(p.poll() is None for p in children):
                row['error'] = 'case_timeout_360s'
            row['returncodes'] = [p.poll() for p in children]
            results = []
            for rank in range(4):
                file = root / f'rank-{rank}/result.json'
                results.append(json.loads(file.read_text()) if file.exists() else {'status': 'missing'})
            row['ranks_results'] = results
            row['status'] = ('success' if row['returncodes'] == [0] * 4
                             and all(r.get('status') == 'success' and r.get('native_zero_verified') is True
                                     for r in results) else 'failed')
        finally:
            for p in children:
                if p.poll() is None:
                    p.kill()
                p.wait()
            for log in logs:
                log.close()
        dump(out / 'summary.json', rows)
        print('[registration-probe] ' + json.dumps(row), flush=True)
    # Negative cases remain failures; the matrix is diagnostic, never a smoke result.
    return 0 if all(row['status'] == 'success' for row in rows) else 1


if __name__ == '__main__':
    sys.exit(worker(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
