#!/usr/bin/env python3
"""No-model Triton diagnostics in the pinned vLLM image and Slurm container."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout[:40000], 'stderr': result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': str(exc)}


def mapped_gpu_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        path = line.split()[-1]
        if path.startswith('/') and any(s in path for s in
                ('libamdhip64', 'libhsa-runtime', 'libamd_comgr', 'libdrm_amdgpu')):
            paths.add(path)
    return sorted(paths)


def environment(torch, triton):
    from triton.runtime import driver
    versions = {}
    for name in ('vllm', 'torch', 'triton', 'rocm', 'rocm-sdk-core', 'amd-aiter'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    keys = ('HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
            'AITER_CACHE_DIR', 'AITER_JIT_DIR', 'FLYDSL_RUNTIME_CACHE_DIR',
            'LD_LIBRARY_PATH', 'LD_PRELOAD', 'PYTORCH_ROCM_ARCH', 'HSA_OVERRIDE_GFX_VERSION',
            'HSA_XNACK', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
            'DEBUG_HIP_DYNAMIC_QUEUES', 'VLLM_USE_V2_MODEL_RUNNER')
    props = torch.cuda.get_device_properties(0)
    target = driver.active.get_current_target()
    data = {'uname': tuple(os.uname()), 'uid': os.getuid(), 'versions': versions,
            'torch_hip_version': torch.version.hip, 'device_name': props.name,
            'gcnArchName': getattr(props, 'gcnArchName', None),
            'visible_device_count': torch.cuda.device_count(), 'logical_device': 0,
            'triton_target': vars(target), 'env': {k: os.environ.get(k) for k in keys},
            'affinity': sorted(os.sched_getaffinity(0))}
    p = Path('/sys/module/amdgpu/version')
    data['host_amdgpu_module_version'] = p.read_text().strip() if p.exists() else None
    data['commands'] = [command(argv) for argv in (
        ['uname', '-a'], ['amd-smi', 'version'], ['rocm-smi', '--showdriverversion'],
        ['hipconfig', '--version'])]
    libraries = mapped_gpu_libraries()
    data['mapped_gpu_libraries'] = libraries
    data['hip_api_versions'] = {}
    for path in libraries:
        if 'libamdhip64' not in path:
            continue
        lib = ctypes.CDLL(path)
        for name in ('hipRuntimeGetVersion', 'hipDriverGetVersion'):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            version = ctypes.c_int()
            rc = fn(ctypes.byref(version))
            data['hip_api_versions'][name] = {'rc': rc, 'value': version.value, 'library': path}
        break
    from triton.runtime.cache import get_cache_manager
    data['effective_triton_cache_dir'] = str(get_cache_manager('atomesh-probe-location').cache_dir)
    return data


def save_kernel(compiled, out):
    out.mkdir(exist_ok=True)
    metadata = compiled.metadata
    dump(out / 'metadata.json', metadata._asdict() if hasattr(metadata, '_asdict') else metadata)
    hashes = {}
    for suffix in ('ttir', 'ttgir', 'llir', 'amdgcn', 'hsaco'):
        value = compiled.asm.get(suffix)
        if value is None:
            continue
        path = out / ('kernel.' + suffix)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(str(value))
        hashes[suffix] = hashlib.sha256(path.read_bytes()).hexdigest()
        if suffix == 'hsaco':
            dump(out / 'elf-header.json', command(['readelf', '-h', str(path)]))
            dump(out / 'elf-notes.json', command(['readelf', '-n', str(path)]))
    dump(out / 'sha256.json', hashes)


def run_worker(out):
    global torch, triton, tl
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        import torch
        import triton
        import triton.language as tl
        torch.cuda.set_device(0)
        env = environment(torch, triton)
        dump(out / 'environment.json', env)
        assert '8d0f05a0c' in env['versions']['vllm'], env['versions']
        assert 'MI355' in env['device_name'], env['device_name']
        assert env['triton_target']['arch'].split(':')[0] == 'gfx950', env['triton_target']
        print('[environment]', json.dumps(env), flush=True)

        @triton.jit
        def write_kernel(output, N: tl.constexpr, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            tl.store(output + offsets, offsets * 3 + 7, offsets < N)

        def launch(name, kernel, grid, args, constants, verify):
            row = {'kernel': name, 'stage': 'compile', 'status': 'running'}
            rows.append(row)
            dump(out / 'results.json', rows)
            try:
                compiled = kernel.warmup(*args, grid=grid, **constants)
                save_kernel(compiled, out / name)
                row['stage'] = 'load_and_launch'
                dump(out / 'results.json', rows)
                kernel[grid](*args, **constants)
                torch.cuda.synchronize()
                row['stage'] = 'verify'
                verify()
                row.update(status='success', stage='complete')
            except Exception:
                row.update(status='failed', traceback=traceback.format_exc())
            dump(out / 'results.json', rows)
            print('[kernel]', json.dumps(row), flush=True)

        # Bounds and nonzero output are checked on CPU, including an odd tail.
        output = torch.empty(2051, dtype=torch.int32, device='cuda')
        expected = torch.arange(2051, dtype=torch.int32) * 3 + 7
        launch('minimal_write', write_kernel, (triton.cdiv(2051, 256),), (output,),
               {'N': 2051, 'BLOCK': 256},
               lambda: torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0))

        # Import installed functions: do not patch vLLM or substitute copies.
        from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel, StagedWriteTensor
        from vllm.v1.worker.utils import _zero_kv_blocks_kernel
        for name, fn in [('_apply_write_kernel', _apply_write_kernel),
                         ('_zero_kv_blocks_kernel', _zero_kv_blocks_kernel)]:
            source = Path(fn.fn.__code__.co_filename)
            dump(out / (name + '-source.json'), {'path': str(source),
                 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})

        tensor = lambda x, dtype=torch.int32: torch.tensor(x, dtype=dtype, device='cuda')
        output = torch.full((3, 6), -7, dtype=torch.int32, device='cuda')
        expected = torch.full((3, 6), -7, dtype=torch.int32)
        expected[0, 1:3] = torch.tensor([11, 12])
        expected[2, 3:6] = torch.tensor([21, 22, 23])
        args = (output, output.stride(0), tensor([0, 2]), tensor([1, 3]),
                tensor([11, 12, 21, 22, 23]), tensor([2, 5]), None)
        launch('apply_write_device', _apply_write_kernel, (2,), args,
               {'BLOCK_SIZE': 1024, 'MULTI_GROUP': False},
               lambda: torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0))

        # Match total_len.apply_write(): int32 scalar state with native UVA pools.
        staged = StagedWriteTensor(4, dtype=torch.int32, device=torch.device('cuda:0'))
        staged.stage_write_elem(0, 11)
        staged.stage_write_elem(3, 29)
        indices = staged.write_indices.copy_to_uva(staged._staged_write_indices)
        starts = staged.write_starts.copy_to_uva(staged._staged_write_starts)
        lengths = staged.write_cu_lens.copy_to_uva(staged._staged_write_cu_lens)
        contents = tensor(staged._staged_write_contents)
        expected = torch.tensor([11, 0, 0, 29], dtype=torch.int32)
        def verify_staged():
            torch.testing.assert_close(staged.gpu.cpu(), expected, rtol=0, atol=0)
            staged.stage_write_elem(2, 41)
            staged.apply_write()
            torch.cuda.synchronize()
            torch.testing.assert_close(staged.gpu.cpu(), torch.tensor([11, 0, 41, 29], dtype=torch.int32))
        launch('apply_write_uva', _apply_write_kernel, (2,),
               (staged.gpu, staged.gpu.stride(0), indices, starts, contents, lengths, None),
               {'BLOCK_SIZE': 1024, 'MULTI_GROUP': False}, verify_staged)

        # Two segments, padding, untouched blocks, and multi-chunk pages.
        strides, pages = [1536, 2048], [1300, 1800]
        segments = [torch.full((4, stride), 7, dtype=torch.int32, device='cuda') for stride in strides]
        expected_segments = [s.cpu() for s in segments]
        for e, page in zip(expected_segments, pages):
            e[[0, 2], :page] = 0
        def verify_zero():
            for actual, expected in zip(segments, expected_segments):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        launch('zero_kv_blocks', _zero_kv_blocks_kernel, (2, 2, 2),
               (tensor([s.data_ptr() for s in segments], torch.uint64), tensor(strides, torch.int64),
                tensor(pages, torch.int64), tensor([0, 2], torch.int64)), {'BLOCK_SIZE': 1024}, verify_zero)
        dump(out / 'mapped-libraries-final.json', mapped_gpu_libraries())
    except Exception:
        rows.append({'kernel': 'worker_setup_or_native_import', 'status': 'failed',
                     'traceback': traceback.format_exc()})
        dump(out / 'results.json', rows)
        print(traceback.format_exc(), flush=True)
    return 0 if len(rows) == 4 and all(r['status'] == 'success' for r in rows) else 1


def run_suite(out):
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE.parents[1] / 'benchmark/vllm_m31/experiment.json').read_text())
    from pd_vllm_m31 import role_env
    base = dict(os.environ)
    base.update(plan['common_env'])
    base.update(VLLM_PLUGINS='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    rows = []
    for cache in ('default', 'fresh'):
        for role in ('prefill', 'decode'):
            dest = out / (cache + '-' + role)
            dest.mkdir(exist_ok=True)
            env = role_env(base, role, '127.0.0.1', plan[role]['env'])
            # No Store, no communication group, no model creation in this probe.
            if cache == 'fresh':
                env['TRITON_CACHE_DIR'] = str(Path(os.environ['XDG_CACHE_HOME']) / 'triton-probe' / dest.name)
                if Path(env['TRITON_CACHE_DIR']).exists():
                    raise RuntimeError('Fresh probe cache already exists; use a new allocation')
            row = {'case': dest.name, 'status': 'running'}
            rows.append(row)
            dump(out / 'summary.json', rows)
            with (dest / 'worker.log').open('w') as log:
                try:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--worker', '--out', str(dest)], env=env, stdout=log,
                        stderr=subprocess.STDOUT, timeout=180, cwd='/tmp')
                    row['returncode'] = result.returncode
                    result_file = dest / 'results.json'
                    results = json.loads(result_file.read_text()) if result_file.exists() else []
                    row['kernels'] = results
                    row['status'] = ('success' if result.returncode == 0 and len(results) == 4
                        and all(r['status'] == 'success' for r in results) else 'failed')
                except subprocess.TimeoutExpired:
                    row.update(status='failed', error='worker_timeout_180s')
            dump(out / 'summary.json', rows)
            print('[triton-probe]', json.dumps(row), flush=True)
    return 0 if all(r['status'] == 'success' for r in rows) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or Path(os.environ['RUN_DIR']) / 'triton-probe'
    sys.exit(run_worker(out) if args.worker else run_suite(out))
