#!/usr/bin/env python3
"""Single-node native vLLM M3.1 PD experiment under the ATOMesh Slurm runner.

Uses the pinned image as installed. Owns only the subprocesses it starts;
all artifacts use the existing RUN_DIR and benchmark_results contracts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[1] / "benchmark/vllm_m31"
PROCESSES: list[tuple[str, subprocess.Popen, Path]] = []
# Explicit script path also supports CPU unit-test imports via importlib.
sys.path.insert(0, str(HERE))
from pd_vllm_memory import StoreMemoryUnavailable, memory_failure



def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def flags(values):
    args = []
    for name, value in values.items():
        if value is True:
            args.append("--" + name)
        elif value is False:
            args.append("--no-" + name)
        elif value is not None:
            args.extend(["--" + name, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    return args


def role_env(common, role, ip, overrides=None):
    env = dict(common)
    # These belong to the role/worker, never to inherited launcher defaults.
    for key in ('GLIBC_TUNABLES', 'MC_MAX_MR_SIZE', 'MC_TE_FILTERS', 'MC_DISABLE_HIP'):
        env.pop(key, None)
    prefix = f"ATOMESH_{role.upper()}_ENV_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env[key[len(prefix):]] = value.replace("${ROLE_IP}", ip)
    env.update(overrides or {})
    if role == 'decode':
        env.pop('MOONCAKE_CONFIG_PATH', None)
        env.pop('GLIBC_TUNABLES', None)
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    env["HIP_VISIBLE_DEVICES"] = "0,1,2,3" if role == "prefill" else "4,5,6,7"
    return env


def verify_attention_configuration(plan, logs, out):
    """Accept the requested cache dtypes and the actual native attention path."""
    expected = ('nvfp4', 'nvfp4', 'nvfp4', 'fp8')
    actual = (plan['common_args']['kv-cache-dtype'],
              plan['prefill']['attention-config']['indexer_kv_dtype'],
              plan['decode']['attention-config']['indexer_kv_dtype'],
              plan['speculative_config']['kv_cache_dtype'])
    evidence = {'cache_dtypes': dict(zip(('main', 'prefill_indexer', 'decode_indexer', 'draft'), actual)),
                'roles': {}, 'status': 'failed'}
    marker = 'MiniMax M3 sparse attention selected ROCm NVFP4'
    for role in ('prefill', 'decode'):
        with (logs / f'{role}.log').open(errors='replace') as stream:
            selected = [line.strip() for line in stream if marker in line]
        evidence['roles'][role] = {'rocm_nvfp4': bool(selected), 'evidence': selected}
    ok = actual == expected and all(x['rocm_nvfp4'] for x in evidence['roles'].values())
    evidence['status'] = 'success' if ok else 'failed'
    dump(out / 'attention-verification.json', evidence)
    if not ok:
        raise RuntimeError('Required NVFP4/NVFP4/FP8 cache or ROCm NVFP4 attention path was not verified')


def start(name, cmd, env, logs):
    log = logs / f"{name}.log"
    print(f"[{name}] {shlex.join(map(str, cmd))}", flush=True)
    # A pipe gives the many engine workers one log writer. Retain a local copy
    # so an interrupted/shared-filesystem write cannot hide the first exception.
    local = Path(tempfile.mkdtemp(prefix=f'atomesh-{name}-')) / log.name
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd="/tmp")

    def collect():
        with local.open('wb', buffering=0) as backup, log.open('wb', buffering=0) as shared:
            while chunk := proc.stdout.read1(65536):
                backup.write(chunk)
                try:
                    shared.write(chunk)
                except OSError as exc:
                    print(f'[logs] {name} shared log write failed: {exc}; local={local}', flush=True)
                if any(marker in chunk for marker in (b'ERROR', b'Fatal Python', b'Traceback', b'Error:')):
                    print(f'[{name}-diagnostic] ' + chunk.decode(errors='replace'), flush=True)
        proc.stdout.close()

    proc._atomesh_log_thread = threading.Thread(target=collect, daemon=True)
    proc._atomesh_local_log = local
    proc._atomesh_log_thread.start()
    PROCESSES.append((name, proc, log))
    return proc


def check_services():
    for name, proc, path in PROCESSES:
        if name == "agentx":
            continue
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"{name} exited {proc.returncode}: {path.read_text(errors='replace')[-6000:]}")
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                tail = stream.read().decode(errors="replace")
            for marker in ("Fatal Python error:", "Engine core initialization failed", "EngineCore failed"):
                if marker in tail:
                    raise RuntimeError(f"{name}: {marker}; {tail[-6000:]}")


def request(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode()


def wait_http(url, timeout):
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        check_services()
        try:
            request(url)
            print(f"[ready] {url}", flush=True)
            return
        except (OSError, urllib.error.URLError) as exc:
            last = str(exc)
        time.sleep(5)
    raise TimeoutError(f"{url}: {last}")


def cleanup():
    # First stop all clients/engines, then wait before stopping the Store master.
    for name, proc, log in reversed(PROCESSES):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        collector = getattr(proc, '_atomesh_log_thread', None)
        if collector is not None:
            collector.join(timeout=10)
            if not collector.is_alive():
                try:
                    shutil.copyfile(proc._atomesh_local_log, log)
                except OSError as exc:
                    print(f'[logs] {name} final copy failed: {exc}', flush=True)


def on_signal(signum, frame):
    raise RuntimeError(f"Interrupted by signal {signum}")


def reset_gpu_cache(url):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        check_services()
        result = json.loads(request(url + "/reset_prefix_cache", {}, timeout=30))
        if result.get("success") is True:
            return result
        time.sleep(2)
    raise TimeoutError(f"GPU prefix reset still refused: {url}")


def store_load_bytes(metrics):
    return sum(float(line.split()[-1]) for line in metrics.splitlines()
               if line.startswith("vllm:mooncake_store_operation_bytes_total{")
               and 'operation="load_get"' in line and 'status="ok"' in line)


def verify_store_load(url, before, out, name):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        check_services()
        metrics = request(url + "/metrics")
        (out / f"{name}.prom").write_text(metrics)
        delta = store_load_bytes(metrics) - before
        if delta > 0:
            return delta
        time.sleep(2)
    raise RuntimeError(f"{name}: no successful Store GET bytes observed")


def smoke(model, model_path, p_url, d_url, proxy_url, out):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rows = []
    for target in (1024, 32768, 131072):
        prefix = ("Reference notes: one two three four five six seven eight nine ten.\n" * (target // 10 + 1))
        ids = tokenizer.encode(prefix, add_special_tokens=False)[:target - 32]
        prompt = tokenizer.decode(ids) + "\nQuestion: What is two plus two? Answer briefly.\nAnswer:"
        body = {"model": model, "prompt": prompt, "max_tokens": 32,
                "temperature": 0, "ignore_eos": True, "stream": False, "seed": 42}
        # Exercise remote transfer before the direct-D reference can populate its cache.
        before = request(d_url + "/metrics")
        via_pd = json.loads(request(proxy_url + "/v1/completions", body, timeout=1200))
        after = request(d_url + "/metrics")
        (out / f"smoke-{target}-decode-before.prom").write_text(before)
        (out / f"smoke-{target}-decode-after.prom").write_text(after)
        # Drop D's local prefixes before computing the independent reference.
        reset = reset_gpu_cache(d_url)
        direct = json.loads(request(d_url + "/v1/completions", body, timeout=1200))
        # Drop both GPU prefix caches. The repeated PD request must exercise CPU reuse.
        reset_gpu_cache(p_url)
        reset_gpu_cache(d_url)
        load_before = store_load_bytes(request(p_url + "/metrics"))
        repeated = json.loads(request(proxy_url + "/v1/completions", body, timeout=1200))
        loaded = verify_store_load(p_url, load_before, out, f"smoke-{target}-store-reload")
        a, b, c = (v["choices"][0]["text"] for v in (via_pd, direct, repeated))
        # Exact greedy matches are recorded only: under ignore_eos the tokens
        # after the short answer are near ties that differ between P and D
        # prefill numerics (jobs 11054, 11075). Each path must answer correctly.
        answers = {name: bool(re.search(r"\b(4|four)\b", text[:48], re.IGNORECASE))
                   for name, text in (("pd", a), ("direct", b), ("after_reset", c))}
        row = {"target_tokens": target, "actual_tokens": len(tokenizer.encode(prompt)),
               "pd": via_pd, "direct_decode_reference": direct, "after_gpu_reset": repeated,
               "store_get_bytes": loaded, "matches_reference": a == b, "matches_after_reset": a == c,
               "answers_correct": answers, "reset_response": reset}
        rows.append(row)
        dump(out / "smoke.json", rows)
        (out / f"smoke-{target}-prefill-after-reset.prom").write_text(request(p_url + "/metrics"))
        print(f"[smoke] tokens={target} answers_correct={answers} reference_match={a == b} "
              f"store_reload_match={a == c}", flush=True)
        if not all(answers.values()):
            raise RuntimeError(f"PD/Store answer incorrect at {target} tokens: {answers}")

    dump(out / "smoke-store-scope.json", {
        "store_roles": ["prefill"], "decode_connector": "MooncakeConnector",
        "prefill_store_reload": "passed", "decode_to_prefill_store_reuse": "not_applicable",
        "reason": "Decode has no Store connector and does not write Store objects.",
    })


def main():
    if os.environ.get("ATOMESH_PD_WORKER_LAYOUT") != "single_node":
        raise ValueError("This experiment requires single-node 1P1D")
    if os.environ.get("CONC_LIST", "").count(","):
        raise ValueError("One concurrency per CI cell is required for independent cache state")
    if (os.environ.get("PREFILL_TP_SIZE") != "4"
            or os.environ.get("DECODE_TP_SIZE") != "4"
            or os.environ.get("xP") != "1" or os.environ.get("yD") != "1"):
        raise ValueError("This recipe supports exactly 1P1D TP4+TP4")
    plan = json.loads((CONFIG / "experiment.json").read_text())
    out = Path(os.environ["RUN_DIR"])
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    results = out / "benchmark_results"
    results.mkdir(exist_ok=True)
    common = dict(os.environ)
    for key, value in os.environ.items():
        if key.startswith("ATOMESH_ENV_"):
            common[key[len("ATOMESH_ENV_"):]] = value
    common.update(plan["common_env"])
    common["VLLM_PLUGINS"] = ""
    for key in ('GLIBC_TUNABLES', 'MC_MAX_MR_SIZE', 'MC_TE_FILTERS', 'MC_DISABLE_HIP',
                'MOONCAKE_CONFIG_PATH'):
        common.pop(key, None)
    ip = os.environ.get("NODE0_ADDR", "127.0.0.1")
    common["VLLM_HOST_IP"] = ip
    offset = int(os.environ.get("ATOMESH_SERVICE_PORT_OFFSET", "0"))
    ports = {"prefill": int(os.environ.get("PREFILL_PORT", 8100)) + offset,
             "decode": int(os.environ.get("DECODE_PORT", 8200)) + offset,
             "proxy": int(os.environ.get("ROUTER_PORT", 8000)) + offset,
             "master": 26051 + offset, "bootstrap": 8998 + offset}
    for port in ports.values():
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
    common["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(ports["bootstrap"])
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        common.pop(key, None)
    import importlib.metadata as md
    import mooncake.store
    import mooncake.engine
    import torch
    assert torch.cuda.device_count() == 8, "Expected one full eight-GPU allocation"
    assert mooncake.engine.SUPPORT_HIP
    assert all("MI355" in torch.cuda.get_device_name(i) for i in range(8))
    assert shutil.which("mooncake_master"), "Pinned image is missing mooncake_master"
    versions = {k: md.version(k) for k in ("vllm", "torch", "amd-aiter", "mooncake-transfer-engine-rocm")}
    assert "8d0f05a0c" in versions["vllm"], versions
    hardware = {"host": socket.gethostname(), "ip": ip, "versions": versions,
                "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
                "infiniband": sorted(p.name for p in Path('/sys/class/infiniband').iterdir())}
    dump(out / "image-hardware.json", hardware)
    print(json.dumps(hardware), flush=True)
    mem = dict(line.split(":", 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024
    store_bytes = plan['offload']['total_pool_bytes'] + plan['offload']['total_private_buffer_bytes']
    if available <= store_bytes + 128 * 1024**3:
        print(f'[store-memory][WARN] MemAvailable={available / 1024**3:.1f} GiB; '
              'continuing to actual Store registration', flush=True)
    # Reuse the cluster cache, while the model revision remains explicit.
    # Bound download memory; Xet's concurrent shard reconstruction was killed
    # with rc=137 during the first run, before Store or engines were started.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    memory = {"proc_meminfo": Path('/proc/meminfo').read_text(),
              "proc_cgroup": Path('/proc/self/cgroup').read_text()}
    for name in ('memory.max', 'memory.current', 'memory.events', 'memory.peak'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            memory[name] = path.read_text()
    dump(out / "memory-before-download.json", memory)
    print("[model] resolving pinned snapshot, HTTP streaming, max_workers=2", flush=True)
    from huggingface_hub import snapshot_download
    model_root = Path(os.environ["MODEL_PATH"])
    cache_root = model_root.parent / ".hf-vllm-m31"
    if not model_root.is_absolute():
        raise ValueError("Catalog must supply the shared model root")
    cache_root.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(plan['model'], revision=plan['model_revision'], cache_dir=str(cache_root), max_workers=2)
    assert (Path(model_path) / 'dspark/config.json').exists()
    plan["resolved_model_path"] = model_path
    plan["runtime_ports"] = ports
    plan["status"] = "running"
    plan["concurrency"] = int(os.environ["CONC_LIST"])
    dump(out / "resolved-experiment.json", plan)
    store = json.loads((CONFIG / "mooncake-store.json").read_text())
    store["master_server_address"] = f"{ip}:{ports['master']}"
    expected_nics = {f'rdma{i}' for i in range(8)}
    if not expected_nics.issubset(hardware['infiniband']):
        raise RuntimeError(f"Expected per-GPU RDMA rails {sorted(expected_nics)}, got {hardware['infiniband']}")
    # Empty device_name lets MC_TE_FILTERS select one physical GPU rail in each
    # worker, as in successful CI 40376309047. Never pass the eight-HCA list.
    store['device_name'] = ''
    dump(out / "mooncake-store.json", store)
    common['ATOMESH_VLLM_STORE_CONFIG_PATH'] = str(out / "mooncake-store.json")
    common['PYTHONPATH'] = str(HERE) + (os.pathsep + common['PYTHONPATH'] if common.get('PYTHONPATH') else '')
    common['ATOMESH_VLLM_MEMORY_DIR'] = str(out)
    master = start('mooncake-master', ['mooncake_master', '--port', str(ports['master'])], common, logs)
    deadline = time.monotonic() + 60
    while True:
        check_services()
        try:
            with socket.create_connection((ip, ports['master']), timeout=2):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError('Store master did not bind')
            time.sleep(2)
    mode = common.get('ATOMESH_VLLM_MODE', 'benchmark')
    spec = dict(plan['speculative_config'])
    if mode == 'smoke':
        spec.pop('synthetic_acceptance_length', None)
        spec['rejection_sample_method'] = 'block'
    for role in ('prefill', 'decode'):
        env = role_env(common, role, ip, plan[role]['env'])
        if role == 'prefill':
            env['MOONCAKE_CONFIG_PATH'] = common['ATOMESH_VLLM_STORE_CONFIG_PATH']
        env["ATOMESH_VLLM_ROLE"] = role
        if mode == 'smoke':
            # /reset_prefix_cache is a development endpoint in this vLLM.
            env['VLLM_SERVER_DEV_MODE'] = '1'
        cfg = json.loads((CONFIG / f'{role}-kv.json').read_text())
        dump(out / f'{role}-kv.json', cfg)
        args = dict(plan['common_args'])
        args.update({'served-model-name': plan['model'], 'port': ports[role], 'host': '0.0.0.0',
                     'enable-prefix-caching': True, 'enable-dbo': role == 'prefill', 'attention-config': plan[role]['attention-config'],
                     'speculative-config': spec, 'kv-transfer-config': cfg})
        if role == 'prefill':
            args.update({'enable-dbo': True, 'dbo-prefill-token-threshold': 16384,
                         'dbo-decode-token-threshold': 32768})
        cmd = ['vllm', 'serve', model_path, *flags(args)]
        recorded_env = {k: env[k] for k in sorted(env)
                        if k in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'GLIBC_TUNABLES', 'PYTHONPATH', 'ATOMESH_VLLM_ROLE',
                                 'ATOMESH_VLLM_MEMORY_DIR', 'ATOMESH_VLLM_STORE_CONFIG_PATH')
                        or k.startswith(('VLLM_', 'MC_', 'MOONCAKE_', 'NCCL_',
                                         'HIP_VISIBLE', 'DEBUG_HIP'))}
        dump(out / f'{role}-launch.json', {'argv': cmd, 'env': recorded_env})
        start(role, cmd, env, logs)
    urls = {role: f'http://{ip}:{ports[role]}' for role in ('prefill', 'decode')}
    try:
        for url in urls.values():
            wait_http(url + '/health', int(os.environ.get('WAIT_SERVER_TIMEOUT', 7200)))
    except RuntimeError:
        # Only Store registration ENOMEM during startup qualifies for memory
        # recovery; ordinary model, transport and correctness failures stay fatal.
        for path in (logs / 'prefill.log', logs / 'decode.log'):
            text = path.read_text(errors='replace') if path.exists() else ''
            if ('Failed to register memory' in text and 'Cannot allocate memory [12]' in text
                    and 'Initialize MooncakeDistributedStore failed' in text):
                memory_failure('rdma_registration_enomem', out)
        raise
    verify_attention_configuration(plan, logs, out)
    proxy_url = f'http://{ip}:{ports["proxy"]}'
    start('proxy', [sys.executable, str(HERE / 'pd_vllm_proxy.py'),
          '--host', '0.0.0.0', '--port', str(ports['proxy']), '--prefill', urls['prefill'],
          str(ports['bootstrap']), '--decode', urls['decode']], common, logs)
    wait_http(proxy_url + '/health', 300)
    dump(out / 'metadata-rank-0.json', {'backend': 'vllm', 'model': os.environ['MODEL_NAME'],
         'model_path': model_path, 'topology': '1p1d', 'display_topology': '1P1D-TP4-TP4',
         'prefill_ips': ip, 'decode_ips': ip, 'prefill_ports': str(ports['prefill']),
         'decode_ports': str(ports['decode']), 'pd_worker_layout': 'single_node'})
    if mode == 'smoke':
        smoke(plan['model'], model_path, urls['prefill'], urls['decode'], proxy_url, out)
        # Smoke is a correctness artifact, not a performance number.
        return
    venv = Path(f'/tmp/m31-agentx-{os.environ.get("SLURM_JOB_ID", "local")}')
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    subprocess.run([str(venv/'bin/pip'), 'install', 'agentx-harness==1.0.12'], check=True)
    conc = int(os.environ['CONC_LIST'])
    duration = int(os.environ.get('AIPERF_BENCHMARK_DURATION', 3600))
    # Pin the HF loader's supported class-level revision inside the client venv.
    # This changes dataset selection only; image packages and replay stay intact.
    loader = next(venv.glob("lib/python*/site-packages/aiperf/dataset/loader/base_hf_dataset.py"))
    loader_source = loader.read_text()
    needle = "hf_revision: str | None = None"
    pinned_line = f"hf_revision: str | None = {plan['benchmark']['dataset_revision']!r}"
    if loader_source.count(needle) == 1:
        loader.write_text(loader_source.replace(needle, pinned_line))
    elif loader_source.count(pinned_line) != 1:
        raise RuntimeError("Unexpected AgentX dataset loader source")
    dump(out / "harness-dataset-pin.json", {"version": "1.0.12", "file": str(loader),
        "change": "Set BaseHFDatasetLoader.hf_revision", "revision": plan['benchmark']['dataset_revision']})
    artifact = results / f'aiperf-m31-1p1d-c{conc}'
    artifact.mkdir(exist_ok=True)
    cmd = [str(venv/'bin/agentx'), 'profile', '--url', proxy_url, '--model', plan['model'],
           '--tokenizer', model_path, '--tokenizer-trust-remote-code', '--concurrency', str(conc),
           '--benchmark-duration', str(duration), '--random-seed', '42', '--server-metrics',
           urls['prefill']+'/metrics', urls['decode']+'/metrics', '--output-artifact-dir', str(artifact),
           '--public-dataset', plan['benchmark']['dataset']]
    dump(out/'benchmark-command.json', cmd)
    # The runner env file declares unset AIPERF_* knobs as empty strings, which
    # AIPerf's settings reject (job 11156); let AgentX apply its own defaults.
    agentx_env = {k: v for k, v in common.items() if not (k.startswith('AIPERF_') and v == '')}
    proc = start('agentx', cmd, agentx_env, logs)
    deadline = time.monotonic() + duration + 3600
    while proc.poll() is None:
        check_services()
        if time.monotonic() > deadline:
            raise TimeoutError('AgentX exceeded measurement plus warmup/setup budget')
        time.sleep(5)
    if proc.returncode:
        raise RuntimeError(f'AgentX failed {proc.returncode}')
    raw = artifact/'profile_export_aiperf.json'
    data = json.loads(raw.read_text())
    dump(out/'measured-input-config.json', data.get('input_config'))
    measured = data.get('input_config', {})
    expected_phase = {"duration": duration, "concurrency": conc,
                      "warmup_requests_per_lane": 10, "grace_period": 30.0,
                      "agentic_warmup_grace_period": 1800.0,
                      "trajectory_start_min_ratio": 0.25, "trajectory_start_max_ratio": 0.75}
    phase = measured.get("phases", [{}])[0]
    for key, value in expected_phase.items():
        if phase.get(key) != value:
            raise RuntimeError(f"Workload drift: {key}={phase.get(key)!r}, expected {value}")
    dataset = measured.get("datasets", [{}])[0]
    if dataset.get("entries") != 393 or dataset.get("random_seed") != 42:
        raise RuntimeError(f"Dataset configuration drift: {dataset}")
    summary_env = dict(common, BACKEND='vllm', AGGREGATED='0', AIPERF_MAX_CONTEXT_LENGTH='1048576')
    filename = f'pd-vllm-{os.environ["MODEL_NAME"]}-1p1d-isl1048576-osl1024-conc{conc}-0.8.json'
    subprocess.run([sys.executable, str(HERE.parent/'aiperf_dashboard.py'), str(raw),
                    str(results/filename), str(conc)], env=summary_env, check=True)
    for role, url in urls.items():
        (out / f'{role}-final.prom').write_text(request(url+'/metrics'))


def smoke_then_sweep(run_phase=None, *, smoke_only=False, concurrencies=(32, 48, 64)):
    """Use one allocation; each phase owns fresh engines and a fresh CPU Store."""
    run_phase = run_phase or main
    keys = ('RUN_DIR', 'CONC_LIST', 'ATOMESH_ENV_ATOMESH_VLLM_MODE')
    saved = {key: os.environ.get(key) for key in keys}
    root = Path(os.environ['RUN_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    phases = []
    try:
        selected = [('smoke', 'smoke', 1)]
        if not smoke_only:
            selected += [(f'c{c}', 'benchmark', c) for c in concurrencies]
        for name, mode, concurrency in selected:
            os.environ.update(RUN_DIR=str(root / name), CONC_LIST=str(concurrency),
                              ATOMESH_ENV_ATOMESH_VLLM_MODE=mode)
            phases.append({'phase': name, 'concurrency': concurrency, 'status': 'running'})
            dump(root / 'suite-status.json', phases)
            print(f'[suite] starting {name} with fresh engines and Store', flush=True)
            try:
                try:
                    run_phase()
                finally:
                    cleanup()
                    PROCESSES.clear()
            except BaseException:
                phases[-1]['status'] = 'failed'
                dump(root / 'suite-status.json', phases)
                raise
            phases[-1]['status'] = 'success'
            dump(root / 'suite-status.json', phases)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    os.environ['ATOMESH_ALLOCATION_RUN_DIR'] = os.environ['RUN_DIR']
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    try:
        mode = os.environ.get('ATOMESH_ENV_ATOMESH_VLLM_MODE')
        if mode in ('smoke', 'smoke_then_sweep'):
            smoke_then_sweep(smoke_only=mode == 'smoke')
        elif mode == 'smoke_then_c32':
            smoke_then_sweep(concurrencies=(32,))
        else:
            main()
    except StoreMemoryUnavailable as exc:
        print(f'[store-memory] {exc}; returning memory retry status 86', flush=True)
        raise SystemExit(86)
    finally:
        cleanup()
