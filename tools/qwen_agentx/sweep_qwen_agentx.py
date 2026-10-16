#!/usr/bin/env python3
"""Sequential fresh-server baseline points on the already authorized GPU job."""
import json
import os
import pathlib
import signal
import subprocess
import time

root = pathlib.Path('/home/yifehuan/data/qwen38_agentx_native_20261016')
scripts = pathlib.Path('/home/yifehuan/glm5_tileRT/data')
client = '/var/tmp/qwen_agentx_client/bin/python'
previous = 'tp2_k3_c1_formal'
while True:
    path = root/previous/'manifest.json'
    if path.exists() and 'exit_code' in json.loads(path.read_text()):
        break
    time.sleep(10)

def stop_server(tag):
    pid = int((root/tag/'server.pid').read_text())
    try:
        command = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return
    if b'atom.entrypoints.openai_server' not in command:
        raise RuntimeError(f'PID {pid} no longer points to our ATOM server')
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+180
    while pathlib.Path(f'/proc/{pid}/cmdline').exists():
        command = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
        if not command:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Graceful server shutdown incomplete; inspect processes')
        time.sleep(2)

def audit(tag):
    run = root/tag
    subprocess.run([client, str(scripts/'summarize_qwen_agentx.py'), str(run)], check=True,
                   stdout=(run/'summary.log').open('w'))
    subprocess.run([client, str(scripts/'inferencex_agentx_reference/validate_agentic_result.py'),
                    str(run/'aiperf_artifacts'), '--failed-request-threshold', '.1'], check=True)
    export = json.loads((run/'aiperf_artifacts/profile_export_aiperf.json').read_text())
    if not export['metadata'].get('submission_valid'):
        raise RuntimeError(f'{tag} failed scenario validation: {export["metadata"]}')
    print(json.dumps({'point_completed': tag, 'requests':export['request_count'],
                      'throughput':export['output_token_throughput']}), flush=True)

audit(previous)
for tp, k, concurrency, tag in [(8,3,1,'tp8_k3_c1_formal'),
                               (8,3,4,'tp8_k3_c4_formal'),
                               (2,3,4,'tp2_k3_c4_formal')]:
    stop_server(previous)
    server = subprocess.Popen(['bash', str(scripts/'start_qwen_agentx_native.sh'),
                                str(tp),str(k),tag])
    print(json.dumps({'starting':tag,'server_pid':server.pid}),flush=True)
    result = subprocess.run([client,str(scripts/'run_qwen_agentx.py'),'--tag',tag,
                             '--concurrency',str(concurrency),'--duration','900'])
    if result.returncode:
        raise RuntimeError(f'Client failed for {tag}; server retained for diagnosis')
    audit(tag)
    previous = tag
print('Formal sweep complete; final server retained for isolated profiler.',flush=True)
