#!/usr/bin/env python3
"""Capture a short diagnostic window in a separate AgentX replay."""
import json
import pathlib
import subprocess
import time
import urllib.request

root = pathlib.Path('/home/yifehuan/data/qwen38_agentx_native_20261016')
run = root/'tp2_k3_c4_profile'
run.mkdir(exist_ok=True)
proc = subprocess.Popen(['/var/tmp/qwen_agentx_client/bin/python',
    '/home/yifehuan/glm5_tileRT/data/run_qwen_agentx.py','--tag',run.name,
    '--concurrency','4','--duration','120','--warmup-requests','1'])
deadline = time.monotonic()+1800
while True:
    log = run/'client.log'
    if log.exists() and 'Phase profiling (profiling) started' in log.read_text():
        break
    if proc.poll() is not None or time.monotonic()>deadline:
        raise RuntimeError('Diagnostic client exited or profiling phase did not start')
    time.sleep(2)

def post(endpoint):
    request = urllib.request.Request('http://localhost:8000/'+endpoint, method='POST')
    with urllib.request.urlopen(request,timeout=300) as response:
        data = response.read().decode()
        (run/(endpoint+'.json')).write_text(data)
        print(data,flush=True)

# Restrict profiling to a short window; formal benchmark is already finished.
time.sleep(20)
post('start_profile')
time.sleep(10)
post('stop_profile')
rc = proc.wait()
print(json.dumps({'client_exit_code':rc, 'diagnostic_only':True}),flush=True)
raise SystemExit(rc)
