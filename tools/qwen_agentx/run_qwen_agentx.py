#!/usr/bin/env python3
"""Run the pinned AgentX protocol and preserve independent server counters."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--concurrency', type=int, required=True)
p.add_argument('--duration', type=int, default=900)
p.add_argument('--warmup-requests', type=int)
args = p.parse_args()
root = pathlib.Path('/home/yifehuan/data/qwen38_agentx_native_20261016')
run = root / args.tag
run.mkdir(parents=True, exist_ok=True)
checkpoint = pathlib.Path('/shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8')
url = 'http://localhost:8000'

def get(endpoint):
    with urllib.request.urlopen(url + endpoint, timeout=30) as response:
        return response.read().decode()

deadline = time.monotonic() + 1800
while True:
    try:
        models = json.loads(get('/v1/models'))
        break
    except Exception:
        if time.monotonic() > deadline:
            raise RuntimeError('Server not ready after 1800s; inspect server process and log')
        time.sleep(5)
model = models['data'][0]['id']
cli = '/var/tmp/qwen_agentx_client/bin/agentx'
cmd = [cli, 'profile', '--url', url, '--model', model,
       '--tokenizer', str(checkpoint), '--tokenizer-trust-remote-code',
       '--public-dataset', 'semianalysis_cc_traces_weka_062126_256k',
       '--concurrency', str(args.concurrency), '--benchmark-duration', str(args.duration),
       '--max-context-length', '262144', '--server-metrics', url + '/metrics',
       '--output-artifact-dir', str(run / 'aiperf_artifacts')]
if args.duration < 900:
    cmd += ['--unsafe-override']
if args.warmup_requests is not None:
    cmd += ['--warmup-requests-per-lane', str(args.warmup_requests)]

manifest = {'argv': cmd, 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'agentx_harness': importlib.metadata.version('agentx-harness'),
            'inferencex_revision': '77a21c0aa849d1bd3cdcb23ee33554ebeb4d88e4',
            'corpus_revision': '8fecd2fc56694469f758f0afbbb6335ad3043740',
            'native_prefix_caching': False, 'discovery': args.duration < 900,
            'models': models, 'checkpoint': str(checkpoint), 'repositories': {}}
for repo in ['atom', 'aiter']:
    path = '/home/yifehuan/glm5_tileRT/' + repo
    manifest['repositories'][repo] = {
        'head': subprocess.check_output(['git', '-c', 'safe.directory=' + path, '-C', path, 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git', '-c', 'safe.directory=' + path, '-C', path, 'status', '--porcelain'], text=True)}
manifest['checkpoint_metadata_sha256'] = {
    name: hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()
    for name in ['config.json', 'generation_config.json', 'model.safetensors.index.json']
    if (checkpoint/name).exists()}
corpus = pathlib.Path('/root/.cache/huggingface/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots') / manifest['corpus_revision'] / 'traces.jsonl'
manifest['corpus_sha256'] = hashlib.file_digest(corpus.open('rb'), 'sha256').hexdigest()
(run/'manifest.json').write_text(json.dumps(manifest, indent=2))
stop = threading.Event()

def snapshot(label):
    for endpoint, name in [('/metrics', 'metrics.prom'), ('/debug/mtp_stats', 'mtp.json'),
                           ('/debug/cache_stats', 'cache.json')]:
        try:
            (run/f'{label}.{name}').write_text(get(endpoint))
        except Exception as exc:
            (run/f'{label}.{name}.error').write_text(repr(exc))

def monitor():
    while not stop.wait(30):
        snapshot('sample_' + str(int(time.time())))

snapshot('before')
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
print(json.dumps({'run': str(run), 'argv': cmd}), flush=True)
with (run/'client.log').open('w') as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
stop.set()
thread.join(timeout=35)
snapshot('after')
manifest['exit_code'] = result.returncode
manifest['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
(run/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps({'exit_code': result.returncode, 'run': str(run)}), flush=True)
raise SystemExit(result.returncode)
