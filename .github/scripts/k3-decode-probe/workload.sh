#!/usr/bin/env bash
set -euo pipefail
port="$1"
results="$2/decode-probe"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${results}"
exec > >(tee "${results}/workload.log") 2>&1
client=(python3 -u "${script_dir}/client.py" --port "${port}" --prompts "${results}/prompts.jsonl" --timeout 600)
reset_prefix() {
  python3 - "${port}" <<'PY'
import json
import sys
import time
import urllib.request

url = f'http://127.0.0.1:{sys.argv[1]}/reset_prefix_cache'
for _ in range(60):
    with urllib.request.urlopen(urllib.request.Request(url, method='POST'), timeout=5) as response:
        if json.load(response).get('success'):
            print('D-PROBE local prefix reset succeeded', flush=True)
            break
    time.sleep(1)
else:
    raise SystemExit('D-PROBE prefix cache is still held')
PY
}
echo 'D-PROBE preparing 48 mixed long prefixes'
"${client[@]}" --mode prepare --n 48 --lengths 12289,16385,24577,65537,131073,262145
echo 'D-PROBE fresh local prefill and generation'
"${client[@]}" --mode generate --concurrency 48 --max-tokens 16 --logprobs 5 --out "${results}/fresh.jsonl"
reset_prefix
curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${port}/metrics" > "${results}/final.metrics"
echo 'D-PROBE completed'
