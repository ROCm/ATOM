#!/usr/bin/env bash
set -euo pipefail
port="$1"
results="$2/prefill-probe"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${results}"
exec > >(tee "${results}/workload.log") 2>&1
client=(python3 -u "${script_dir}/client.py" --port "${port}" --prompts "${results}/prompts.jsonl" --timeout 3000)
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
            print('P-PROBE local prefix reset succeeded', flush=True)
            break
    time.sleep(1)
else:
    raise SystemExit('P-PROBE prefix cache is still held')
PY
}
echo 'P-PROBE preparing 64 prefixes through one million tokens'
"${client[@]}" --mode prepare --n 64 --lengths 65537,131073,262145,524289,786433,1044481
echo 'P-PROBE remote prefill with simulated consumer release'
"${client[@]}" --mode prefill --concurrency 64 --logprobs 5 --out "${results}/prefill.jsonl"
reset_prefix
curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${port}/metrics" > "${results}/final.metrics"
echo 'P-PROBE completed'
