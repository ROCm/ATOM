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
echo 'D-PROBE preparing eight long prefixes for forced preemption'
"${client[@]}" --mode prepare --n 8 --lengths 262145
echo 'D-PROBE injecting one KV allocation failure after decode starts'
"${client[@]}" --mode generate --concurrency 8 --max-tokens 2048 --out "${results}/forced.jsonl"
python3 - "${results}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def preemptions(name):
    return sum(float(line.split()[-1]) for line in (root / name).read_text().splitlines()
               if line.startswith(('vllm:num_preemptions_total{', 'vllm:num_preemptions_total ')))
count = preemptions('forced.after.metrics') - preemptions('forced.before.metrics')
assert count > 0, 'No preemption observed'
summary = json.loads((root / 'forced.summary.json').read_text())
assert summary['requests'] == summary['ok'] == 8
report = {'preemptions': count, 'generation': summary,
          'scope': 'One injected allocation failure on D; no PD source ownership'}
(root / 'preempt-summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
PY
reset_prefix
curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:${port}/metrics" > "${results}/final.metrics"
echo 'D-PROBE completed'
