#!/usr/bin/env bash
# One complete original c512 workload, called twice on one GPU allocation.
set -euo pipefail
: "${TRIAL:?}" "${EXPECTED_AITER_SHA:?}" "${EXPECTED_IMAGE_ID:?}" "${PINNED_IMAGE:?}"
out="eplb-diagnostics/$TRIAL"
mkdir -p "$out"
image_id=$(docker inspect --format '{{.Image}}' atom-eplb)
test "$image_id" = "$EXPECTED_IMAGE_ID"
docker image inspect "$PINNED_IMAGE" --format '{{json .RepoDigests}}' > "$out/repo-digests.json"
docker exec -i -e EXPECTED_AITER_SHA -e EXPECTED_ATOM_SHA="$(git rev-parse HEAD)" \
  -e TRIAL atom-eplb python3 - <<'PY'
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import torch
from atom.benchmarks.results.metadata import installed_package

assert Path(importlib.util.find_spec('atom').origin).resolve().parent == Path('/workspace/atom')
sha = subprocess.check_output(['git', '-c', 'safe.directory=/workspace', 'rev-parse', 'HEAD'], text=True).strip()
assert sha == os.environ['EXPECTED_ATOM_SHA'], sha
aiter = installed_package('amd-aiter')
assert aiter['sha'] == os.environ['EXPECTED_AITER_SHA'], aiter
marker = Path('/models/deepseek-ai/DeepSeek-V4-Pro/.hf-revision').read_text().strip()
assert marker == 'b5968e9190ef611bbf34a7229255be88a0e937c1', marker
packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
data = dict(atom_sha=sha, aiter=aiter, model_revision=marker, torch=torch.__version__,
            hip=torch.version.hip, nccl=torch.cuda.nccl.version(), packages=packages,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            devices=[dict(name=torch.cuda.get_device_name(i), pci=str(torch.cuda.get_device_properties(i)))
                     for i in range(torch.cuda.device_count())])
Path('eplb-diagnostics', os.environ['TRIAL'], 'runtime.json').write_text(json.dumps(data, indent=2))
print('Verified ATOM', sha, 'AITER', aiter['sha'], 'model', marker)
print('torch', data['torch'], 'HIP', data['hip'], 'NCCL/RCCL', data['nccl'])
PY
docker exec atom-eplb python3 .github/scripts/sample_eplb_host.py "/workspace/$out/host.jsonl" &
sampler=$!
finish() {
  local code=$?
  trap - EXIT
  docker exec atom-eplb pkill -TERM -f '[s]ample_eplb_host.py' || true
  wait "$sampler" || true
  docker cp atom-eplb:/tmp/atom_server.log "$out/server.log" || true
  docker cp atom-eplb:/tmp/atom_client.log "$out/client.log" || true
  echo "$code" > "$out/exit-code.txt"
  exit "$code"
}
trap finish EXIT
bash .github/scripts/collect_gpu_info.sh atom-eplb docker linux-atom-pitt-mi350x-8
docker exec -i atom-eplb bash -l <<'SH'
.github/scripts/atom_test.sh launch /models/deepseek-ai/DeepSeek-V4-Pro \
  --kv_cache_dtype fp8 -tp 8 --index_cache_dtype fp4 --no-enable_prefix_caching \
  --enable-dp-attention --enable-expert-parallel --eplb-enable --moe-backend mega \
  --eplb-config '{"num_redundant_experts":0,"placement_policy":"naive","rebalance_interval":200,"load_window_size":200,"rebalance_min_balancedness":2.0}'
SH
docker exec \
  -e RESULT_FILENAME="eplb-$TRIAL" -e DOCKER_IMAGE="$PINNED_IMAGE" \
  -e ATOM_IMAGE_DIGEST="$image_id" -e GPU_NAME="AMD Instinct MI355X" -e ROCM_VERSION=7.2.4 \
  -e GITHUB_REPOSITORY -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT \
  -e GITHUB_JOB -e GITHUB_SHA -e GITHUB_WORKFLOW_REF \
  atom-eplb bash -lc '.github/scripts/atom_test.sh benchmark /models/deepseek-ai/DeepSeek-V4-Pro'
docker exec -i -e TRIAL atom-eplb python3 - <<'PY'
import json, os
from pathlib import Path
trial = os.environ['TRIAL']
r = json.loads(Path(f'eplb-{trial}.json').read_text())
assert r['completed'] == r['num_prompts'] == 5120, r.get('completed')
print('Validated 5120/5120 requests', r.get('total_token_throughput'), r.get('mean_tpot_ms'))
PY
docker exec atom-eplb python3 -m atom.benchmarks.results summary-package \
  "benchmark-bundles/eplb-$TRIAL" --output "$out/summary"
