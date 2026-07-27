# Agentic Dataset Benchmark with PD Disaggregation and LMCache Offload

This recipe shows how to run the SemiAnalysis/Weka agentic replay workload
against ATOM in PD-disaggregated mode while enabling LMCache KV cache offload on
the prefill node.

The setup is based on the workflow from
[ROCm/ATOM PR #1586](https://github.com/ROCm/ATOM/pull/1586). The important
configuration is that the prefill node uses the `multi` KV connector:

- `mooncake` with `kv_role=kv_producer` sends prefill KV to the decode node.
- `lmcache_offload` with `kv_role=offload` stores and reloads reusable prompt KV
  on the prefill node.

The decode node remains a normal Mooncake consumer.

## Topology

```text
Client / AIPerf
    |
    v
atomesh router (:8030)
    |
    +--> Prefill node (:8010)
    |       ATOM native backend
    |       Mooncake producer + LMCache offload
    |
    +--> Decode node (:8020)
            ATOM native backend
            Mooncake consumer
```

The commands below use MiniMax-M3-MXFP4 with TP=4 on each server:

- Prefill node: GPUs `0,1,2,3`
- Decode node: GPUs `4,5,6,7` for single-node testing, or `0,1,2,3` on a
  separate decode node

Adjust paths, IPs, GPU lists, and ports for your deployment.

## Prerequisites

- ATOM container with ATOM, atomesh, Mooncake, and LMCache available. Reference recipes include [pd_disaggregation_guide.md](https://github.com/ROCm/ATOM/blob/main/recipes/pd_disaggregation_guide.md) and [LMCache CPU/NVMe KV Cache Offload (ATOM standalone)](https://github.com/ROCm/ATOM/tree/main/atom/kv_transfer/offload#how-to-run).
- RDMA connectivity between prefill and decode nodes for Mooncake KV transfer.
- MiniMax-M3-MXFP4 model available at the same path on both servers.
- AIPerf installed from a SemiAnalysis-compatible commit.
- `PYTHONHASHSEED=0` set consistently when using LMCache prefix hashing.

For a host-side AIPerf setup helper, see `scripts/setup_sa_aiperf_venv.sh`.

## Common Environment

Run this inside the ATOM container on each server process host.

```bash
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

# Avoid stale compiled kernels from previous experiments.
rm -rf /root/.cache/atom/* 2>/dev/null || true
```

## Step 1: Start the Prefill Node

The prefill node is both a Mooncake producer and an LMCache offload user. LMCache
is configured through `LMCACHE_*` environment variables.

```bash
export LOG_PATH=${LOG_PATH:-/workspace/logs/minimax_m3_pd_lmcache_prefill.log}
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"

export HIP_VISIBLE_DEVICES=0,1,2,3
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=200
export LMCACHE_CHUNK_SIZE=256

# Optional local NVMe tier:
# export LMCACHE_LOCAL_DISK=/nvme/lmcache
# export LMCACHE_MAX_LOCAL_DISK_SIZE=2000

python -m atom.entrypoints.openai_server \
  --model /mnt/models/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 \
  --server-port 8010 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 65536 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache":true,"index_topk_freq":4}' \
  --kv-transfer-config '{"kv_connector":"multi","connectors":[{"kv_connector":"mooncake","kv_role":"kv_producer","handshake_port":6301},{"kv_connector":"lmcache_offload","kv_role":"offload"}]}' \
  2>&1 | tee "${LOG_PATH}"
```

Notes:

- `--no-enable_prefix_caching` keeps the benchmark focused on PD transfer and
  LMCache offload reuse rather than ATOM's native HBM prefix cache.
- `LMCACHE_MAX_LOCAL_CPU_SIZE` controls the CPU offload tier size in GiB.
- `LMCACHE_CHUNK_SIZE` must be compatible with `--block-size`.

## Step 2: Start the Decode Node

The decode node is a normal Mooncake KV consumer. It does not need LMCache
offload for this benchmark.

For single-node testing, use GPUs `4,5,6,7`. For multi-node testing, run the same
command on the decode node and use GPUs `0,1,2,3`.

```bash
export LOG_PATH=${LOG_PATH:-/workspace/logs/minimax_m3_pd_lmcache_decode.log}
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"

export HIP_VISIBLE_DEVICES=4,5,6,7
export MAX_CONTEXT_LENGTH=262144

python -m atom.entrypoints.openai_server \
  --model /mnt/models/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 \
  --server-port 8020 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 262144 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","vision_tower","multi_modal_projector","patch_merge_mlp","*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache":true,"index_topk_freq":4}' \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
  2>&1 | tee "${LOG_PATH}"
```

## Step 3: Verify KV Transfer Info

Before starting the router, check that both endpoints report the expected roles.

```bash
curl -s http://<PREFILL_IP>:8010/kv_transfer_info | python3 -m json.tool
# Expect Mooncake producer information and the prefill endpoint.

curl -s http://<DECODE_IP>:8020/kv_transfer_info | python3 -m json.tool
# Expect kv_role=kv_consumer for the decode endpoint.
```

For single-node testing, `PREFILL_IP` and `DECODE_IP` can be the same node IP.

## Step 4: Start the atomesh Router

Start the router on a node that can reach both ATOM servers.

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

atomesh launch \
  --host 0.0.0.0 --port 8030 \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:8010" \
  --decode  "http://${DECODE_IP}:8020" \
  --policy random \
  --backend atom \
  --log-level info \
  --disable-health-check \
  --disable-circuit-breaker \
  2>&1 | tee /workspace/logs/atomesh_pd_lmcache_router.log
```

Send all benchmark traffic to the router at `http://<router-ip>:8030`.

## Step 5: Smoke Test

Run a small chat request through the router before launching AIPerf.

```bash
curl -sS -X POST http://127.0.0.1:8030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/mnt/models/MiniMax-M3-MXFP4",
    "messages": [{"role": "user", "content": "Write one sentence about KV cache reuse."}],
    "max_completion_tokens": 32,
    "temperature": 0
  }'
```

## Step 6: Prepare AIPerf

If AIPerf is not already installed in the container, prepare an isolated venv.

```bash
export AIPERF_DIR=${AIPERF_DIR:-/workspace/codes/aiperf}
export AIPERF_VENV=${AIPERF_VENV:-/workspace/venvs/aiperf-sa}
export SA_AIPERF_COMMIT=${SA_AIPERF_COMMIT:-0d2aa0572ac685943d38c580675c4a61023581d3}

mkdir -p "$(dirname "${AIPERF_DIR}")" "$(dirname "${AIPERF_VENV}")"
if [[ ! -d "${AIPERF_DIR}/.git" ]]; then
  git clone https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_DIR}"
fi

cd "${AIPERF_DIR}"
git fetch https://github.com/SemiAnalysisAI/aiperf.git "${SA_AIPERF_COMMIT}"
git checkout --detach "${SA_AIPERF_COMMIT}"
python3 -m venv "${AIPERF_VENV}"
"${AIPERF_VENV}/bin/python" -m pip install --upgrade pip
"${AIPERF_VENV}/bin/python" -m pip install -e .
"${AIPERF_VENV}/bin/aiperf" --version

"${AIPERF_VENV}/bin/python" - <<'PY'
from aiperf.plugin.enums import PublicDatasetType
names = [
    item.value
    for item in PublicDatasetType
    if "semianalysis" in item.value.lower() or "weka" in item.value.lower()
]
print("SemiAnalysis/Weka datasets:", names)
PY
```

## Step 7: Run the Agentic Dataset Benchmark

The smoke defaults below use a short benchmark duration. Increase
`BENCHMARK_DURATION`, `AGENTIC_CACHE_WARMUP_DURATION`, and `NUM_DATASET_ENTRIES`
for a full run.

```bash
export AIPERF_VENV=${AIPERF_VENV:-/workspace/venvs/aiperf-sa}
export SERVER_URL=${SERVER_URL:-http://127.0.0.1:8030}
export ENDPOINT=${ENDPOINT:-/v1/chat/completions}
export MODEL_PATH=${MODEL_PATH:-/mnt/models/MiniMax-M3-MXFP4}
export TOKENIZER=${TOKENIZER:-${MODEL_PATH}}
export PUBLIC_DATASET=${PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126_256k}
export OUTPUT_DIR=${OUTPUT_DIR:-/workspace/results/aiperf_m3_pd_lmcache_256k/smoke}

export CONCURRENCY=${CONCURRENCY:-1}
export BENCHMARK_DURATION=${BENCHMARK_DURATION:-20}
export AGENTIC_CACHE_WARMUP_DURATION=${AGENTIC_CACHE_WARMUP_DURATION:-1}
export MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-262144}
export NUM_DATASET_ENTRIES=${NUM_DATASET_ENTRIES:-39}
export TRAJECTORY_START_MIN_RATIO=${TRAJECTORY_START_MIN_RATIO:-0.25}
export TRAJECTORY_START_MAX_RATIO=${TRAJECTORY_START_MAX_RATIO:-0.25}
export FAILED_REQUEST_THRESHOLD=${FAILED_REQUEST_THRESHOLD:-0.50}
export RANDOM_SEED=${RANDOM_SEED:-42}
export SLICE_DURATION=${SLICE_DURATION:-1.0}

mkdir -p "${OUTPUT_DIR}"
"${AIPERF_VENV}/bin/aiperf" profile \
  --unsafe-override \
  --scenario inferencex-agentx-mvp \
  --url "${SERVER_URL}" \
  --endpoint "${ENDPOINT}" \
  --endpoint-type chat \
  --streaming \
  --model "${MODEL_PATH}" \
  --concurrency "${CONCURRENCY}" \
  --benchmark-duration "${BENCHMARK_DURATION}" \
  --random-seed "${RANDOM_SEED}" \
  --failed-request-threshold "${FAILED_REQUEST_THRESHOLD}" \
  --trajectory-start-min-ratio "${TRAJECTORY_START_MIN_RATIO}" \
  --trajectory-start-max-ratio "${TRAJECTORY_START_MAX_RATIO}" \
  --agentic-cache-warmup-duration "${AGENTIC_CACHE_WARMUP_DURATION}" \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "${TOKENIZER}" \
  --tokenizer-trust-remote-code \
  --max-context-length "${MAX_CONTEXT_LENGTH}" \
  --num-dataset-entries "${NUM_DATASET_ENTRIES}" \
  --slice-duration "${SLICE_DURATION}" \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset "${PUBLIC_DATASET}" \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

For a full SA-style run, reuse the command above with these environment
overrides:

```bash
export BENCHMARK_DURATION=1800
export AGENTIC_CACHE_WARMUP_DURATION=600
export CONCURRENCY=1
export NUM_DATASET_ENTRIES=393
export MAX_CONTEXT_LENGTH=262144
export OUTPUT_DIR=/workspace/results/aiperf_m3_pd_lmcache_256k/full
```

## Validation Checklist

- Prefill logs show the `multi` connector with both Mooncake producer and
  `lmcache_offload`.
- Prefill logs show LMCache configuration, including `LMCACHE_LOCAL_CPU=True`
  and `LMCACHE_CHUNK_SIZE=256`.
- Decode logs show `kv_role=kv_consumer`.
- Router logs show requests routed through PD disaggregation.
- AIPerf log reports warmup/profiling progress and exports artifacts under
  `OUTPUT_DIR`.
- For offload visibility, run with `OFFLOAD_PROFILE=1` on the prefill node and
  look for `[OFFLOAD-SAVE-PROF]` and `[OFFLOAD-LOAD-PROF]` lines.

## Troubleshooting

### Scheduler assertion on producer + offload

In a PD + LMCache setup, the prefill node is a P/D producer and can also reload
KV from LMCache. LMCache reload completion must use the offload-specific
`finished_loading` / `failed_loading` completion states, not the P/D consumer
`finished_recving` / `failed_recving` states. Mixing these semantics can trigger
the scheduler assertion:

```text
Only consumer should update recving KV status
```

This is the issue addressed by PR #1586.

### CPU offload capacity

Increase `LMCACHE_MAX_LOCAL_CPU_SIZE` if LMCache logs memory pressure or stores
fewer chunks than expected. Optional NVMe capacity can be added with
`LMCACHE_LOCAL_DISK` and `LMCACHE_MAX_LOCAL_DISK_SIZE`.

### Hash consistency

Set `PYTHONHASHSEED=0` on all participating processes. Inconsistent prefix
hashing can make lookup misses look like offload failures.

### Mooncake shared libraries

Most ATOM containers already configure Mooncake's shared library path. If the
server fails to import or load Mooncake native libraries, add the Mooncake
package directory and ROCm libraries to `LD_LIBRARY_PATH` before starting the
servers:

```bash
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

### Releasing GPU resources

Stop the ATOM servers, router, and AIPerf client after the run:

```bash
pkill -f 'atom.entrypoints.openai_server' || true
pkill -f 'atomesh launch' || true
pkill -f 'aiperf profile' || true
rocm-smi
```
