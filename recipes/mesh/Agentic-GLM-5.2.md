# GLM-5.2 Agentic Dataset Benchmark with PD Disaggregation and LMCache Offload

This recipe shows how to run the SemiAnalysis/Weka agentic replay workload
against GLM-5.2-MXFP4 in ATOM PD-disaggregated mode while enabling LMCache KV
cache offload on the prefill node.

The setup follows the agentic PD workflow introduced by
[ROCm/ATOM PR #1586](https://github.com/ROCm/ATOM/pull/1586) and the nightly CI
structure from [ROCm/ATOM PR #1623](https://github.com/ROCm/ATOM/pull/1623).
The prefill node uses the `multi` KV connector:

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
    |       GLM-5.2-MXFP4, TP8
    |       Mooncake producer + LMCache CPU offload
    |
    +--> Decode node (:8020)
            GLM-5.2-MXFP4, TP8
            Mooncake consumer
```

The commands below use one 8-GPU prefill node and one 8-GPU decode node. Both
nodes must expose the same model path. Adjust paths, IPs, and ports for your
deployment.

## Prerequisites

- An ATOM container with ATOM, atomesh, Mooncake, and LMCache available.
- RDMA connectivity between the prefill and decode nodes for Mooncake KV
  transfer.
- `GLM-5.2-MXFP4` available at the same path on both nodes.
- Enough host memory for the configured 200 GiB LMCache CPU tier.
- AIPerf installed from the pinned SemiAnalysis-compatible commit.
- `PYTHONHASHSEED=0` set consistently on all processes using LMCache prefix
  hashing.

Reference material:

- [PD disaggregation guide](../pd_disaggregation_guide.md)
- [LMCache CPU/NVMe KV cache offload](../../atom/kv_transfer/offload/README.md)
- [MiniMax-M3 agentic PD recipe](Agentic-MiniMax-M3.md)

## Common Environment

Run these settings inside the ATOM container on both service nodes:

```bash
export MODEL_PATH=${MODEL_PATH:-/mnt/models/GLM-5.2-MXFP4}

# If ATOM and AITER are editable source trees rather than installed packages:
# export PYTHONPATH=/workspace/aiter:/workspace/ATOM
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export AITER_LOG_LEVEL=WARNING

# Avoid stale compiled kernels from previous experiments.
rm -rf "${HOME}/.cache/atom/"* 2>/dev/null || true
```

## Step 1: Start the Prefill Node

The prefill node uses all eight local GPUs. It is both a Mooncake producer and
an LMCache offload user.

```bash
export LOG_PATH=${LOG_PATH:-/workspace/logs/glm52_prefill_8010.log}
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=200
export LMCACHE_CHUNK_SIZE=256

# Optional local NVMe tier:
# export LMCACHE_LOCAL_DISK=/nvme/lmcache
# export LMCACHE_MAX_LOCAL_DISK_SIZE=2000

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8010 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --kv_cache_dtype fp8 \
  --block-size 16 \
  --max-model-len 1048576 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 131072 \
  --attn-prefill-chunk-size 131072 \
  --long-prefill-token-threshold 131072 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
  --kv-transfer-config '{"kv_connector":"multi","connectors":[{"kv_connector":"mooncake","kv_role":"kv_producer","handshake_port":6301},{"kv_connector":"lmcache_offload","kv_role":"offload"}]}' \
  2>&1 | tee "${LOG_PATH}"
```

Notes:

- The 131072-token prefill budget and chunk size are intended for the 1M-context
  Weka workload.
- Native HBM prefix caching remains enabled. LMCache acts as the lower CPU
  offload tier when reusable blocks are evicted from HBM.
- `LMCACHE_CHUNK_SIZE=256` is an integer multiple of `--block-size 16`.

## Step 2: Start the Decode Node

Run this command on a separate 8-GPU node:

```bash
export LOG_PATH=${LOG_PATH:-/workspace/logs/glm52_decode_8020.log}
mkdir -p "$(dirname "${LOG_PATH}")"
: > "${LOG_PATH}"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_JOBS=16

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8020 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --kv_cache_dtype fp8 \
  --block-size 16 \
  --max-model-len 1048576 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 262144 \
  --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
  2>&1 | tee "${LOG_PATH}"
```

The decode node does not need LMCache offload for this benchmark.

## Step 3: Verify KV Transfer Information

Before starting the router, verify that both endpoints report their expected
roles:

```bash
curl -sS http://<PREFILL_IP>:8010/kv_transfer_info | python3 -m json.tool
# Expect the Mooncake producer and prefill endpoint.

curl -sS http://<DECODE_IP>:8020/kv_transfer_info | python3 -m json.tool
# Expect kv_role=kv_consumer.
```

Also confirm that the model is loaded by checking GPU memory utilization on
both nodes:

```bash
rocm-smi --showmemuse
```

## Step 4: Start the atomesh Router

Start the router on a host that can reach both ATOM servers:

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

atomesh launch \
  --host 0.0.0.0 --port 8030 \
  --pd-disaggregation \
  --prefill "http://${PREFILL_IP}:8010" \
  --decode "http://${DECODE_IP}:8020" \
  --policy random \
  --backend atom \
  --log-level info \
  --disable-health-check \
  --disable-circuit-breaker \
  2>&1 | tee /workspace/logs/atomesh_glm52_pd_lmcache_router.log
```

Send all benchmark traffic to `http://<router-ip>:8030`.

## Step 5: Smoke Test

Run a small request through the router before launching AIPerf:

```bash
curl -sS -X POST http://127.0.0.1:8030/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/mnt/models/GLM-5.2-MXFP4",
    "messages": [{"role": "user", "content": "Write one sentence about KV cache reuse."}],
    "max_completion_tokens": 32,
    "temperature": 0
  }'
```

## Step 6: Prepare AIPerf

Prepare an isolated AIPerf environment if one is not already available:

```bash
export AIPERF_DIR=${AIPERF_DIR:-/workspace/codes/aiperf}
export AIPERF_VENV=${AIPERF_VENV:-/workspace/venvs/aiperf-sa}
export SA_AIPERF_COMMIT=${SA_AIPERF_COMMIT:-0d2aa0572ac685943d38c580675c4a61023581d3}

mkdir -p "$(dirname "${AIPERF_DIR}")" "$(dirname "${AIPERF_VENV}")"
if [[ ! -d "${AIPERF_DIR}/.git" ]]; then
  git clone https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_DIR}"
fi

git -C "${AIPERF_DIR}" fetch \
  https://github.com/SemiAnalysisAI/aiperf.git "${SA_AIPERF_COMMIT}"
git -C "${AIPERF_DIR}" checkout --detach "${SA_AIPERF_COMMIT}"

rm -rf "${AIPERF_VENV}"
python3 -m venv "${AIPERF_VENV}"
"${AIPERF_VENV}/bin/python" -m pip install --upgrade pip
"${AIPERF_VENV}/bin/python" -m pip install -e "${AIPERF_DIR}"
"${AIPERF_VENV}/bin/aiperf" --version
```

## Step 7: Run the Agentic Dataset Benchmark

The defaults below reproduce the 1-hour InferenceX-aligned GLM-5.2 workload.
Set `CONCURRENCY` to one of the nightly points: `1`, `2`, `4`, `16`, or `32`.

```bash
export AIPERF_VENV=${AIPERF_VENV:-/workspace/venvs/aiperf-sa}
export SERVER_URL=${SERVER_URL:-http://127.0.0.1:8030}
export ENDPOINT=${ENDPOINT:-/v1/chat/completions}
export MODEL_PATH=${MODEL_PATH:-/mnt/models/GLM-5.2-MXFP4}
export PUBLIC_DATASET=${PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126}

export CONCURRENCY=${CONCURRENCY:-1}
export BENCHMARK_DURATION=${BENCHMARK_DURATION:-3600}
export AGENTIC_CACHE_WARMUP_DURATION=${AGENTIC_CACHE_WARMUP_DURATION:-600}
export WARMUP_GRACE_PERIOD=${WARMUP_GRACE_PERIOD:-1800}
export MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-1048576}
export NUM_DATASET_ENTRIES=${NUM_DATASET_ENTRIES:-393}
export TRAJECTORY_START_MIN_RATIO=${TRAJECTORY_START_MIN_RATIO:-0.25}
export TRAJECTORY_START_MAX_RATIO=${TRAJECTORY_START_MAX_RATIO:-0.75}
export FAILED_REQUEST_THRESHOLD=${FAILED_REQUEST_THRESHOLD:-0.10}
export RANDOM_SEED=${RANDOM_SEED:-42}
export SLICE_DURATION=${SLICE_DURATION:-1.0}

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300

export OUTPUT_BASE=${OUTPUT_BASE:-/workspace/results/aiperf_glm52_pd_lmcache_1m}
export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_BASE}/c${CONCURRENCY}-$(date +%Y%m%d_%H%M%S)}

SERVER_METRICS_ARGS=()
if [[ -n "${PREFILL_IP:-}" || -n "${DECODE_IP:-}" ]]; then
  SERVER_METRICS_ARGS+=(--server-metrics)
  [[ -n "${PREFILL_IP:-}" ]] && \
    SERVER_METRICS_ARGS+=("http://${PREFILL_IP}:8010/metrics")
  [[ -n "${DECODE_IP:-}" ]] && \
    SERVER_METRICS_ARGS+=("http://${DECODE_IP}:8020/metrics")
fi

mkdir -p "${OUTPUT_DIR}"
"${AIPERF_VENV}/bin/aiperf" profile \
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
  --warmup-grace-period "${WARMUP_GRACE_PERIOD}" \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "${MODEL_PATH}" \
  --tokenizer-trust-remote-code \
  --max-context-length "${MAX_CONTEXT_LENGTH}" \
  --num-dataset-entries "${NUM_DATASET_ENTRIES}" \
  --slice-duration "${SLICE_DURATION}" \
  "${SERVER_METRICS_ARGS[@]}" \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset "${PUBLIC_DATASET}" \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

Do not use `--unsafe-override` for the 3600-second performance run. It is only
appropriate for short smoke tests below the scenario's minimum valid duration.

## Nightly Concurrency Sweep

The nightly performance suite measures these points independently:

```text
1, 2, 4, 16, 32
```

Run Step 7 once for each value. Restart the prefill server, decode server, and
router before every point so each measurement begins with fresh HBM and LMCache
state. The repository's nightly configuration models these as five separate
jobs for the same reason.

For a short functional smoke test, use:

```bash
export CONCURRENCY=1
export BENCHMARK_DURATION=20
export AGENTIC_CACHE_WARMUP_DURATION=1
export NUM_DATASET_ENTRIES=39
export TRAJECTORY_START_MAX_RATIO=0.25

# Add --unsafe-override to the AIPerf command only for this smoke run.
```

## Validation Checklist

- Prefill logs show a `multi` connector containing both the Mooncake producer
  and `lmcache_offload`.
- Prefill logs show `LMCACHE_LOCAL_CPU=True`, a 200 GiB CPU tier, and
  `LMCACHE_CHUNK_SIZE=256`.
- Decode logs show `kv_role=kv_consumer`.
- Router logs show requests routed through PD disaggregation.
- AIPerf reports `submission_valid=true` for the 3600-second run.
- AIPerf reports the `semianalysis_cc_traces_weka_062126` dataset, 393 entries,
  a `0.25-0.75` trajectory range, and zero unsafe overrides.
- Artifacts include `profile_export_aiperf.json`, `profile_export_aiperf.csv`,
  the raw JSONL records, and `aiperf.log`.
- If server metrics are reachable, the export contains series from both the
  prefill and decode `/metrics` endpoints.

## Troubleshooting

### Scheduler assertion on producer + offload

The prefill node is both a P/D producer and an LMCache reload consumer. LMCache
reload completion must use the offload-specific `finished_loading` and
`failed_loading` states, not P/D consumer receive states. Mixing the semantics
can trigger:

```text
Only consumer should update recving KV status
```

This behavior is addressed by PR #1586.

### Long-context warmup does not drain

The 1M Weka corpus can leave long requests in flight after cache warmup. Keep
`--warmup-grace-period 1800`; this is a maximum wait and exits early when all
warmup requests drain. If failures persist, inspect request errors before
increasing the timeout.

### Dataset configuration timeout

Loading and reconstructing all 393 Weka trajectories can take several minutes.
Keep both configuration timeout variables at 1800 seconds:

```bash
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
```

### CPU offload capacity

Increase `LMCACHE_MAX_LOCAL_CPU_SIZE` if LMCache reports memory pressure or
stores fewer chunks than expected. Make sure the host has enough free memory
before increasing the value.

### Server metrics are unavailable

Prometheus metrics are optional for request-level AIPerf results. If the client
cannot reach either `/metrics` endpoint, leave `PREFILL_IP` and `DECODE_IP`
unset so `--server-metrics` is omitted, then validate network routing
separately.

### Hash consistency

Set `PYTHONHASHSEED=0` on all participating processes. Inconsistent prefix
hashing can make lookup misses look like offload failures.

### Mooncake shared libraries

If Mooncake native libraries cannot be loaded, add the package and ROCm library
directories to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$(python3 -c \
  "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

### Releasing resources

Stop all benchmark processes after a run:

```bash
pkill -f 'atom.entrypoints.openai_server' || true
pkill -f 'atomesh launch' || true
pkill -f 'aiperf profile' || true
rocm-smi
```
