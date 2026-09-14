# GLM-5.2 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against
GLM-5.2-MXFP4 with ATOM on AMD MI355X GPUs. It covers both a standalone TP4
server on 4 GPUs and an optimized, single-node PD-disaggregated deployment on
8 GPUs. Both deployments use:

- `amd/GLM-5.2-MXFP4`
- FP8 KV cache
- MTP with three speculative tokens
- synthetic draft acceptance fixed to the InferenceX reference target
- the SemiAnalysis Weka AgentX workload

Their parallelism and cache configurations differ and are documented in their
respective sections below.

The workload uses the AIPerf scenario `inferencex-agentx-mvp` and public dataset
`semianalysis_cc_traces_weka_062126`. It replays long-context, multi-turn coding
traces with subagent fan-out rather than a fixed ISL/OSL workload.

The optimized single-node PD setup is documented below. For the general
multi-node PD workflow, see
[`mesh/Agentic-GLM-5.2.md`](mesh/Agentic-GLM-5.2.md).

## 1. Start the ATOM Server

### PD Disaggregated Deployment (1P1D)

The optimized topology uses all 8 GPUs of one MI355X node:

```text
AIPerf
  |
  v
atomesh router (:8000)
  |-- Prefill (:8010), GPU 0-3: TP1 × PP4
  `-- Decode  (:8020), GPU 4-7: TP4 × DCP4
```

The validated prefill and decode configurations are:

| Item | Prefill node | Decode node |
|---|---|---|
| GPU count | 4 | 4 |
| Parallelism | TP1 × PP4 | TP4 × DCP4 |
| PP partition | `20,20,20,18` | N/A |
| KV cache | FP8, block size 16 | FP8, block size 16 |
| GPU memory utilization | 0.85 | 0.85 |
| Maximum sequences | 512 | 512 |
| Compilation | Level 3, enforce eager | Level 3 |
| CUDAGraph | Disabled by enforce eager | Full mode, configured sizes up to 256 |
| Batched-token budget | 8192 | 16384 (default) |
| LMCache | 256 GiB CPU tier, 256-token chunks | Disabled |
| Native prefix caching | Enabled | Enabled |
| Speculative decoding | MTP3 | MTP3 |

Both nodes use MXFP4 weights, online PTPC FP8 quantization,
`ATOM_MLA_PAGE_SIZE=1`, `ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047`,
`ATOM_ONLINE_QUANT_STREAMING=0`, and `ATOM_USE_TRITON_MLA=0`. The prefill node
enables `OFFLOAD_PROFILE=1` and `OFFLOAD_MIN_LOAD_TOKENS=0`.

#### Start the PD Deployment

Cold-start the deployment for each concurrency point. Each script below is
self-contained and can be run from a separate terminal. They write `prefill.log`,
`decode.log`, and `mesh.log` in the current directory.

Start prefill and decode first. **Do not start atomesh until both workers are
ready** — otherwise the router may fail health checks or route traffic to servers
that are still loading weights. After each worker starts, wait until its endpoint
responds (model load can take several minutes):

```bash
curl -sf http://127.0.0.1:8010/v1/models   # prefill ready
curl -sf http://127.0.0.1:8020/v1/models   # decode ready
```

If a curl fails, check the corresponding log (`prefill.log` or `decode.log`) and
retry once the server is up.

##### Start Prefill

```bash
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}

rm -rf "${HOME}/.cache/atom/"*

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export ATOM_MLA_PAGE_SIZE=1
export ATOM_ONLINE_QUANT_STREAMING=0
export ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047
export ATOM_USE_TRITON_MLA=0
export MAX_JOBS=16
export ATOM_HOST_IP=127.0.0.1
export LD_LIBRARY_PATH="$(python3 -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

env \
  HIP_VISIBLE_DEVICES=0,1,2,3 \
  VLLM_PP_LAYER_PARTITION=20,20,20,18 \
  LMCACHE_LOCAL_CPU=True \
  LMCACHE_MAX_LOCAL_CPU_SIZE=256 \
  LMCACHE_CHUNK_SIZE=256 \
  OFFLOAD_PROFILE=1 \
  OFFLOAD_MIN_LOAD_TOKENS=0 \
  nohup python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --enable_prefix_caching \
    --online_quant_config \
      '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
    --level 3 \
    --method mtp \
    --num-speculative-tokens 3 \
    --server-port 8010 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 4 \
    --enforce-eager \
    --max-num-batched-tokens 8192 \
    --kv-transfer-config \
      '{"kv_connector":"multi","connectors":[{"kv_connector":"mooncake","kv_role":"kv_producer","proxy_ip":"127.0.0.1","handshake_port":6301,"protocol":"rdma"},{"kv_connector":"lmcache_offload","kv_role":"offload"}]}' \
    >prefill.log 2>&1 &
```

##### Start Decode

```bash
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export ATOM_MLA_PAGE_SIZE=1
export ATOM_ONLINE_QUANT_STREAMING=0
export ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB=2047
export ATOM_USE_TRITON_MLA=0
export MAX_JOBS=16
export ATOM_HOST_IP=127.0.0.1
export LD_LIBRARY_PATH="$(python3 -c \
  'import sysconfig; print(sysconfig.get_path("purelib"))')/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

env \
  HIP_VISIBLE_DEVICES=4,5,6,7 \
  nohup python3 -m atom.entrypoints.openai_server \
    --model "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --enable_prefix_caching \
    --online_quant_config \
      '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
    --level 3 \
    --method mtp \
    --num-speculative-tokens 3 \
    --server-port 8020 \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 4 \
    --cudagraph-mode FULL \
    --cudagraph-capture-sizes \
      '[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]' \
    --kv-transfer-config \
      '{"kv_connector":"mooncake","kv_role":"kv_consumer","proxy_ip":"127.0.0.1","handshake_port":6301,"protocol":"rdma"}' \
    >decode.log 2>&1 &
```

##### Start ATOMesh

Run this only after both prefill (`:8010`) and decode (`:8020`) pass the curl checks
above.

```bash
nohup atomesh launch \
  --host 0.0.0.0 \
  --port 8000 \
  --pd-disaggregation \
  --prefill http://127.0.0.1:8010 6301 \
  --decode http://127.0.0.1:8020 \
  --policy random \
  --backend atom \
  --log-level info \
  --disable-circuit-breaker \
  --prometheus-port 29100 \
  >mesh.log 2>&1 &
```

### PD Mixed Deployment (Standalone)

Start a fresh server for each concurrency point.

The validated standalone configuration is:

| Item | Value |
|---|---|
| Hardware | 4×MI355X (`gfx950`) |
| Model | `amd/GLM-5.2-MXFP4` |
| Parallelism | TP4 |
| KV cache | FP8 |
| Prefix cache | Enabled |
| CPU offload | LMCache, 200 GiB, 256-token chunks |
| Speculative decoding | Native MTP, 3 draft tokens |
| Synthetic acceptance rate | `0.6633` |
| Expected acceptance length | `1 + 3 × 0.6633 = 2.9899` tokens/forward |
| Profiling duration | 3,600 seconds |
| Warmup | 10 additional one-token requests per lane |
| AIPerf | `0.12.0` (`agentx-v1.0.4`) |

#### GLM-5.2 MXFP4 with MTP

```bash
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

# LMCache-related settings
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=200
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MIN_LOAD_TOKENS=8192

export TP=${TP:-4}
export CONC=${CONC:-8}

case "${CONC}" in
  1)  CUDAGRAPH_CAPTURE_SIZES='[1,2]' ;;
  2)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4]' ;;
  4)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8]' ;;
  8)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16]' ;;
  10) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20]' ;;
  12) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24]' ;;
  16) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24,28,32]' ;;
  *)
    echo "Unsupported CONC=${CONC}" >&2
    exit 2
    ;;
esac

python -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --online_quant_config \
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
  --kv-transfer-config \
    '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
  --tensor-parallel-size "${TP}" \
  --max-num-seqs "$((CONC * 2))" \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --num-speculative-tokens 3 \
  --method mtp \
  --spec-decode-acceptance-rate 0.6633 \
  --max-num-batched-tokens 16384 \
  2>&1 | tee "server-glm52-mtp3-synth-c${CONC}.log"
```

##### Synthetic Acceptance Semantics

`--spec-decode-acceptance-rate 0.6633` fixes the mean draft-token acceptance ratio:

```text
accepted draft tokens / total draft tokens ≈ 0.6633
expected tokens per target forward = 1 + 3 × 0.6633 ≈ 2.99
```

The draft model and target verification still run. This override controls which real draft tokens are committed, so performance comparisons do not depend on each engine's measured draft-head quality.

This mode is **performance-only**. Disable `--spec-decode-acceptance-rate` for SWE-bench, GSM8K, or any correctness evaluation because synthetic acceptance does not preserve model accuracy.

##### Use GPU Prefix Caching Without LMCache

To use only the native GPU prefix cache, unset the LMCache-related environment variables before starting the server:

```bash
unset PYTHONHASHSEED
unset LMCACHE_LOCAL_CPU
unset LMCACHE_MAX_LOCAL_CPU_SIZE
unset LMCACHE_CHUNK_SIZE
unset OFFLOAD_MIN_LOAD_TOKENS
```

Also remove the `--kv-transfer-config` argument from the server command.

#### GLM-5.2 MXFP4 Without MTP

```bash
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

# LMCache-related settings
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=200
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MIN_LOAD_TOKENS=8192

export TP=${TP:-4}
export CONC=${CONC:-8}

case "${CONC}" in
  1)  CUDAGRAPH_CAPTURE_SIZES='[1,2]' ;;
  2)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4]' ;;
  4)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8]' ;;
  8)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16]' ;;
  10) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20]' ;;
  12) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24]' ;;
  16) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24,28,32]' ;;
  *)
    echo "Unsupported CONC=${CONC}" >&2
    exit 2
    ;;
esac

python -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8000 \
  --kv_cache_dtype fp8 \
  --online_quant_config \
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
  --kv-transfer-config \
    '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
  --tensor-parallel-size "${TP}" \
  --max-num-seqs "$((CONC * 2))" \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --max-num-batched-tokens 16384 \
  2>&1 | tee "server-glm52-c${CONC}.log"
```

## 2. Run the AgentX Profile

### PD Mixed Deployment (Standalone)

Run this once per concurrency point against a newly started server:

```bash
export CONC=${CONC:-10}
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}
export OUTPUT_DIR=${OUTPUT_DIR:-results/glm52-agentx-c${CONC}}

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "${OUTPUT_DIR}"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "${MODEL_PATH}" \
  --concurrency "${CONC}" \
  --benchmark-duration 3600 \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "${MODEL_PATH}" \
  --tokenizer-trust-remote-code \
  --max-context-length 1048576 \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --server-metrics http://127.0.0.1:8000/metrics \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

### PD Disaggregated Deployment (1P1D)

Use the same AIPerf command. The client URL remains
`http://127.0.0.1:8000`, which is the atomesh endpoint. Replace the
single-server metrics argument with both ATOM endpoints:

```bash
--server-metrics \
  http://127.0.0.1:8010/metrics \
  http://127.0.0.1:8020/metrics
```

## Accuracy

Synthetic acceptance is performance-only. For accuracy evaluation, either use
the non-MTP standalone command or use MTP without
`--spec-decode-acceptance-rate`.

```bash
export MODEL_PATH=${MODEL_PATH:-amd/GLM-5.2-MXFP4}

python3 -m lm_eval \
  --model local-chat-completions \
  --model_args \
    "model=${MODEL_PATH},base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=16384,temperature=0,top_p=1
```

Validated standalone GSM8K 5-shot result:

```text
local-chat-completions ({'model': 'amd/GLM-5.2-MXFP4', 'base_url': 'http://0.0.0.0:8000/v1/chat/completions', 'api_key': 'EMPTY', 'eos_string': '</s>', 'max_retries': 5, 'num_concurrent': 16, 'timeout': 1800, 'tokenized_requests': False, 'max_length': 1048576}), gen_kwargs: ({'max_tokens': 16384, 'temperature': 0, 'top_p': 1}), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9674|±  |0.0049|
|     |       |strict-match    |     5|exact_match|↑  |0.9659|±  |0.0050|
```

Validated GSM8K 5-shot accuracy for the PD-disaggregated deployment above
(full 1319 samples, real MTP3 without synthetic acceptance):

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9613|±  |0.0053|
|     |       |strict-match    |     5|exact_match|↑  |0.9621|±  |0.0053|
```
