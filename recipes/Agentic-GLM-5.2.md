# GLM-5.2 AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against GLM-5.2-MXFP4 with ATOM on 4×AMD MI355X GPUs. The validated configuration uses:

- `amd/GLM-5.2-MXFP4`
- TP4
- FP8 KV cache
- MTP with four speculative tokens
- forced acceptance length fixed to the current benchmark target
- native GPU prefix caching plus a 512 GiB/rank LMCache CPU tier
- the SemiAnalysis Weka AgentX workload

The workload uses the AIPerf scenario `inferencex-agentx-mvp` and public dataset `semianalysis_cc_traces_weka_062126`. It replays long-context, multi-turn coding traces with subagent fan-out rather than a fixed ISL/OSL workload.

For PD-disaggregated serving, see [`mesh/Agentic-GLM-5.2.md`](mesh/Agentic-GLM-5.2.md). This document covers a single standalone ATOM server.

## Validated Configuration

| Item | Value |
|---|---|
| Hardware | 4×MI355X (`gfx950`) |
| Model | `amd/GLM-5.2-MXFP4` |
| Parallelism | TP4 |
| KV cache | FP8 |
| Prefix cache | Enabled |
| CPU offload | LMCache, 512 GiB per TP rank (2 TiB total), 256-token chunks |
| Speculative decoding | Native MTP, 4 draft tokens |
| Forced acceptance length | `3.33` tokens/forward |
| Profiling duration | 3,600 seconds |
| Warmup | 10 additional one-token requests per lane |
| AIPerf | `0.12.0` (`agentx-v1.0.4`) |

## 1. Start the ATOM Server

Start a fresh server for each concurrency point.

Use TP + MTP for small concurrency (`C2`–`C10`), and TP + DCP for large
concurrency (`C16` and above).

### GLM-5.2 MXFP4 with TP + MTP (small concurrency)

For small-concurrency runs, TP4 + MTP4 uses four speculative tokens and the
current benchmark's forced acceptance length of `3.33`.

```bash
export MODEL_PATH=${MODEL_PATH:-models/GLM-5.2-MXFP4}

export PYTHONNOUSERSITE=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

# TP4 GPU and NUMA placement used by the current MI355X runs.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1,3,5,7}
export ATOM_NUMA_NODE=${ATOM_NUMA_NODE:-0,0,1,1}
export ATOM_NUMA_BIND=1
export ATOM_AUTO_NUMA_BIND=0
export ATOM_CRASH_ON_NUMA_BIND_FAILURE=1

# LMCache-related settings
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_NUMA_MODE=auto
export LMCACHE_MAX_LOCAL_CPU_SIZE=512  # GiB per TP rank; 2 TiB total for TP4
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
  --gpu-memory-utilization 0.95 \
  --kv_cache_dtype fp8 \
  --online_quant_config \
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*"]}' \
  --kv-transfer-config \
    '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
  --tensor-parallel-size "${TP}" \
  --max-num-seqs "$((CONC * 2))" \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --num-speculative-tokens 4 \
  --method mtp \
  --spec-decode-acceptance-length 3.33 \
  --max-num-batched-tokens 16384 \
  2>&1 | tee "server-glm52-mtp4-c${CONC}.log"
```

#### Forced Acceptance Semantics

`--spec-decode-acceptance-length 3.33` fixes the mean acceptance length, the draft model and target verification still run. This override controls which real draft tokens are committed, so performance comparisons do not depend on each engine's measured draft-head quality.

This mode is **performance-only**. Disable
`--spec-decode-acceptance-length` for SWE-bench, GSM8K, or any correctness
evaluation because forced acceptance does not preserve model accuracy.

#### Use GPU Prefix Caching Without LMCache

To use only the native GPU prefix cache, unset the LMCache-related environment variables before starting the server:

```bash
unset PYTHONHASHSEED
unset LMCACHE_LOCAL_CPU
unset LMCACHE_NUMA_MODE
unset LMCACHE_MAX_LOCAL_CPU_SIZE
unset LMCACHE_CHUNK_SIZE
unset OFFLOAD_MIN_LOAD_TOKENS
```

Also remove the `--kv-transfer-config` argument from the server command.


### GLM-5.2 MXFP4 with TP + DCP (large concurrency)

For large-concurrency runs, TP4 + DCP4 reuses the same four GPUs to shard the
decode KV cache and increase the available decode batch capacity; speculative
decoding is disabled.

```bash
export MODEL_PATH=${MODEL_PATH:-models/GLM-5.2-MXFP4}

export PYTHONNOUSERSITE=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

# TP4 GPU and NUMA placement used by the current MI355X runs.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1,3,5,7}
export ATOM_NUMA_NODE=${ATOM_NUMA_NODE:-0,0,1,1}
export ATOM_NUMA_BIND=1
export ATOM_AUTO_NUMA_BIND=0
export ATOM_CRASH_ON_NUMA_BIND_FAILURE=1

# LMCache-related settings
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_NUMA_MODE=auto
export LMCACHE_MAX_LOCAL_CPU_SIZE=512  # GiB per TP rank; 2 TiB total for TP4
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MIN_LOAD_TOKENS=8192

export TP=${TP:-4}
export DCP=${DCP:-4}
export CONC=${CONC:-64}
export ATOM_MLA_PAGE_SIZE=1
export ATOM_DCP_REPLICATE_INDEX_CACHE="${ATOM_DCP_REPLICATE_INDEX_CACHE:-0}"

if (( CONC < 16 )); then
  echo "DCP mode expects large CONC (>=16); got CONC=${CONC}" >&2
  exit 2
fi
CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8'
for ((size=12; size <= CONC * 2; size += 4)); do
  CUDAGRAPH_CAPTURE_SIZES+=",${size}"
done
CUDAGRAPH_CAPTURE_SIZES+=']'

python -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --server-port 8000 \
  --gpu-memory-utilization 0.95 \
  --kv_cache_dtype fp8 \
  --online_quant_config \
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","*expert*"]}' \
  --kv-transfer-config \
    '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
  --tensor-parallel-size "${TP}" \
  --decode-context-parallel-size "${DCP}" \
  --max-num-seqs "$((CONC * 2))" \
  --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --max-num-batched-tokens 16384 \
  2>&1 | tee "server-glm52-dcp${DCP}-tp${TP}-c${CONC}.log"
```

## 2. Run the AgentX Profile

Run this once per concurrency point against a newly started server:

```bash
export CONC=${CONC:-10}
export MODEL_PATH=${MODEL_PATH:-models/GLM-5.2-MXFP4}
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


## Accuracy

Forced acceptance is performance-only. For accuracy evaluation, either use the
non-MTP server command or use MTP without
`--spec-decode-acceptance-length`/`--spec-decode-acceptance-rate`. Then run:

```bash
export MODEL_PATH=${MODEL_PATH:-models/GLM-5.2-MXFP4}

python3 -m lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --output_path ./eval_out-tta1J8 --log_samples --model_args 'model=${MODEL_PATH},base_url=http://0.0.0.0:8015/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=1048576' --gen_kwargs
```

Validated DCP4 GSM8K 20-shot result:

```text
local-chat-completions ({'model': '/shared/data/amd_int/models/GLM-5.2-MXFP4', 'base_url': 'http://0.0.0.0:8015/v1/chat/completions', 'api_key': 'EMPTY', 'eos_string': '</s>', 'max_retries': 5, 'num_concurrent': 64, 'timeout': 1800, 'tokenized_requests': False, 'max_length': 1048576}), gen_kwargs: ({'max_tokens': 16384, 'temperature': 0, 'top_p': 1}), limit: None, num_fewshot: 20, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9689|±  |0.0048|
|     |       |strict-match    |    20|exact_match|↑  |0.9697|±  |0.0047|
```

Validated MTP4+TP4 GSM8K 5-shot result:

```text
local-chat-completions ({'model': '/shared/data/amd_int/models/GLM-5.2-MXFP4', 'base_url': 'http://0.0.0.0:8015/v1/chat/completions', 'api_key': 'EMPTY', 'eos_string': '</s>', 'max_retries': 5, 'num_concurrent': 64, 'timeout': 1800, 'tokenized_requests': False, 'max_length': 1048576}), gen_kwargs: ({'max_tokens': 16384, 'temperature': 0, 'top_p': 1}), limit: None, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9704|±  |0.0047|
|     |       |strict-match    |     5|exact_match|↑  |0.9712|±  |0.0046|
```
