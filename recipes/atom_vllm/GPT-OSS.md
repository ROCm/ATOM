# GPT-OSS with vLLM-ATOM Plugin Backend

This recipe shows how to run `openai/gpt-oss-120b` with the vLLM-ATOM plugin backend. For background on the plugin backend, see [vLLM plugin backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-v0.19.0-nightly_20260517
```

## Step 2: Launch vLLM Server

The vLLM-ATOM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
TP=1
MODEL="openai/gpt-oss-120b"
if [ "${TP}" = "1" ]; then
    GPU_MEM_UTIL_ARGS=""
else
    GPU_MEM_UTIL_ARGS="--gpu-memory-utilization 0.5"
fi

export SAFETENSORS_FAST_GPU=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_CACHE_ROOT=/root/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export GPTOSS_USE_GENERIC_SWIGLU_MXFP4_LAYOUT=1
if [ "${TP}" != "1" ]; then export AITER_QUICK_REDUCE_QUANTIZATION=INT4; fi

rm -rf /root/.cache
vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size "${TP}" \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    ${GPU_MEM_UTIL_ARGS} \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark

```bash
TP=1
ISL=1000
OSL=100
CONC=4
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-$PWD/results_gptoss}/${RUN_ID}"
NAME="${RUN_ID}-gpt-oss-120b-aw-tp${TP}-${ISL}-${OSL}-${CONC}"

mkdir -p "${RESULTS_DIR}"

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model openai/gpt-oss-120b \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --temperature 0.0 \
    --num-prompts "$(( CONC * 10 ))" \
    --max-concurrency "${CONC}" \
    --trust_remote_code \
    --num-warmups "$(( 2 * CONC ))" \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el \
    --result-dir "${RESULTS_DIR}" \
    --result-filename "${NAME}.json"
```
## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=openai/gpt-oss-120b,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
