# DeepSeek-V3.2 with vLLM-ATOM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V3.2` with the vLLM-ATOM plugin backend. For background on the plugin backend, see [vLLM plugin backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The vLLM-ATOM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
TP=4
if [ "${TP}" = "4" ]; then
    KV_CACHE_DTYPE=auto
else
    KV_CACHE_DTYPE=fp8
fi

vllm serve deepseek-ai/DeepSeek-V3.2 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark

Users can use the default vllm bench commands for performance benchmarking.

```bash
vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model deepseek-ai/DeepSeek-V3.2 \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 100 \
    --max-concurrency 4 \
    --num-prompts 40 \
    --trust_remote_code \
    --num-warmups 8 \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```
