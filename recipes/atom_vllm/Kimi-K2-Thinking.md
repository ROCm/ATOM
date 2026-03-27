# Kimi-K2-Thinking with ATOM vLLM Plugin Backend

This recipe shows how to run `Kimi-K2-Thinking` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

This model uses remote code, so the launch command keeps `--trust-remote-code`.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```


## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
# Disbale moe dual stream path. TODO: fix hang issue when captuing much more cuda graph. 
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0

vllm serve amd/Kimi-K2-Thinking-MXFP4 \
    --host localhost \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark
Users can use the default vllm bench command for performance benchmarking.
```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model amd/Kimi-K2-Thinking-MXFP4 \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust_remote_code \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=amd/Kimi-K2-Thinking-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
