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
# use quick allreduce to reduce TTFT (may impact accuracy)
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

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
Use the ATOM benchmark guide aligned command form (`vllm` backend + `/v1/completions` endpoint + `temperature=0.0`) for reproducible results.
```bash
vllm bench serve \
    --backend vllm \
    --base-url http://localhost:8000 \
    --endpoint /v1/completions \
    --model amd/Kimi-K2-Thinking-MXFP4 \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --temperature 0.0 \
    --request-rate inf \
    --ignore-eos \
    --max-concurrency 64 \
    --num-prompts 640 \
    --num-warmups 128 \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=amd/Kimi-K2-Thinking-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
