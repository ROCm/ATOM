# Kimi-K2-Thinking with ATOM vLLM Plugin Backend

This recipe shows how to run `Kimi-K2-Thinking` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

This model uses remote code, so the launch command keeps `--trust-remote-code`.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```


## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).
We adopt [amd/Kimi-K2-Thinking-MXFP4-AttnFP8](https://huggingface.co/amd/Kimi-K2-Thinking-MXFP4-AttnFP8) for better performance by leveraging FP8 weights for attention layers.

```bash
# use quick allreduce to reduce TTFT
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
TP=4

vllm serve amd/Kimi-K2-Thinking-MXFP4-AttnFP8 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size "${TP}" \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark
Users can use the default vllm bench command for performance benchmarking.
```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model amd/Kimi-K2-Thinking-MXFP4-AttnFP8 \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0.0 \
    --max-concurrency "${CONC}" \
    --num-prompts "$(( CONC * 8 ))" \
    --trust_remote_code \
    --num-warmups "${CONC}" \
    --request-rate inf \
    --ignore-eos \
    --disable-tqdm \
    --save-result \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=amd/Kimi-K2-Thinking-MXFP4-AttnFP8,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
The reference values of corresponding metrics:
```bash
local-completions ({'model': 'amd/Kimi-K2-Thinking-MXFP4-AttnFP8', 'base_url': 'http://localhost:8000/v1/completions', 'num_concurrent': 16, 'max_retries': 3, 'tokenized_requests': False}), gen_kwargs: ({}), limit: None, num_fewshot: 3, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9340|±  |0.0068|
|     |       |strict-match    |     3|exact_match|↑  |0.9325|±  |0.0069|
```