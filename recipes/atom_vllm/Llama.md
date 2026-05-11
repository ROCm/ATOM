# Llama-3.1 (8B + 405B FP8) with ATOM vLLM Plugin Backend

This recipe shows how to run `meta-llama/Llama-3.1-8B-Instruct` and `Meta-Llama-3.1-405B-Instruct-FP8/` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

Both models are gated on Hugging Face. Make sure your environment has access first:

```bash
huggingface-cli login
```

### Llama-3.1-8B-Instruct (TP=1)

```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

### Meta-Llama-3.1-405B-Instruct-FP8/ (TP=8)

```bash
vllm serve meta-llama/Llama-3.1-405B-Instruct-FP8/ \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --load-format safetensors \
    --allow-deprecated-quantization \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark

Users can use the default `vllm bench` command for performance benchmarking:

```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust-remote-code \
    --percentile-metrics ttft,tpot,itl,e2el
```

## Step 4: Accuracy Validation

### Llama-3.1-8B-Instruct

```bash
lm_eval --model local-completions \
        --model_args model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

Measured result (2026-04-03, TP=1, `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`):

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.7551|±  |0.0118|
|     |       |strict-match    |     3|exact_match|↑  |0.6694|±  |0.0130|


### Meta-Llama-3.1-405B-Instruct-FP8/

```bash
lm_eval --model local-completions \
        --model_args model=meta-llama/Llama-3.1-405B-Instruct-FP8/,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

Measured result (2026-04-07, TP=8, plain vLLM baseline with `VLLM_PLUGINS=''`, `--load-format safetensors --allow-deprecated-quantization`):

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  |0.9507|±  |0.0060|
|     |       |strict-match    |     3|exact_match|↑  |0.9158|±  |0.0076|


## CI Alignment Notes

- `Llama-3.1-8B-Instruct` uses TP=1 in nightly accuracy (`RUN_LLAMA8_TP1`) with `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`.
- `Meta-Llama-3.1-405B-Instruct-FP8/` uses TP=8 in nightly accuracy (`RUN_LLAMA405_FP8_TP8`).
