# GPT-OSS with ATOM vLLM Plugin Backend

This recipe shows how to run `GPT-OSS-120B` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

GPT-OSS-120B is a single-GPU model, so `--tensor-parallel-size` defaults to 1 and can be omitted.

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

vllm serve openai/gpt-oss-120b \
    --host localhost \
    --port 8000 \
    --attention-backend ROCM_AITER_FA \
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
    --model openai/gpt-oss-120b \
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

### Reference Result (TP1, 1K/1K, Concurrency 64)

Latest measured result with the command form above with docker of `docker.io/rocm/atom-dev:vllm-v0.19.0-nightly_20260414` 

| ISL | OSL | Concurrency | Num Prompts | TTFT (ms) | TPOT (ms) | Output Throughput (tok/s) | Total Throughput (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 64 | 640 | 139.96 | 10.70 | 5584.06 | 11339.76 |

More performance details: [ATOM Benchmark Dashboard (ATOM-vLLM / gpt-oss-120b)](https://rocm.github.io/ATOM/benchmark-dashboard/#backend=ATOM-vLLM&model=gpt-oss-120b)

### Optional: Enable Profiling
If you want to collect profiling trace, you can use the same API as default vLLM to add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${your-profiler-dir}")
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=openai/gpt-oss-120b,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

### Reference Accuracy Result (gsm8k, fewshot=3)

```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |Value |Stderr|
|-----|------:|----------------|-----:|-----------|-----:|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|0.4329|0.0136|
|     |       |strict-match    |     3|exact_match|0.2328|0.0116|
```
