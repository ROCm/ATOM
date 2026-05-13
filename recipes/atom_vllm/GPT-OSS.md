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
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1 \
vllm serve openai/gpt-oss-120b \
    --host localhost \
    --port 8000 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.5 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark
Users can use the default vllm bench commands for performance benchmarking.
```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model openai/gpt-oss-120b \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --percentile-metrics ttft,tpot,itl,e2el
```

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
