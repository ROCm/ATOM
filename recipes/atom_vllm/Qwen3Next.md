# Qwen3-Next with ATOM vLLM Plugin Backend

This recipe shows how to run `Qwen3-Next-80B-A3B-Instruct-FP8` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Qwen3-Next-80B-A3B-Instruct-FP8 (TP=1, MI355X)

```bash
export ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0

vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --max-model-len 16384 \
    --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching
```

**Important**: `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1` is required for Qwen3-Next because it uses a hybrid architecture with both linear attention (GatedDeltaNet) and full attention layers. This env var ensures full attention layers use vLLM's default implementation.

## Step 3: Performance Benchmark

Users can use the default vllm bench commands for performance benchmarking.

```bash
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --dataset-name random \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --random-range-ratio 0.8 \
    --max-concurrency 64 \
    --num-prompts 640 \
    --trust_remote_code \
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
        --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

## Key Environment Variables

- `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`: **Required** - disables ATOM attention plugin to use vLLM's implementation for full attention layers

## Architecture Notes

Qwen3-Next uses a hybrid architecture combining:
- **Linear Attention**: GatedDeltaNet layers for efficient long-context modeling
- **Full Attention**: Standard multi-head attention layers for enhanced accuracy

The model alternates between these layer types, requiring careful handling of both attention mechanisms.


