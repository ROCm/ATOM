# GLM-5 with ATOM vLLM Plugin Backend

This recipe shows how to run a `GLM-5` model with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

GLM-5 features sparse MLA, and is architecturally similar to DeepSeek-V3.2. Its architecture is exposed through `GlmMoeDsaForCausalLM` to be picked up by ATOM OOT.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
vllm serve zai-org/GLM-5-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
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
    --model zai-org/GLM-5-FP8 \
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

The sparse MLA mechanism contains an indexer that selects the top-k tokens it deems most relevant for each query from the KV cache. For GLM-5, the top-2048 tokens are selected from the context by the indexer. To evaluate its accuracy, it is recommended to use requests with context longer than 2048 so that the indexer can be tested. In `lm_eval`, this can be set by increasing the `num_fewshot=20` to increase the context length.


```bash
lm_eval --model local-completions \
        --model_args model=zai-org/GLM-5-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3 \
        --tasks gsm8k \
        --num_fewshot 20
```
