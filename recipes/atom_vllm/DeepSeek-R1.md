# DeepSeek-R1 with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-R1-0528` or `amd/DeepSeek-R1-0528-MXFP4` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, users can refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Deepseek with FP8
Users can use this command to launch server on AMD Instinct MI300X, MI325X and MI355X platforms.

```bash
vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu_memory_utilization 0.9 \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching 
```

### Deepseek with MXFP4
AMD Instinct MI355X GPU support MXFP4 computation instruction and users can use the following command to launch server on MI355X platform. For MXFP4 model weight, we suggest using the model weight quantized from AMD Quark.

```bash
vllm serve amd/DeepSeek-R1-0528-MXFP4 \
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
    --model amd/DeepSeek-R1-0528-MXFP4 \
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
        --model_args model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
