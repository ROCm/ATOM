# GLM-4-MoE with ATOM vLLM Plugin Backend

This recipe shows how to run a `GLM-4-MoE` model with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

The checkpoint used here should expose the `Glm4MoeForCausalLM` architecture so it can be picked up by the ATOM OOT model override.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

vllm serve zai-org/GLM-4.7-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --no-enable-prefix-caching
```

### GLM-4.7-FP8 MTP (TP=4/TP=8, MI355X)

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

vllm serve zai-org/GLM-4.7-FP8 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 1 \
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
    --model zai-org/GLM-4.7-FP8 \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 100 \
    --temperature 0.0 \
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

### Optional: Enable Profiling
If you want to collect profiling trace, you can use the same API as default vLLM to add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${your-profiler-dir}")
```

## Step 4: Accuracy Validation

```bash
lm_eval --model local-completions \
        --model_args model=zai-org/GLM-4.7-FP8,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```
