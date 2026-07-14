# GLM-5 with ATOM vLLM Plugin Backend

This recipe shows how to run `GLM-5.1` and `GLM-5.2` models with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

GLM-5 features sparse MLA, and is architecturally similar to DeepSeek-V3.2. Its architecture is exposed through `GlmMoeDsaForCausalLM` to be picked up by ATOM OOT. GLM-5.2 additionally uses IndexShare: `"shared"` layers reuse the preceding `"full"` layer's DSA indexer.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

vllm serve zai-org/GLM-5.1-FP8 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching
```

### GLM-5.2-MXFP4 Nightly Recipe

The nightly OOT accuracy recipe uses TP4 with FP8 KV cache and 20-shot GSM8K, matching the IndexShare coverage in CI:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve amd/GLM-5.2-MXFP4 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### GLM-5.2-MXFP4 MTP3 Recipe
```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve amd/GLM-5.2-MXFP4 \
    --host localhost \
    --port 8004 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 16384 \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "model.layers.[0-9].mlp.*expert*", "model.layers.[1-6][0-9].mlp.*expert*", "model.layers.7[0-7].mlp.*expert*"]}}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```



## Step 3: Performance Benchmark
Users can use the default vllm bench commands for performance benchmarking.
```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model zai-org/GLM-5.1-FP8 \
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

### Optional: Enable Profiling
If you want to collect profiling trace, you can use the same API as default vLLM to add `--profiler-config "$profiler_config"` to the `vllm serve` command above.

```bash
profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":true,"torch_profiler_record_shapes":true}' \
    "${your-profiler-dir}")
```

## Step 4: Accuracy Validation

The sparse MLA mechanism contains an indexer that selects the top-k tokens it deems most relevant for each query from the KV cache. For GLM-5, the top-2048 tokens are selected from the context by the indexer. To evaluate its accuracy, it is recommended to use requests with context longer than 2048 so that the indexer can be tested. In `lm_eval`, this can be set by increasing the `num_fewshot=20` to increase the context length.


```bash
MODEL_PATH=amd/GLM-5.2-MXFP4

lm_eval --model local-completions \
        --model_args model="${MODEL_PATH}",base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 20
```
