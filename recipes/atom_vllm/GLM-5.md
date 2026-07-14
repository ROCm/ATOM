# GLM-5 with ATOM vLLM Plugin Backend

This recipe shows how to run `GLM-5` (including `GLM-5.1` and `GLM-5.2`) models with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

GLM-5 features sparse MLA, and is architecturally similar to DeepSeek-V3.2. Its architecture is exposed through `GlmMoeDsaForCausalLM` to be picked up by ATOM OOT. GLM-5.2 is the pivot version of GLM-5 family that additionally uses IndexShare: `"shared"` layers reuse the preceding `"full"` layer's DSA indexer.

Refer to the [GLM-5.2-FP8 Recipe](#glm-52-fp8-recipe) and [GLM-5.2-MXFP4 Recipe](#glm-52-mxfp4-recipe) for deployment details of the latest GLM-5.2 model.

## Pull the Docker Image
Use the latest image for all the recipes below.
```bash
docker pull rocm/atom-dev:vllm-latest
```

## GLM-5.1-FP8 Recipe
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

## GLM-5.2-FP8 Recipe
GLM-5.2-FP8 is supported on both MI350X and MI300X GPUs. The recipes may vary between the two GPU platforms due to differences in their capabilities. Check recipes below for more details.
### Deployment on MI350X GPUs
```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve zai-org/GLM-5.2-FP8 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "layer_quant_config":{"model.layers.*.mlp.experts":"mxfp8"}, "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}}' \
```
Note:
- Quick allreduce is used to accelerate the communication and flydsl sort enabled for MOE layers.
- FP8 kv cache is used for memory efficiency and performance benefits.
- Online quantization is used to convert the original FP8 weights to kernel-friendly formats, i.e., attention linear weights in PTPC-FP8 and expert weights in MXFP8, enabling optimal hardware performance.
- The MTP feature is also supported. To enable it, add `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` to the command above.

### Deployment on MI300X/MI308X GPUs
```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1

vllm serve zai-org/GLM-5.2-FP8 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate"]}}'
```
Note:
- TP=8 is needed on MI300X GPUs due to the memory limit.
- Online quantization is used to convert both attention and MoE from the original FP8 weights to PTPC-FP8.
- Similarly, add `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` to enable MTP. Add `--max-model-len 16384` in this usecase as well to fit the memory limit.


## GLM-5.2-MXFP4 Recipe
GLM-5.2-MXFP4 is only supported on MI350X GPUs.

### Deployment on MI350X GPUs
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
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}}'
```

To run MTP with MXFP4 model, the `online-quant-config` needs to be updated to quant the bf16 draft layer to PTPC-FP8 for better performance. 
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
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "model.layers.[0-9].mlp.*expert*", "model.layers.[1-6][0-9].mlp.*expert*", "model.layers.7[0-7].mlp.*expert*"]}}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```



## Step 3: Performance Benchmark
Users can use the default vllm bench commands for performance benchmarking.
```bash
ISL=1000
OSL=100
CONC=4
MODEL_PATH=amd/GLM-5.2-MXFP4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model $MODEL_PATH \
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
