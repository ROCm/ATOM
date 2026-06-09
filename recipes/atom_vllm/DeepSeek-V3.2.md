# DeepSeek-V3.2 with vLLM-ATOM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V3.2` with the vLLM-ATOM plugin backend. For background on the plugin backend, see [vLLM plugin backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The vLLM-ATOM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
TP=4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0

vllm serve deepseek-ai/DeepSeek-V3.2 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --no-enable-prefix-caching
```

### DeepSeek-V3.2 MTP (TP=4/TP=8, MTP=1/MTP=3, MI355X)

```bash
TP=4
MTP=3

vllm serve deepseek-ai/DeepSeek-V3.2 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP}}" \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching
```

### DeepSeek-V3.2 PTPC (TP=4, MI355X)

```bash
TP=4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0

vllm serve amd/DeepSeek-V3.2-mtp-ptpc \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --no-enable-prefix-caching
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
    --model deepseek-ai/DeepSeek-V3.2 \
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

```bash
lm_eval --model local-completions \
        --model_args model=deepseek-ai/DeepSeek-V3.2,base_url=http://localhost:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 20
```