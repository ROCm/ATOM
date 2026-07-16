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
VLLM_ROCM_USE_AITER=1 \
VLLM_USE_V2_MODEL_RUNNER=1 \
vllm serve openai/gpt-oss-120b \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
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
    --model openai/gpt-oss-120b \
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
lm_eval --model local-chat-completions --apply_chat_template \
        --model_args model=openai/gpt-oss-120b,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=65,max_retries=3,max_gen_toks=2048,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 3
```

Here is the reference value:
```bash
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     3|exact_match|↑  | 0.88|±  |0.0088|
|     |       |strict-match    |     3|exact_match|↑  | 0.31|±  |0.0128|
```
