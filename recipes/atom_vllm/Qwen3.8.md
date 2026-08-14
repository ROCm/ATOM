# Qwen3.8 with vLLM-ATOM Plugin Backend

This recipe shows how to run `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` with the vLLM-ATOM plugin backend. For background on the plugin backend, see [vLLM plugin backend](../../docs/vllm_plugin_backend_guide.md). For how this configuration was arrived at, see [Qwen3.8 bring-up findings](../../docs/qwen3_8_vllm_plugin_bringup.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The vLLM-ATOM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### Qwen3.8-2.4T-A95B-Quark-MXFP4 (TP=8, MI355X)

```bash
TP=8
export VLLM_ROCM_USE_AITER=1
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export ATOM_USE_CUSTOM_ALL_GATHER=0
export ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=0
export GATED_DELTA_RULE_TRITON_AUTOTUNE=1

vllm serve amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size "${TP}" \
    --kv-cache-dtype fp8 \
    --async-scheduling \
    --trust-remote-code \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --gpu-memory-utilization 0.9 \
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
    --model amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 \
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
        --model_args model=amd/Qwen3.8-2.4T-A95B-Quark-MXFP4,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
        --tasks gsm8k \
        --num_fewshot 5 \
        --gen_kwargs max_gen_toks=2048
```
