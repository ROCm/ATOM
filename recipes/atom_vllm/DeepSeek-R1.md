# DeepSeek-R1 with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-R1-0528` or `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, users can refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

### DeepSeek-R1 FP8

Users can use this command to launch the server on AMD Instinct MI300X, MI325X, and MI355X platforms.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

vllm serve deepseek-ai/DeepSeek-R1-0528 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

### DeepSeek-R1 MXFP4

AMD Instinct MI355X supports MXFP4 computation. For MXFP4 weights, use the AMD Quark-converted checkpoint that matches the nightly vLLM-ATOM benchmark setup.

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

vllm serve amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host localhost \
    --port 8000 \
    --async-scheduling \
    --load-format fastsafetensors \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching
```

## Step 3: Performance Benchmark

Users can use the default `vllm bench` command for performance benchmarking.

```bash
ISL=1000
OSL=100
CONC=4

vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --model amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --dataset-name random \
    --random-input-len "${ISL}" \
    --random-output-len "${OSL}" \
    --random-range-ratio 0.0 \
    --num-prompts "$(( CONC * 8 ))" \
    --max-concurrency "${CONC}" \
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
        --model_args model=amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True \
        --tasks gsm8k \
        --num_fewshot 3
```
