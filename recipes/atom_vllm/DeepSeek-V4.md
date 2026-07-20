# DeepSeek-V4 with ATOM vLLM Plugin Backend

This recipe shows how to run `deepseek-ai/DeepSeek-V4-Flash` with the ATOM vLLM plugin backend. For background on the plugin backend, see [ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).


## Step 1: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and general usage flow compatible with upstream vLLM. For general server options and API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

```bash
MODEL=deepseek-ai/DeepSeek-V4-Pro
TP=8

export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1

vllm serve "${MODEL}" \
    --host localhost \
    --port 8001 \
    --dtype auto \
    --tensor-parallel-size "${TP}" \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 512 \
    --tokenizer-mode deepseek_v4 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

Notes:
- `ATOM_USE_TRITON_MOE=1` enables the Triton MoE path used by this configuration.
- `--tokenizer-mode deepseek_v4` selects the DeepSeek-V4 tokenizer mode required by the vLLM-ATOM adapter.
- Keep `--max-num-seqs` at or below `512` for this configuration; larger values may OOM.
- The command above serves on port `8001`; update the accuracy command below if you change the port.
- The profiler writes torch traces to `./vllm_profile`. Remove `--profiler-config` if profiling is not needed.

## Step 3: Performance Benchmark

Users can use the default vLLM bench command for performance benchmarking.

```bash
vllm bench serve \
    --backend vllm \
    --base-url http://127.0.0.1:8001 \
    --endpoint /v1/completions \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 100 \
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

## Step 4: Accuracy Validation

The accuracy can be verified on the GSM8K dataset with `lm_eval`:

```bash
lm_eval \
  --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Flash,base_url=http://localhost:8001/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
```
