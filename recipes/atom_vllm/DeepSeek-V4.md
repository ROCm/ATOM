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
    --kv-cache-dtype fp8 \
    --tensor-parallel-size "${TP}" \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 512 \
    --tokenizer-mode deepseek_v4 \
    --async-scheduling \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

The command above turns on every optimization the ATOM-vLLM DeepSeek-V4 backend supports: the fp8 2-buffer KV cache, cross-request prefix caching, and MTP speculative decoding. Drop the corresponding flag from any feature you do not want.

Notes:
- `--tokenizer-mode deepseek_v4` selects the DeepSeek-V4 tokenizer mode required by the vLLM-ATOM adapter.
- Keep `--max-num-seqs` at or below `512` for this configuration; larger values may OOM.
- The command above serves on port `8001`; update the accuracy command below if you change the port.

Feature flags:
- `--kv-cache-dtype fp8` enables the fp8 2-buffer KV cache (fp8 NoPE pool + a parallel bf16 RoPE pool), which lowers KV-cache memory and raises the token budget / max concurrency. It requires AMD `gfx950`/`gfx1250`; on other GPUs the plugin automatically falls back to a bf16 KV cache, so the flag is safe to leave on. Omit it to force bf16.
- `--speculative-config '{"model":"...","num_speculative_tokens":1}'` enables MTP speculative decoding. The draft is the model's own next-token-prediction layer, so `model` points to the same checkpoint. `num_speculative_tokens` must be `<= num_nextn_predict_layers` (which is `1` for DeepSeek-V4-Flash).

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
