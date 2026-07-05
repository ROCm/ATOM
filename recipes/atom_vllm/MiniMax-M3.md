# MiniMax-M3 with ATOM vLLM Plugin Backend

This recipe shows how to run MiniMax-M3 sparse checkpoints with the ATOM vLLM
plugin backend. For background on the plugin backend, see
[ATOM vLLM Plugin Backend](../../docs/vllm_plugin_backend_guide.md).

MiniMax-M3 uses the ATOM-owned model implementation and vLLM attention adapters
for both dense and sparse attention layers.

## Step 1: Pull the OOT Docker

```bash
docker pull rocm/atom-dev:vllm-latest
```

## Step 2: Launch vLLM Server

The ATOM vLLM plugin backend keeps the standard vLLM CLI, server APIs, and
general usage flow compatible with upstream vLLM. For general server options and
API usage, refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

The example below serves the MXFP8 checkpoint on four GPUs. Use your local
checkpoint path or the corresponding model id for `MODEL`.

```bash
MODEL=/path/to/MiniMax-M3-MXFP8
TP=4
PORT=8001

vllm serve "${MODEL}" \
    --dtype auto \
    --load-format auto \
    --host localhost \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --no-async-scheduling \
    --kv-cache-dtype auto \
    --no-enable-prefix-caching \
    --language-model-only \
    --no-trust-remote-code \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}}' \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

For the MXFP4 checkpoint, change `MODEL` and omit the MXFP8 online quantization
config:

```bash
MODEL=/path/to/MiniMax-M3-MXFP4

vllm serve "${MODEL}" \
    --dtype auto \
    --load-format auto \
    --host localhost \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --block-size 128 \
    --no-async-scheduling \
    --kv-cache-dtype auto \
    --no-enable-prefix-caching \
    --language-model-only \
    --no-trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

To validate FP8 KV cache, set `--kv-cache-dtype fp8` in either command.

Notes:
- Keep `--block-size 128`; MiniMax-M3 sparse attention assumes 128-token sparse
  blocks.
- `--no-trust-remote-code` is expected because ATOM registers the MiniMax-M3
  model classes used by the vLLM plugin path.
- `--language-model-only` serves the language model path for MiniMax-M3 VL
  checkpoints.

## Step 3: Accuracy Validation

The accuracy can be verified on GSM8K with the chat-completions API:

```bash
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=${MODEL},base_url=http://localhost:${PORT}/v1/chat/completions,num_concurrent=32,max_gen_toks=2048" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn
```

Reference average results from five local GSM8K runs are shown below.

| Config | `flexible-extract` avg | `strict-match` avg |
| --- | ---: | ---: |
| MIXFP8 | 0.9503 | 0.9510 |
| MIXFP4 | 0.9399 | 0.9407 |
| MIXFP8-kv_fp8 | 0.9480 | 0.9487 |
| MIXFP4-kv_fp8 | 0.9439 | 0.9445 |
