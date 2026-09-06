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
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
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
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}}' \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

For the MXFP4 checkpoint, change `MODEL` and omit the MXFP8 online quantization
config:

```bash
MODEL=/path/to/MiniMax-M3-MXFP4
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
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
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
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

## Step 4: EAGLE3 Speculative Decoding

MiniMax-M3 sparse serving supports an EAGLE3 draft model
(`Inferact/MiniMax-M3-EAGLE3`, a 1-layer MHA Llama drafter that shares the
target's embedding and `lm_head`). Attach it with `--speculative-config`:

```bash
MODEL=/path/to/MiniMax-M3-MXFP8
DRAFT=/path/to/MiniMax-M3-EAGLE3
TP=8
PORT=8900
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# MiniMaxM3Sparse defaults to vLLM's V2 model runner on ROCm; ATOM's EAGLE3
# integration targets the V1 runner, so force V1 (env only, no source edit).
export VLLM_USE_V2_MODEL_RUNNER=0
vllm serve "${MODEL}" \
    --served-model-name minimax-m3 \
    --host localhost \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 128 \
    --block-size 128 \
    --no-async-scheduling \
    --kv-cache-dtype auto \
    --no-enable-prefix-caching \
    --language-model-only \
    --no-trust-remote-code \
    --enforce-eager \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}}' \
    --speculative-config '{"method": "eagle3", "model": "'"${DRAFT}"'", "num_speculative_tokens": 3}'
```

Notes:
- **Force the V1 model runner** (`VLLM_USE_V2_MODEL_RUNNER=0`). On ROCm,
  MiniMaxM3Sparse defaults to the V2 runner, which bypasses ATOM's EAGLE3
  patches — the draft then runs with a batch bug and acceptance collapses to
  ~`1/concurrency`. V1 restores normal acceptance.
- **`--enforce-eager` is expected with speculative decoding.** The M3 sparse
  attention backends declare `UNIFORM_SINGLE_TOKEN_DECODE` CUDAGraph support,
  but spec-verify runs a multi-token (`num_spec + 1`) query, so vLLM disables
  CUDAGraph and runs eager regardless of `cudagraph_mode`. This is backend-
  driven, not a regression.
- The speculative path is **lossless**: because verify accepts a draft token
  only when it matches the target's own argmax, GSM8K accuracy with the draft
  attached equals the no-draft baseline.

### Speculative decoding results

GSM8K (5-shot, chat-completions, concurrency 16, MXFP8, TP=8, MI355X):

| Config | `flexible-extract` | `strict-match` | accept rate | accepted len / step | draft toks / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| no draft (baseline) | 0.9515 | 0.9522 | — | — | — |
| EAGLE3 (`num_speculative_tokens=3`) | 0.9515 | 0.9522 | 0.588 | 2.76 | 3.0 |

Accuracy is identical to the baseline (lossless), acceptance is ~59% (mean
accepted length ~2.76), and the drafter emits exactly 3 tokens per step.
