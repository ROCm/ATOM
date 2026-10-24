# MiniMax-M3 / MiniMax-M3.1 MXFP4/MXFP8 Usage Guide

[MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) and [MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) are supported by the native ATOM OpenAI-compatible server path, as is MiniMax-M3.1-Flash with its DSpark
draft, text and image input, from either the release checkpoint or the AMD Quark MXFP4 export
([MiniMax-M3.1-Flash](#minimax-m31-flash-on-4xmi355-gpus)).

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:latest
```

## MXFP4 on 4xMI355 GPUs

### Launching Server

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' 2>&1 | tee "${run_name}-server.log"
```

## MXFP8 on 4xMI355 GPUs

### Launching Server

For the MXFP8 model, online quant is used to convert the linear weights in attention module and first 3 dense MLP layers to PTPC FP8 format, which are originally equipped with 1*32 block scale.
The MoE weights keep unchanged. Check **--online_quant_config** in the script below for more details.

```bash
model_path=${model_path:-MiniMaxAI/MiniMax-M3-MXFP8}
run_name=${run_name:-m3-mxfp8}
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --max-num-batched-tokens 32768 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' 2>&1 | tee "${run_name}-server.log"
```


### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee "${run_name}-bs65-accuracy.log"
```

Validated MXFP4 GSM8K result:

```text
local-chat-completions ({'model': 'amd/MiniMax-M3-MXFP4', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9363|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9371|±  |0.0067|
```

Validated MXFP8 GSM8K result:

```text
local-chat-completions ({'model': 'MiniMaxAI/MiniMax-M3-MXFP8', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9484|±  |0.0061|
|     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061|
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and
latency:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency="${CONC}" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

Reference MXFP4 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 73.27 | 260.77 | 791.33 | 7.50 | 8.33 | 502.35 | 4515.86 |
| 8 | 80 | 85.64 | 295.52 | 1144.91 | 8.78 | 9.29 | 864.87 | 7693.44 |
| 16 | 160 | 114.35 | 383.04 | 2200.03 | 11.73 | 12.84 | 1280.47 | 11555.95 |
| 32 | 320 | 163.86 | 512.32 | 4477.16 | 16.74 | 19.12 | 1807.32 | 16161.65 |
| 64 | 640 | 242.49 | 831.98 | 8566.28 | 25.00 | 29.83 | 2432.75 | 21928.25 |

Reference MXFP8 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 82.00 | 268.02 | 564.13 | 8.43 | 8.66 | 448.82 | 4034.60 |
| 8 | 80 | 103.52 | 323.33 | 1284.59 | 10.67 | 11.31 | 715.51 | 6364.77 |
| 16 | 160 | 143.25 | 414.95 | 2411.41 | 14.80 | 16.44 | 1022.17 | 9224.81 |
| 32 | 320 | 208.34 | 565.02 | 4936.02 | 21.42 | 24.16 | 1421.47 | 12711.25 |
| 64 | 640 | 305.81 | 893.93 | 9610.43 | 31.69 | 37.31 | 1929.04 | 17387.94 |

## EAGLE3 Speculative Decoding

EAGLE3 runs a small single-layer draft model alongside the MiniMax-M3 target to
propose multiple tokens per step, which the target then verifies. It is lossless
with respect to the target's greedy output. The draft checkpoint is
[`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3).
Enable it by adding three flags to any of the server commands above:

- `--method eagle3`
- `--draft-model Inferact/MiniMax-M3-EAGLE3`
- `--num-speculative-tokens 3`

### Launching Server

The following starts the MXFP4 target with the EAGLE3 draft on 4xMI355 (the FP4
server command above plus the three speculative-decoding flags):

```bash
model_path=amd/MiniMax-M3-MXFP4
draft_path=Inferact/MiniMax-M3-EAGLE3

export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --kv_cache_dtype fp8 \
  --max-num-batched-tokens 32768 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --method eagle3 \
  --draft-model "$draft_path" \
  --num-speculative-tokens 3 2>&1 | tee m3-mxfp4-eagle3-server.log
```

### Accuracy Test

Run GSM8K 5-shot with `lm_eval` (identical to the non-speculative test):

```bash
model_path=amd/MiniMax-M3-MXFP4
model_path=MiniMaxAI/MiniMax-M3-MXFP8
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee m3-mxfp4-eagle3-bs65-accuracy.log
```

Validated MXFP4+EAGLE GSM8K result:

```text
| Case | ATOM Commit | GSM8K flexible-extract | GSM8K strict-match | Accept ratio | Avg toks/fwd | Accepted / Total Draft |
|---|---:|---:|---:|---:|---:|---:|
| `fp4_eagle_tp4` | `9fc48338` | `0.9469 ± 0.0062` | `0.9477 ± 0.0061` | `73.36%` | `3.20` | `90229 / 123000` |

MiniMax-M3 Eagle accepted tokens distribution:
`{0: 14.40%, 1: 12.00%, 2: 12.73%, 3: 60.87%}`
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency="${CONC}" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --use-chat-template \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

Reference MXFP4 EAGLE3 results from our run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 43.38 | 287.09 | 755.46 | 4.27 | 7.78 | 850.53 | 7653.56 |
| 8 | 80 | 59.31 | 343.81 | 1516.38 | 5.93 | 10.85 | 1251.08 | 11146.00 |
| 16 | 160 | 78.17 | 430.34 | 2680.95 | 7.91 | 15.58 | 1876.30 | 16928.43 |
| 32 | 320 | 125.69 | 609.24 | 5304.23 | 12.60 | 23.81 | 2355.93 | 21132.49 |
| 64 | 640 | 198.58 | 966.20 | 10476.78 | 19.97 | 40.44 | 2973.94 | 26857.80 |

## MiniMax-M3.1-Flash on 4xMI355 GPUs

MiniMax-M3.1-Flash keeps the MiniMax-M3 backbone (every M3 tensor exists in M3.1 with the
same name and shape) with three model changes: lightning-indexer sparse attention in
**all 60 layers** (M3: layers 0–2 are dense), RoPE θ 10M, and a 1M-token context. Its
speculative-decoding draft is a separate **DSpark** block drafter shipped in the
checkpoint's `dspark/` sub-directory.

This recipe runs the AMD Quark MXFP4 export
[amd/MiniMax-M3.1-Flash-Quark-MXFP4](https://huggingface.co/amd/MiniMax-M3.1-Flash-Quark-MXFP4),
which ships the routed experts as MXFP4.

| Weights | Quark MXFP4 export | At runtime (recipe below) |
|---|---|---|
| Routed experts (57 layers × 128) | MXFP4 (E2M1, E8M0 scale per 32) | as shipped |
| Shared expert | MXFP8 (FP8 E4M3, E8M0 scale per 32) | re-quantized to MXFP4 and fused into the MoE (`ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1`) |
| Attention q/k/v/o, indexer, dense MLP (layers 0–2) | MXFP8 (Quark `per_block`, `block_size [1, 32]`) | per-token / per-channel FP8 (`ptpc_fp8` online quant), as for MiniMax-M3 |
| Vision tower, projector | BF16 | as shipped |
| DSpark draft (`dspark/`) | MXFP8 linears, BF16 rest | as shipped |

### What runs

- **PTPC FP8 attention.** The same `--online_quant_config` as the MiniMax-M3 recipes above
  re-quantizes attention and the three dense MLPs from MXFP8 to PTPC FP8 at start-up
  (`126 layers online-quantized` in the log: 60 layers × qkv/o plus 3 dense layers ×
  gate_up/down). Drop it to keep attention in native MXFP8.
- **Fused shared expert (FSE).** The checkpoint's shared expert is MXFP8 and cannot share the
  MXFP4 MoE as is. With `ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1` (off by default) ATOM stages it at
  load, re-quantizes it MXFP8 → MXFP4 and writes it into the fused slot, so the MoE runs as the
  tuned 129-expert / top-5 MXFP4 kernel (no `no tuned FlyDSL config` warning in the log).
  Without it the shared expert runs as its own MXFP8 MLP next to a 128-expert / top-4 MoE.
  The fusion needs MXFP4 routed experts in the checkpoint.
- **DSpark, CUDA-graph captured.** See [DSpark Speculative Decoding](#dspark-speculative-decoding-minimax-m31)
  below. Expect `Draft passes warmed at [1, ..., 128] ... ['DSparkProposer._block_backbone (captured)']`.
- **Image input.** With the checkpoint's `vision_config`, ATOM runs the vision tower natively
  and scatters the image embeddings over the image tokens; send images as `image_url` content
  parts to `/v1/chat/completions`. Text-only requests take the same path.
- **Mono decode is not used.** The MiniMax-M3 mono kernels assume dense layers 0–2; the log
  reads `MiniMax-M3 mono decode off: DSpark speculative decoding` (with DSpark) or
  `... off: layers 0..2 must be dense full-attention layers` (without).

**Environment.** Measured on 4xMI355X in `rocm/atom-dev:minimax-m31-flash`
(`rocm/atom-dev:nightly_202610181506` with these MiniMax-M3.1 changes):

```bash
docker pull rocm/atom-dev:minimax-m31-flash
```

### Launching Server

Quark MXFP4 export, with DSpark. The online-quantization config re-quantizes the attention and
dense-MLP linears to PTPC FP8; the experts are already MXFP4, so the whole MoE block is left out:

```bash
model_path=${model_path:-amd/MiniMax-M3.1-Flash-Quark-MXFP4}   # includes dspark/
run_name=${run_name:-m31-flash-mxfp4-dspark}
export AITER_LOG_LEVEL=WARNING
export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --served-model-name minimax-m3.1 \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --max-model-len 131072 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --method dspark \
  --draft-model "$model_path/dspark" \
  --num-speculative-tokens 7 2>&1 | tee "${run_name}-server.log"
```

Without speculative decoding, drop the last three flags. Notes:

- `--default-chat-template-kwargs '{"thinking_mode": "enabled"}'` turns thinking on; the
  accuracy below is measured with it.
- `--block-size 128` is required with DSpark: the draft's KV pool shares the target's pages.
- Set `--max-model-len` explicitly; the model's default is 1M tokens.
- `--hf-overrides '{"use_index_cache": true, ...}'` (index-cache sharing, used for MiniMax-M3
  above) is not used for M3.1; its accuracy on M3.1 has not been measured.
- A second instance on the same node needs a different `--server-port` and engine `--port`
  (default 8006).
- DSpark acceptance is logged as `[MTP Stats]` lines and served cumulatively at
  `GET /debug/mtp_stats`.

### Accuracy Test

Day-0 accuracy with the launch command above (PTPC FP8 attention, fused shared expert, DSpark
with 7 drafts, FP8 KV and index cache, thinking on), lm-eval through the chat endpoint:

| Checkpoint | GSM8K flexible / strict | AIME 2026 mean@16 | AIME 2025 mean@16 | MMMU-Pro standard / vision |
|---|---:|---:|---:|---:|
| Quark MXFP4 export | 0.955 / 0.914 | 0.919 | 0.915 | 0.791 / 0.682 |

- GSM8K: 5-shot, multi-turn few-shot, temperature 0, `reasoning_effort=max`, up to 16,384 new
  tokens, no stop strings (command below).
- AIME 2025 / 2026: 0-shot, mean over 16 seeds, temperature 1.0, top-p 0.95, up to 98,304 new
  tokens, answer taken from the last `\boxed{}`.
- MMMU-Pro: standard (10 options) and vision subsets, temperature 1.0, top-p 0.95, up to 65,536
  new tokens, strict letter match.

```bash
lm_eval \
  --model local-chat-completions \
  --model_args "model=minimax-m3.1,base_url=http://127.0.0.1:8000/v1/chat/completions,tokenizer=$model_path,tokenized_requests=False,num_concurrent=16,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "temperature=0,reasoning_effort=max,max_gen_toks=16384,until=<|no_stop_string|>" \
  --log_samples \
  --output_path "./gsm8k-${run_name}" 2>&1 | tee "${run_name}-gsm8k.log"
```

#### Switch ablation (GSM8K, Quark MXFP4 export)

Measured earlier without `--served-model-name` and `--default-chat-template-kwargs`, with
lm-eval's default generation settings (`model=$model_path`, `--batch_size 65`, no `--gen_kwargs`).

MiniMax-M3.1 usually states its final answer as `\boxed{\$70{,}000}` or `**Answer: 12 years**`
instead of M3's `#### 70000`, which lm-eval's regex extractors partly misread (hence the lower
`strict-match`). Every run is therefore also re-scored from its logged samples
(`samples_gsm8k_*.jsonl`, the `flexible-extract` rows, one per question) with an
answer-format-robust extractor, the column marked *: drop the `<mm:think>` block; take the
number inside the last `\boxed{}`, else after the last `####`, else after the last `Answer:`,
else in the last `**bold**` span, else the last number of the response; remove `$`, `\$`,
LaTeX `{,}` and thousands separators first; count the question correct when that number equals
the GSM8K target numerically.

Validated results (1,319 questions, standard error ≈ ±0.005). Every row: Quark MXFP4 export,
FP8 KV and index cache, prefix caching off, `AITER_QUICK_REDUCE_QUANTIZATION=INT4`.
Acceptance is the cumulative `/debug/mtp_stats` read right after the GSM8K run (7 draft tokens
per verify step):

| Attention | Shared expert | DSpark | flexible-extract | strict-match | Answer-format-robust* | Acceptance | Avg toks/fwd |
|---|---|---|---:|---:|---:|---:|---:|
| `ptpc_fp8` | fused (→ MXFP4) | 7 | 0.9484 | 0.8848 | **0.9613** (1268/1319) | 51.78% | 4.62 |
| `ptpc_fp8` | fused (→ MXFP4) | off | 0.9477 | 0.9045 | **0.9621** (1269/1319) | — | 1.00 |
| native MXFP8 | fused (→ MXFP4) | 7 | 0.9416 | 0.8719 | **0.9591** (1265/1319) | 49.81% | 4.49 |
| `ptpc_fp8` | independent MXFP8 MLP | 7 | 0.9454 | 0.9090 | **0.9613** (1268/1319) | 49.63% | 4.47 |

DSpark accepted tokens distribution on GSM8K (accepted drafts per verify step, PTPC + FSE row):
`{0: 14.51%, 1: 14.06%, 2: 12.76%, 3: 10.38%, 4: 8.54%, 5: 7.16%, 6: 6.36%, 7: 26.24%}`

All four configurations agree within 6 questions (0.9591–0.9621, about one standard error);
DSpark on or off moves the score by one question.

### DSpark Speculative Decoding (MiniMax-M3.1)

The DSpark draft in `dspark/` (`architectures: ["DSparkMiniMaxDraftModel"]`, MXFP8 linears,
2.3 GB) is a parallel *block* drafter: one backbone pass drafts 7 tokens, a low-rank Markov head
samples them left to right, and the target verifies the block in one forward. It needs no
conversion — point `--draft-model` at that directory.

| Draft component | Shape |
|---|---|
| Context | target hidden states of layers 3, 17, 31, 45, 59 (each layer's output, before the final norm) → `fc` [6144, 5 × 6144] → `hidden_norm` |
| Backbone | 5 layers: GQA attention (64 q / 4 kv heads, head_dim 128, per-head Gemma q/k norm, partial RoPE 64 of 128, θ 10M, sliding window 4096) + dense SwiGLU-OAI MLP (12288) |
| Block | `[anchor, MASK × 6]`, non-causal over `[context \| block]`; embedding and LM head shared with the target |
| Heads | Markov head (rank 256); confidence head (loaded, used only with `--dspark-config '{"confidence_schedule": true}'`) |

How ATOM runs it:

- The draft's KV is a sibling pool with the target's page layout (5 layers, FP8 KV like the
  target). Each target forward writes the context K/V rows for the tokens it processed; the
  block attends that context plus its own K/V.
- The target layers' outputs are tapped with forward hooks (all-reduced across TP, plus the
  residual). This is why mono decode, which replaces those layers, is off under DSpark.
- Block attention. The block's K/V are written into the draft cache at the block's slots and
  the non-causal `[context | block]` attention runs as one q_len = 1 paged decode row per block
  token (the per-row decode the target's sparse layers use), through page tables built outside
  the graph in pre-allocated buffers. The block pass therefore has fixed shapes and is
  CUDA-graph captured for every batch size up to `--max-num-seqs`. The per-row decode takes one
  KV head per rank, so DSpark needs TP ≥ 4 (the draft has 4 KV heads) and refuses smaller TP.

## PD Disaggregation

For PD-disaggregated serving (1P+1D, 2P+1D DPA, with/without EAGLE3), see
[recipes/mesh/MiniMax-M3.md](mesh/MiniMax-M3.md).
