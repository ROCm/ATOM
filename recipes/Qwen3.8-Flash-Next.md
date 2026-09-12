# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) is a
multimodal hybrid model. Its checkpoint declares `model_type: qwen4_exp` and
`Qwen4ExpForConditionalGeneration`; ATOM registers those names and implements
the model under `atom/models/qwen3_8_flash_next.py`.

It shares Qwen3.5's backbone layout — 3 Gated-DeltaNet layers to every 1
full-attention layer — and adds four things Qwen3.5 does not have:

- **Hyper-Connections.** There is no `input_layernorm` / `post_attention_layernorm`.
  Four parallel residual streams are carried between layers as one flat
  `[tokens, 4 * hidden]` tensor, and every sub-layer is wrapped in a
  `mix` / `combine` pair that owns the normalization.
- **QSA (Query-Sparse Attention).** All 12 full-attention layers are sparse: a
  4-head indexer scores mean-pooled groups of 4 past keys, keeps the best 512,
  and the real 24-head GQA reads only the ~2051 positions those expand to.
  Below `indexer_budget` tokens of context this is exactly dense attention.
- **PLE n-gram memory** on layer 1: a 320M-row hashed table (102 GB, BF16 even
  in the FP8 release) read through a gate against the current hidden state.
- **MTP** draft layer — present in the checkpoint, not yet wired up in ATOM.

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All the operations below are executed inside the container.

`transformers>=5.16.1` is required — that is the first release carrying
`Qwen4ExpConfig`, and ATOM builds this model's vision config through
`AutoConfig`.

## Launching server

### FP8 on 2xMI355X GPUs (TP2 + EP)

```bash
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  -tp 2 --enable-expert-parallel --kv-cache-dtype bf16 \
```

### FP8 on 1xMI355X GPU

The FP8 weights are ~237 GB, which fits one 288 GB card. Expert parallelism is
not needed at TP1 and must be dropped.

```bash
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  -tp 1 --block-size 64 --kv-cache-dtype bf16 \
  --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.98 --server-port 8000
```

Single-GPU costs ~6% of concurrent throughput and a 5x smaller KV pool, and
nothing in latency: decode is launch-bound, and only 10 of 512 experts are read
per token, so halving the weights per card does not halve the traffic that
matters.

### Flags that are not optional

- **`--enable-expert-parallel` whenever TP > 1.** `moe_intermediate_size` is
  640, and tensor parallelism divides it: 640/2, 640/4 and 640/8 are none of
  them multiples of the 64 the AITER CK MoE GEMM needs, nor of the 128 the FP8
  block scales need. There is no usable pure-TP split. Expert parallelism keeps
  the intermediate whole at 640 and shards the 512 experts instead. Without the
  flag the server dies at load with
  `The output_size of gate's and up's weight = 320 is not divisible by weight
  quantization block_n = 128`.
- **`--kv-cache-dtype bf16`.** The QSA kernels require BF16 Q/K/V.
- **`--block-size` divisible by `indexer_compress_ratio` (4).**
- **Do not pass `--served-model-name`.** The multimodal processor is loaded from
  the served name, so an alias makes `AutoProcessor.from_pretrained("<alias>")`
  fail on the first image request.
- **`--gpu-memory-utilization 0.98`**, and check `num_kvcache_blocks` in the log
  after every launch — see below.

### Check the KV pool after every launch

The FP8 load can leave ~130 GB of non-torch device memory still held at the
moment ATOM profiles free memory, which collapses the KV pool:

```
Memory budget: ... peak_torch=117.26GB, non_torch=131.54GB, available_for_kv=3.97GB,
               num_kvcache_blocks=584
```

It is transient — steady state is ~128 GB of 288 GB — and it does not reproduce
on every launch. `--gpu-memory-utilization 0.98` compensates and is still
physically safe. Expected values:

| Config | `num_kvcache_blocks` |
|---|---|
| TP2 + EP | ~156000 |
| TP1 | ~30000 |

A few hundred blocks means the profile caught the transient; restart.

## Image request

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-Flash-Next-FP8",
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "file:///app/image.png"}},
      {"type": "text", "text": "Describe this image."}
    ]}],
    "max_tokens": 2048,
    "temperature": 0
  }'
```

`url` accepts `file://`, `http(s)://` and `data:` base64.

Two things to get right:

- **Give `max_tokens` at least 1024.** This is a thinking model; truncating
  before `</think>` closes leaves the reasoning parser nothing to split, and the
  whole chain of thought lands in `content` instead of `reasoning_content`.
- **The answer is in `content`, the thinking in `reasoning_content`.**

## Accuracy

`lm-eval`'s gsm8k task carries `until=["Question:", ...]`, which a thinking
model trips over while restating the problem — it truncated 61/1319 answers and
cost ~4 points. Override it, and use chat mode:

```bash
CHAT=1 NUM_CONCURRENT=48 GEN_KWARGS="max_gen_toks=4096,until=<|im_end|>" \
  bash scripts/run_gsm8k_eval.sh Qwen/Qwen3.8-Flash-Next-FP8 8000 5
```

Reference values, full 1319 questions, 5-shot chat:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9674|±  |0.0049|
|     |       |strict-match    |     5|exact_match|↑  |0.9644|±  |0.0051|
```

Measured on TP2 + EP. TP1 scored 0.9727 / 0.9689 and the BF16 checkpoint
0.9697 / 0.9659 — the spread across configurations is run-to-run noise, not a
quantization or sharding effect. Two runs of identical code differ by ~0.15pp:
greedy decoding over a long chain of thought amplifies 1-ULP BF16 differences,
so treat anything inside ~0.5pp as unchanged.

## Notes

- Prefix caching is on by default and works for text. Multimodal prefills are
  never chunked (the vision encoder covers the whole prompt), and the scheduler
  enforces that, so a multimodal prompt must fit `--max-num-batched-tokens` in
  one go.
- `atom/model_ops/attentions/qwen3_8_flash_next_attn.py` is the **backend** —
  cache sizing, allocation and per-step metadata. The attention itself is in
  `atom/model_ops/qwen3_8_flash_next/qsa_attention.py`.
- Each QSA layer owns three paged caches on the same block table: the main K/V,
  the raw (un-normalized) index key, and the pooled compressed key. A fourth,
  the per-token MRoPE positions, is allocated only when a multimodal config is
  present, since a compressed key's group position cannot be recomputed
  arithmetically once the three MRoPE rows differ.
