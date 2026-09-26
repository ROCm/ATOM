# Qwen3.8-Flash-Next Usage Guide

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is a
multimodal MoE model combining Gated DeltaNet (GDN), sparse attention (QSA),
and n-gram memory. ATOM supports text and image inference with BF16 or FP8
weights and BF16 activations/KV cache.

## Preparing the environment

Follow the [ATOM installation instructions](../README.md). Use the
dependencies pinned by ATOM, including `transformers==5.16.1`, which provides
the model configuration and multimodal processor.

## Serving FP8 on one GPU

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1
```

The API listens on port 8000 by default. Use
`Qwen/Qwen3.8-Flash-Next-FP8` as the `model` in API requests.

BF16 is the only supported KV cache format. The default block size already
satisfies `indexer_compress_ratio` (4); no block-size override is needed.

Choose a GPU with enough memory for the weights, recurrent state, and KV
cache. Adjust `--max-num-seqs`, `--max-model-len`, and
`--max-num-batched-tokens` for the available memory and workload.
In particular, `--max-num-seqs` defaults to 512; lowering it reduces the
preallocated recurrent-state memory. These are resource controls, not
model-specific requirements.

## Native MTP

Use the draft weights included in the same checkpoint:

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --trust-remote-code \
  -tp 1 \
  --method mtp --num-speculative-tokens 2 \
  --no-enable_prefix_caching
```

Draft depths 1, 2, and 3 are supported. The checkpoint contains one draft
layer, reused at each step; no separate draft model is needed. The target
and draft share the embedding and output head.

## Agentic serving on one MI350X (prefix caching + MTP)

For agent workloads (e.g. Claude Code through `/v1/messages`: long shared prefixes, many short turns, several concurrent sub-agents),
**prefix caching is what makes turns cheap** and can be combined with native MTP. Measured on 1× MI350X (VF), FP8, TP1, 1M context,
agent turn = 100k cached tokens + 4k new tokens, 256 output tokens:

| | prefix caching off | prefix caching on |
|---|---|---|
| 1 agent, time to first token | 6.2 s | **0.53–0.58 s** |
| 6 agents, time to first token (median) | 31 s | **~1.9 s** |

Numerical check with MTP + prefix caching: 12/12 identical answers (5 markers spread over 58k tokens; cold, exact repeat and
new-question turns).

```bash
python -m atom.entrypoints.openai_server \
  --model Qwen/Qwen3.8-Flash-Next-FP8 --trust-remote-code -tp 1 \
  --method mtp --num-speculative-tokens 3 \
  --state-checkpoint-interval-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,8,16]" \
  --max-num-seqs 16 --kv_cache_dtype bf16
```

- **`--state-checkpoint-interval-tokens`**: a GDN state checkpoint must exist at the resume point. `-1` (prompt-end anchors only) serves
  conversations that only grow at the end, but sibling sub-agents that share a long prefix and diverge in the middle then recompute
  everything. With `16384`, a real Claude Code session (auto mode + sub-agents) reached **94% prefix-cache hits**; the default `8192`
  costs ~5–8% prefill throughput for the extra rungs.
- **`--cudagraph-capture-sizes`**: capture every batch size you expect (the default list skips 3, 5, 6).
- **Persist the Triton/comgr caches** (`~/.triton`, `~/.cache`) across container restarts and send a warm-up request after start:
  first-time JIT compilation of a new shape stalls a request for 5–25 s.
- MTP depth: 3 was best for 1–3 concurrent agents, 2 for 4–5 (measured 240 / 545 / 745 tok/s aggregate decode at 1 / 3 / 6 agents).
- Clients that send `role: system` messages in the middle of `messages` (Claude Code 2.1.x) get HTTP 500 ("System message must be at the
  beginning"); converting them in place (e.g. into a `<system-reminder>` block of the preceding user turn) keeps the prefix stable —
  moving them to the top-level `system` field invalidates the cache on every new reminder.
- Concurrent requests with an identical prefix do not share it until the first one finishes prefill; serializing them at the proxy
  (release on first token) raised hits from 77% to 88% in the same session.

## Usage notes and limitations

- For image requests, use `--no-enable_prefix_caching` and sufficient cache
  capacity to avoid preemption until the image caching/recompute limitations
  are resolved. Image prompts must fit within `--max-num-batched-tokens`.
- Preserve the checkpoint's quantization exclusions and `ple_embedding_dtype`
  when present. GDN inputs support unquantized weights or per-channel FP8 QKV/Z
  with unquantized B/A. GDN input online quantization is not supported.
- PTPC (`compressed-tensors`) checkpoint loading is supported, but end-to-end
  execution on gfx950 is blocked by the default small-batch MoE dispatch:
  its second-stage kernel requires a 256-aligned intermediate size, whereas
  this model uses 640. The FP8 example above uses block-wise quantization.
- Chunked text prefill and full decode CUDA graphs are supported. Piecewise
  compilation/CUDA graphs are not supported.
- Only native MTP speculation is supported. Pipeline parallelism, context
  parallelism, DP attention, TBO, and external KV transfer/offload are not
  supported.
- This example covers FP8 with TP1. Pure TP2 is not currently validated;
  it can fail MoE warmup on gfx950 due to an MoE dispatch limitation.
- Video inference and long-context accuracy have not been validated.

## GSM8K evaluation

Install the evaluation client:

```bash
python -m pip install 'lm-eval[api]'
```

With the server running, evaluate all 1,319 test questions using 5-shot chat:

```bash
lm_eval --model local-chat-completions \
  --model_args 'model=Qwen/Qwen3.8-Flash-Next-FP8,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=900,tokenized_requests=False,trust_remote_code=True' \
  --tasks gsm8k --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
  --log_samples --output_path ./results/gsm8k
```

The model emits reasoning before its final answer, so allow enough output
tokens to avoid truncation. The explicit stop string replaces GSM8K's default
`Question:` stop, which can prematurely end a reasoning response.
The harness scores the final `content`, not `reasoning_content`.
The timeout accommodates long requests under load; it does not change the
generation limit or scoring. Set evaluation concurrency to suit the server's
capacity.
