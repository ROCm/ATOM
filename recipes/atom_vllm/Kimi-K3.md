# Kimi-K3 with ATOM vLLM Plugin Backend

This recipe serves the text-only Kimi-K3 backbone
(`KimiK3ForConditionalGeneration`) through the ATOM vLLM out-of-tree plugin.
Kimi-K3 combines KDA recurrent-attention layers, MLA full-attention layers, and
an MXFP4 latent MoE.

The validated configuration requires eight MI355 (gfx950) GPUs with TP8.

## Prerequisites

Use the ATOM vLLM OOT image and install the KDA dependency used by the native
Kimi-K3 implementation:

```bash
docker pull rocm/atom-dev:vllm-latest
pip install "fla-core==0.5.1" "flash-linear-attention==0.5.1"
```

Install the target ATOM checkout into the same environment:

```bash
pip install -e /path/to/ATOM --no-deps
```

## Launch

```bash
MODEL=/path/to/Kimi-K3

# A4W4 MOE require
export AITER_SITUV2_A4W4=1
# A8W4 MOE require
# export AITER_SITUV2_A8W4=0
# export ATOM_MOE_GU_ITLV=0

export VLLM_USE_BREAKABLE_CUDAGRAPH=0

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}}'
```

The plugin keeps KDA temporal state in fp32, registers every KDA layer through
vLLM's hybrid/Mamba cache contract, and uses ATOM's MLA backend for full
attention. vLLM may increase the physical attention block size so its MLA and
KDA pages have equal byte size; this is expected.

Prefix caching needs `--mamba-cache-mode align`. KDA recurrent state cannot be
reconstructed from the paged MLA cache alone, so the two have to agree on
block boundaries before a prefix hit can be reused; without that mode prefix
caching has to stay off. Leaving it off makes any long-shared-prefix benchmark
meaningless — every request re-prefills the whole prompt.

## Smoke test

```bash
curl http://127.0.0.1:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "/path/to/Kimi-K3",
      "prompt": "Question: What is 17 + 25? Answer:",
      "max_tokens": 32,
      "temperature": 0
    }'
```

The deterministic response starts with `42`.

## Accuracy validation

```bash
lm_eval \
    --model local-completions \
    --model_args "model=${MODEL},base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --output_path /app/logs_claude/kimi_k3_vllm_graph_clean_gsm8k
```

Validated on the full 1319-example GSM8K test set with TP8 and
`FULL_AND_PIECEWISE` CUDA Graph:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9553|±  |0.0057|
|     |       |strict-match    |     5|exact_match|↑  |0.9553|±  |0.0057|
```

Raw result JSON is written below
`/app/logs_claude/kimi_k3_vllm_graph_clean_gsm8k/`.

Use a freshly started server for each reported accuracy run, matching the
native Kimi-K3 validation protocol. Back-to-back evaluations on a warm server
are not used as baselines for this model.

## Speculative decoding (DSpark)

Kimi-K3's DSpark draft is a separate checkpoint,
[Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark). It
drafts a whole block in one parallel backbone pass, which the target then
verifies. The launch is the one above with `--speculative-config` added; nothing
else changes:

```bash
MODEL=/path/to/Kimi-K3
DRAFT=/path/to/Kimi-K3-DSpark

export AITER_SITUV2_A4W4=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --speculative-config "{\"method\":\"dspark\",\"model\":\"${DRAFT}\",\"num_speculative_tokens\":7}" \
    --additional-config '{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*self_attn.[qkv]_conv1d*", "*block_sparse_moe.experts*", "*block_sparse_moe.routed_expert_*", "*vision_tower*", "*mm_projector*"]}}'
```

This needs an aiter build carrying the non-causal MLA decode kernels for an fp8
KV cache; without them the draft has no kernel to run its block through.

**Leave `kv_cache_dtype` out of the speculative config.** The draft then
inherits the engine's fp8, which is what lets prefix caching stay on. Pinning
the draft to bf16 doubles its page against a fp8 target page, and since vLLM
allocates one page size across all KV cache groups the draft layers have to
scale their own block size down to compensate — a block size prefix caching
then rejects for disagreeing with its hash block size.

The draft block is drafted **non-causally**: every position attends to the whole
block. Nothing has to be passed for this. The draft checkpoint carries neither
`layer_types` nor a `dflash_config`, so vLLM resolves the draft to non-causal on
its own, and the plugin carries that per-KV-cache-group flag down into both the
persistent MLA work descriptors and the decode kernel so the two agree.

`FULL_AND_PIECEWISE` applies to the target as usual.

**Clear the compile cache whenever `num_speculative_tokens` changes.** The cache
key does not include it, so a new width silently replays graphs captured for the
old one and the run dies with an opaque `HIP error: unknown error`:

```bash
rm -rf /app/.cache/atom/* /app/.cache/vllm/* /app/.cache/inductor/* /root/.cache/atom/* /root/.cache/vllm/*
```

### Accuracy

Full 1319-example GSM8K 5-shot, TP8, fp8 KV, 64 concurrent,
`FULL_AND_PIECEWISE`, prefix caching on, `num_speculative_tokens=7`:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9492|±  |0.0060|
|     |       |strict-match    |     5|exact_match|↑  |0.9484|±  |0.0061|
```

Over that run the draft held ~51% acceptance at a mean accepted length of ~4.6
tokens per target forward, with per-position acceptance decaying smoothly from
0.93 at position 0 to 0.15 at position 6.

### Known issue: prefill slows down sharply with the draft enabled

Decode is unaffected, but prefill is not. Serving the same 245,912 computed
prompt tokens (four unique 64K-token prompts, no prefix reuse) takes 15.8 s
without a draft and 438.6 s with one, so TTFT on long-prompt workloads inflates
by more than an order of magnitude while ITL barely moves (23 ms vs 32 ms).
Sampling puts the time in the *target's* KDA prefill path, not in the draft's
own pass. Short-prompt serving is not affected. Under investigation.

## Current scope

- Text generation only; the vision tower and multimodal projector are skipped.
- TP8 on MI355/gfx950 is the validated deployment.
- Prefix caching is on via `--mamba-cache-mode align`; async scheduling is off.
- DSpark speculative decoding is supported; see the section above, including the
  open prefill-throughput issue.
