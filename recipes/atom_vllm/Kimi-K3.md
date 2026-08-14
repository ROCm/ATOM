# Kimi-K3 with ATOM vLLM Plugin Backend

This recipe serves the text-only Kimi-K3 backbone
(`KimiK3ForConditionalGeneration`) through the ATOM vLLM out-of-tree plugin.
Kimi-K3 combines KDA recurrent-attention layers, MLA full-attention layers, and
an MXFP4 latent MoE.

The validated configuration requires eight MI355 (gfx950) GPUs with TP8.

## Prerequisites

Use the ATOM vLLM OOT image. The KDA recurrence runs on aiter, which the image
already carries, so no extra package is needed:

```bash
docker pull rocm/atom-dev:vllm-latest
```

Install the target ATOM checkout into the same environment:

```bash
pip install -e /path/to/ATOM --no-deps
```

## Launch

```bash
MODEL=/path/to/Kimi-K3

vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --language-model-only \
    --kv-cache-dtype fp8 \
    --max-model-len 16384 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --additional-config '{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}}' 
```

The plugin keeps KDA temporal state in fp32, registers every KDA layer through
vLLM's hybrid/Mamba cache contract, and uses ATOM's MLA backend for full
attention. vLLM may increase the physical attention block size so its MLA and
KDA pages have equal byte size; this is expected.

Prefix caching must stay disabled because KDA recurrent state cannot be
reconstructed from the paged MLA cache alone.

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

## Speculative decoding with DSpark

Kimi-K3 ships a DSpark draft, which proposes a block of `N` tokens in one
non-causal pass and has the target verify all of them in the next step. Add
`--speculative-config` to the launch above, and turn prefix caching on with
`--mamba-cache-mode align` so the KDA and MLA pages agree on block boundaries:

```bash
DRAFT=/path/to/Kimi-K3-DSpark

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
    --gpu-memory-utilization 0.85 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    --speculative-config '{"method":"dspark","model":"'"${DRAFT}"'","num_speculative_tokens":2}' \
    --additional-config '{"online_quant_config":{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}}'
```

`num_speculative_tokens=2` with `--gpu-memory-utilization 0.85` is the
validated pair: the draft's own paged cache has to fit alongside the target's,
and 0.85 leaves 30.1 GiB of KV cache against the 35.7 GiB the target alone
gets at the same utilization. The draft inherits the engine's fp8 cache dtype;
setting it to bf16 halves the draft's page size relative to the target's, which
prefix caching's hash block size rejects.

Both graph modes stay in force: the target captures FULL graphs for pure decode
and Piecewise for prefill, and the draft block captures its own FULL graphs.

### Validated accuracy and acceptance

Full 1,319-example GSM8K, 5-shot, 64 concurrent, TP8, `FULL_AND_PIECEWISE`,
fresh server per run:

```text
                           flexible-extract   strict-match   wall clock
DSpark, N=2                        0.9507         0.9500        177 s
DSpark, N=2, repeat                0.9530         0.9522        171 s
no speculation                     0.9545         0.9545        229 s
no speculation, V1 runner          0.9515         0.9500        244 s
```

vLLM forces its V2 model runner whenever the speculative method is `dspark`, so
the like-for-like baseline is the third row: the launch above with
`--speculative-config` dropped and `VLLM_USE_V2_MODEL_RUNNER=1` put back. The
last row is that same baseline on the default V1 runner, and is here only to
show that the runner accounts for 15 s of the gap and speculation for the rest.

All four runs sit within a standard error of each other, so speculation costs
no measurable accuracy. Do not read a regression from any single pair of runs:
repeats of the same build have come out 0.014 apart, against a single run's
±0.006 standard error, because continuous batching decides which requests share
a step and a greedy argmax can flip on the last bit of a reduction. Against the
like-for-like baseline drafting is worth about 1.3x wall clock; GSM8K's short
answers leave little room for it to pay off, so that is a floor, not the
headline speedup.

Draft acceptance over those runs, reported by vLLM's SpecDecoding metrics:

```text
Mean acceptance length:      2.61 - 2.78  (of 3)
Per-position acceptance:     0.89 - 0.95, 0.72 - 0.84
Avg draft acceptance rate:   86.1%, 86.1%  (whole run, each of the two)
```

A collapse to roughly 17% acceptance means the draft's context rows are being
written to the fp8 cache as integers rather than as bytes -- the target still
answers correctly, so accuracy alone does not catch it. Watch the per-position
rates, not just GSM8K.

Acceptance degrades at very long contexts (128K and beyond), where the draft's
YaRN extrapolation runs out; the target's own accuracy is unaffected.

## Current scope

- Text generation only; the vision tower and multimodal projector are skipped.
- TP8 on MI355/gfx950 is the validated deployment.
- Asynchronous scheduling is supported. Prefix caching is off by default and
  needs `--mamba-cache-mode align` to be turned on, as the DSpark launch does.
- DSpark speculative decoding is supported; see above.
