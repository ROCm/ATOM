# Kimi-K3 with ATOM vLLM Plugin Backend

This recipe serves multimodal Kimi-K3 (`KimiK3ForConditionalGeneration`)
through the ATOM vLLM out-of-tree plugin. Kimi-K3 combines a MoonViT3d vision
tower with KDA recurrent-attention layers, MLA full-attention layers, and an
MXFP4 latent MoE.

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

### Validated accuracy and acceptance

Full 1,319-example GSM8K, 5-shot, 64 concurrent, TP8, `FULL_AND_PIECEWISE`,
fresh server per run:

```text
                           flexible-extract   strict-match   wall clock
DSpark, N=2                        0.9507         0.9500        177 s
```

Draft acceptance over those runs, reported by vLLM's SpecDecoding metrics:

```text
Mean acceptance length:      2.61 - 2.78  (of 3)
Per-position acceptance:     0.89 - 0.95, 0.72 - 0.84
Avg draft acceptance rate:   86.1%, 86.1%  (whole run, each of the two)
```

## LMCache KV offload (byte codec)

K3 offloads KV through `AtomLMCacheOffloadConnector` with a second leg for the
KDA recurrent state. It has its own recipe, because it needs three K3-specific
settings that are each a silent or hard failure if wrong, and because it is the
only one of these models whose offload gain has a **sign** that depends on how
the HBM pool is sized:

**[Kimi-K3 — LMCache KV offload on the vLLM plugin](Kimi-K3-LMCache-Byte-Offload.md)**

Measured there: what the tier can do is set by whether the workload's **reuse
distance** lands above what HBM keeps, and the pool size is only one of the two
ways to move that. On an agentic trace replay at the default unpinned pool the
tier supplies 0.30% of prompt tokens and the ON/OFF difference is not measurable
(two rulers straddle zero, n=1 per arm) — vLLM's pool already answers 85.05%,
and on the plugin path the connector is only asked about what the pool missed.
On a controlled-prefix client at the *same* unpinned pool the tier supplies ~82%
and throughput gains **+20.5 / +22.4 / +27.8%** at conc 16 / 20 / 24 (n=1 per
arm). Pinning the pool moves the same edge from the other side: +10.81% req/s at
`tier / pool = 2.0`, against −3.59% at `tier / pool = 0.81`.

## Current scope

- Text and image inputs are supported through the Kimi-K3 multimodal processor,
  vision tower, and projector.
- TP8 on MI355/gfx950 is the validated deployment.
- Asynchronous scheduling is supported. Prefix caching is off by default and
  needs `--mamba-cache-mode align` to be turned on, as the DSpark launch does.
- DSpark speculative decoding is supported; see above.
- LMCache KV offload is supported with prefix caching and
  `--mamba-cache-mode align`; see
  [Kimi-K3-LMCache-Byte-Offload.md](Kimi-K3-LMCache-Byte-Offload.md). It needs
  the hybrid memory allocator
  left on (the connector declares `SupportsHMA`) and the boundary-state
  hand-off, both of which the pinned vLLM 0.28 provides.
