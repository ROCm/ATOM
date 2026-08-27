# GLM-5.3-Flash Usage Guide (gfx950 / MI355)

GLM-5.3-Flash (`Glm5NextForConditionalGeneration`, `zai-org/GLM-5.3-Flash`) is a
**hybrid linear/sparse-MLA MoE** model. ATOM serves the text backbone; the
vision tower under `model.visual.*` is skipped.

| Property | Value |
|---|---|
| Layers | 45 (+1 MTP, not loaded) |
| Layer mix | 34 KDA linear-attention + 11 sparse-MLA, 3:1 pattern |
| MLA | **NoPE** — `qk_nope_head_dim=256`, `qk_rope_head_dim=0`, `v_head_dim=256`, `kv_lora_rank=512` |
| Indexer | `index_topk=2048`, `index_kpool=4` (pooled K cache) |
| Residual | mHC hyper-connections, `hc_mult=4` |
| MoE | 288 routed + 1 shared, top-8, sigmoid / `noaux_tc`, `swiglu_limit=10.0` |
| Quantization | FP8 block 128×128 (MoE + MLA projections); KDA, indexer, `kv_b_proj`, norms and mHC are BF16 |

Total checkpoint 328 GB → ~40 GB/GPU at `-tp 8`.

## Launching the server

```bash
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model /data/amd_int/models/GLM-5.3-Flash \
  --kv_cache_dtype fp8 -tp 8 \
  --max-model-len 2048
```

## Accuracy

`lm_eval --tasks gsm8k --num_fewshot 3` over the full 1319 questions, TP8,
`--kv_cache_dtype fp8`, `--max-model-len 2048`:

| Filter | exact_match |
|---|---|
| flexible-extract | 0.9204 ± 0.0075 |
| strict-match | 0.9212 ± 0.0074 |

## Context-length limit (important)

**`--max-model-len` must not exceed `index_topk` (2048) today.**

The indexer's pooled (`index_kpool=4`) top-k is not implemented yet; ATOM falls
back to token-granular selection. Below `index_topk` that fallback is not an
approximation — top-k then selects *every* pool, so expanding the pools yields
every token position regardless of the pooled K values, and the two are exactly
equal. Past `index_topk` they genuinely differ, so `Glm5NextIndexer` raises
`NotImplementedError` rather than returning a quietly wrong answer.

Lifting this means implementing the pooled path: the softmax-pool +
Hadamard + FP8 compress kernel, the per-request tail cache for the in-progress
pool, and pool→token expansion. DeepSeek-V4's `Compressor`
(`atom/models/deepseek_v4.py`) already does the same softmax-pool-with-`ape`
compression and drives `deepgemm_fp8_paged_mqa_logits` at pool granularity, so
it is the natural starting point.

## Notes

- The NoPE MLA runs on ATOM's standard 576-wide KV path with the 64-lane rope
  block held at zero (`_ROPE_PAD` in `atom/models/glm5_next.py`). Rotating the
  zero vector yields the zero vector, so this is bit-for-bit NoPE; it costs
  64/576 of the MLA KV cache on 11 of 46 layers.
- The KDA layers reuse `kimi_k3.KimiKDAAttention` with its low-rank output gate
  (`g_a_proj`/`g_b_proj`) rather than Kimi's full-rank `g_proj`.
- If accuracy looks off, try `--no-enable_prefix_caching`: the KDA recurrent
  state is per-request, and Kimi-K3 disables prefix caching for that reason.
- **The shared expert must be counted in `get_expert_mapping()`.** With
  `n_shared_experts=1` aiter fuses it into the routed tensor, so `w13_weight`
  gets `n_routed_experts + n_shared_experts` (289) slots and the loader renames
  `mlp.shared_experts.*` to `mlp.experts.288.*`. A mapping built with only
  `n_routed_experts` leaves slot 288 holding uninitialized `torch.empty`
  memory, which the shared expert then contributes to EVERY token of EVERY MoE
  layer. Nothing reports it -- `w13_weight` is a single parameter, so writing
  any slot marks the whole tensor loaded, and the weight-load diagnostic stays
  silent. The symptom is distinctive: grammar stays fluent (attention and the
  dense layers are untouched) while factual recall collapses, and identical
  greedy requests return different text as the allocator recycles that memory.
  This cost most of a debugging session; check it first when porting a MoE
  model whose shared expert is fused.
