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

`lm_eval` over the full 1319 GSM8K questions, TP8, `--kv_cache_dtype fp8`,
`--max-model-len 2048`:

| Protocol | flexible-extract | strict-match |
|---|---|---|
| **chat, 3-shot `--fewshot_as_multiturn`** (use this) | **0.9659** | **0.9651** |
| raw `/v1/completions`, 3-shot | 0.9257 | 0.9257 |
| chat, 0-shot | 0.8362 | 0.0000 |

For reference, SGLang publishes 0.9704-0.9757 for this model on
H100/H200/GB300, measured with its own `sgl-eval run gsm8k` harness.

**Measure in chat mode WITH few-shot, or the number is meaningless.** lm_eval's
gsm8k filters are built around the few-shot `#### N` convention, while this
model answers in markdown/LaTeX and appends a `**Check:** ...` verification
line. Drop the shots and strict-match finds no `####` at all (0.0000) while
flexible-extract's "last number in the text" lands inside the verification tail
-- 0.8362 on a model that is actually right far more often. An extraction-free
audit of those same 1319 replies finds the correct value present in 1292 of
them (97.95%), with only 26 genuinely wrong.

`--num_fewshot` is no longer pinned at 3: kpool lifts the context cap, and both
5-shot @4096 and 16-shot @8192 run clean (see "Context length").

## Context length

`--max-model-len` is no longer capped at `index_topk`. The pooled (kpool) indexer
path is implemented: the index cache stores one compressed key per `index_kpool`
(4) tokens, top-k runs over `index_topk // index_kpool` (512) pools, each selected
pool expands back to its 4 token positions, and the trailing incomplete pool is
always appended unscored.

Accuracy past the old cap, `lm_eval` gsm8k over all 1319 questions, TP8, fp8 KV:

| Protocol | prompt tokens | flexible-extract | strict-match |
|---|---|---|---|
| chat, 3-shot, `--max-model-len 2048` | ~389 | 0.9644 | 0.9651 |
| chat, 5-shot, `--max-model-len 4096` | ~645 | 0.9674 | 0.9682 |
| **chat, 16-shot, `--max-model-len 8192`** | **2763-3591** | **0.9659** | **0.9659** |

The 16-shot row is the one that actually measures the pooled path: it is the only
setting whose prompts *all* exceed `index_topk`, so pooled scoring and pooled top-k
decide what the model attends to on every question. Below 2048 `attention_mla` runs
dense MLA and the indexer's selection is computed but never used -- a short-context
benchmark says nothing about pooled selection, only that the pooled *writes* did no
harm.

Retrieval itself is checked by `scripts/run_longctx_needle.py`, which runs each
needle depth twice with a different secret in an otherwise identical prompt. The
answer has to track the control, so a guess or an inference from the question
cannot pass; and the script fails rather than reports green if any prompt came in
under `index_topk`, since below that threshold nothing about selection is being
measured. Passing at depths 0.1/0.5/0.9, ctx=5571:

```bash
PYTHONPATH=$PWD ATOM_GLM5_KPOOL=1 python3 scripts/run_longctx_needle.py \
    --model /data/amd_int/models/GLM-5.3-Flash \
    --kv_cache_dtype fp8 -tp 8 --max_model_len 8192 \
    --gpu_memory_utilization 0.85 --no-enable_prefix_caching
```

### How pooled entries are stored

One pool covers 4 tokens and one block covers 16, so 16 pools span 4 token blocks.
Pooled entries live 16-per-block in every 4th block of the request's own block
table -- `pool p -> block_table[4 * (p // 16)], row p % 16` -- so the KV allocator,
the paging and the cache shape are all unchanged. Three of every four blocks'
index-cache regions go unused; reclaiming them is a future optimization.

Keeping 16 rows per block is **required**, not a convenience:
`deepgemm_fp8_paged_mqa_logits` is correct only in the preshuffled layout, and
preshuffle needs `KVBlockSize % 16 == 0`. With `Preshuffle=False` it disagrees with
the flat `fp8_mqa_logits` kernel by ~100% at *every* block size, and aiter's assert
only guards the preshuffle case -- so a 4-row-per-block cache would be silently
mis-scored rather than rejected.

### Not supported with kpool

- **DCP / PCP.** Both shard tokens round-robin across ranks, which does not commute
  with pooling four *consecutive* tokens into one key. Raises rather than
  mis-indexing.
- **Speculative decode.** The decode path assumes one token per request. GLM-5.3's
  MTP layer is not loaded, so this is unreachable today.

`ATOM_GLM5_KPOOL=0` restores token-granular selection, which still refuses a
context past `index_topk`.

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
