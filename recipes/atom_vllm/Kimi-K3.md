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

Kimi-K3 offloads KV through `AtomLMCacheOffloadConnector`, the same connector
MiniMax-M3 and GLM-5.2 use, plus a second leg for the KDA recurrent state. See
[MiniMax-M3 — LMCache KV offload](MiniMax-M3-LMCache-Byte-Offload.md) for the
LMCache build steps and the tier-sizing arithmetic; everything there applies
here unchanged. What follows is only what K3 adds.

### Why K3 needs more than the M3 path

K3 is hybrid, so vLLM splits the cache into two *kinds* of group — MLA full
attention and KDA recurrent state. It does not build one group per kind. vLLM
builds **equal-sized** groups, sized by the largest layer family, so K3's 29 MLA
layers and 69 KDA layers come out as **four** groups: one attention group of 29
and **three** mamba groups of 23 (measured: `groups=23,23,23`). All three mamba
groups hold one logical state — one token boundary, one block id each — and none
of them is restorable alone.

A restored MLA prefix is correct only if the KDA state at the **same token
boundary** is restored with it — half a restore is not a crash and not a log
line, it is wrong output.

The two groups therefore move by two different mechanisms:

| group | moved by | addressed by |
|---|---|---|
| MLA | `DenseKVByteCodec`, whole blocks per chunk | the request's block table, positionally |
| KDA | `StateByteCodec`, one opaque page per boundary | vLLM's explicit boundary hand-off |

The KDA group cannot be read positionally at all. In `--mamba-cache-mode align`
a mamba block table is not append-only — a superseded state block is freed and
nulled, and speculative blocks relocate in place — so indexing it by
`token // block_size` can land on a null, freed, or live speculative block and
persist those bytes under a valid prefix hash. The only safe source is vLLM's
explicit hand-off — `SchedulerOutput.partial_tail_offloads` on the pinned 0.28
(`KVCacheManager.take_partial_tail_offloads`), the same
`{req_id: [(group_id, block_id, boundary_tokens)]}` payload under
`kv_connector_block_state.boundary_state_offloads` on 0.29 — which names the
exact block holding a committed boundary state. The connector reads whichever
spelling the running vLLM provides.

Correctness of the pair is enforced on **lookup**, not on save: the reported
external hit is capped at the largest chunk boundary whose KDA state the index
still claims. A KDA state that was never stored, or that LMCache evicted,
shortens the prefix instead of corrupting it.

### Launch

Add to the DSpark launch above (prefix caching and `--mamba-cache-mode align`
are already there and are both **mandatory** for offload):

```bash
export PYTHONHASHSEED=0              # mandatory, see Gotchas in the M3 recipe
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20 # GiB **per TP rank** -- TP8 x 20 = 160 GiB pinned
export LMCACHE_CHUNK_SIZE=1536       # multiple of the *effective* block size,
                                     # which vLLM raises to 1536 -- see below
export OFFLOAD_MIN_LOAD_TOKENS=256   # default 8192 disables the tier for chat-sized prompts

vllm serve "${MODEL}" \
    ... the DSpark flags above ... \
    --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'
```

Three settings are K3-specific and each one is a hard failure if wrong:

- **`--mamba-cache-mode align` is mandatory**, not just useful for DSpark. Any
  other mode keeps the recurrent state where this connector has no hand-off for
  it, so a boundary block id would be a guess. The connector refuses to start
  rather than guess.
- **`"kv_load_failure_policy":"recompute"`** must be set. vLLM's default is
  `fail`, which turns an offload-tier miss into a user-visible 500. The KDA leg
  reports a load error deliberately whenever a recurrent state does not come
  back — that is the mechanism that keeps a half-restored prefix from being
  served — so under the default policy an ordinary eviction fails the request.
- **`LMCACHE_CHUNK_SIZE` must be a multiple of the mamba block size, which is
  not the `--block-size` you passed.** vLLM raises the attention block size so
  the attention page is at least as large as the mamba page, logging `Setting
  attention block size to 1536 tokens to ensure that attention page size is >=
  mamba page size` (`vllm/platforms/interface.py`). `--block-size 128` does not
  prevent this: it only fixes the *requested* size, and the hybrid alignment
  step overrides it afterwards. On K3 the effective size is **1536**, so read it
  out of the log rather than assuming the flag won. Only
  chunk-aligned boundaries are stored and only chunk-aligned boundaries are
  probed on lookup, so a chunk that ends between two boundaries can never
  produce a usable pair. The connector validates this at construction and names
  both numbers if it does not hold.

Hybrid models also require vLLM's hybrid memory allocator, which vLLM
auto-disables for a connector that does not declare `SupportsHMA` — so without
that declaration K3 plus `--kv-transfer-config` does not mis-save, it does not
boot. The connector declares it; the check below confirms HMA stayed on.

### Verify it is actually on

On top of the four checks in the M3 recipe:

```bash
# the recurrent leg found its groups. These are TWO different lines from two
# different processes, and each one alone leaves the other side unchecked:
# the EngineCore prints the scheduler-side line, every worker prints its own.
grep "ATOM LMCache offload: recurrent state leg on group" server.log  # EngineCore
grep "ATOM LMCache offload: recurrent state tier up"      server.log  # 1 per worker

# the dense leg strides by blocks, not tokens. `leading dim` is 1536x num_blocks
# on K3 because the MLA backend asks for a kernel block size of 1; the two being
# equal here would mean the codec is striding per token.
grep "ATOM LMCache offload: registered" server.log  # 1 per worker

# HMA must NOT have been turned off -- this line means the connector was not
# recognised as SupportsHMA and the recurrent leg is not running
grep "Turning off hybrid kv cache manager" server.log   # expect no match
```

### Status

**Boot: validated** on 8xMI355 TP8 (2026-09-18). Three mamba groups on 0,1,2 and
the attention group on 3, the hybrid memory allocator left on, and the
deterministic smoke above answering 42. Measured geometry, for comparison
against a future boot:

```text
Setting attention block size to 1536 tokens ...   # effective block size, not 128
Available KV cache memory: 37.85 GiB              # per rank
GPU KV cache size: 1,887,436 tokens, Maximum concurrency ... 28.80x
registered 29 layers, num_blocks=1584 (leading dim 2433024, block_size=1536)
recurrent state leg on group(s) 0,1,2 (mamba_block=1536, hash_block=1536, chunk=1536)
recurrent state tier up, 69 layers, entry=58.22 MiB, ... groups=23,23,23
bytes_per_block=25657344 chunk=1536
```

Two numbers there are worth checking rather than skimming, because both fail
silently if they are wrong:

- `bytes_per_block` must be `block_size x bytes_per_token`, here
  `1536 x 16,704 = 25,657,344`. It is what the codec charges for one block-table
  entry. If it comes back as the per-token figure instead, every entry is being
  read as one token and the restored bytes come from the wrong rows — no error,
  no log, just wrong output.
- `leading dim` is `1536 x num_blocks`, not `num_blocks`. The MLA backend asks
  vLLM for a kernel block size of 1, so the cache is allocated one row per token
  while the block ids a connector receives stay manager ids. The two being equal
  would mean this model is not on the kernel-block path and the check above is
  the one that matters.

**Accuracy and hit rate: not yet run.** The numbers earlier in this section are
requirements, not measurements. When they are run, use the two-pass method: a
single SAVE-only pass measures nothing, so salt the prefixes to defeat the GPU
prefix cache, size the HBM pool below the working set to force read-back, and
take the noise floor from the OFF arm's own two-pass delta.

Size both ends before spending GPU time. Per rank K3 costs **56,448 B/token** —
`29 x 576 = 16,704` for attention plus `58.22 MiB / 1536 = 39,744` for the
recurrent boundary state, so the recurrent leg is 2.4x the attention leg and
dominates the tier. A tier smaller than the HBM pool has nothing to do: at the
defaults above the pool holds ~1.89M prefix tokens while `20 GiB` of tier holds
`20 GiB / 56,448 = 371k`, five times less, and the plugin path only ever offers
the connector the tail that missed in HBM. Shrink the pool with
`--num-gpu-blocks-override` and raise `LMCACHE_MAX_LOCAL_CPU_SIZE` until the
window `[HBM pool, tier)` is non-empty, and keep both identical across the two
arms.

## Current scope

- Text and image inputs are supported through the Kimi-K3 multimodal processor,
  vision tower, and projector.
- TP8 on MI355/gfx950 is the validated deployment.
- Asynchronous scheduling is supported. Prefix caching is off by default and
  needs `--mamba-cache-mode align` to be turned on, as the DSpark launch does.
- DSpark speculative decoding is supported; see above.
- LMCache KV offload is supported with prefix caching and
  `--mamba-cache-mode align`; see above. It needs the hybrid memory allocator
  left on (the connector declares `SupportsHMA`) and the boundary-state
  hand-off, both of which the pinned vLLM 0.28 provides.
