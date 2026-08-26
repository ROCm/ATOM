# The DCP KV relayout moved DSA index pages as if a token were contiguous

Root cause and fix for two symptoms that turned out to be one bug: a DCP decode
node that answers nothing correctly past `index_topk`, and a DSA top-k kernel
that costs 1.6-1.8 ms per call. Both come from the mooncake producer writing
the indexer cache with the MLA latent's addressing.

Measured 2026-08-26 on MI355X, GLM-5.2-MXFP4, PP4xTP1 prefill -> TP4xDCP4
decode over mooncake PD.

## The index page is not token-addressable

`indexer_k_quant_and_cache(..., preshuffle=True)` writes a page, not a run of
tokens (`cache_kernels.cu:1364`). For GLM-5.2 (`index_head_dim` 128, one fp32
scale per token, `aligned_index_dim` 144, `block_size` 16) a 16-token page is

    [ 2048 B fp8 keys, MFMA-16x16 tiled ][ 64 B fp32 scales ][ 192 B padding ]

Every token's key bytes are interleaved into the tile, and every token's scale
lives in one shared plane after *all* of the page's keys. No 144-byte window of
that page is a token.

`_execute_block_transfer` addressed all block regions the same way:

    src_addr = src_base + token_index * (per_block_bytes / block_size)

which is exactly right for the MLA latent `[num_slots, 1, 576]` and wrong for
every index page.

## What that costs, byte for byte

`indexer_k_quant_and_cache` on the real kernel, against the page each plan
produces (16-token source pages, dcp_size 4):

Replicated index cache -- source page `s` is copied whole to destination byte
offset `s * 2304`, but its keys belong at `s * 2048` and its scales at
`8192 + s * 64`:

    page: values [0,8192) scales [8192,8448) pad [8448,9216)
      value bytes wrong :  6084 / 8192
      scale bytes wrong :   256 / 256
        sub-page 0 values:     0 / 2048 wrong
        sub-page 1 values:  2029 / 2048 wrong
        sub-page 2 values:  2025 / 2048 wrong
        sub-page 3 values:  2030 / 2048 wrong
      scales as read : finite-max 9.913e+33
      scales expected: finite-max 0.01562
      page-aware move wrong bytes: 0

Sharded index cache at `interleave_size=1` -- each rank pulls its 16 global
tokens 144 bytes at a time into its own 16-token page:

      value bytes wrong :  2009 / 2048
      scale bytes wrong :    63 / 64
      scales as read : max|.| 9.779e+36
      scales expected: max|.| 0.01562

Only sub-page 0 of the replicated page survives, because only it starts at a
page boundary. The whole destination scale plane is filled with fp8 key bytes
reinterpreted as float32, which is where the 1e33-1e36 magnitudes come from.

## Why it shows up as both a wrong answer and a slow kernel

Every dequantized key of a transferred token is multiplied by a garbage scale,
so the indexer logits plane over the prompt is noise: `docs/dcp_decode_topk_logits_dump.md`
recorded 5-7 % NaN, magnitudes to 1e35, and a valid suffix exactly as long as
the number of tokens the decode node had generated itself. Sparse selection
then picks 2 048 arbitrary prompt positions -- uniform across every decile of
the context, none in the last 512, none in the first 8 -- and the answer falls
apart.

The same plane is what makes `radix_topk_one_block_kernel` slow. An all-NaN
region collapses into a single radix bucket, so the pass-0 histogram cannot
narrow the candidate set and the kernel runs its non-compact worst case. The
1.6 ms top-k and the retrieval failure are the same defect measured two ways.

## Why every existing check passed

- `top_k_per_row` short-circuits when the row is shorter than k, so any context
  under `index_topk` (2 048) degenerates to dense and cannot see the bug. GSM8K
  never reaches 2 048 tokens; its 0.9295 / 0.9356 scores say nothing here.
- Under a **sharded** index cache each rank's row is `g_ctx / W` long, so the
  short-circuit holds until the *global* context passes `W * index_topk` =
  8 192 tokens. The 6.9 k-token needle sweep that recorded "sharded is clean,
  8/8" sat below that threshold on every prompt; it did not exercise sparse
  selection at all. `results/dsa_dump3_sharded_20260826/` confirms it -- every
  mirrored local row is ~1 765 long.
- The DCP transfer tests verified the MLA latent, whose layout the plan is
  correct for.

## The fix

`plan_replicated_index` plans the replicated index region **in bytes**, one key
run and one scale run per source page, so each source page lands as the writer
would have written it:

    src: [page s keys]                    -> dst: page + sub*key_plane
    src: [page s scales]                  -> dst: page + W*key_plane + sub*scale_plane

The two plane sizes come from the region itself: `KVTransferRegion` carries
`key_plane_bytes` and `scale_plane_bytes`, filled in by `aiter_mla.get_kv_transfer_tensors`
from `index_head_dim` and the page's token count. They are `None` for
token-contiguous regions, which keep the token plan unchanged.

A sub-page interleave on an **unreplicated** index region has no valid plan at
all, so it is now a hard error rather than silent corruption:

    A DCP interleave of 1 shards the DSA index cache below a page, which its
    preshuffled layout does not allow. Run the decode node with
    ATOM_DCP_REPLICATE_INDEX_CACHE=1, or with an interleave of 16.

`scripts/start_glm52_pp4pd_dcp.sh` already defaults `REPLICATE_INDEX_CACHE=1`,
so the default DCP PD configuration takes the fixed path.

### Cost

The index plan can no longer coalesce: a page's key run stops one scale plane
and one padding run short of the next page, so consecutive block ids never make
the source side contiguous. It emits `2 * n_src_pages` descriptors per region
instead of ~1. For a 1 779-block request that is ~3 558 descriptors across 21
full-indexer layers, ~74.7 k, against the ~555 k the MLA regions already emit at
`interleave_size=1` -- about 13 % more descriptors in the batch.

## Validation

Same probe, same endpoint, same seeds as the failing record:

| context | before | after |
|---|---|---|
| 6 951-6 957 tok, seed 23, PD mesh | 0/4 | **4/4** |
| 23 049-23 056 tok, seed 991, depths 0.1/0.5/0.9 | -- | **6/6** |
| 46 048-46 056 tok, seed 991, depths 0.1/0.5/0.9 | fail | **6/6** |

Mirrored logits planes (`ATOM_DSA_LOGITS_DUMP_LAYERS=0,6,10,74`, steps 100-400):
every row 100 % finite over its whole length, all four replicated sub-slots
landed, one contiguous valid run from token 0. Before the fix the valid run was
the decode node's own generated suffix and nothing else.

GSM8K, full 1319 samples, 5-shot, on the fixed relayout:

| eval form | flexible-extract | strict-match |
|---|---|---|
| `/v1/completions`, no chat template (this repo's older script) | 0.9303 | 0.9272 |
| the `recipes/GLM-5.md` form (chat template, `max_tokens=16384`) | 0.9583 | 0.9591 |

The first row is the one comparable to this branch's earlier DCP PD records
(0.9295, 0.9356) and shows no regression. The second is the form the recipe
publishes its TP4 reference against (0.9742 / 0.9727); GSM8K prompts are a few
hundred tokens, far below the `W * index_topk` = 8192 threshold at which the
DCP top-k stops short-circuiting, so neither row can see this bug class -- they
are here to show the fix costs nothing, not to prove it works. The needle rows
above are what prove it.

Artifacts: `results/dcp_index_relayout_fix_20260826/`.

### The kernel time was the same defect

A 3-minute kineto capture on the fixed relayout, 20 minutes into the same
agentic replay that produced the 2026-08-25 baseline, at the same concurrency
and with the same number of top-k calls in the window:

| | broken relayout | page-aware relayout |
|---|---|---|
| `radix_topk_one_block_kernel` per call | 1.601 ms | **0.400 ms** |
| its share of decode GPU time | 50.2 % | **19.7 %** |
| decode step p90 | 68.87 ms | **45.08 ms** |

No other bucket moved by more than noise. The 4x was a NaN radix bucket the
select could not split, manufactured by the transfer -- so the "top-k is 50 % of
decode" finding and the wrong-answer finding were one bug, not two. See
`docs/dcp_decode_topk_bottleneck.md` for the full before/after breakdown.
