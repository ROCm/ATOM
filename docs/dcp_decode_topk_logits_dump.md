# What the DSA decode top-k actually sees

**Corrected 2026-08-26.** The plane is wrong because the mooncake DCP relayout
moves DSA index pages with the MLA latent's token addressing, and the resulting
NaN mass is also what makes the top-k slow -- so the two findings below are one
bug, not two. See `docs/dcp_index_cache_page_relayout.md`.

Mirrored the DSA indexer's decode logits plane out of a live GLM-5.2 PD run and
measured it against the radix kernel's cost model. Two results: the plane is
*benign* for the kernel, and the plane is *wrong* over every token that came
from the prefill node.

Run: 2026-08-26 06:02-07:02 UTC, MI355X, GLM-5.2-MXFP4, PP4xTP1 prefill ->
TP4xDCP4 decode over mooncake PD, `ATOM_DCP_REPLICATE_INDEX_CACHE=1`, aiperf
replaying `cc-traces-weka-062126` at concurrency 32.
Artifacts: `results/dsa_logits_1h_20260826/` -- 40 snapshots, 8 rows each,
decode steps 400..16000, 300 rows over 2 048 tokens.

## How the plane was captured

Decode runs under a FULL whole-forward CUDA graph, so the Python body of
`sparse_attn_indexer` executes only while the graph is captured -- a `.cpu()`
in it would read one warmup batch and nothing else. `atom/utils/dsa_logits_dump.py`
instead allocates its mirror during the eager warmup forward that precedes each
capture and captures a `copy_` into it, so every replay refreshes the mirror and
the eager tail of `run_model` reads back the plane the top-k just consumed. The
record call sits between `deepgemm_fp8_paged_mqa_logits` and
`top_k_per_row_decode` (`deepseek_v2.py:1851`), inside a custom-op body, which
is compile-safe. It is off unless `ATOM_DSA_LOGITS_DUMP` names a directory, and
only rank 0 mirrors -- under a replicated index cache all four DCP ranks compute
the same plane and would race on one path.

## What the plane looks like (measured on the broken relayout)

Over all 300 rows:

| | min | p50 | max |
|---|---|---|---|
| row length | 3 253 | 73 827 | 694 598 |
| pass-0 buckets populated (of 4096) | 1 185 | 2 354 | 3 047 |
| median bucket occupancy | -- | ~15 | -- |
| tie mass at the k-th key | 1 | **1** | 1 |
| survivors in the k-th bucket | 1 | 74 | 321 |

`docs/dcp_decode_topk_bottleneck.md` guessed at two data-dependent costs. Both
are absent:

- **No tie pile-up.** The k-th largest key is unique in every row measured, so
  `last_filter_stable_fast` never sorts an over-subscribed boundary.
- **No bucket concentration.** More than half of the 4096 pass-0 buckets are
  populated and the median holds ~15 values, because the plane spans ~235
  binades of exponent. The one crowded bucket is the NaN bucket (see below),
  and it sits at the bottom of the order where no pass after 0 revisits it.

Those two are absent, but the plane is not benign: the crowded NaN bucket is
5-7 % of the row, and a radix pass cannot split it. That is the data-dependent
cost, and it disappears with the relayout fix. On top of it sits the kernel's
shape: `radix_topk_one_block_kernel` launches one workgroup per row, compact is
disabled so every pass re-scans the full row, and with `BitsPerPass=12` on 256
CUs that is 3 passes plus a stable final collect -- roughly four full-row scans
driven by a single workgroup, with only `batch_size` (~16-32) workgroups on a
256-CU part. A synthetic A/B (`dsa_logits_analyze.py --bench`: the dumped plane
vs randn vs a wide-exponent fill vs a single repeated value, all at the same
lengths) is the causal test and needs a free GPU.

## The prefill node's tokens have no valid indexer keys

Every row splits cleanly in two. A suffix carries plausible logits -- finite,
|v| < 1e4, the -inf padding out to `ceil64(ctx)` exactly where it belongs. Everything
before it is 1e35-scale noise with 5.4-6.9 % NaN and a 235-binade exponent
spread.

The suffix is the tokens the decode node generated itself. Slot 0 of the
snapshot series, sampled every 2 400 decode steps:

| step | 400 | 2 800 | 5 200 | 7 600 | 10 000 | 12 400 |
|---|---|---|---|---|---|---|
| valid suffix | 2 | 671 | 2 513 | 5 042 | 7 571 | 10 098 |

It grows by one entry per decode step. At step 400 that row already held a
100 K-token context of which **2** entries were valid. The transferred prompt --
the whole point of PD -- contributes no usable indexer logits, so the sparse
top-k picks 2 048 essentially arbitrary prompt positions. The selected indices
confirm it: uniform across every decile of the context, none in the last 512
positions, none in the first 8. Real DSA selection is sink-heavy and
recency-heavy.

Root cause: the mooncake producer writes the DSA index cache with the MLA
latent's addressing, which no byte range of a preshuffled index page obeys.
Every transferred token's keys are dequantized against a scale plane filled
with key bytes. See `docs/dcp_index_cache_page_relayout.md` for the byte-level
proof and the fix. A plane recorded here is a symptom of that bug and is not
the "real data" reference for a top-k benchmark: after the fix the same layers
mirror 100 % finite over the whole row.

### Why no accuracy test has caught this

`top_k_per_row` short-circuits when the row is shorter than k. Every GSM8K
context is under 2 048 tokens, so top-k selects everything, sparse degenerates
to dense, and the indexer's output cannot affect the answer. The 0.9295 and
0.9356 GSM8K scores recorded for the DCP PD path say nothing about whether the
indexer cache survives the transfer. The test that would catch it is
`scripts/needle_in_haystack.py` at a context past 2 048 -- and past ~16 K, where
2 048 of N stops being most of the context.

## Fixed on the way in

`build_for_cudagraph_capture` (`aiter_mla.py`) did not set `token_to_seq_idxs`
for the non-MTP sparse branch, so a non-MTP DCP decode server died during
capture at `dcp_ops.py:1035` with `'NoneType' object has no attribute
'contiguous'`. Commit 20c63290 added the identity map to the runtime metadata
builder only; the MTP branch already had one, which is why the last validation
run did not see it.
