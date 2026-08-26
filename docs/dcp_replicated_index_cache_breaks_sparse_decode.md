# ATOM_DCP_REPLICATE_INDEX_CACHE=1 breaks decode past index_topk

**Superseded 2026-08-26.** It *is* the KV transfer, and the sharded path is
broken the same way -- it just hides below a threshold this page never crossed.
Root cause, byte-level proof and fix:
`docs/dcp_index_cache_page_relayout.md`. The measurements below stand; the
section "It is not the KV transfer" does not.

Measured 2026-08-26 on MI355X, GLM-5.2-MXFP4, PP4xTP1 prefill -> TP4xDCP4
decode over mooncake PD. `scripts/needle_in_haystack.py`, 6 000-word haystacks
(~6 950 tokens), `--unique-prefix` so nothing is served from a prefix cache.

With the index cache replicated, a DCP decode rank returns degenerate text for
any context longer than `index_topk` (2 048). With it sharded, the same
prompts come back correct. Retrieval accuracy is the only thing that separates
the two: throughput, TTFT, and `err=0` all look healthy either way, and so does
GSM8K, because a GSM8K context never reaches 2 048 tokens and the sparse path
never runs.

## The measurement

| config | endpoint | prompts | retrieved |
|---|---|---|---|
| replicated | mesh proxy (PD) | 4 | **0/4** |
| replicated | decode direct (no RDMA) | 2 + 4 | **0/6** |
| replicated | prefill node (no DCP) | 4 | 4/4 |
| sharded | mesh proxy (PD) | 4 + 4 | 8/8 |
| sharded | decode direct (no RDMA) | 4 + 4 | 6/8 |

The failures are not subtle. A broken answer is a run of `!` -- token 0, the
argmax of a flat or NaN logit vector -- with occasional real tokens:

    The!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

The first token is fine because the prefill node produced it; every token the
DCP node generates after that is garbage.

## It is the sparse path, and only the sparse path

Sweeping the haystack down through `index_topk` on the replicated node puts
the boundary exactly where `top_k_per_row` stops short-circuiting:

| prompt tokens | result |
|---|---|
| 511 | pass |
| 1 777 | pass |
| 2 927 | **fail** |
| 6 951 - 46 050 | **fail** |

Below 2 048 the row is shorter than k, top-k returns everything, and the
indexer cannot affect the answer. Above it, selection starts and the answer
falls apart.

## It *is* the KV transfer (this section was wrong)

The obvious suspect was the mooncake DCP relayout, which moves the indexer
cache a token at a time on the assumption that a token is contiguous. That is
the bug. Both measurements offered against it were artifacts:

- the "decode node hit directly" prompts were served out of the node's prefix
  cache (`cached_tokens=28,416`), so they read back the index cache a *previous
  PD transfer* had already corrupted. With a seed that forces
  `cached_tokens=0`, a locally prefilled context retrieves correctly;
- the sharded PD sweep never reached the sparse path. Each rank's row is
  `g_ctx / W` long, so `top_k_per_row` short-circuits until the global context
  passes `W * index_topk` = 8 192 tokens, and every prompt in that sweep was
  ~6.9 k.

## It is prompt-dependent, which is why it stayed hidden

Two needle seeds at the same length on the same replicated node:

    seed 7,  6 948 - 6 957 tokens, 6 prompts -> 6/6 pass
    seed 23, 6 950 - 6 957 tokens, 4 prompts -> 0/4 fail

Identical token counts appear in both sets with opposite outcomes, so it is
neither length nor alignment. It is deterministic per prompt -- seed 23 fails
again after the node has served hundreds of requests, and seed 7 passes on a
node that has served none -- so it is not a warmup or state effect either.

A bug that fires on some prompts and not others, only past 2 048 tokens, and
only in a mode short-context evals cannot reach, is invisible to every check
this configuration has passed: GSM8K 0.9356 with the replicated cache, a clean
1-hour agentic replay at concurrency 32 with `err=0`, and the DCP transfer
tests, which verify the MLA latent.

## What this explains

`docs/dcp_decode_topk_logits_dump.md` records a decode logits plane that is
~97 % implausible -- 6.2 % NaN, exponents spread over 235 binades, magnitudes
to 1e33. Those dumps come from a replicated-cache run whose median input was
93 k tokens, so every sampled step was in the failing regime. The plane is a
symptom of this bug, and it is **not** a valid reference for what a healthy
DCP logits distribution looks like.

It also means the top-k cost measured in `docs/dcp_decode_topk_bottleneck.md`
(1.60 ms per call, 50.8 % of decode GPU time) was measured on a plane full of
garbage. The kernel is data-dependent, so that number needs re-measuring on
the sharded configuration before it can be attributed to the replicated
cache's wider plane.

## Where to look next (answered)

The replicated cache gives every rank the whole 64-token indexer page of each
virtual block while the MLA KV stays sharded 1/4. Producing that page locally
requires each rank to hold indexer keys for tokens whose latents live on other
ranks. The candidates, in order:

1. whether the local write path (`indexer_k_quant_and_cache` via the decode
   node's own prefill) ever populates the three-quarters of each page that
   belong to other ranks;
2. whether the global top-2 048 selected from the full-width plane is filtered
   to owned slots correctly (`_dcp_owned_counts_gpu`,
   `_dcp_sparse_kv_indptr_gpu` in `aiter_mla.py:486`);
3. whether a rank that ends up owning zero selected slots for a query returns
   an LSE the cross-rank merge can consume.

None of the three. The replicated page is written correctly by the decode
node's own kernel and corrupted on arrival by the relayout, which copies each
16-token source page as one 2 304-byte run into a 9 216-byte destination page
whose keys and scales live in separate planes.

Reproduce the failure by reverting `plan_replicated_index` to the token plan:

    REPLICATE_INDEX_CACHE=1 bash scripts/start_glm52_pp4pd_dcp.sh
    python scripts/needle_in_haystack.py --url http://127.0.0.1:8020/v1/chat/completions \
        --words 6000 --trials 4 --unique-prefix --seed 23
