# Experimental OPUS-layout DSA FP4 Mono

The full DSA Mono has an MXFP4 index-cache path on gfx950. This is one main
indexer kernel, including Q projection, K LayerNorm/RoPE, Q/K quantization,
cache publication, scoring, TopK and slot output. The existing `step_begin`
launch is still required once per model step. It is not the four-kernel
short-context hybrid.

## Opt In

Use `ATOM_DSA_FP4_OPUS=1` together with `--index-cache-dtype fp4 --block-size 64`.
Keep the checkpoint quantization configuration and do not add online weight
quantization. Mono uses the checkpoint's BF16 `wq_b`; a quantized `wq_b` or
quantized Q latent remains unsupported by Mono.

The native prefill/fallback path also uses the OPUS layout when this flag is
enabled. It requires AITER's `mqa_logits_mxfp4_config` / `mqa_logits_mxfp4` API
from PR 6952. All TP/DCP ranks must use the same flag from process startup.
Never change the flag on a live cache or reuse a legacy FlyDSL FP4 cache.
Without the flag, the legacy FP4 cache layout is unchanged. FP8 also receives
the wide-width planner and short-context improvements in this change.

## Cache ABI

- Packed E2M1 values: contiguous uint8 `[groups, 4, 64, 16]`.
- E8M0 scales: contiguous uint8 `[groups, 2, 32, 4]`.
- One scale per 32 dimensions; even dimension in the low nibble.
- Data offset within a 64-row group: `(chunk * 64 + row) * 16 + byte`.
- Scale offset: `(chunk % 2) * 128 + (row % 32) * 4 + (chunk // 2) * 2 + row // 32`.
- A physical row costs 64 data bytes plus 4 scale bytes, 68 B total, compared
  with 144 B in the padded FP8 DSA cache.
- Native query scales are `[tokens, 2, 32, heads/16]`, as in AITER PR 7128.
  Mono's private query scratch need not share this external representation.
- Every cache page's base address is formed in 64 bits, including pools above
  4 GiB; only in-page buffer offsets are 32-bit.

## Scope

Mono retains its 32-head, 128-dimension, BF16-Q contract. NoPE and RoPE64
(interleaved/NeoX) are supported. Plain decode builds extend through 256 rows.
Uniform qlen-6 verification extends through 256 slots, requiring complete
requests: at most 252 query rows (BS42). Other verify lengths, including
qlen-8, retain the 64-total-row limit.
The standalone kernel accepts complete 64-row page groups; the production
DSA FP4 pool currently requires block size 64.

DCP1 and DCP=TP with interleave 1 are supported. The FP8-specific 512-CTA
tuning is not enabled for FP4. PCP, prefill micro-batching, plugin backends
and cache transfer/offload are not supported by the experimental OPUS DSA
route and are rejected. Whole-model quality has not been qualified.

For DCP1 qlen-6 verification (and the existing narrow qlen-8 domain), a
device-side whole-batch predicate selects independent rank-local work when
all active requests are at most 8192 tokens. Mixed or longer batches retain
peer collaboration. DCP8 keeps communication because the cache is sharded.
The predicate is evaluated inside the captured graph, not at capture time.
The standalone DSA runner uses a matching conditional reset; runners that
share buffers with whole-model Mono kernels retain their original fence.

## Validation Status

Tests include independent tensor-permutation checks, native/Mono cache-byte
comparisons, direct OPUS logits/TopK checks, graph replay, negative K slots,
the all-selected short-context path, BS256 and cache reads/writes past 4 GiB.
GLM checkpoint indexer tests cover DCP1, DCP2, qlen1/6 and 8K/160K contexts.
These are indexer tests, not whole-model generation or long-context accuracy
evaluations.

Same-Q comparisons agree with OPUS. The existing small-batch native Q GEMM
and Mono GEMV can round differently; FP4 amplifies threshold crossings.
This and shape-dependent performance are reasons the FP4 route is opt-in,
not a default replacement. See [the qlen-6 report](dsa_mono_qlen6_benchmark.md)
for the FP8/FP4 comparison, known short-context regressions and reproduction.
