# Experimental OPUS-layout DSA FP4 Mono

The full DSA Mono has an MXFP4 index-cache path on gfx950. This is one main
indexer kernel, including Q projection, K LayerNorm/RoPE, Q/K quantization,
cache publication, scoring, TopK and slot output. The existing `step_begin`
launch is still required once per model step. It is not the four-kernel
short-context hybrid.

## Opt In

Use `--index-cache-layout mfma --index-cache-dtype fp4 --block-size 64`.
Keep the checkpoint quantization configuration and do not add online weight
quantization. Mono uses the checkpoint's BF16 `wq_b`; a quantized `wq_b` or
quantized Q latent remains unsupported by Mono.

Admitted GLM/K3 indexers prepare `wq_b` once before model warmup using AITER's
standard BF16 `(16,16)` shuffle. Native prefill/fallback GEMM and both Mono
Q paths share that same Parameter and storage address. There is no persistent
row-major or private head-packed copy and no per-step repack. This requires
the companion AITER BF16 `flydsl_hgemm(..., bpreshuffle=True)` extension and
shuffled tuning rows; older AITER leaves the indexer unmodified on native.
Online weight-update runners are refused before shuffling. This does not
change checkpoint values, cache dtype/layout or unrelated model GEMMs.

The native prefill/fallback path uses the same resolved layout. It requires
AITER's `mqa_logits_mxfp4_config` / `mqa_logits_mxfp4` API from PR 6952 and
the HIP `scale_layout="opus"` mode of `indexer_qk_rope_quant_and_cache` from
[AITER PR 7798](https://github.com/ROCm/aiter/pull/7798). This replaces the
Triton companion PR 7730 without changing the ATOM call or OPUS cache ABI.
Use the matching AITER Python sources and rebuilt `module_cache` extension;
an older cached binary is not compatible with the extended HIP signature.
PR 7128's V4 query and compressor writers alone do not provide this entry
point. Mono still performs its own fused cache write; this dependency selects
the native prefill/fallback writer, not a different Mono kernel.
`Config` resolves the layout once and passes it to every TP/DCP worker and
the offload namespace. Never change it on a live cache or reuse a legacy
FlyDSL FP4 cache. The default remains `preshuffle`. FP8 also receives
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
tuning is not enabled for FP4. Standalone `lmcache_mp` supports both planes,
including TP8+DCP8. PCP, prefill micro-batching, plugin backends and other
cache-transfer connectors are rejected by the experimental OPUS DSA route.

For DCP1 qlen-6 verification (and the existing narrow qlen-8 domain), a
device-side whole-batch predicate selects independent rank-local work when
all active requests are at most 8192 tokens. Mixed or longer batches retain
peer collaboration. DCP8 keeps communication because the cache is sharded.
The predicate is evaluated inside the captured graph, not at capture time.
The standalone DSA runner uses a matching conditional reset; runners that
share buffers with whole-model Mono kernels retain their original fence.
Intermediate wide graph buckets are opt-in via `--cudagraph-capture-sizes`,
for example `[64,128,160,192,224,256]`; global capture defaults are unchanged.

### Short-Context Qlen-6 Tuning

On unmasked 256-CU gfx950 devices, replicated 32x128-head indexers with
BF16 Q LoRA 1024/1536/2048 and max_model_len <= 262144 additionally use the
short-query specialization. Its runtime predicate remains inside the graph:
long or mixed batches keep the original collaborative projection/scoring path.

- At 96 or more query rows, each CTA projects complete heads for a 16/32-row
  tile and quantizes from LDS, eliminating the cross-CTA Q/quantization rendezvous.
- A one-time, bit-preserving BF16 permutation supplies both native and Mono.
  Removing the former short-path copy saves 8 MiB per indexer for Q LoRA
  1024, 12 MiB for K3.1 Pro's Q LoRA 1536, or 16 MiB for GLM's Q LoRA 2048,
  per GPU. Initialization still needs temporary shuffle storage. DCP8 had no
  private short-path copy, so it does not receive this memory saving.
- Short MQA work is balanced across the resident CTAs; FP8 uses a wider
  gfx950 MFMA with unit internal scales and unchanged external Q/K scales.
- The planner handles regular short batches directly; padding, contexts below
  the TopK threshold, and mixed/long contexts retain the generic planner.
- The standalone short reset clears only flags that the local step can read.

The OPUS FP4 cache ABI above is unchanged. There is still one main Mono
kernel plus the required reset, not a native fallback or four-kernel hybrid.
The performance reports cover GLM-5.3-MXFP4 and K3.1 Pro; they are not a
performance guarantee for every admitted shape or model.
Historical tables below predate shared weights. The accepted native GEMM
tradeoff and separate shared-weight validation are recorded below.

### Shared-Weight Validation

On gfx950 / FlyDSL 0.5.0, 630 focused ATOM tests passed. Forty TP8 indexer
graph cases cover GLM/K3, FP8/FP4, qlen1/6, 8K DCP1 and 160K DCP1/DCP8;
old/shared Mono Q and sorted TopK hashes match. Eight rank-local checks and
80 same-graph context transitions passed. Repeated weight preparation adds
zero persistent bytes, with one weight-sized temporary shuffle allocation.

Checkpoint BF16 GEMM timings use the shipped tuning rows, without an external
configuration overlay. Units are microseconds; positive delta is slower.

| Model | M | N | K | Raw native | Shared shuffled | Delta |
|---|---:|---:|---:|---:|---:|---:|
| K3.1 Pro | 2048 | 4096 | 1536 | 24.946 | 27.818 | +2.872 |
| GLM-5.3 | 2048 | 4096 | 2048 | 31.686 | 35.421 | +3.735 |
| K3.1 Pro | 16384 | 4096 | 1536 | 150.826 | 157.434 | +6.609 |
| GLM-5.3 | 16384 | 4096 | 2048 | 198.917 | 197.080 | -1.837 |

This native GEMM cost is an accepted memory tradeoff, not a universal
no-regression claim. Nine of the 40 current Mono cases remain slower than
current native. These are indexer graph checks, not whole-model graphs;
whole-model accuracy and live LMCache reload were not rerun for shared weights.

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
not a default replacement. See the
[GLM-5.3-MXFP4 report](dsa_mono_qlen6_benchmark.md) and
[K3.1 Pro report](dsa_mono_k3_1_pro_qlen6_benchmark.md)
for the FP8/FP4 comparisons, measurement scope and reproduction.
