# DeepSeek-V4.1 Flash FP4 Mono indexer

The standalone persistent indexer is enabled by default on supported gfx950
FP4 TP2/TP4 deployments. TP8 requires explicit `ATOM_DSV41_MONO_INDEXER=1`
until TP8 GPU and answer-accuracy qualification is available.
Set `ATOM_DSV41_MONO_INDEXER=0` to retain the embedded/native indexer.
Native decode and ordinary low-concurrency layer Mono share the indexer
kernel. A successfully installed DSpark HI hook and its per-step fallback retain their existing indexer. An install-time HI refusal under `auto` enters ordinary decode and follows the indexer flag.

The kernel reads attention-normalized BF16 hidden rows and MXFP8 query
latents, then fuses query projection, GPT-J tail RoPE, head-weight projection,
FP4 query quantization, paged scoring and stable Top512/Top2048 selection.
FULL layers use the request page table. REINDEX layers use the earlier
candidate-block table and lift the selection back to compressed row IDs.
Both OPUS `mfma` and eight-row `preshuffle` FP4 index-cache layouts are
supported. Query, scale and head-weight buffers use uncached device memory
for producer/consumer communication across XCDs.

Ordinary layer Mono exports normalized hidden rows and query latents from
K1, runs compression and the standalone indexer, then passes its selected
physical rows to K2. Token-parallel K1 gathers each rank's query latents
before the independent launch. The native hook activates for one supported
decode forward. Both paths share one 64-row workspace for logits, block
maxima, projection outputs, selections and indexer scratch. The layer path
uses its first 48 rows. The workspace is allocated at model installation,
before warmup and KV sizing, so its PyTorch and uncached allocations reduce
the measured KV budget. A layer-served forward does not prepare the native
runner; retained candidate data is never cleared by the other path.

## TP work and selection

The replicated cache permits the K3 work planner's three modes:

- **Local:** each rank computes the complete indexer. Short requests avoid
  peer exchange. In addition, build buckets with at most eight query rows
  and capacity at most 262144 select a world-1 binary: measured peer merge
  and synchronization overhead exceeds the scoring saved at these widths.
- **Owned:** independent requests are assigned to ranks. The owner computes
  the request's scores and publishes its selected keys to every reader.
- **Split:** few long requests, or a dominant sufficiently long request, use
  contiguous 128-key-aligned K ranges across ranks. Top512 merges retained
  rank-local roots. Candidate-block Top2048 uses K3's distributed four-pass
  radix directly over the original block logits. Each rank maps the global
  winner IDs through its own page table.

The larger buckets use the same live request ownership/split decisions as
K3. Groups share a partition derived from their final query's bound; earlier
causal queries clip that range. Stable ties retain ascending global key IDs.
Consumer acknowledgements protect winner buffers from reuse across layers.
Native metadata fuses peer reset with step planning, and its first-layer
fence overlaps projection. The layer route reuses its existing peer arena
and step fence. Local native buckets do not reset the peer mailboxes.

REINDEX's compressed candidate domain has at most 16384 keys. It uses a
separate bounded local binary with capacity 18432, sharing the projection
and output buffers. This avoids the larger FULL/candidate selector's resource
footprint. Only synchronization flag regions are reset between steps;
split radix histograms are cleared conditionally when splitting can occur.

The inherited K3 qlen1/8K short-decode optimization remains in this branch,
behind the default-off `ATOM_MONO_K3_SHORT_DECODE=1` switch. The switch only
changes eligible K3 TP8 replicated FP4 qlen1 builds; DSV4.1 routing does not
read it. K3 GPU and numeric qualification is a separate open item.
DSV4.1 adapts its scheduling and lifetime rules to Flash's shuffled MXFP8
query projection, GPT-J RoPE, Top512 and candidate blocks. Its TP partition
operates on replicated caches; K3's DCP-sharded-cache timings are a different
workload and are not an identical performance comparison.

Non-producing FULL layers use a dedicated Top512-only binary and scratch arena.
The build bounds the compressed key domain using the smallest FULL ratio,
sizes local leaves up to 8K and rounds root input widths to per-thread quads.
Owned FULL requests use 8K leaves. Candidate-block and REINDEX routes keep their
original binaries; the K3 request ownership, K partition and ACK protocol remain
the source of runtime work decisions.

## Admission and fallback

Admission requires 256 gfx950 CUs, shuffled `(16,16)` MXFP8 query weights,
BF16 RoPE tables and Flash geometry: 32 index heads, dimension 128, Top512,
eight-row blocks and Top2048 candidate blocks. The independent kernel accepts
at most 64 query rows; ordinary layer Mono retains its 48-row cap. Builds
with private scratch are refused collectively. Prefill, incompatible inputs,
unsupported widths and refused builds use the existing path.

FP8 index caches, attention DP and DSpark HI retain their existing indexers.
At TP2 without attention DP, FP4 widths of 40 rows or more use native layers
with the standalone indexer: the ordinary layer route was slower there in
the earlier measurements. No aiter source or query-interface change is needed.

## DP attention and HI routing

The standalone indexer is disabled under DP attention, including when
`ATOM_DSV41_MONO_INDEXER=1` is explicitly set. Each attention-DP rank owns
different requests and cache rows, so the replicated-cache TP ownership and
K-splitting protocol cannot merge these ranks' selections. Supported DP
layer Mono keeps its existing embedded indexer; unsupported layer steps use
native attention. Non-DP standalone-indexer timings do not qualify DP gains.

Successful HI installation keeps its own indexer on both HI-served steps and
its per-step fallback. An install-time refusal with `ATOM_DSV41_MONO_HI=auto`
instead enters ordinary decode and follows the standalone-indexer flag.
FP8 KV with an FP4 index cache is eligible for the ordinary route; it is the
FP8 **index** cache that keeps the existing indexer.

The shared workspace is reserved before KV sizing, so its cost reduces KV
capacity. Uncached arenas follow the existing process-lifetime policy in
`atom/mono/runtime/uncached.py`; a later owner refusal retains those arenas.
Measure the flag-0/flag-1 KV block counts with the same model, cache dtype,
maximum context, sequence cap and memory utilization when assessing that cost.

## Numerical qualification

GPU differential checks compare scoring and selection exactly against the
native scorer on identical quantized queries and head weights. Projection
is checked separately. The native multi-row BF16 GEMM rounds partial sums
while Mono accumulates FP32, so complete projection chains can move
near-boundary TopK membership and greedy outputs need not be token-identical.
Formal GSM8K and long-context results must therefore compare answer accuracy
against native, rather than require identical generated tokens. TP4 GSM8K
qualification on c6f06f5f3 covered all 1,319 questions for each route.

The tested aiter row-group scorer's long-context fast path mis-scores a
partial final six-query group. Differential tests qualify those tails against
its one-query scorer using the same pages. This test workaround does not
modify aiter or establish that the native partial-group bug is fixed.

Manual qualification from the ATOM repository root:

```bash
export PYTHONPATH=.
export ATOM_DSV41_CHECKPOINT=/data/models/deepseek-ai/DeepSeek-V4.1-Flash
export ATOM_MONO_ENABLE=1 ATOM_DSV41_MONO_INDEXER=1
pytest -q tests/models/deepseek_v41/test_mono_indexer.py

# TP4 integrated exact selection and same-graph mode changes.
V=tests/models/deepseek_v41/mono_indexer_validation
HIP_VISIBLE_DEVICES=1,3,2,0 torchrun --standalone --nproc-per-node=4 "$V/tp_indexer_probe.py"
HIP_VISIBLE_DEVICES=1,3,2,0 torchrun --standalone --nproc-per-node=4 "$V/tp_indexer_switch_probe.py"
# Repeat with TP_FORCE_COOP=1 to qualify cross-TP even at adaptive local widths.

HIP_VISIBLE_DEVICES=1,3,2,0 TP_POINTS=1:8192:1,1:163840:1,12:163840:6 \
  torchrun --standalone --nproc-per-node=4 "$V/tp_indexer_benchmark.py"

# Actual short and ~25.6K prompt generation, supported level3/FULL graphs.
HIP_VISIBLE_DEVICES=1,3,2,0 ATOM_DSV41_MONO_HI=0 \
  torchrun --standalone --nproc-per-node=4 "$V/validate_tp4_cross_level3.py"
```

The HIP order above is specific to the validation machine. Select devices
appropriate to the deployment. TP8 in the new cross-TP qualification is
compile-only; this round does not use the reserved final two physical GPUs.

## Timing methodology

Indexer measurements use real checkpoint projection weights with synthetic
hidden/query/cache inputs, FP4 `mfma`, contexts 8192/163840, and
FULL/candidate/REINDEX paths. The FULL optimization benchmark compares
native, the baseline without FULL specialization, and optimized Mono in seven
interleaved samples, 500 calls per sample in CUDA Graphs,
CUDA Events, and the maximum TP-rank time. Report the median and spread;
record loaded clocks and exclude concurrent model loading/builds/workloads.

Sequence batch is `query rows / qlen`. FULL uses ratio2: a raw 160K sequence
contains 80K index keys. Candidate uses ratio1; REINDEX uses its compressed
candidate domain. `kernel_tp=1` means independent local indexer math on each
model TP rank. A larger binary world can still take the short-request local
path; the benchmark records the actual `execution` mode separately.

These are indexer timings, excluding TTFT, scheduling, serving throughput
and whole-model decoding. The instrumented real-model generation check
contains live projection comparisons and cannot be used as a timing result.
DSV4.1's runtime supports compilation levels 0/3 and rejects 1/2. The current
real-model check uses supported level3/FULL CUDA Graphs. TP4 formal GSM8K and actual 8K/160K whole-model serving were qualified
on c6f06f5f3; the PR preserves their complete results. The default-on
long-context follow-up is recorded separately below.


## Current FULL optimization qualification (2026-11-19)

The benchmark uses real checkpoint projection weights and synthetic inputs/cache, seven paired samples of 500 graph calls, and the maximum TP-rank event time. It compares native, the original selection path with FULL specialization disabled, and the optimized path. All comparisons use the same inputs in one process.

| Context | qlen | Sequence BS | Baseline μs | Optimized μs | Reduction vs baseline | Reduction vs native |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 1 | 1 | 18.569 | 16.268 | 12.39% | 27.19% |
| 163840 | 1 | 1 | 34.292 | 29.407 | 14.24% | 29.01% |
| 8192 | 1 | 8 | 21.247 | 19.036 | 10.41% | 19.43% |
| 163840 | 1 | 8 | 41.058 | 35.988 | 12.35% | 22.88% |
| 8192 | 1 | 16 | 22.975 | 20.775 | 9.58% | 14.15% |
| 163840 | 1 | 16 | 44.855 | 42.242 | 5.83% | 14.81% |
| 8192 | 1 | 24 | 23.769 | 21.545 | 9.36% | 13.46% |
| 163840 | 1 | 24 | 46.874 | 44.818 | 4.39% | 25.14% |
| 8192 | 1 | 32 | 24.022 | 21.801 | 9.25% | 13.37% |
| 163840 | 1 | 32 | 47.920 | 45.350 | 5.36% | 26.95% |
| 8192 | 1 | 40 | 23.881 | 22.072 | 7.57% | 14.19% |
| 163840 | 1 | 40 | 51.772 | 48.962 | 5.43% | 23.47% |
| 8192 | 6 | 1 | 23.152 | 21.139 | 8.70% | 19.73% |
| 163840 | 6 | 1 | 41.177 | 35.714 | 13.27% | 22.05% |
| 8192 | 6 | 8 | 26.685 | 24.391 | 8.60% | 13.14% |
| 163840 | 6 | 8 | 52.549 | 49.464 | 5.87% | 21.69% |

All 48 TP4 main points, six TP4 qlen6/BS2 supplements, and 18 TP2 representative points have faster medians than native. All 504 paired samples are faster than native. Every measured baseline/optimized binary has zero private scratch. Candidate and REINDEX retain their original kernels; the unchanged 160K qlen6/BS2 REINDEX point measured 1.29% slower than the paired baseline, but 6.43% faster than native.

Validation:

- Final dispatch/build-key/resource suite: 55 passed, including TP2/4/8 FULL binaries with zero private scratch (TP8 compile-only).
- Related CPU/GPU numerical and handoff regression: 473 passed, 18 subtests. CPU import checks with real aiter/FlyDSL blocked: 124 passed, 2 skipped, 18 subtests.
- TP4 and TP2 each passed 12 real-weight exact-selection cases. TP4 adaptive and forced cross-TP same-graph mode changes each passed 63 transitions.
- Real TP4 checkpoint, level3/FULL Graph: all 16 short/~25.6K prompt cases at BS1/8 completed. Both Mono routes checked all eight live head-weight projections with zero error markers. Native plus standalone indexer matched native on 18/18 generated sequences; ordinary layer Mono matched 17/18; native repeat matched 18/18. One long-prompt BS8 layer Mono sequence differed. Formal GSM8K was separately qualified on c6f06f5f3; instrumented decode_ms is not a performance result.
- After the complete model results were written, aiter IPCBuffer/CustomAllreduce destructors printed ignored PyTorch schema errors during interpreter shutdown; torchrun exited 0. The full log is retained.
- Black, Ruff and git diff --check passed.

BS1 reaches K3's earlier approximate 25.6%–38.1% FULL reference. BS8/16 and some qlen6 points remain below it; projection, TopK and TP/DCP cache geometry differ. TP4 formal GSM8K and actual 8K/160K serving measurements are available in the PR qualification results. These results do not establish bitwise equivalence or qualify TP2/TP8 model accuracy.

Run the paired retained-version benchmark with:

```bash
V=tests/models/deepseek_v41/mono_indexer_validation
HIP_VISIBLE_DEVICES=1,3,2,0 TP_POINTS=1:8192:1,1:163840:1,8:163840:1 \
  torchrun --standalone --nproc-per-node=4 "$V/tp_full_optimization_benchmark.py"
```

## Default-on TP4 long-context qualification (2026-11-19)

The GPU runs use `d2aa3faeef6aec4b6f2967ff70af2c1e7273220b`. Documentation-only follow-ups do not
change the code qualified here. Real DeepSeek-V4.1-Flash weights on
MI355X/gfx950, physical GPUs 0–3, TP4, BF16 KV and FP4/mfma index cache;
level3/FULL CUDA Graph, no HI/DSpark. Both Mono routes leave
`ATOM_MONO_ENABLE` and `ATOM_DSV41_MONO_INDEXER` unset; all four ranks
recorded enabled defaults. The ordinary layer runner recorded
`indexer_apart=true` and all capture widths built without refusal.

The fixed paired set contains 352 RULER inputs (11 tasks, 16 samples
per task/budget, 128K and extended 160K budgets) and 64 LongBench-v2
documents selected before inference by actual length and subdomain
(24/24/16 in 32K–64K/64K–128K/128K–160K). RULER uses lm-eval 0.4.13's
original generators and string_match_all. LongBench-v2 uses a fixed
generative answer-letter prompt; this is a subset, not the full official
benchmark score. No source truncation; every API prompt length checked.
Common settings: temperature 0, 512 output-token cap, HTTP concurrency 8.

| Route | RULER score (512) | LB raw parser (512) | LB raw parser (2048) | Caps: primary 416 / follow-up 64 |
|---|---:|---:|---:|---:|
| native | 99.754% | 37/64 | 37/64 | 0 / 0 |
| native_indexer | 99.754% | 37/64 | 34/64 | 1 / 0 |
| layer_indexer | 99.754% | 35/64 | 38/64 | 3 / 0 |

The original extractor missed some Markdown answer formats. A uniform
reference-independent normalization of every route/budget is also retained:

| Route | LongBench normalized answers (512) | LongBench normalized answers (2048) |
|---|---:|---:|
| native | 37/64 | 37/64 |
| native_indexer | 37/64 | 36/64 |
| layer_indexer | 35/64 | 38/64 |

The independent-only 2048 route differs on one answer choice versus
native (36/64 versus 37/64). These are small-subset observations, not proof
of per-question equivalence or formal non-inferiority. RULER aggregates
are equal; all normalized paired changes are retained in the PR results.

Two primary layer-Mono LongBench arithmetic responses exhausted the
512-token budget before producing their final answer. A separate uniform
2048-token follow-up repeats all 64 documents on all three routes with
unchanged prompts/scoring. Both runs and their caps remain recorded; the
follow-up does not replace the original primary samples.

Budget is not actual input length: most 160K retrieval/tracking inputs
are roughly 163K tokens; the original CWE generator produces
132,546–132,762 tokens at that budget, and FWE varies between
146,091–162,630. These aggregation cases are not labeled exact-160K.
Paired changes and capped response details are retained in the PR and
qualification archive; observations do not establish formal non-inferiority.

The default-on dispatch/HI suite passed 123 tests; selected CPU regressions
passed 454 tests plus 18 subtests; import-boundary checks with real
aiter/FlyDSL blocked passed 124 tests, 2 skipped, 18 subtests. These
suites overlap. A separate 16-concurrent short passkey smoke passed 16/16.
TP2/TP8 long-context accuracy, FP8 and HI GPU qualification are outside
this follow-up. No aiter source/query-interface changes were made.

Dataset SHA256: `5f0b75439f4dea8ec8705847df9d72c9f8572b12e2eaea5b8eea07e4a96c9560`.

Full model, indexer performance and paired results:
[PR #3524 qualification](https://github.com/ROCm/ATOM/pull/3524).
