# DeepSeek-V4.1 Flash FP4 Mono indexer

The standalone persistent indexer is enabled by default on supported gfx950
FP4 deployments: `ATOM_MONO_ENABLE=1 ATOM_DSV41_MONO_INDEXER=1`.
Set `ATOM_DSV41_MONO_INDEXER=0` to retain the embedded/native indexer.
Native decode and ordinary low-concurrency layer Mono share the indexer
kernel. The DSpark HI route and its fallback retain their existing indexer.

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
decode forward. It avoids preparing an unused second buffer set when the
ordinary layer runner serves that forward.

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

The inherited K3 qlen1/8K short-decode optimization remains in this branch.
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
real-model check uses supported level3/FULL CUDA Graphs. Formal GSM8K and
actual 8K/160K whole-model serving performance remain to be qualified.


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
- Real TP4 checkpoint, level3/FULL Graph: all 16 short/~25.6K prompt cases at BS1/8 completed. Both Mono routes checked all eight live head-weight projections with zero error markers. Native plus standalone indexer matched native on 18/18 generated sequences; ordinary layer Mono matched 17/18; native repeat matched 18/18. One long-prompt BS8 layer Mono sequence differed. Formal GSM8K remains unqualified; instrumented decode_ms is not a performance result.
- After the complete model results were written, aiter IPCBuffer/CustomAllreduce destructors printed ignored PyTorch schema errors during interpreter shutdown; torchrun exited 0. The full log is retained.
- Black, Ruff and git diff --check passed.

BS1 reaches K3's earlier approximate 25.6%–38.1% FULL reference. BS8/16 and some qlen6 points remain below it; projection, TopK and TP/DCP cache geometry differ. TP4 formal GSM8K and actual 8K/160K serving measurements are available in the PR qualification results. These results do not establish bitwise equivalence or qualify TP2/TP8 model accuracy.

Run the paired retained-version benchmark with:

```bash
V=tests/models/deepseek_v41/mono_indexer_validation
HIP_VISIBLE_DEVICES=1,3,2,0 TP_POINTS=1:8192:1,1:163840:1,8:163840:1 \
  torchrun --standalone --nproc-per-node=4 "$V/tp_full_optimization_benchmark.py"
```
