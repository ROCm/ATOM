# DeepSeek-V4.1 Flash FP4 Mono indexer

Set `ATOM_MONO_ENABLE=1 ATOM_DSV41_MONO_INDEXER=1` at process startup to opt
in to a standalone persistent indexer on supported gfx950 deployments.
`ATOM_DSV41_MONO_INDEXER` defaults to **0**, pending formal model accuracy
qualification. Native decode and ordinary low-concurrency layer Mono share
the indexer kernel. The DSpark HI route retains its existing implementation.

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
This is why the route remains opt-in pending formal GSM8K.

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
FULL/candidate/REINDEX paths. Each point uses seven alternating
native/local/automatic-Mono samples, 500 calls per sample in CUDA Graphs,
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


## Current qualification results (2026-11-19)

On g15 with the DeepSeek-V4.1-Flash checkpoint and unchanged aiter `b5b4bf978160b2d495e59b163e932fadc584675d`:

- Final relevant regression: 451 passed, 18 subtests. CPU installation/import checks with real aiter/FlyDSL imports blocked: 124 passed, 2 skipped, 18 subtests.
- Selected unit/GPU metadata/dispatch/build suite: 199 passed, 18 subtests.

TP2 and TP4 each passed 12 integrated exact-selection cases and 63 same-graph request-mode changes for both the adaptive and forced cross-TP policy. Isolated root merge passed 48 cases and distributed radix passed 24, including ties/IEEE patterns and delayed consumers. Six TP2/4/8 long builds (rows1/48) have zero private scratch; TP8 is compile-only.

Real TP4 checkpoint generation at supported level3/FULL CUDA Graph: 16 short and ~25.6K-prompt cases at BS1/8 completed. Both Mono routes checked live head-weight projections on all eight indexer layers with zero error markers. Each Mono route matched the first native run on 18/18 sequences; native repeat matched 17/18. This instrumented check is not GSM8K or a latency benchmark.

TP4 main matrix: 48 points, 48 faster medians, 0 within +/-0.5%; latency reductions 3.55% to 43.29%. TP2 representative matrix: 18 points, 18 faster medians, 0 within +/-0.5%; latency reductions 7.90% to 44.29%.

TP4 qlen1 latency reductions versus native:

| Sequence BS | 8K FULL | 8K candidate | 8K REINDEX | 160K FULL | 160K candidate | 160K REINDEX |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.33% | 30.60% | 17.35% | 17.63% | 43.29% | 16.38% |
| 8 | 9.74% | 25.83% | 15.72% | 12.21% | 39.88% | 14.61% |
| 16 | 5.51% | 21.53% | 7.85% | 10.15% | 42.43% | 6.83% |
| 24 | 4.46% | 18.87% | 6.51% | 23.02% | 36.49% | 7.68% |
| 32 | 5.14% | 18.18% | 3.55% | 23.76% | 40.49% | 4.19% |
| 40 | 6.78% | 22.28% | 8.01% | 20.06% | 40.34% | 11.10% |

TP4 qlen6 latency reductions (query rows = sequence BS x 6):

| Sequence BS | 8K FULL | 8K candidate | 8K REINDEX | 160K FULL | 160K candidate | 160K REINDEX |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.99% | 23.73% | 20.78% | 10.89% | 40.03% | 16.60% |
| 8 | 5.17% | 19.39% | 7.13% | 17.04% | 33.73% | 7.29% |

TP2 qlen1 representative points:

| Sequence BS | 8K FULL | 8K candidate | 8K REINDEX | 160K FULL | 160K candidate | 160K REINDEX |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.74% | 30.01% | 16.15% | 18.22% | 44.29% | 15.50% |
| 8 | 9.70% | 27.65% | 16.14% | 13.53% | 39.64% | 14.51% |
| 40 | 7.90% | 23.09% | 8.56% | 9.11% | 21.82% | 11.04% |

TP4 qlen6/BS2 supplements exercise actual K splitting at 160K for FULL/candidate; REINDEX remains local:

| Sequence BS | 8K FULL | 8K candidate | 8K REINDEX | 160K FULL | 160K candidate | 160K REINDEX |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.42% | 17.62% | 12.14% | 9.63% | 31.02% | 7.44% |

K3's earlier 160K/DCP8 FULL reference was about 25.6%-38.1%. The low-batch FULL results here do not reach that reference. Different projection, TopK and TP/DCP cache geometry prevent an identical comparison. The adaptive local route removes the peer overhead measured at these small widths. Original pre-adaptive results and tuning attempts are archived separately.
