# GLM-5.3-MXFP4: DSA Mono qlen-6 FP8 and OPUS FP4

For K3.1 Pro results, see the separate
[K3.1 Pro report](dsa_mono_k3_1_pro_qlen6_benchmark.md).

## Scope

- Eight MI355X GPUs; TP8, qlen=6, BS=1/8/16/24/32/40, plus BS42 boundary checks.
- Real GLM-5.3-MXFP4 layer-0 indexer weights with synthetic inputs/history.
- BF16 Q projection; checkpoint configuration unchanged; no online weight quantization.
- FP8 page128 and FP4 page64 (OPUS gfx950 MFMA layout).
- Fixed max_model_len=163840; max_num_seqs=40 for the main short sweep,
  42 for the long sweeps and separate short BS42 boundary checks.
- Complete indexer: Q projection, cache writing, MQA and TopK/slot conversion.
- Mono includes reset; DCP8 native includes candidate exchange communication.
- K/head-weight projection is common precomputed input outside both timings.
- Nine interleaved repeats, 64 graph replays, eight indexer calls per graph.
- Median of slowest-TP-rank CUDA-event latency. Traces confirm every Mono call.
- Production runner serving path, without the isolated candidate module loader.
  Sampled GPU-process monitoring found no foreign-job overlap in accepted runs.

These are indexer microbenchmarks, not end-to-end model throughput or model
accuracy results. The full-model reset is once per forward step, not per
layer. All results below supersede the pre-optimization table in the initial
draft. The native baseline implementation and benchmark inputs are unchanged.

## Short-context changes

The graph-local short path still runs one Mono main kernel plus reset.
It does not invoke native kernels or the archived four-kernel hybrid.

For qlen=6 on the supported gfx950 configuration, the short path now uses
complete-head tiled BF16 Q projection with local quantization, balanced MQA
work assignment, a fast regular-batch planner, and a compact flag reset.
FP8 scoring uses a wider MFMA with unit internal scales. The FP4 OPUS cache
layout is unchanged. Long/mixed contexts retain the collaborative path;
the choice is made from device context lengths inside the captured graph.

Q weights are permuted once without changing their BF16 bits or the original
native weight. This costs 16 MiB per GLM indexer layer (8 MiB for Q LoRA 1024).
See [the implementation scope and cache ABI](dsa_fp4_mono.md).

## Results

Microseconds, lower is better. Parentheses show latency reduction relative
to native: 100 * (native - Mono) / native. The final column uses
100 * (FP8 Mono - FP4 Mono) / FP8 Mono. Positive means faster, negative
means slower. Percentages use unrounded measurements.

### 8K, DCP1 (no DCP)

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 24.15 -> 23.08 (+4.45%) | 26.04 -> 22.08 (+15.22%) | +4.34% |
| 8 | 32.40 -> 30.20 (+6.79%) | 31.62 -> 29.75 (+5.90%) | +1.48% |
| 16 | 39.10 -> 33.55 (+14.20%) | 36.59 -> 31.25 (+14.60%) | +6.86% |
| 24 | 46.62 -> 43.80 (+6.05%) | 42.23 -> 41.90 (+0.80%) | +4.34% |
| 32 | 51.35 -> 47.20 (+8.09%) | 47.70 -> 44.68 (+6.34%) | +5.34% |
| 40 | 52.78 -> 51.29 (+2.81%) | 50.71 -> 48.24 (+4.88%) | +5.95% |

### 160K, DCP1 (no DCP)

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 84.98 -> 58.95 (+30.63%) | 67.01 -> 59.03 (+11.91%) | -0.13% |
| 8 | 129.49 -> 60.85 (+53.01%) | 99.66 -> 57.21 (+42.59%) | +5.97% |
| 16 | 255.22 -> 88.65 (+65.27%) | 188.14 -> 80.18 (+57.38%) | +9.56% |
| 24 | 352.41 -> 106.51 (+69.78%) | 213.03 -> 96.21 (+54.84%) | +9.67% |
| 32 | 446.81 -> 146.63 (+67.18%) | 186.72 -> 131.94 (+29.34%) | +10.01% |
| 40 | 512.20 -> 169.75 (+66.86%) | 217.08 -> 150.22 (+30.80%) | +11.51% |

### 160K, DCP8

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 80.99 -> 44.21 (+45.41%) | 77.23 -> 43.62 (+43.52%) | +1.35% |
| 8 | 110.02 -> 69.32 (+36.99%) | 100.10 -> 65.27 (+34.79%) | +5.84% |
| 16 | 140.93 -> 103.76 (+26.37%) | 126.69 -> 97.20 (+23.28%) | +6.33% |
| 24 | 171.67 -> 123.71 (+27.94%) | 149.14 -> 113.77 (+23.72%) | +8.03% |
| 32 | 202.61 -> 148.15 (+26.88%) | 171.69 -> 133.60 (+22.18%) | +9.82% |
| 40 | 222.22 -> 169.14 (+23.88%) | 196.54 -> 154.93 (+21.17%) | +8.40% |

### BS42 boundary

All six boundary cases also won nine of nine paired repetitions. These are
252 query rows, the largest complete qlen-6 batch below the 256-slot limit.

| Context | DCP | FP8: native -> Mono | FP4: native -> Mono |
|---:|---:|---:|---:|
| 8192 | 1 | 61.28 -> 52.50 (+14.32%) | 52.43 -> 49.28 (+6.00%) |
| 163840 | 1 | 684.59 -> 181.22 (+73.53%) | 224.88 -> 159.99 (+28.86%) |
| 163840 | 8 | 228.97 -> 169.42 (+26.01%) | 201.38 -> 155.58 (+22.74%) |

## Phase diagnosis

A separate BS40/qlen-6/8K/DCP1 FP8 timestamp capture explains the change.
These are instrumented same-CTA intervals, not the uninstrumented event
timings above. Work assignment changes between variants; neither medians
nor maxima should be added to estimate kernel latency.

| Phase | Before (us) | After (us) | Statistic |
|---|---:|---:|---|
| Planner | 2.64 | 0.72 | median |
| Q/K stage | 22.36 | 16.00 | median; after includes Q quantization |
| Separate Q quantization/synchronization | 7.20 | 0.04 | median; quantization moved into Q stage |
| MQA stage | 38.32 | 24.72 | maximum CTA interval |
| Waiting before selection | 19.56 | 5.64 | maximum CTA interval |
| TopK selection | 4.40 | 4.32 | median |

The main gains are the Q/quantization path and the MQA scheduling tail,
not a large speedup of TopK itself. This capture has no usable PMC/ATT
counters and does not establish hardware MFMA or memory-bandwidth saturation.

## Validation and limitations

- 492 focused host/GPU regression tests passed; Ruff, Black, syntax compilation
  and diff checks passed.
- 42 timed cases: 36 main table points and six BS42 boundary cases.
  Every case beat its own native baseline in all nine paired repetitions.
- No native fallback counted as Mono; traces show eight Mono main calls per graph.
- Cache/scale byte equality, request/causal bounds, global TopK counts,
  and repeated graph output consistency passed.
- 140 length-transition patterns, three replays per native/Mono graph
  on all eight ranks: 8K, 8193, a long request in the last tile, 160K, then 8K.
- 14 short cases ran rank0 alone with remote addresses zeroed; peer mailbox,
  epoch and shared winner slab remained unchanged.
- Same-Q native/Mono TopK overlap is 100% for all 14 short cases.
  Minimum original short-case overlap is 98.7060547%; Q projection rounding
  differs in the small-batch paths.
- Minimum same-Q native/Mono TopK overlap over all 42 cases: 100%.
- Minimum original overlap over all cases: 96.7122396% (FP4 BS1, 160K/DCP8).
  The long-path native GEMM and Mono GEMV can round Q differently, as before.
  This is not a full-model quality pass or an FP4-vs-FP8 accuracy qualification.
- Qlen-6 admission accepts multiples of six through 252; rejects 256/258/384.
  Qlen-8 is not widened. The main kernel retains 256 resident CTAs.
- Earlier 294 GPU planner scenarios and 12 BS11/42 boundary checks remain
  recorded in the initial draft. The new regression suite also covers packed
  weight bits, balanced score coverage, compact-reset bounds, and fast/generic
  planner row contracts across short, padded and long batches.

All measured short and long points now beat native. The smallest short
margin is FP4 BS24 at +0.80%; measurement noise or a different workload can
change such a small margin. This is not a universal speed guarantee for
every admitted shape, model, device or context distribution. Whole-model
quality and throughput still need qualification; the PR remains a draft.

## Reproduction

Requires an AMD gfx950 setup with ATOM, FlyDSL, and the OPUS FP4 AITER API
from ROCm/aiter#6952 and #7128. The measured AITER checkout was
`0bff524d7c252a11f68ee59091b2f52daaf9fadb`; the native ATOM baseline was
`498133ed9` plus the OPUS FP4 native-cache plumbing in this change.
Set `MODEL` to your GLM-5.3-MXFP4 checkpoint. No model weights are included.
Run only on idle GPUs; the commands do not stop other jobs.

```bash
python tests/benchmarks/dsa_indexer/check_planner.py --output /tmp/dsa_planner.json
for cache in fp8 fp4; do
  opus=0
  [ "$cache" = fp4 ] && opus=1
  ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
    tests/benchmarks/dsa_indexer/benchmark.py \
    --model "$MODEL" --output /tmp/dsa_indexer_results \
    --cache "$cache" --dcp 1 --contexts 8192 --batches 1 8 16 24 32 40 \
    --repeats 9 --replays 64 --boundary --tag "short_${cache}"
  for dcp in 1 8; do
    ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
      tests/benchmarks/dsa_indexer/benchmark.py \
      --model "$MODEL" --output /tmp/dsa_indexer_results \
      --cache "$cache" --dcp "$dcp" --contexts 163840 \
      --batches 1 8 16 24 32 40 42 --repeats 9 --replays 64 --boundary \
      --tag "long_dcp${dcp}_${cache}"
  done
  ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
    tests/benchmarks/dsa_indexer/benchmark.py \
    --model "$MODEL" --output /tmp/dsa_indexer_results \
    --cache "$cache" --dcp 1 --contexts 8192 --batches 42 \
    --repeats 9 --replays 64 --boundary --tag "short_boundary_${cache}"
done
```

Each case saves raw rank timings, correctness diagnostics and compressed
native/Mono Chrome traces. Aggregated values and source tags are in
[measured_qlen6.json](../tests/benchmarks/dsa_indexer/measured_qlen6.json).
Separate instrumented phase captures are diagnostic only and are excluded
from the performance tables. Original experiment artifacts and the unrelated
dirty working tree are preserved.
