# K3.1 Pro: DSA Mono qlen-6 FP8 and OPUS FP4

## Scope and provenance

- Eight MI355X GPUs; TP8, qlen=6, BS1/8/16/24/32/40 plus BS42 (252 query rows).
- Real K3.1 Pro first DSA-layer indexer weights, synthetic activations/history.
- Checkpoint model_type is `kimi_k31`; the text configuration is `kimi_linear`.
  The display name is K3.1 Pro. Existing checkpoint paths, runtime identifiers,
  and historical source tags are retained for reproducibility.
- Weight prefix: `language_model.model.layers.3.self_attn.indexer`.
- BF16 Q LoRA 1536, 32 heads x 128 dimensions, TopK 2048, NoPE (RoPE dimension 0).
- Checkpoint configuration unchanged; no online weight quantization.
- FP8 page128 and OPUS gfx950 MFMA FP4 page64; cache/scale layouts unchanged.
- Fixed max_model_len=163840. max_num_seqs=40 for the main short sweep, 42 for
  long sweeps and the separate short BS42 checks.
- Complete indexer timing includes Q projection, cache writing, MQA,
  TopK/output and Mono reset; DCP8 native includes candidate exchange.
  Common precomputed K/head-weight projection is outside both timings.
- Nine interleaved repeats, 64 graph replays, eight indexer calls per graph;
  median of the slowest-rank CUDA-event latency.
- Actual runner serving path. Traces confirm Mono calls, not native fallback.
  Accepted runs have no sampled foreign-GPU-job overlap.

**The 28 DCP1 points are newly measured after enabling the 1536 short
specialization. The 14 DCP8 points are carried forward from the earlier
`369dfea5b` sweep and were not rerun after that update.** The specialization
is gated on replicated caches (DCP1); the DCP8 path is unchanged.

These are indexer microbenchmarks, not whole-model throughput or model-quality
results. The tested AITER revision is
`0bff524d7c252a11f68ee59091b2f52daaf9fadb`, as in the
[GLM-5.3-MXFP4 report](dsa_mono_qlen6_benchmark.md). The native baseline is
ATOM `498133ed9` plus this PR's OPUS native-cache plumbing, not today's main.

## Short-context change

Admit Q LoRA 1536 alongside 1024/2048 in the existing gfx950 short-query
specialization. The packed BF16 projection already handles complete
512-column slabs, and the existing NoPE path preserves the native query/cache
rounding conventions. The checkpoint weights are permuted once, without
changing their bits or the original native weights; this adds 12 MiB per
K3.1 Pro indexer layer.

For qlen=6, the short path uses the fast planner, balanced local MQA work,
and, at 96 or more query rows, complete-head tiled Q projection with local
quantization. It is still one main Mono kernel plus reset, not the archived
four-kernel path.

The device-side predicate selects independent rank-local execution for short
batches inside the same captured graph. Long/mixed batches retain the existing
collaborative algorithm. No Python-side switch on live context lengths is added.

## Results

Microseconds, lower is better. Parentheses:
`100 * (native - Mono) / native`.
Final column:
`100 * (FP8 Mono - FP4 Mono) / FP8 Mono`.
Positive means faster. Percentages use unrounded timings.

### 8K, DCP1 (no DCP), after optimization

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 22.85 -> 21.71 (+5.00%) | 25.33 -> 22.86 (+9.76%) | -5.27% |
| 8 | 30.40 -> 28.79 (+5.31%) | 29.81 -> 29.12 (+2.31%) | -1.17% |
| 16 | 38.72 -> 30.24 (+21.90%) | 36.87 -> 29.65 (+19.57%) | +1.93% |
| 24 | 43.34 -> 39.94 (+7.85%) | 41.08 -> 37.32 (+9.17%) | +6.56% |
| 32 | 47.96 -> 43.32 (+9.68%) | 47.34 -> 40.64 (+14.15%) | +6.18% |
| 40 | 50.56 -> 47.21 (+6.62%) | 49.83 -> 44.58 (+10.53%) | +5.56% |
| 42 | 59.74 -> 49.13 (+17.76%) | 50.78 -> 45.36 (+10.68%) | +7.67% |

All 14 short cases beat native in all nine paired repetitions. The smallest
margin is FP4 BS8 at +2.31%; this is not a universal speed guarantee.

At BS40, the previous Mono took 77.32 us (FP8) and 70.75 us (FP4).
The optimized Mono takes 47.21 us and 44.58 us, latency reductions of 38.94%
and 36.98% relative to the previous Mono, respectively.

### 160K, DCP1 (no DCP), regression after optimization

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 83.12 -> 58.15 (+30.04%) | 66.27 -> 58.49 (+11.74%) | -0.59% |
| 8 | 128.47 -> 59.75 (+53.49%) | 99.51 -> 56.27 (+43.45%) | +5.83% |
| 16 | 253.15 -> 87.23 (+65.54%) | 135.25 -> 79.59 (+41.15%) | +8.75% |
| 24 | 401.23 -> 104.44 (+73.97%) | 159.49 -> 94.51 (+40.74%) | +9.50% |
| 32 | 443.47 -> 143.65 (+67.61%) | 186.87 -> 129.99 (+30.44%) | +9.51% |
| 40 | 508.77 -> 167.01 (+67.17%) | 216.02 -> 148.33 (+31.33%) | +11.19% |
| 42 | 698.91 -> 177.48 (+74.61%) | 224.96 -> 156.55 (+30.41%) | +11.80% |

All 14 long DCP1 cases beat native in all nine paired repetitions. Relative
to the previous Mono, measured long-context latency changes are within 1.36%.

### 160K, DCP8, carried forward from the earlier sweep

These values were measured at `369dfea5b`, before the short admission update.
They are included for comparison, not presented as a new post-update rerun.

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 77.66 -> 43.32 (+44.22%) | 76.36 -> 43.34 (+43.25%) | -0.05% |
| 8 | 108.71 -> 67.77 (+37.66%) | 97.84 -> 66.46 (+32.07%) | +1.93% |
| 16 | 140.86 -> 102.26 (+27.40%) | 127.61 -> 94.56 (+25.90%) | +7.52% |
| 24 | 169.15 -> 123.28 (+27.12%) | 148.12 -> 111.77 (+24.54%) | +9.34% |
| 32 | 200.58 -> 145.19 (+27.61%) | 171.35 -> 128.97 (+24.73%) | +11.18% |
| 40 | 219.84 -> 165.55 (+24.69%) | 196.16 -> 148.00 (+24.55%) | +10.60% |
| 42 | 226.90 -> 169.46 (+25.31%) | 200.53 -> 152.44 (+23.98%) | +10.04% |

## Phase diagnosis

Separate FP8 rank3 timestamp captures at qlen=6, 8K/DCP1. These are
instrumented same-CTA durations, not the event timings in the tables.
Only projection-active CTAs contribute to Q medians. MQA includes operand
readiness waits; its active work assignment changes between variants.
These medians and maxima must not be added to estimate whole-kernel latency.

| BS | Q/cache/quant median: before -> after (us) | MQA/wait max: before -> after (us) | TopK/output median: before -> after (us) |
|---:|---:|---:|---:|
| 16 | 14.64 -> 7.96 | 12.48 -> 18.04 | 6.12 -> 6.00 |
| 40 | 27.40 -> 11.84 | 38.24 -> 23.88 | 6.32 -> 6.36 |

The direct improvement is in Q projection/quantization. At BS40 the MQA tail
also shrinks; BS16's score tail grows as CTAs reach scoring at different times.
TopK is essentially unchanged. No PMC/ATT hardware-saturation claim is made.

## Validation and limitations

- 499 focused indexer host/GPU tests passed for the optimization. Ruff, Black
  and diff checks passed. New tests cover bit-preserving 1536 packing and
  FP8/FP4 replicated/DCP admission.
- All 28 newly measured points beat native in every one of nine paired repeats.
- Traces confirm eight Mono main calls per graph, with Q LoRA 1536 / RoPE0.
- Cache/scale byte equality, causal/request bounds, global TopK counts and
  eager/replay output consistency passed.
- 70 new same-graph length-transition patterns, three replays per arm on eight
  ranks: 8K, 8193, mixed short/160K, 160K, then 8K.
- 14 new short rank0-only/null-peer tests passed; peer mailbox, epoch and winner
  slab remained unchanged.
- Same-Q native/Mono TopK overlap is 100% in every newly measured case.
  Minimum original overlap is 97.55859375%; native GEMM and Mono GEMV can
  round Q differently. This is not a whole-model or FP4-vs-FP8 quality pass.
- The carried-forward DCP8 cases had 100% same-Q overlap and minimum original
  overlap 97.5016276%, plus 70 earlier graph-transition patterns.
- DCP8 performance and whole-model generation were not rerun for this update.

The first long FP8 and first short BS42 FP4 runs were excluded because the
final GPU-monitor sample failed to parse warnings mixed with JSON after
results were saved. Both were rerun; the monitor now separates stdout/stderr.
Original records remain preserved. Exclusion was not based on performance.

## Reproduction

Use the same gfx950/FlyDSL/OPUS AITER environment as the GLM report. Set
`MODEL` to the K3.1 Pro checkpoint. The benchmark reads nested text and
quantization configuration and accepts the first DSA-layer weight prefix.
No weights are included; no checkpoint file or runtime model identifier is renamed.
Run only on idle GPUs; do not stop other jobs.

```bash
PREFIX=language_model.model.layers.3.self_attn.indexer
for cache in fp8 fp4; do
  opus=0
  [ "$cache" = fp4 ] && opus=1
  ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
    tests/benchmarks/dsa_indexer/benchmark.py \
    --model "$MODEL" --weight-prefix "$PREFIX" --output /tmp/dsa_indexer_results \
    --cache "$cache" --dcp 1 --contexts 8192 --batches 1 8 16 24 32 40 \
    --repeats 9 --replays 64 --boundary --tag "k3_1_pro_short_${cache}"
  for dcp in 1 8; do
    ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
      tests/benchmarks/dsa_indexer/benchmark.py \
      --model "$MODEL" --weight-prefix "$PREFIX" --output /tmp/dsa_indexer_results \
      --cache "$cache" --dcp "$dcp" --contexts 163840 \
      --batches 1 8 16 24 32 40 42 --repeats 9 --replays 64 --boundary \
      --tag "k3_1_pro_long_dcp${dcp}_${cache}"
  done
  ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
    tests/benchmarks/dsa_indexer/benchmark.py \
    --model "$MODEL" --weight-prefix "$PREFIX" --output /tmp/dsa_indexer_results \
    --cache "$cache" --dcp 1 --contexts 8192 --batches 42 \
    --repeats 9 --replays 64 --boundary --tag "k3_1_pro_boundary_${cache}"
done
```

The DCP8 command is provided for reproduction; the DCP8 numbers above are
from the earlier sweep. Every case saves raw rank timings, diagnostics and
compressed native/Mono Chrome traces. See
[measured_k3_1_pro_qlen6.json](../tests/benchmarks/dsa_indexer/measured_k3_1_pro_qlen6.json)
for aggregated data, source tags, phase evidence and per-group provenance.
