# DSA Mono qlen-6: FP8 and OPUS FP4

## Scope

- Eight MI355X GPUs; TP8, qlen=6, BS=1/8/16/24/32/40.
- Real GLM-5.3-MXFP4 layer-0 indexer weights with synthetic inputs/history.
- BF16 Q projection; checkpoint configuration unchanged; no online weight quantization.
- FP8 page128 and FP4 page64 (OPUS gfx950 MFMA layout).
- Fixed max_model_len=163840 and max_num_seqs=40.
- Complete indexer: Q projection, cache writing, MQA and TopK/slot conversion.
- Mono includes reset; DCP8 native includes candidate exchange communication.
- K/head-weight projection is common precomputed input outside both timings.
- Nine interleaved repeats, 64 graph replays, eight indexer calls per graph.
- Median of slowest-TP-rank CUDA-event latency. Traces confirm every Mono call.

These are indexer microbenchmarks, not end-to-end model throughput or model
accuracy results. The full-model reset is once per forward step, not per
layer. Wider runner buffers change reset cost versus the earlier 64-row
prototype, so small-BS numbers should not be directly compared to that run.

## Results

Microseconds, lower is better. Parentheses show latency reduction relative
to native: 100 * (native - Mono) / native. The final column uses
100 * (FP8 Mono - FP4 Mono) / FP8 Mono. Positive means faster, negative
means slower. Percentages use unrounded measurements.

### 8K, DCP1 (no DCP)

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 24.17 -> 24.53 (-1.50%) | 26.02 -> 23.09 (+11.26%) | +5.91% |
| 8 | 32.41 -> 31.85 (+1.72%) | 31.20 -> 31.15 (+0.17%) | +2.22% |
| 16 | 39.07 -> 40.90 (-4.67%) | 36.54 -> 38.39 (-5.06%) | +6.13% |
| 24 | 46.73 -> 53.59 (-14.67%) | 42.49 -> 49.15 (-15.68%) | +8.28% |
| 32 | 51.37 -> 58.97 (-14.79%) | 47.69 -> 54.01 (-13.26%) | +8.41% |
| 40 | 53.18 -> 80.35 (-51.10%) | 50.84 -> 72.26 (-42.11%) | +10.07% |

### 160K, DCP1 (no DCP)

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 84.52 -> 58.85 (+30.37%) | 66.71 -> 58.85 (+11.79%) | +0.01% |
| 8 | 130.04 -> 60.71 (+53.31%) | 99.34 -> 57.11 (+42.52%) | +5.94% |
| 16 | 255.12 -> 87.81 (+65.58%) | 187.21 -> 79.37 (+57.60%) | +9.60% |
| 24 | 353.05 -> 105.60 (+70.09%) | 212.23 -> 95.77 (+54.88%) | +9.31% |
| 32 | 447.24 -> 145.92 (+67.37%) | 187.24 -> 132.49 (+29.24%) | +9.20% |
| 40 | 511.79 -> 169.06 (+66.97%) | 216.98 -> 150.32 (+30.72%) | +11.09% |

### 160K, DCP8

| BS | FP8: native -> Mono | FP4: native -> Mono | FP4 Mono vs FP8 Mono |
|---:|---:|---:|---:|
| 1 | 81.36 -> 43.95 (+45.99%) | 77.42 -> 44.47 (+42.55%) | -1.20% |
| 8 | 109.99 -> 69.32 (+36.97%) | 100.28 -> 64.46 (+35.72%) | +7.01% |
| 16 | 141.28 -> 103.05 (+27.06%) | 127.00 -> 97.40 (+23.31%) | +5.48% |
| 24 | 171.43 -> 125.98 (+26.51%) | 148.16 -> 113.45 (+23.43%) | +9.95% |
| 32 | 202.93 -> 149.71 (+26.22%) | 171.98 -> 133.68 (+22.27%) | +10.70% |
| 40 | 222.21 -> 168.34 (+24.24%) | 196.66 -> 151.83 (+22.79%) | +9.81% |

## Validation and limitations

- 294 GPU planner cases, including qlen-6 BS11/16/22/32/40/42 and qlen-1 BS256,
  replicated/sharded caches, three ranks and seven length patterns.
- Qlen-6 admission accepts multiples of six through 252; rejects 256/258/384.
  Qlen-8 is not widened. The main kernel retains 256 resident CTAs.
- 12 full-kernel boundary cases: BS11 and BS42, both caches, all three modes.
- 36 timed cases; no native fallback counted as Mono.
- Cache/scale byte equality, request/causal bounds, global TopK counts,
  and repeated graph output consistency passed.
- 40 boundary length-transition patterns, three replays per native/Mono graph
  on all eight ranks: 8K, 8193, a long request in the last tile, 160K, then 8K.
- Four short boundary cases ran rank0 alone with remote addresses zeroed;
  peer mailbox, epoch and shared winner slab remained unchanged.
- Minimum same-Q native/Mono TopK overlap: 100%.
- Minimum original native/Mono overlap: 96.7122396% (FP4 BS1, 160K/DCP8).
  The native GEMM and Mono GEMV can round Q differently. This is not a
  full-model quality pass and must not be described as identical original outputs.

Long context improves at all measured points. At 8K/DCP1, larger batches
remain slower than native. FP4 Mono is generally faster than FP8 Mono,
but that does not imply either beats its native baseline at short context.
These limitations are why the change is submitted for draft review.

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
  for dcp in 1 8; do
    contexts="8192 163840"
    [ "$dcp" = 8 ] && contexts="163840"
    opus=0
    [ "$cache" = fp4 ] && opus=1
    ATOM_DSA_FP4_OPUS=$opus torchrun --standalone --nproc-per-node=8 \
      tests/benchmarks/dsa_indexer/benchmark.py \
      --model "$MODEL" --output /tmp/dsa_indexer_results \
      --cache "$cache" --dcp "$dcp" --contexts $contexts \
      --tag "dcp${dcp}_${cache}"
  done
done
```

For boundary and same-graph transition checks, add `--boundary --batches 11 42`.
Each case saves raw rank timings, correctness diagnostics and compressed
native/Mono Chrome traces. The benchmark now calls the runner's serving
entry point; the measured kernel and selection code are unchanged from
the isolated candidate used for these tables. No four-kernel hybrid is included.
