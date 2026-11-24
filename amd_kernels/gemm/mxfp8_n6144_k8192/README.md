# MXFP8 GEMM, N 6144 x K 8192 (`o_proj_full` of the sequence-parallel prefill, TP4)

## What it does
The full attention output projection of MiniMax-M3.1-Flash (K 8192 = 64 heads x 128, N 6144, bf16 out) that the TP4
sequence-parallel prefill runs on each rank's token shard: 1024-8192 rows per rank for 4096-32768-token steps. The rows
and kernels of this shape, registered with the dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md)
(`gemm._SHAPE_HOSTS`); layouts and the numerics contract are that folder's. Callers: `mxfp8_gemm` with row-major codes
and preshuffled scales (`collectives/seq_parallel_handoffs_h6144_q2048_ranks4`, the SP o_proj). No TP4 call has fewer
than 1024 rows; the rows below keep the decode chain's choices.

## Entry points
- The common entries (`mxfp8_gemm`, `mxfp8_linear`, `warmup_mxfp8_gemm_weight`); the warm-up also compiles the
  split-2 row's ready-word clear.
- `host.select`: this shape's config for a row count ahead of the tuned table (None: the table decides); it is the
  dispatch.
- `host.LAUNCHERS`: the launcher of each kernel `select` names.

## Performance
One MI355X. Layer 39 `o_proj_full` weights in 8 cold copies (> 400 MB), 16-100 seeded inputs,
`mxfp8_gemm` with row-major codes and scales (the scale shuffle is timed), alternated rounds, graph replay, median µs;
eager in brackets.

| Rows | M | µs | Baseline µs | Baseline |
|---|---|---|---|---|
| 417-1024 | 417 / 768 / 1024 | 42.01 / 46.33 / 52.15 (43.12 / 46.80 / 52.63) | 54.06 / 56.74 / 56.49 (54.66 / 57.77 / 57.27) | `4wave_nt` 128x256 / same / a 128x192 tile |
| 1281-1536 | 1536 | 68.42 (68.37) | 75.82 (76.74) | `4wave_nt` 256x256 |
| 1537-2048 | 2048 | 81.35 (79.79) | 82.75 (85.05) | `4wave_nt` 256x256 |
| 2049-4608 | 3072 | 126.45 (127.36) | 167.31 (169.79) | `aiter` 128x256 |
| 289-416 | - | -7.5..-9.3% | - | 96x96, 2 K-waves (decode chain) |

5 of 380 timed graph rounds of the split-2 kernel ran 2-5x slow; 3000 eager calls showed none. The cause is not known.

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n6144_k8192 -q`
- `python3 benchmarks/gemm/mxfp8_n6144_k8192/benchmark_n6144_k8192.py --checkpoint <model> --ms 417,1024`
