# MXFP8 GEMM, N 2560 x K 6144 (QKV projection, TP4)

## What it does
The QKV projection of the MiniMax-M3.1-Flash sparse attention layer at TP4: N 2560 = q (16 heads) | k | v | index-q |
index-k (one head each) x 128, K 6144, bf16 out. The rows and kernels of this shape, registered with the dispatcher in
[`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); layouts and the numerics contract are that
folder's. Callers: the fused norm + quant producer writes (16, 16) codes in the tile `prequant_code_tile_rows`
names; the deferred partials feed the split-K reduce + norm / RoPE / NVFP4 insert of
`attention/norm_rope_nvfp4_insert_d128_rot64_p128` (wrapped by
`attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128`).

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_prequant`, `prequant_code_tile_rows` +
  `mxfp8_gemm_shuffled`, which returns `SkPartials` on a deferred row); `warmup_mxfp8_gemm_weight` also compiles the
  deferred rows.
- `host.select`: this shape's config for a row count ahead of the tuned table (None: the table decides).
- `host.select_prequant`: the config for producer codes where it differs from `select`.
- `host.select_deferred`: the split-K partial row that the fused norm / RoPE / insert chain reduces.
- `host.LAUNCHERS`: the launcher of each kernel these name.

The `host.py` selectors are the dispatch; `mxfp8_common`'s `gemm.py` calls them.

## Performance
One MI355X. Graph replay, layer 39 rank 0 weights in 26 cold copies (> 400 MB), 100 seeded inputs,
3 x 5 alternated rounds, median µs; "chain" = GEMM + norm / RoPE / NVFP4 insert, the caller's unit.

| Rows | M | Unit | µs | Baseline µs | Baseline |
|---|---|---|---|---|---|
| 1-16 deferred | 8 | chain | 7.70 | 8.51 | standalone GEMM + insert |
| 33-80 deferred | 80 | chain | 10.88 | 11.93 | standalone 32x32 2 K-waves + insert |
| 97-128 deferred | 128 | chain | 12.08 | 14.73 | standalone GEMM + split-K reduce + insert |
| 193-288 deferred | 256 | chain | 15.56 | 19.51 | standalone GEMM + split-K reduce + insert |
| 385-1536 | 1024 | chain | 27.62 | 33.39 | decode 96x128 4 K-waves |
| 1537-2048 | 2048 | chain | 45.52 | 53.44 | `4wave` 128x256 |
| 2049-3072 | 3072 | GEMM | 45.30 | 46.31 | `4wave` 128x256, cached stores |
| 3073-4800 | 4096 | chain | 67.88 | 72.14 | `4wave_nt` 256x256 |
| 4801-10240 | 8192 | GEMM | 105.54 | 106.69 | `4wave` 256x256 |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n2560_k6144 -q`
- `python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_qkv_insert.py --checkpoint <model> --ms 128,1024`
- `python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_rect_n2560_k6144.py --checkpoint <model>`
