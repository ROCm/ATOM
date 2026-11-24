# MXFP8 GEMM, N 6144 x K 2048 (o_proj, TP4)

## What it does
The attention output projection of MiniMax-M3.1-Flash at TP4: K 2048 = 16 heads x 128, N 6144, bf16 out; the target's
sparse attention layers and the DSpark draft's attention. The rows and kernels of this shape, registered with the
dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); layouts and the numerics contract
are that folder's. The attention kernels' MXFP8 epilogues write the o_proj input in the tile
`prequant_code_tile_rows` names.

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_shuffled`, `mxfp8_gemm_prequant`,
  `warmup_mxfp8_gemm_weight`).
- `mxfp8_linear` and `mxfp8_gemm` at this shape only can write a caller-owned contiguous bf16 [M, 6144] arena on the
  activation device in place (same tiles and bits) and return it; a bad shape, dtype, device, stride or an input
  alias raises before any write. The caller owns its lifetime and orders readers after the producing stream.
- `host.select`: this shape's config for a row count ahead of the tuned table (None: the table decides).
- `host.select_prequant`: the config for producer codes where it differs from `select`.
- `host.decode_launcher`: the decode row whose activation stages through LDS (None: the common decode kernel).
- `host.LAUNCHERS`: the launcher of each kernel these name.

The `host.py` selectors are the dispatch; `mxfp8_common`'s `gemm.py` calls them.

## Performance
One MI355X. Graph replay, layer 39 rank 0 weights in 32 cold copies (> 400 MB), 100 seeded inputs, the
caller's call form (codes in the producer's layout), 3 x 5 alternated rounds, median µs per GEMM.

| Rows | M | µs | Baseline µs | Baseline |
|---|---|---|---|---|
| 1-32 | 8 / 32 | 3.91 / 4.40 | 4.30 / 5.05 | the `decode_fq` tiles on codes, 2 K-waves |
| 33-48 / 65-80 | 48 / 80 | 4.87 / 5.97 | 5.08 / 6.30 | 32x48 (two m-blocks) / 96x32 |
| 97-128 | 128 | 6.39 | 6.64 | direct 64x48 decode |
| 129-160 / 257-320 / 385-448 | 160 / 320 / 448 | 7.04 / 8.88 / 10.61 | 7.64 / 9.92 / 11.77 | 96x48 / 96x96 / 128x96, 2 K-waves |
| 769-1024 | 1024 | 17.43 | 19.65 | `4wave` 256x128 (`benchmark_n6144_k2048.py`) |
| 1281-3328 | 2560 / 3328 | 29.68 / 43.02 | 33.79 / 47.57 | `aiter` 128x256 |
| 3329-3968 | 3456 / 3968 | 44.52 / 47.53 (eager 49.56) | 51.03 / 52.22 (eager 52.95) | `aiter` 128x256 |
| `out=` arena | 2048 | 26.50 | 36.25 | allocated output + explicit copy |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n6144_k2048 -q`
- `python3 benchmarks/gemm/mxfp8_n6144_k2048/benchmark_n6144_k2048.py --checkpoint <model>` (769-1024)
- `python3 benchmarks/gemm/mxfp8_n6144_k2048/benchmark_oproj_out.py --checkpoint <model>` (`out=`)
