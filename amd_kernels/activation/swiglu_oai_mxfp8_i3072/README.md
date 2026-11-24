# SwiGLU-OAI + MXFP8 quant, I 3072 (gfx950)

## What it does

One launch for the SwiGLU-OAI activation of a split-layout gate_up row (`[m, 2 x 3072]` bf16, gate | up, unit column
stride, any row stride) and the MXFP8 quant of the result in the layout the down GEMM reads: e4m3 codes row-major
`[m, 3072]` or in the decode GEMM's (16, 16) tiles with rows padded to the tile, E8M0 scales
`[ceil(rows / 32) * 32, 96]` in AITER's shuffled layout (padding rows and scale padding unwritten; the GEMM masks them).
Caller: the M3.1 dense FFN at TP4 (I = 3072 per rank): the DSpark draft's 5 layers and the target's dense layers 0-2,
through `decoder/dense_mxfp8_h6144_i3072_ranks4` `mxfp8_swiglu_oai_mlp` above 8 rows. Numerics (`reference.py`): `form`
is the caller's, never chosen by row count. `bf16_chain` = the draft's aten chain (`clamp`, `sigmoid`, `mul`, `add`,
each rounded to bf16); `fp32` = the target's `swiglu_oai_split` (fp32 math, one bf16 rounding); both use
`1 / (1 + exp(-x))` with the accurate exp and an IEEE divide. Then the Triton MXFP8 quantizer (`_mxfp8_quantize_kernel`)
on the bf16 value: scale `clamp(ceil(log2(amax / 448)) + 127, 0, 254)` per 32 columns, e4m3 codes of
`act * 2^(127 - scale)`. Codes and scales equal the reference bit for bit on finite inputs.

## Entry points

- `swiglu_oai_mxfp8`: the down GEMM's codes and scales of a gate_up row block, row-major or in (16, 16) tiles; the
  caller takes the tile from `gemm.mxfp8_common.gemm.prequant_code_tile_rows` and runs the down GEMM with
  `gemm.mxfp8_common.gemm.mxfp8_gemm_prequant`. The output buffers may be the caller's.
- `warmup_swiglu_oai_mxfp8`: compiles all 8 variants (2 kernels x 2 forms x 2 code layouts; the row count is a
  runtime argument). Call it before serving or graph capture.

`host.swiglu_oai_mxfp8` picks the kernel (`kernel.py` or `kernel_partitioned.py`) by row count.

## Performance

One MI355X, graph replay, 100 distinct inputs per graph, us per call. Unit timing of `kernel.py` vs the
unfused chain (the 7 aten kernels + the Triton quantizer in the down GEMM's layout). Rows are 7 x requests.

| Rows | Tile | aten chain + quantizer | swiglu_oai_mxfp8 | Ratio |
|---|---|---|---|---|
| 7 | 16 | 19.36 | 2.95 | 6.6x |
| 224 | 128 | 28.10 | 3.54 | 7.9x |
| 448 | 128 | 32.93 | 4.94 | 6.7x |
| 896 | 0 | 40.44 | 6.09 | 6.6x |
| 1792 | 0 | 64.42 | 10.16 | 6.3x |

At 17-128 rows the partitioned kernel cuts the whole dense FFN (gate_up -> SwiGLU + quant -> down, rank 0 slices)
by 0.3-0.5 us against `kernel.py`: target layer 1 at 80 / 128 rows 23.42 / 25.08 -> 23.01 / 24.76 us, draft layer 0
at 70 / 112 rows 23.43 / 25.22 -> 22.97 / 24.78 us; the down outputs are bitwise equal.

## Tests

```bash
python3 -m pytest tests/activation/swiglu_oai_mxfp8_i3072 -q
python3 benchmarks/activation/swiglu_oai_mxfp8_i3072/benchmark_swiglu_oai_mxfp8.py
```
