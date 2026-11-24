# Gemma RMSNorm + bf16-row MXFP8, hidden 6144

## What it does

The plain Gemma RMSNorm (no residual) followed by the bf16-row MXFP8 quantizer, in one launch. It serves the
MiniMax-M3.1-Flash TP4 DSpark draft: the context hidden norm before the context K / V GEMM
(`DFlashQwen3Model` / `MiniMaxM3DSparkModel.precompute_and_store_context_kv`) and the first draft layer's input norm
before QKV (`DFlashQwen3DecoderLayer`, the embedding stays the residual). Input bf16 `[M, 6144]` with unit column
stride (any row stride), weight bf16 `[6144]` contiguous, fp32 eps, M >= 0. Codes e4m3 row-major (`code_tile_rows=0`)
or in the dense decode GEMM's (16, 16) tiles (`code_tile_rows` a positive multiple of 16, rows padded to it); scales
E8M0 in AITER's shuffled layout, padded to 32 rows; padding rows are not written. Contract (`reference.py`): fp32 sums,
`x * rsqrt(mean(x^2) + eps) * (1 + w)`, rounded to bf16 (per-row cosine >= 0.999999 vs the reference), then 1x32 E8M0
/ E4M3 quantization of that bf16 value at every M. The kernel also keeps the served Triton plain norm's square-sum
tree, so its bf16 values equal the Triton norm's bit for bit. The caller picks the tile with
`prequant_code_tile_rows(M, N, 6144)` (`gemm/mxfp8_common`).

## Entry points

- `gemma_rmsnorm_mxfp8`: norm + quant -> (codes, scales); an optional contiguous bf16 `[M, 6144]` buffer also
  receives the normed rows.
- `warmup_gemma_rmsnorm_mxfp8`: compiles all eight variants (2 kernels x 2 code layouts x with / without the bf16
  output); call it before graph capture. Eager and graph replay run the same kernels.

`host.gemma_rmsnorm_mxfp8` picks the 12-wave kernel (`kernel.py`) or the four-wave one (`kernel_fourwave.py`) by row
count.

## Performance

MI355X. Graphs of 100 distinct inputs, ten replays, alternating rounds, row-major codes, median [min-max]
us. Baseline: the Triton plain norm + the quantizer (two launches).

| rows | Triton norm + quant | this entry |
|---:|---:|---:|
| 8 | 4.100 [4.098-4.106] | 2.693 [2.686-2.698] |
| 80 | 4.881 [4.879-4.913] | 2.900 [2.889-2.909] |
| 128 | 5.524 [5.519-5.531] | 2.967 [2.964-2.978] |
| 256 | 8.080 [8.076-8.089] | 3.448 [3.445-3.452] |
| 512 | 8.900 [8.882-8.943] | 4.618 [4.587-4.659] |
| 1024 | 13.725 [13.715-13.842] | 6.544 [6.539-6.584] |
| 2048 | 19.690 [19.666-20.268] | 10.358 [10.309-10.744] |

## Tests

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest tests/norm/gemma_rmsnorm_bf16_mxfp8_h6144 -q
python3 benchmarks/norm/gemma_rmsnorm_bf16_mxfp8_h6144/benchmark_norm.py
```

One GPU: zero / odd rows, every draft decode batch, row strides, exponents across +-5 binades, every code layout and
padding, quantization of the bf16 output, poisoned graph outputs, changed inputs, warm-up coverage, MFMA co-residency.
