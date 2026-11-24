# MXFP8 GEMM: the dispatcher and the shared kernels (gfx950)

## What it does
`out [M, N] = x [M, K] @ W^T` for the MXFP8 dense linears of MiniMax-M3.1-Flash at TP4. W: E4M3 codes [N, K] + E8M0
per-32 scales [N, K / 32], preshuffled at load ((16, 16) codes, AITER `shuffle_scale_a16w4` scales). x: bf16 / fp16
[M, K], K % 256 == 0, quantized per 1 x 32 block. Shapes (N x K): 2560 x 6144 QKV, 6144 x 2048 o_proj, 6144 x 8192
`o_proj_full`, 6144 x 6144 dense gate_up, 6144 x 3072 dense down, 1536 x 6144 / 6144 x 768 shared-expert gate_up /
down, 2304 x 6144 draft QKV, 1280 x 6144 draft context K / V. Callers: vLLM's MXFP8 linears, the norm + quant
producers, the QKV chain, SP hand-offs. `reference.py`: exact dequant, fp32 sums, one rounding; cos >= 0.99999,
L2 <= 1e-3, bitwise repeats.

## Entry points
- `gemm.preshuffle_mxfp8_weight`, `gemm.preshuffle_mxfp8_weight_`: the load-time weight preshuffle; the in-place form
  returns False off `tuned_shapes()`.
- `gemm.tuned_shapes`: the (N, K) with tuned rows (none for fp16 output); switch only these shapes.
- `gemm.mxfp8_linear`: the GEMM from a bf16 / fp16 activation, quantized in the selected row's code layout.
- `gemm.mxfp8_gemm`: the GEMM from row-major activation codes (never a decode row); row-major scales are shuffled
  first (torch).
- `gemm.prequant_code_tile_rows`: for a code producer, 0 (write row-major codes) or the row tile its (16, 16) codes
  are padded to; `gemm.prequant_buffers`: those buffers.
- `gemm.mxfp8_gemm_shuffled`: the GEMM on (16, 16) codes; it can return `gemm.SkPartials` (fp32 split-K partials)
  for a consumer that sums them.
- `gemm.mxfp8_gemm_prequant`: either form, as `prequant_code_tile_rows` says.
- `gemm.warmup_mxfp8_gemm_weight`: compiles every variant up to a row count, once per (N, K, device, dtype); call it
  before graph capture (a first call does not warm up).
- `gemm.set_dbo_query`: tells the 4-wave rows whether they run inside a DBO microbatch (else `gemm.DBO` answers).
- `gemm.mxfp8_e4m3_quantize_aiter`, `gemm.mxfp8_e4m3_quantize_shuffled`: the activation quantizer, row-major or
  (16, 16) codes.

Dispatch: `gemm._select_config` reads the tuned table `mxfp8_gemm_tuned.py`; a shape folder's `host.select`
(`gemm._SHAPE_HOSTS`) answers first.

## Performance
Graph replay, real TP4 rank 0 weights in cold rotated copies (> 400 MB), µs per call, median.

| N x K | M | µs | Baseline µs, or our delta | Baseline |
|---|---|---|---|---|
| 1536 x 6144 | 1-256 | 4.28-13.13 | -19..-49% / -18..-53% | AITER `any` row / vLLM nightly `dot_scaled` |
| 6144 x 768 | 1-256 | 2.79-7.56 | -20..-62% / -19..-47% | AITER `any` row / vLLM nightly `dot_scaled` |
| shared expert, c16 decode mix | per MoE layer | 11.96 | 19.46 / 18.80 µs | AITER rows / vLLM nightly |
| 2304 x 6144 | 112 / 896 | 8.61 / 27.48 | 10.06 / 35.26 µs | non-temporal W / AITER 128x128 |
| 1280 x 6144 | 256 / 512 / 2048 | 11.75 / 13.75 / 41.09 | 15.53 / 51.29 / 68.37 µs | generic AITER fallback |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_common -q`
- `python3 benchmarks/gemm/mxfp8_common/benchmark_gemm.py --variant <a test_gemm.CASES name>`
