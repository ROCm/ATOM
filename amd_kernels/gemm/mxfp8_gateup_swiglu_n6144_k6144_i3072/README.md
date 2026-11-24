# MXFP8 gate_up + SwiGLU-OAI, N 6144 x K 6144, I 3072 (dense MLP, TP4, 1-8 rows)

## What it does
`act [m, 3072] = SwiGLU-OAI(gate, up)` with `gate | up = x @ W^T` in one launch, for 1-8 rows of the MiniMax-M3.1-Flash
dense MLP at TP4. W is the dense gate_up weight as the common dispatcher holds it: MXFP8 [6144, 6144], gate rows
0-3071 and up rows 3072-6143, (16, 16) codes and AITER-layout scales (`preshuffle_mxfp8_weight`), no copy. x: E4M3
codes in the (16, 16) layout padded to 16 rows ([16, 6144]), E8M0 scales in AITER's layout ([>= 32, 192]). Each dot
product rounds to bf16 before the activation; the bf16 output feeds the down linear's own quantizer. Callers:
`decoder/dense_mxfp8_h6144_i3072_ranks4` (the target's dense layers with form `fp32`, the DSpark draft's MLP with
`bf16_chain`); above 8 rows they run gate_up `mxfp8_linear` + `activation/swiglu_oai_mxfp8_i3072` + down. Contract:
`mxfp8_common/reference.py` `mxfp8_gateup_swiglu_bf16_reference` (fp64 dots rounded to bf16, the caller's form):
per-row cosine >= 0.9999, and the bf16 bits of the materialized GEMM's activation in the same K order.

## Entry points
- `host.mxfp8_gateup_swiglu_bf16`: the bf16 [m, 3072] activation of 1-8 rows in one launch; the caller picks the
  numerics form (`fp32` or `bf16_chain`), never M.
- `host.warmup_mxfp8_gateup_swiglu_bf16`: compiles the forms; call it before graph capture.

`host.py` picks the kernel by form.

## Performance
One MI355X. Graph replay, real layer weights in cold rotated copies (> 400 MB), 100 inputs, both capture
orders x 5 alternated rounds, median µs of the whole FFN (gate_up + SwiGLU + down).

| Unit | M | µs | Baseline µs | Baseline |
|---|---|---|---|---|
| target FFN, `fp32` | 8 | 14.20 | 15.90 | gate_up GEMM + SwiGLU-OAI quant + down |
| draft FFN, `bf16_chain` | 7 | 14.19 | 16.83 | another fused draft FFN path (not in the benchmark) |
| target FFN, `fp32` one K-wave | 1 / 3 / 7 / 8 | 13.45 / 13.65 / 13.86 / 14.02 | 13.70 / 13.83 / 14.10 / 14.18 | the two-K-wave form |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072 -q`
- `python3 benchmarks/gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072/benchmark_gateup_swiglu_bf16.py --rows 1 7 8`
