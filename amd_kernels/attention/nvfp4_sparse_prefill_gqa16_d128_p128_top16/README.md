# NVFP4 sparse prefill attention, GQA16, D128, 128-token pages, top-16

## What it does

MiniMax-M3.1 sparse attention: each query attends its selected 16 blocks of 128 keys (own block causal).
Inputs: Q `[T, 16 Hkv, 128]` (bf16, fp16 for fp16 compute, or prepacked e4m3 u8 for fp8), NVFP4 pages
`[pages, 2 Hkv, 128, 72]` u8 with free page stride, logical picks `[Hkv, T, 16]` i32 (-1 none), block tables, request
ids and positions. Tiles may span requests. Output: bf16, MXFP8 codes + AITER-shuffled scales for o_proj, or
sequence-parallel all-to-all rows (Hkv 1); the fused outputs leave the bf16 output and the padding rows untouched.
TP4: Hkv 1, up to 65536 tokens. TP4 serves fp16 with the lazy rescale (fp16 P against a reference re-based only past
2^8, P <= 256; the fastest at every T); bf16, plain fp16 and fp8 stay available. FP8 uses NV Q8KV4 with P rounded per
32-key chunk; gate it with the NV KL check.
Contract: `reference.py` (`REL_L2_TOL`, `COS_MIN`); fused outputs equal quantization of the kernel's own bf16 output.
Strict xfail: bf16 TQ4 can propagate a NaN V scale past the sequence end through zero P.

## Entry points

- `nvfp4_sparse_prefill`: the attention of a prefill step into bf16, or fused into the MXFP8 o_proj input or the
  sequence-parallel all-to-all rows; graph-safe. An `amd_kernels.warm_up` entry: the first call of each mode warms
  every variant at 8 / 512 / 1024 / 1536 / 2048 tokens.
- `nvfp4_sparse_prefill_bf16`, `nvfp4_sparse_prefill_fp16`, `nvfp4_sparse_prefill_fp8`: the same with the compute
  precision bound. Build the fp8 table outside graph capture.
- `host.warm_up_prefill`: compiles every bf16-output variant of one compute precision now.
- `supports_nvfp4_sparse_prefill`: whether a model fits (gfx950, GQA16, D128, top-16 blocks of 128).

`host.py` `nvfp4_sparse_prefill` picks the kernel by precision, token count and grid size.

## Performance

MI355X, Hkv 1, bf16 Q, 16K prefix, 100 cold inputs, 4 alternated graph-replay rounds, us.
Tokens: 256 / 512 / 1024 / 1536 / 4096 / 8192 / 16384 / 32768. FP8 includes Q packing.

| Compute | quant | pack |
|---|---|---|
| bf16 | 15.7 / 22.2 / 32.6 / 42.8 / 95.1 / 180.2 / 355.9 / 708.7 | 15.6 / 22.0 / 32.5 / 42.5 / 95.0 / 179.6 / 355.0 / 711.3 |
| fp16 | 14.7 / 22.1 / 31.8 / 41.7 / 92.6 / 176.2 / 348.8 / 696.4 | 14.5 / 21.9 / 31.6 / 41.5 / 92.4 / 175.6 / 348.1 / 694.6 |
| fp8 | 17.2 / 24.2 / 34.8 / 44.8 / 99.3 / 187.0 / 361.6 / 712.1 | 17.0 / 23.8 / 34.5 / 44.6 / 98.9 / 185.8 / 360.4 / 708.8 |
| fp16 lazy | 14.7 / 21.0 / 29.8 / 38.8 / 85.0 / 162.9 / 322.4 / 644.0 | 14.6 / 20.7 / 29.7 / 38.6 / 84.6 / 161.9 / 321.2 / 641.7 |

## Tests

```bash
python3 -m pytest tests/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16 -q
python3 benchmarks/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16/benchmark_sparse_prefill.py --fp8-bf16q --fp16-lazy
```
