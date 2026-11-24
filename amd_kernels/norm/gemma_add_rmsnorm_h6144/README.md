# Gemma add + RMSNorm, hidden 6144, and the TP4 all-reduce fused in front

## What it does

Residual add + Gemma RMSNorm of MiniMax-M3.1-Flash in FlyDSL (bf16 x, residual, weight as w + 1; fp32 eps; hidden
6144). `add_rmsnorm` (one GPU): bf16 or MXFP8 out (AITER-layout E8M0 scales; codes row-major or in the decode GEMM's
(16, 16) tiles). `allreduce_add_rmsnorm`: the TP4 all-reduce in front, for the attention tail (one bf16 o_proj
partial per rank) and the MoE tail (which first sums each rank's 4 weighted slots + shared in fp32). Contract
(`reference.py`): chunk c = rows `[c q, (c + 1) q)`, q = ceil(rows / 4), summed over ranks c..c+3 in fp32 with
denormals flushed, one bf16 rounding (AITER's two-stage bits); `s = x + residual` in fp32, residual out `bf16(s)`;
`y = s * rstd * (w + 1)` in Triton's square-sum order, within 1 bf16 ulp of fp64; MXFP8 of the fp32 y at every M.
Fused = one-GPU kernels on `allreduce_rows_reference`.

## Entry points

- `add_rmsnorm.gemma_add_rmsnorm`: one-GPU add + norm, bf16 out.
- `add_rmsnorm.gemma_add_rmsnorm_mxfp8`, `add_rmsnorm.gemma_add_rmsnorm_mxfp8_row_major`: MXFP8 out, tiled or
  row-major codes.
- `add_rmsnorm.gemma_add_rmsnorm_mxfp8_for`: MXFP8 out in the layout a given K 6144 GEMM weight reads;
  `add_rmsnorm.code_tile_rows_for`: that layout's code tile (0: row-major).
- `add_rmsnorm.add_rmsnorm_ok`: whether the one-GPU kernels take these rows (not under torch.compile).
- `add_rmsnorm.warmup_add_rmsnorm_mxfp8`, `add_rmsnorm.warmup_for_gemm`: compile the MXFP8 variants before graph
  capture (`gemm.mxfp8_common.warmup_mxfp8_gemm_weight` calls the latter).
- `allreduce_add_rmsnorm.staged_comm`: the attention tail's communicator, bf16 or MXFP8 out; its `capture()` context
  registers graph inputs so faster tiers read them in place; `supports` / `accepts` say whether a call fits.
- `allreduce_add_rmsnorm.slots4_comm`: the MoE tail's communicator, MXFP8 out.
- `allreduce_add_rmsnorm.unsupported_reason`: why this process cannot run the communicators (None: it can).
- `allreduce_add_rmsnorm.DecoderArNorm`: both communicators (+ one layer's MoE modes): `add_rmsnorm`, a layer's tail
  `post_attention` (norm + FFN -> `FfnPartial`) and `pre_attention` (the next norm); `DenseMlp`: a dense layer's MLP.

Both communicators: 1-4095 rows (larger prefills take the sequence-parallel hand-offs), eager and graphs, one stream
each, `warm_up` on every rank before serving, the same calls in order on all ranks, IPC buffers for the process
lifetime. Tiers by row count: one-shot (each rank sums all), two-stage one row per workgroup, streamed R rows per
workgroup, some exchanges in a lossless 12-bit code; `allreduce_add_rmsnorm/staged_host.py` and `host.py` pick them.

## Performance

Graphs to 512 rows, eager above; MAX over ranks, median us; baseline = AITER's custom all-reduce (RCCL above its
cap), then the norm. Coded tiers (real layer-39 producer + tail, against the same stream uncoded): attention 2048 /
3253 rows 219.1 / 338.2 us, -16..-18%; MoE 1024 rows 340.9 us, -4%; the coded reduce-scatter -3..-65 us more.

| M | bf16: baseline | entry | MXFP8: baseline | entry | MoE tail: baseline | entry |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7.94 | 5.99 | 10.83 | 7.04 | 15.43 | 8.09 |
| 1024 | 128.91 | 119.45 | 129.64 | 118.60 | 148.13 | 124.03 |

## Tests

```bash
python3 -m pytest tests/norm/gemma_add_rmsnorm_h6144 -q   # all-reduce tests: 4 GPUs
python3 benchmarks/norm/gemma_add_rmsnorm_h6144/benchmark_ar_add_rmsnorm_rows.py --kinds bf16,mxfp8 --rounds 7 8 128 1024 4095
python3 benchmarks/norm/gemma_add_rmsnorm_h6144/benchmark_slots4_ar_add_rmsnorm.py result.json   # also benchmark_add_rmsnorm.py
```
