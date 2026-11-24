# Four-rank all-reduce + Gemma norm, bf16-row MXFP8, hidden 6144

## What it does

`Bf16ArNormComm` fuses the TP4 all-reduce of a bf16 partial with the residual add, Gemma RMSNorm and (optionally) the
MXFP8 quant, in one launch, for the MiniMax-M3.1-Flash DSpark draft on TP4: the draft's o_proj / down stay TP partials
(`reduce_results=False`) and the next add + norm + quant is this entry (nine quantized tails and one final bf16 tail per
draft step). Input bf16 `[rows, 6144]` contiguous, rows 1-2048 (the draft runs 7-1792); residual bf16, unit column
stride, row stride divisible by 8; weight bf16 `[6144]` contiguous. Codes row-major or (16, 16)-tiled, padded to
`code_tile_rows`; scales in AITER's shuffled layout, padded to 32 rows; padding rows are not written. Contract
(`reference.py`): owner `c = min(row // ceil(rows / 4), 3)` sums ranks c, c+1, c+2, c+3 (mod 4) in fp32, every operand
and partial sum flushed to signed zero, one bf16 rounding; the residual add is fp32; the norm (weight + 1) uses the
12-wave square tree of the plain norm; the quant always reads **the bf16-rounded norm** (the draft's rule; the target's
attention tail quantizes the fp32 norm and uses `gemma_add_rmsnorm_h6144`). This order equals AITER's two-stage order
when rows % 4 == 0; other counts are an explicit order change. Deterministic; atomics only on flags.

## Entry points

- `Bf16ArNormComm`: the communicator, a collective on four ranks with the device current; ~55 MiB shared per rank.
- `Bf16ArNormComm.warm_up`: collective; compiles the four row tiers, both code layouts, bf16 out, both residual forms.
- `Bf16ArNormComm.ar_add_rmsnorm_mxfp8`: all-reduce + add + norm + quant -> (codes, scales, residual out).
- `Bf16ArNormComm.ar_add_rmsnorm`: the same with bf16 out -> (out, residual out).
- The owned-residual form of both (from 32 rows): the owner adds, norms and quantizes its rows and peers copy the
  result, so the residual is read and written on `owned_rows(rows)` only. Pass a full residual or the residual out of
  an owned call with the same row count. Outputs equal the all-row form bit for bit.
- `Bf16ArNormComm.capture`: context around graph capture; graph calls need it (peer inputs are registered and read in
  place, their lifetime follows PyTorch's graph pool). Eager calls stage into two alternating shared slots.
- `Bf16ArNormComm.close`: collective, after graph use has ended.

One communicator per stream, the same calls in the same order on every rank; the whole grid must be resident.
`host.rows_per_group` picks the rows per workgroup by row count.

## Performance

MI355X TP4. Graphs of 100 distinct inputs, ten replays, five alternating rounds, MAX over ranks, median us.
Baseline: AITER custom all-reduce + a one-GPU fused bf16 norm / quant (`benchmark_allreduce.py` runs the bf16 norm
and the FlyDSL quantizer as two launches).

| rows | 7 | 70 | 224 | 448 | 896 | 1792 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 9.911 | 15.206 | 33.361 | 58.738 | 112.936 | 215.127 |
| this entry, all-row MXFP8 | 7.950 | 13.312 | 29.219 | 53.414 | 101.926 | 197.113 |
| this entry, owned MXFP8 (separate run) | (all-row) | 12.035 | 23.411 | | | 166.827 |

## Tests

```bash
python3 -m pytest tests/norm/allreduce_gemma_bf16_mxfp8_h6144 -q                     # 4 GPUs
python3 benchmarks/norm/allreduce_gemma_bf16_mxfp8_h6144/benchmark_allreduce.py      # 4 GPUs
```
