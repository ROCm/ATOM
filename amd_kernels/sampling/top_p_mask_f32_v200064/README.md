# Top-p mask of fp32 logits rows, vocab 200064: vLLM's split top-p chain in one launch

## What it does

Masks the columns outside the top-p nucleus of each row with -inf, in place, bit for bit what vLLM's
`apply_top_k_top_p_triton(logits, None, p)` does at <= 64 rows: its split top-p pipeline (`_topp_sb_stats_kernel`,
five `_topp_sb_step_kernel` rounds, `_topp_sb_mask_kernel`, Triton 3.8 on gfx950), seven launches and about twice
the time of this one. Per row (p < 1): S slices (32 at <= 8 rows, 16 above, by vLLM's
`_topp_split_count` on 256 CUs) give the max, the softmax sum and the min; five rounds of an 8-pivot geometric ladder
bracket the probability where the kept mass reaches p; the mask keeps `prob > pivot` or the max, and boundary
duplicates in column order. Contract (`reference.py`): the same fp32 operations in the same order (Triton's
`tl.exp` / `tl.log` expansions, each `tl.sum`'s butterfly, the two products the compiler fuses), compared exactly.
Rows with p >= 1 stay as they are. Bit equality with the Triton chain needs vLLM, so it is checked outside this
repository (all row counts 1-64; LM rows at T 0.6-1.5, integer ties, -inf columns, p 1 rows, flat rows).
Caller: the M3.1 TP4 sampler (target top-p, `--enable-batch-sharded-sampling` owned rows).

## Entry points

- `top_p_mask`: masks the columns outside each row's top-p nucleus with -inf, in place (fp32 `[rows, 200064]`, 1-64
  rows); one workspace per device (as vLLM's), so launches go on one stream; graph safe. An `amd_kernels.warm_up`
  entry: the first call compiles both slice counts.
- `top_p_blocker`: why the entry cannot stand in for a model's chain (None: it can).
- `warmup_top_p_mask`: compiles both variants and allocates the workspace; nothing compiles after it.
- `splits_for`: the slice count for a row count and CU count (vLLM's `_topp_split_count`); `host.py` dispatches on it.
- `top_p_mask_reference`: the torch contract. `available`: gfx950 with FlyDSL importable. `V` = 200064.

## Performance

One MI355X; graph replay of 100 distinct inputs (real M3.1 logits rows, `--rows`, T 0.85-1.15,
p 0.95), each a bf16 -> fp32 copy and the mask, copy subtracted; rounds A B B A; us per call; A = vLLM's Triton chain
(7 launches).

| Rows | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| Triton chain | 52.0 | 71.3 | 77.7 | 125.1 | 173.6 | 179.8 | 238.7 | 280.5 |
| this entry | 28.4 | 41.5 | 45.6 | 72.2 | 83.6 | 89.2 | 118.2 | 142.1 |

## Tests

```bash
python3 -m pytest tests/sampling/top_p_mask_f32_v200064 -q
python3 benchmarks/sampling/top_p_mask_f32_v200064/benchmark_top_p_mask.py [--rows <logits rows .pt>]
```
