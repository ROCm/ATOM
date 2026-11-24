# Gumbel block partials of a DSpark Markov draft step: base + bias rows, bf16 cache row, vocab 200064

## What it does

One Markov draft step's Gumbel sampling partials, bit for bit the vLLM Triton `_gumbel_sample_kernel` launch of
`gumbel_sample` on the drafting path (bias rows, one cache column, fp32 noise, temperature applied). Per row r
(200064 columns, 196 blocks of 1024): `y = bf16(base[r] + bias[bias_rows[r]])` (round to nearest even) is stored as
the draft-cache row `cache[slot, step]` (pre-temperature; nothing for slot < 0); then `v = y` at T 0 or slot < 0,
else `v = y / T` (IEEE fp32 division, skipped at T 1) `+ noise`, noise = `-log(-log1p_neg_stable(u))`, u = murmur3
uniform of (seed ^ (step << 32), pos + 2^30, vocab id); per block the noised maximum and its vocab id by Triton's
argmax tree (NaN and tie rules included; columns >= 200064 are -inf).
Caller: `sampling/batch_sharded_markov_v200064_h6144_r256_ranks4` (the M3.1 TP4 DSpark draft); the partials feed
`sampling/gumbel_combine_markov_embed_v200064_r256`. Contract (`reference.py`): the cache rows and the tree are
exact; torch has no gfx950 `v_log_f32`, so the tests compare the fp64 noise within 2 fp32 ulps and the tree exactly on
T 0 rows. Bit equality with the Triton kernel needs vLLM, so it is checked outside this repository.

## Entry points

- `gumbel_partials_bias`: one launch per draft step: each row's bf16 draft-cache row and its per-block noised maximum
  and vocab id (`[rows, 196]` fp32 / int64) into caller buffers, any row count, graph safe. A position offset
  (|offset| < 2^29) lets the caller draw at `sample_pos - 1` without a subtraction launch.
- `blocks_per_wg`: the 1024-column blocks per workgroup for a row count; it is the dispatch (`host.py`).
- `warmup_gumbel_partials_bias`: compiles both tilings (each serves every step and temperature); nothing compiles
  after it.
- `available`: gfx950 with FlyDSL importable. `V` = 200064, `NB` = 196.

## Performance

One MI355X; one launch, graph replay of 20 distinct inputs (HBM-cold),
T 1, median of 5 alternated rounds x 10 replays, us per call; baseline = the vLLM Triton `_gumbel_sample_kernel`.

| Owned rows (requests) | 3 (10) | 4 (16) | 8 (32) | 16 (64) | 32 (128) | 64 (256) |
|---|---|---|---|---|---|---|
| Triton | 5.01 | 5.51 | 7.29 | 11.89 | 18.62 | 31.29 |
| this entry | 4.78 | 5.25 | 6.95 | 11.26 | 17.51 | 29.28 |

25 / 38 VGPRs at 1 / 2 blocks per workgroup, no spills, no scratch.

## Tests

```bash
python3 -m pytest tests/sampling/gumbel_partials_bias_bf16_v200064 -q
python3 benchmarks/sampling/gumbel_partials_bias_bf16_v200064/benchmark_gumbel_partials.py
```
