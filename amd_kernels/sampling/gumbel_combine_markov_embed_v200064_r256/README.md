# Markov draft step tail: Gumbel combine + draft id + next markov_w1 row, vocab 200064, rank 256

## What it does

Ends one DSpark Markov draft step on the owner rank in one launch. Per owned row r: the Gumbel combine of the 196
per-block partials of `sampling/gumbel_partials_bias_bf16_v200064` (the first NaN block wins, otherwise the first block
holding the row maximum: vLLM's `_gumbel_combine_kernel`, `torch.argmax` semantics), `tok = local_argmax[r, win]`; then
`draft_ids[r, step] = tok`, `prev[local_rows[r]] = tok` and the next step's `markov_embed` row, `markov_w1[tok]`, copied
into row r of the next bias GEMM's packed input. One launch in place of four of the DSpark speculator's sequential local
sampling: the Triton combine, the draft-id copy, `prev.index_copy_` and the next `markov_embed(prev)` gather. Caller:
`sampling/batch_sharded_markov_v200064_h6144_r256_ranks4` (M3.1 TP4). Contract (`reference.py`): every output is exact;
the tests require equality.

## Entry points

- `gumbel_combine_markov_embed_packed`: the tail of one draft step for up to 64 owned rows: the winner of the 196
  block partials, its vocab id into the draft ids and `prev`, and its `markov_w1` row into row r (the owned row) of the
  next bias GEMM's packed input (16 / 32 / 64 rows, the `[m16, k32, k8, row16, feature8]` layout of
  `gemm/bf16_n200064_k256`); only valid row bytes are written. Any step; graph safe.
- `warmup_gumbel_combine_markov_embed`: compiles the kernel (it serves every row count and step).
- `available`: gfx950 with FlyDSL importable.

No dispatch: one kernel (`kernel_packed.py`).

## Performance

Timed in the seven-step Markov chain (bias GEMM, Gumbel partials, this tail): `gemm/bf16_n200064_k256`'s README.

## Tests

```bash
python3 -m pytest tests/sampling/gumbel_combine_markov_embed_v200064_r256 -q
```
