# bf16 GEMM, N 50016 x K 6144 (the TP4 lm_head vocabulary shard)

## What it does
`out [M, 50016] = x [M, 6144] @ W^T`, bf16 operands, fp32 sums in ascending K, one bf16 rounding: the TP4 vocabulary
shard of the MiniMax-M3.1-Flash lm_head (`lm_head.weight` bf16 [200064, 6144], rank r holds rows
[50016 r, 50016 (r + 1))). One tensor serves the target lm_head (verify rows) and the DSpark draft lm_head
(requests x 7 rows). W layout: `row` (the checkpoint's shard) or `shuf` (a bit-exact permutation once at load, so one
DMA reads 1 KB in one piece); both give the same bits. No split-K, partials or output atomics: the bits equal
hipBLASLt's / AITER asm's on 3.9e9 realistic logits. Contract (`reference.py`): the fp64 product rounded once to bf16;
per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic; NaN / inf propagate as in an fp32 sum. Callers:
`sampling/batch_sharded_markov_v200064_h6144_r256_ranks4` (target and draft logits).

## Entry points
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout; replace the parameter once
  before graph capture, keep the target / draft alias, free the row-major copy.
- `bf16_gemm_n50016_k6144`: the logits of any row count (launches of 2048 rows), graph-safe, no host sync; with a
  device row map and an exchange slot of `collectives/all_to_all_bf16_n50016_ranks4` it writes the mapped rows into
  the send slot in map order and publishes them (no hidden-row gather, no logits reorder), and the caller then runs
  the exchange's pull. An `amd_kernels.warm_up` entry: the first call of each mode (layout x contiguous / publish)
  compiles its 14 tile variants.
- `warmup_bf16_gemm_n50016_k6144`: the same warm-up for one mode, explicitly (publish: a private slot).
- `m_tile_fragments`, `m_halves`: the M tiling.
- `supports`: whether a weight is the [50016, 6144] bf16 contiguous shard (either layout) and there is no bias.

`host.py` picks the kernel and M tile by row count.

## Performance
One MI355X. Graph replay, the real rank 0 shard, `shuf`, 100 distinct realistic hidden inputs (and row
maps), ten replays, five alternated rounds, median µs. Baseline = vLLM's `rocm_unquantized_gemm_impl` (`wvSplitK` 1-4
rows, hipBLASLt 10-768, AITER asm `hgemm` from 896; VLLM_ROCM_USE_AITER=1).

| M | 1 | 16 | 64 | 80 | 128 | 224 | 256 | 512 | 1024 | 1792 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | 90.6 | 125.7 | 126.3 | 131.7 | 141.2 | 157.6 | 162.4 | 283.5 | 505.3 | 973.1 | 902.8 |
| contiguous | 92.3 | 92.1 | 93.3 | 99.5 | 122.8 | - | - | - | - | - | - |
| indexed, plain stores | - | - | - | 100.13 | 121.61 | 158.66 | 161.16 | 290.36 | 557.31 | 874.45 | 1011.97 |

The indexed row is the publish kernels' body with plain stores (the publish epilogue adds system-scope stores and a
ticket). The target + draft sampling path (four ranks) with both lm_heads on this entry: -8.3..-23.3% against vLLM's
GEMM at 10-256 requests.

## Tests
- `python3 -m pytest tests/gemm/bf16_n50016_k6144 -q` (`test_publish.py`: 4 GPUs)
- `python3 benchmarks/gemm/bf16_n50016_k6144/benchmark_gemm.py [M ...]`
