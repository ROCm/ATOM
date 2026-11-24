# bf16 concat linear, N 1536 (one rank's shard) x K 30720, + the 4-rank gather to N 6144

## What it does
The MiniMax-M3.1 DSpark draft's input fc at TP4: five bf16 taps [rows, 6144] (target layers 3, 17, 31, 45, 59; K
30720 = their concatenation) times `dspark.fc.weight` bf16 [6144, 30720], bf16 [rows, 6144] out on every rank. Each
rank keeps its 1536-row weight slice (packed at load), computes its 1536 columns from the taps read in place (no
`torch.cat`), and the communicator gathers the four shards into the caller's buffer. Every output is one fp32 K sum
(split-K slabs summed in a fixed order) rounded once to bf16; no quantization. Taps: unit column stride, row stride %
8 == 0. Contract (`reference.py`): the fp64 product rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3,
bitwise repeats.
The slab count sets the fp32 summation order, so bits differ between row tiers; within a tier they are deterministic.
Caller: `decoder/dense_mxfp8_h6144_i3072_ranks4` (the draft's split input fc).

## Entry points
- `pack_weight`: the rank's 1536-row slice as a lossless permutation, at load; free the row-major slice.
- `bf16_concat_linear`: the local bf16 [rows, 1536] shard of 1-2048 rows from the taps in place, graph-safe.
- `warmup_concat_linear`: compiles the row tiers up to a row count.
- `comm.ConcatLinearComm`: the four-rank fc into the caller's contiguous bf16 [rows, 6144] buffer (IPC staging, up to
  2048 rows, the slab reduce fused into the gather by default); eager only (its Python slot rotation cannot replay in
  a graph); every rank makes the same calls in the same order; its `warm_up` before serving, its `close` on all ranks.

`host.config_for` picks the tier (tile, K slabs) by row count.

## Performance
Four ranks, eager fc as the draft calls it, real taps, cold rotated weights, MAX over ranks, median µs.
Baseline = `torch.cat` + hipBLASLt + `copy_` on the replicated weight, the same protocol on another MI355X node.

| R (rows) | 1 (8) | 10 (80) | 16 (128) | 32 (256) | 64 (512) | 128 (1024) |
|---|---|---|---|---|---|---|
| hipBLASLt | 65.7 | 82.4 | 97.9 | 130.2 | 245.8 | 355.4 |
| this entry | 26.51 | 36.97 | 43.73 | 62.36 | 103.29 | 163.75 |

## Tests
- `python3 -m pytest tests/gemm/bf16_concat_linear_n1536_k30720 -q` (gather / compile: 4 GPUs)
- `python3 benchmarks/gemm/bf16_concat_linear_n1536_k30720/benchmark_concat_linear.py` (1 GPU, local)
