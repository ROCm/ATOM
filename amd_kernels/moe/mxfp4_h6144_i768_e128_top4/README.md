# MXFP4 MoE with MXFP8 shared expert, H6144 I768 E128 top4 (gfx950)

## What it does

One TP4 rank's MoE block of MiniMax-M3.1-Flash, FlyDSL only: bf16x3 split-K router + biased top-4 of 128 experts
(weights include the routed scale 2.0), MXFP4 x MXFP4 routed experts (I 768 per rank, SwiGLU-OAI: alpha 1.702, limit 7,
up + 1), MXFP8 shared expert. Caller: vLLM's MoE layer (decode graphs to 1024 rows, eager prefill). Input:
normed BF16 `[M, 6144]`, contiguous, M 1-87381. One quant rule at every M: MXFP4 with Quark-even e8m0 per 32 for the
routed input and GEMM1 output, E4M3 with e8m0 `2^ceil(log2(amax / 448))` per 32 for the shared input and FP32 SwiGLU
output; M picks only tiles, launches and sum order. Slot k = `topk_weights[:, k] * expert(x)` over the full I 768,
rounded once to BF16; no output atomics, bitwise repeatable, graph replay = eager. Contract (`reference.py`):
`mxfp4_quant_ref`, `shared_mxfp8_quant_ref` bit for bit, `w4a4_moe_ref` within rel. L2 4e-3, `shared_mxfp8_ref`. Router
partials: K split and sum order follow M and the router kernel (weight err <= 5.1e-7). The one-launch GEMMs share one
per-device ready-counter arena, so two calls must not run concurrently on one device.

## Entry points

- `MoeWeights.from_row_major`: the router and routed experts' checkpoint tensors (and the `SharedWeights`, which
  `mxfp4_moe` needs) in the kernel layouts, plus a 3 MiB line-layout gate copy; once, before graph capture.
- `SharedWeights.from_row_major`: the shared expert's MXFP8 weights and scales as a bit-exact permutation.
- `unsupported_reason`: why a checkpoint's MoE tensors or activation do not fit (None: they fit).
- `mxfp4_moe`: the MoE block of a normed BF16 input or a prepared input: reduced, BF16 `[M, 6144]` (slots 0..3, then
  shared, summed in FP32), or the partials (slots `[M, 4, 6144]`, shared `[M, 6144]`) for the fused TP4 collective
  (`collectives/pull_collectives_ranks4`). An `amd_kernels.warm_up` entry: the first call of a mode (input kind,
  partials) compiles its variants at every `chain.WARM_UP_TOKENS` count in worker processes.
- `warm_up_mxfp4_moe`: at start-up, compiles and loads every variant of a mode into the process, once per mode; one
  layer's weights serve every layer; compile processes default to an eighth of the CPUs, at most 16.
- `prepare_mxfp4_moe`: routes and quantizes the local shard under sequence parallelism into a `MoeInput` (9632
  B/token); the caller all-gathers its tensors in rank order, padding rows included, and builds a `MoeInput` from
  them. Its compact form `MoeInputCompact` (7328 B/token) replaces the FP4 codes by a 1-bit mask, which the entry
  rebuilds bit for bit. Local routing can resolve a near-tie unlike the full-M call (1 row in 32768). A warm-up entry.
- `gather_mxfp4_moe`: packs and gathers the compact payload by a warmed TP4 `AllGatherComm` (equal shards <= 8192 rows).
- `set_dbo_query`: the server's query whether a call runs inside a DBO micro-batch (vLLM:
  `vllm.v1.worker.ubatching.dbo_enabled`); inside one the shared expert stays on the caller's stream, elsewhere its
  dense GEMMs run on one side stream per device with buffers from the caller's pool, joined before the return.

Dispatch is by M only, also under graph capture: `chain.py` picks the router, sort and expert host
(`experts_a4w4_small.py`, `experts_a4w4_bm128.py`, `experts_a4w4.py`), each host its kernels.

## Performance

One MI355X, layer-39 TP4 weights, shared expert on both sides: reduced entry vs the baseline MoE block,
graph replay of 100 inputs (16 above 2048 rows), median of 7 x 10 replays, us; last row: same-buffer A/Bs of the
partials mode. At 64-512 rows a serving trace of the baseline is faster (M128 155.6 us): no win shown

| M | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 102.56 | 115.12 | 133.24 | 153.11 | 171.32 | 198.69 | 233.47 | 354.72 | 468.07 | 576.62 | 815.19 | 1438.66 | 2621.55 |
| this entry | 59.70 | 86.88 | 116.85 | 147.83 | 165.59 | 183.65 | 227.16 | 249.02 | 298.09 | 416.36 | 648.43 | 1160.11 | 2226.09 |
| change | -41.8% | -24.5% | -12.3% | -3.5% | -3.3% | -7.6% | -2.7% | -29.8% | -36.3% | -27.8% | -20.5% | -19.4% | -15.1% |
| partials, same-buffer | 45.7 | | | 126.8 | | | | 224.6 | | 348.7 | | | 1656.0 |

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h6144_i768_e128_top4 -q
python3 benchmarks/moe/mxfp4_h6144_i768_e128_top4/benchmark_shared_moe.py --partials  # entry
python3 benchmarks/moe/mxfp4_h6144_i768_e128_top4/benchmark_prepared_sp.py  # 4 GPUs, SP phase
# also: benchmark_prepared.py, benchmark_sidecar.py
```
