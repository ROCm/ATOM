# Shared Mooncake Store with `store_tokens` routing: measured results

Single node, DeepSeek-V4.1-Flash (MTP, forced acceptance length 3.51), AgentX
trace replay through aiperf, gfx950. Every comparison below ran **at the same
time on the same node**, one group of GPUs per arm, so the arms see the same
machine load. Baseline ("main") is session-affinity routing with HBM prefix
caching only (no external tier, no LMCache). Percentages are relative to that
arm; TTFT/ITL lower is better, throughput higher is better.

## What is being measured

| | main | Store arm |
|---|---|---|
| Routing | `ATOM_DP_SESSION_AFFINITY=1`, `least_tokens` | `--dp-load-balance store_tokens` |
| KV tier | HBM prefix cache | HBM + one Mooncake Store shared by all DP ranks |
| Offload settings | - | branch defaults (see below), no environment variables |

Launch: `--kv-transfer-config` with `mooncake_store` (see `README.md`) plus
`--dp-load-balance store_tokens`. `ATOM_FRONTEND_PREFIX_HASH` already defaults
to 1.

Branch defaults that this result depends on:

| Setting | Default | Why |
|---|---|---|
| `OFFLOAD_MIN_LOAD_TOKENS` | 512 | A shared tier should serve what HBM would; 8192 re-prefilled ~11% more. |
| `OFFLOAD_LOAD_STREAM_HIGH_PRIORITY` | 1 | Load unpack 144 ms -> 3.6 ms mean on a busy GPU. |
| `OFFLOAD_STAGING_ON_COMPUTE_STREAM` | 1 | Store arms on side streams froze the engine for minutes in 5 of the 9 runs in which it was tried; no freeze in the 20+ Store arms run on the compute stream since. |
| `ATOM_OFFLOAD_EARLY_LOAD` / `_STATE` | 1 / 1 | Start the Store read when the lookup answers instead of one delayer cycle later. |

## Best measured configuration: DP2 (TP2 x DP2), concurrency 128, 900 s

Branch defaults, early tier load on. Three independent same-time batches:

| Batch | Throughput | TTFT mean | TTFT p50 | TTFT p90 | TTFT p99 | ITL |
|---|---|---|---|---|---|---|
| u09 (node A) | +2.2% | -24.5% | +5.1% | -31.3% | -65.2% | +0.9% |
| q29e el1 (node B) | +1.7% | -22.4% | +9.9% | -27.0% | -60.6% | +1.6% |
| q29e el2 (node B) | +2.0% | -25.4% | +10.5% | -30.4% | -61.3% | +3.3% |
| s09q eq512 (node A) | +2.7% | -24.1% | +17.5% | -30.1% | -74.3% | +0.7% |

Summary: throughput **+1.7% to +2.7%**, mean TTFT **-22% to -25%**, p90 TTFT
**-27% to -31%**, p99 TTFT **-61% to -74%**. Median TTFT is 5-17% higher and
ITL within about +-3% (a same-config repeat on the same node moves ITL between
-0.8% and +3.3%, so it is noise).

Effect of the individual pieces (DP2, same batch as the row above it):

| Change | Throughput | TTFT mean | TTFT p50 | Note |
|---|---|---|---|---|
| least_requests routing -> `store_tokens` | -3.7% -> -0.3% | +15.7% -> -27.4% | +23.3% -> +1.4% | Balance prefill by work, not request count. |
| prefill-delayer visibility fix | - | - | - | One-rank prefills 23% -> 5-7%; prefill steps 7.2% -> 5.5% (main 5.5%). |
| early tier load off -> on | +4.8% -> +2.2% | -24.6% -> -24.5% | **+22.5% -> +5.1%** | Removes one delayer cycle from a Store request. |

## DP4 (TP4, DP-attention), concurrency 256, 1800 s

| Run | Throughput | TTFT mean | TTFT p50 | TTFT p90 | TTFT p99 | ITL | Note |
|---|---|---|---|---|---|---|---|
| e29 (before early load) | -0.9% | -21.2% | +12.0% | -27.2% | -50.3% | +6.8% | |
| f09 | -0.1% | -11.6% | +15.6% | -13.6% | -35.2% | +3.8% | |
| i09 | +0.7% | -15.8% | +32.0% | -19.5% | -57.9% | +3.0% | |
| t29d (branch defaults) | +4.8% | +3.0% | **+80.9%** | -13.9% | -60.9% | -6.2% | main had 66 s of stalls; **open issue below** |

The DP4 rows are different code versions (e29: host-side event waits only; f09: same; i09: compute stream plus the since-removed delayer waits; t29d: the current branch), so they
show the trend, not one configuration.

**Open issue:** with early tier load on, the DP4 median TTFT regressed
(+80.9%). The step trace shows the prefill delayer holds longer on the Store arm
(mean hold 25 steps against 17.6, 58% of steps held against 41%) and each
released prefill is heavier (236 ms against 165 ms). Cause not yet identified;
an early-load on/off DP4 pair is being measured. Until then treat the DP2
numbers above as the validated result and the DP4 median as unresolved.

## Reproducing

Benchmark arms used `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`, Mooncake Store
128 GB (DP2) or 512 GB (DP4) over RDMA, `OFFLOAD_PROFILE=1`. Store usage after
1800 s at DP4/c256 was 190 GB of 512 GB with no eviction.
