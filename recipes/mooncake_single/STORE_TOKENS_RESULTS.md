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

## Latest: unaligned HBM hit floored to the Store chunk (head 0b0aef722)

Store loads start on a 256-token chunk boundary. A request whose HBM prefix hit
ends inside a chunk used to skip the early tier load and take the unaligned
handoff (recompute to the next boundary, load, prefill the rest: two delayer
releases). About 38% of Store-examined requests were in that case. Now the hit is
lowered to the chunk floor and the few tokens in between are recomputed
(`BlockManager.lower_hit_to_chunk`; only for connectors that declare
`lowers_hbm_hit_to_chunk`, i.e. Mooncake Store), when the Store holds at least
`OFFLOAD_MIN_LOAD_TOKENS` more than the lowered hit, there is no joint boundary,
per-request cache classes can be rebuilt by state replay, the SWA gate keeps the
lowered hit, and the pool has page units for the blocks given back.

Same-time DP2 (TP2) pair, concurrency 128, 1800 s, `OFFLOAD_PROFILE` off, no
environment variables:

| | main | Store | change |
|---|---|---|---|
| Completed requests | 8326 | 8420 | +1.1% |
| Output throughput (tok/s) | 3785.7 | 3846.2 | **+1.6%** |
| TTFT mean (ms) | 1961 | 1441 | **-26.5%** |
| TTFT p50 (ms) | 988 | 1008 | **+2.0%** |
| TTFT p90 (ms) | 4415 | 2899 | **-34.3%** |
| TTFT p99 (ms) | 12123 | 6963 | **-42.6%** |
| ITL (ms) | 15.66 | 15.64 | **-0.1%** |

The DP2 median TTFT gap went from +7% to +8% (previous head) to +2%.

Correctness checks (DP2, real decoding, no forced acceptance length):

* gsm8k 3-shot: main 0.9272, Store 0.9287 (stderr 0.007).
* Retrieval after eviction (48 prompts of 2-7k tokens, chat endpoint, greedy;
  round 1 cold, then 6M tokens of other prompts to evict HBM, then round 2 served
  through Store loads): main 39 / 37 correct, Store 37 / 36 correct (round 1 / round 2).
  Greedy text is not bit-identical between rounds on either arm (10/48 identical),
  so the comparison is on answer correctness. The test does not count how often
  the floor fired.
* `tests/test_lower_hit_to_chunk.py`: 11 unit tests (lowering, state replay,
  joint boundary, cache classes, pool exhaustion, early-load integration).

## Final results before the floor change (branch head b1ccf5b97, 1800 s, no environment variables)

Same-time, same-node pairs, 1800 s profiling window each. The Store arm runs the
branch defaults; main is session-affinity routing with HBM prefix caching only.

| | 128 concurrency, DP2 (TP2) | 256 concurrency, DP4 (TP4) |
|---|---|---|
| Completed requests, main / Store | 8306 / 8336 | 15424 / 15499 |
| Output throughput (tok/s), main -> Store | 3769.9 -> 3782.8 (**+0.3%**) | 6891.8 -> 6943.9 (**+0.8%**) |
| TTFT mean (ms) | 1891 -> 1481 (**-21.7%**) | 2560 -> 2401 (**-6.2%**) |
| TTFT p50 (ms) | 991 -> 1062 (**+7.1%**) | 1161 -> 1740 (**+49.9%**) |
| TTFT p90 (ms) | 4366 -> 2867 (**-34.3%**) | 5570 -> 4737 (**-15.0%**) |
| TTFT p99 (ms) | 10051 -> 7664 (**-23.7%**) | 25758 -> 10021 (**-61.1%**) |
| ITL (ms) | 15.98 -> 16.20 (**+1.4%**) | 14.06 -> 14.33 (**+1.9%**) |

No stall of 5 s or more in any arm. Store usage at the end: 190 GB of 512 GB (DP4),
97 GB of 256 GB (DP2); no capacity pressure.

A longer DP2 run of the same pair (3600 s, `OFFLOAD_PROFILE=0`, 512 GB Store):

| | 128 concurrency, DP2, 3600 s |
|---|---|
| Completed requests, main / Store | 15223 / 15410 |
| Output throughput (tok/s) | 4007.4 -> 4044.4 (**+0.9%**) |
| TTFT mean / p50 / p90 / p99 (ms) | 2028 / 925 / 4654 / 12957 -> 1380 / 999 / 2622 / 6889 (**-31.9% / +7.9% / -43.7% / -46.8%**) |
| ITL (ms) | 14.77 -> 14.97 (**+1.4%**) |

Reading: throughput is unchanged. The tail improves at both scales (p99 -24% to
-47% at DP2, -61% at DP4) and the whole distribution improves at DP2 except the
median (+7% to +8%). ITL is within about 2% of main (+1.4% at DP2, +1.9% at
DP4), which is not counted as a regression; p99 ITL is 11-13% higher. The
median TTFT is the open item: +7% to +8% at DP2, +50% at DP4 (the 900 s DP4
runs below were better, median +32% and mean -17%; cause of the difference not
identified).

Benchmark settings: `OFFLOAD_PROFILE` stays at its default (off). A same-time
test showed it does not change ITL (+2.4% on, +2.3% off), but it adds timing and
logging to every transfer, so benchmark with it off (the latest DP2 run and the DP4 run of the floor change were).
Step traces show Store decode steps about 1-2 ms slower than main (median
25-26 ms against 23-24 ms), about 0.4 ms of which is scheduler CPU in the engine
core.

The sections below are the development measurements that led to this version
(shorter 900 s windows, intermediate code).

## Development measurement: DP2 (TP2 x DP2), concurrency 128, 900 s

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
| t29d (branch defaults, before the pending-token fix) | +4.8% | +3.0% | +80.9% | -13.9% | -60.9% | -6.2% | main had 66 s of stalls |
| w09d (same code, same-time main) | +2.0% | +9.3% | +81.9% | -5.6% | -47.7% | -3.0% | confirms t29d |
| **y09d (current branch, 900 s)** | **+5.6%** | **-16.8%** | **+32.3%** | **-22.1%** | **-52.0%** | -0.5% | pending-token fix; same-time main |

The DP4 rows are different code versions (e29: host-side event waits only; f09: same; i09: compute stream plus the since-removed delayer waits; t29d: the current branch), so they
show the trend, not one configuration.

**What the DP4 median regression was.** With early tier load on, a request
whose Store load had finished stayed in `WAITING_FOR_REMOTE_KVS` until the next
prefill release, and the prefill delayer's fill signal
(`Scheduler._waiting_new_token_count`) counted it as zero tokens. The group's
fill stayed under the target (fill < 0.9 on 97% of hold steps, against 28% on
main), so 38% of releases came from the stall bound (6% on main) and median TTFT
rose by 64-82%. Counting what follows the loaded prefix (b8aae1324) halved the
median gap (+82% -> +32%) and moved mean TTFT from +9% to -17%.

**Still open:** median TTFT is 32% (900 s) to 50% (1800 s) above main at DP4 (DP2: 5-17%). With the
fix, 36% of releases are still stall releases. The ranks are nearly always all
prefillable, but each holds only a little pending work (aggregate fill 0.71 against the 0.9
target), because `store_tokens` spreads requests evenly while session affinity
makes a few large chunks. Under test: `ATOM_PREFILL_DELAYER_TARGET_FILL` and
`ATOM_PREFILL_DELAYER_STALL_TICKS`.

## Reproducing

Benchmark arms used `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`, Mooncake Store
256 GB (DP2) or 512 GB (DP4) over RDMA, `OFFLOAD_PROFILE` off. Store usage after
1800 s at DP4/c256 was 190 GB of 512 GB with no eviction.
