# Where GLM-5.2 SparseKV decode time actually goes

Kineto trace, 2026-08-08, decode node (TP4, M48/RATIO=14, GPU cold tier on),
rank 0, ~3.2 s window over a live decode batch of 5-16 requests at ~60K context.
Captured with `scripts/trace_decode_window.py`; raw traces in
`results/trace_decode/`.

**Caveat on the regime.** The window ran at batch 5-16; the production c48 round
sustains 17-41. Per-layer fixed work does not scale with batch but the cold-pool
gather does, so at production batch the gather's share is likely *higher* than
what is measured here, not lower. Treat these as a lower bound on the gather.

## Headline

| bucket | GPU ms | share |
|---|---|---|
| **SparseKV cold-pool machinery** | **2481** | **43.7%** |
| — `sparsekv_gather_planned` | 1861 | 32.6% |
| — `sparsekv_swap_and_translate` (detect) | 479 | 8.4% |
| — `sparsekv_backup` | 141 | 2.5% |
| MoE (gemm1/gemm2 a4w4, mfma_moe1/2, sort_quant, grouped_topk) | ~1075 | 18.8% |
| TP all-reduce (`allreduce_fusion_kernel_1stage`) | 402 | 7.0% |
| DSA indexer (`radix_topk_one_block`, `paged_mqa_logits`) | 224 | 3.9% |
| MLA attention (`mla_a8w8` + `kn_mla_reduce`) | 298 | 5.2% |
| everything else (hgemm, rope, rmsnorm, quant) | ~1200 | 21% |

The swap path costs **2.3× the entire MoE** and **8× the MLA attention it exists
to feed**. That is the headroom.

## The step is not launch-bound

Kernel time sums to 5683 ms over 4812 ms of wall clock with at least one kernel
running — an overlap factor of 1.18×. The GPU is saturated. Fusing kernels to
save launch overhead is not the lever here; moving less data is.

## CUDAGraph padding is not a problem

The capture sizes are `[1, 2, 4, 8, 16, 32, 48]`, and the production batch
distribution (mean 17) pads 29% of graph slots. That number is real and
irrelevant — step time barely tracks batch at all:

| graph bs | real bs | mean step |
|---|---|---|
| 8 | 5 | 18.03 ms |
| 8 | 8 | 19.19 ms |
| 16 | 9 | 15.58 ms |
| 16 | 13 | 17.14 ms |
| 16 | 16 | 17.20 ms |

+78% real work inside graph 16 costs +10% time, and graph 16 at real 9 is
*faster* than graph 8 at real 5. Adding finer capture buckets would buy
approximately nothing. Do not spend effort here.

The corollary matters much more: **step time is nearly flat in batch, so decode
throughput is close to linear in sustained batch.** That is exactly the measured
concurrency curve (53 → 61 → 215 → 287 tok/s for c8 → c16 → c32 → c48). Anything
that raises sustained batch is worth more than anything that shaves a kernel.

## The dual-source gather scans the miss list twice

`_gather_planned_dual` issues two `sparsekv_gather_planned` launches per layer
per step — one per home — and each walks the *entire* recorded miss list,
skipping the entries belonging to the other home. Separating them by issue
order:

| pass | launches | total | mean |
|---|---|---|---|
| A — host home (issued first) | 14,898 | 1195 ms | 80.2 µs |
| B — GPU home (issued second) | 14,898 | 666 ms | 44.7 µs |

Pass B reads HBM, where the data movement is nearly free, yet it costs 666 ms —
**12% of all decode GPU time, most of it re-scanning a list pass A already
walked.** One kernel taking both base pointers and both translation tables, and
dispatching each miss to its recorded home, would do the same data movement with
one scan and one launch per layer. This is the cleanest win available.

## Only half the gather is hidden

The IndexShare prefetch issues shared layers' gathers on a side stream. It works,
but partially:

- gather wall span: 1861 ms
- overlapped with other kernels: 871 ms (46.8%)
- **exposed on the critical path: 990 ms — 20.6% of the busy wall clock**

So a fifth of decode wall time is the cold pool, fully in the way. Deepening the
overlap (more layers prefetched ahead, or starting the host-home gather earlier
than the anchor's own detect) is worth as much as making the gather cheaper.

## An objective the joint sizing does not model

The closed form in `atom/sparsekv/sizing.py` splits the HBM budget to **maximize
the batch ceiling** (`index_pages = host_pages + gpu_cold_pages`, and an index
page costs a quarter of a GPU cold page). It has no term for what a token's home
costs at *gather* time: a host-home token crosses PCIe on every miss, a GPU-home
token does not.

At M48/RATIO=14 the split lands at 344,106 host pages against 37,918 GPU pages —
**9.9% of pages on the GPU tier**, yet the host pass costs 1195 ms against the
GPU pass's 666 ms. Shifting budget toward the GPU tier trades ceiling (which
throughput needs, per the linear-in-batch result above) against gather cost
(which step time needs). Nobody has measured which side of the optimum the
current split sits on. `GPU_COLD` share is the knob; a sweep at fixed M is the
experiment.

## Ranked

1. **Merge the two gather passes into one kernel.** ~12% of decode GPU time is a
   redundant scan. Contained change, no scheduling risk.
2. **Sweep the host/GPU cold split at fixed M.** Decides whether the sizing
   objective needs a gather-cost term at all. Pure config, no code.
3. **Extend the prefetch depth.** 990 ms of gather is exposed; the mechanism to
   hide it already exists and currently covers 47%.
4. **Detect at 119 µs × 21 anchors/step** (8.4%) — worth a look only after the
   gather work lands.
5. **CUDAGraph capture sizes — do nothing.** Measured, not a cost.

---

# What the optimization changed (2026-08-08, same day)

Two changes to `sparsekv_gather_planned`, measured with the same load and window
(`--requests 24 --ctx-tokens 60000`, batch 5-16):

| variant | gather/step | kernel/step | **wall/step** | launches/step |
|---|---|---|---|---|
| two passes, `grid=n` (baseline) | 9.74 ms | 29.76 ms | 25.20 ms | 156 |
| merged into one pass | 4.95 ms | 25.61 ms | 25.19 ms | 78 |
| merged + `grid.y` over the miss list | **0.36 ms** | **20.53 ms** | **20.53 ms** | 78 |

**The merge alone bought nothing in wall time.** It cut GPU kernel work 14% and
halved the launch count, both real, but step time did not move: the pass it
removed was the one the IndexShare prefetch had been hiding (overlap fell from
1.18x to 1.02x), so the critical path was untouched. Worth recording — "less GPU
work" and "faster" are not the same claim, and only the second one was asked for.

**The grid was the actual bug.** The gather launched `grid = n`, one block per
query token: at a decode batch of 16 that is 16 blocks of 4 warps on a 256-CU
device — under 6% of the machine — with each warp walking a hundred rows in
series. Adding a second grid dimension over the miss list, sized to keep ~2048
blocks resident, took the gather from 9.74 to 0.36 ms/step and step wall time
from 25.20 to **20.53 ms (-18.5%)**.

## The gather is now near its ceiling

`scripts/bench_sparsekv_gather.py`, n=16 queries x 2048 misses of 576 B:

| host share | before | after | achieved | vs contiguous ceiling |
|---|---|---|---|---|
| 100% host | 1.473 ms | 0.380 ms | 49.6 GB/s | **88%** of 56.7 GB/s H2D |
| 90% | 1.461 ms | 0.342 ms | 55.2 GB/s | |
| 50% | 1.314 ms | 0.197 ms | 95.8 GB/s | |
| 0% (all GPU tier) | 0.514 ms | 0.013 ms | 1476 GB/s | 37% of 4031 GB/s D2D |

88% of the contiguous host-to-device rate for a *scattered* 576 B gather leaves
nothing worth chasing in this kernel. Further gains on the host path have to come
from moving fewer bytes, not from a faster gather.

## The profile after

| bucket | share |
|---|---|
| MoE | 23.3% |
| SparseKV swap (was 43.7%) | **21.6%** |
| — detect (`swap_and_translate`) | 11.9% |
| — promote / initial-hot gather (`sparsekv_swap_in`) | 4.4% |
| — backup | 2.4% |
| — planned gather | ~0.4% |
| TP all-reduce | 9.9% |
| MLA attention | 6.5% |
| DSA indexer | 5.1% |

MoE is now the largest single bucket. Within SparseKV the remaining target is
**detect**: one block per query token, and its LRU victim search binary-searches
the recency range, rescanning all 8193 hot slots on every step of the search. It
cannot be split across blocks the way the gather was — the search is a block-wide
reduction over the whole hot set — so the levers are threads per block and a
cheaper victim-selection algorithm.

Validation for the above: 83 unit tests, `check_sparsekv_row_bounds.py`,
needle 4K-114K 12/12, GSM8K 200 flexible 0.970 / strict 0.965 (pre-change
0.945/0.940).

## What did not work: a wider detect block

The detect kernel is one block per query token, and its LRU victim search
binary-searches the recency range, rescanning all 8193 hot slots on every step —
at 256 threads that is 32 slots per thread per step. Raising just that kernel's
block to 1024 (the kernel is written against `blockDim.x` throughout, so it
looked safe) **faults within three decode steps**:

```
Memory access fault by GPU node-9 on address 0x76dd1aa6a000. Reason: Unknown.
```

Reverted. Two things make this worth recording rather than retrying:

- The row bound was armed and reported **zero** out-of-range rows, so this is
  *not* the out-of-range `cold_row` failure mode from
  `sparsekv_decode_row_bound_fault.md`. The address is 4 KiB aligned, so it is
  still a host-pool dereference, but by a path the bound does not cover.
- It means the detect kernel has an undiagnosed dependence on its block size.
  `vic[]` is filled with an unbounded `atomicAdd` index (`vic[idx] = s` with no
  `idx < topk` check); the invariant that keeps it in range —
  `count(last_used < tau) < m <= topk` — rests on the tau binary search
  converging, which is a block-wide reduction. That is where to look.

Do not change the detect block size without root-causing this first. And it is a
live lead for the still-open decode fault: it shows a host-address fault can be
reached without tripping the row bound.

## End-to-end: the win is real, and it does not show up in throughput

Full c48 round on M48/RATIO=14, same config as the pre-optimization baseline:

| | before (`postfix_c48_m48_r14`) | after (`gather_opt_c48`) |
|---|---|---|
| **itl p50 / p95** | 60 / 70 ms | **30 / 36 ms** |
| **intvty p50** (per-user tok/s) | 17 | **33** |
| output tok/s | 288.0 | 265.6 |
| peak decode batch | 41 | 24 |
| index pool peak | 91% | 57% |
| admission DEFERs | 3 | 0 |
| TTFT p50 | 99.6 s | 158.6 s |
| prefix-cache hit | 37.5% | 30.8% |
| failures | 0 | 0 |

**The system is prefill-bound, and was already.** Both runs computed exactly
**94.2M prefill tokens at 26.0K tok/s** and scheduled 24,006 vs 24,000 prefill
batches — the prefill node is pinned at its ceiling and did identical work in
both. The output-token difference is a workload difference, not a regression:
the second run drew a lower prefix-cache hit (30.8% vs 37.5%), so the same
94.2M tokens of prefill compute served 1109 requests instead of 1261.

So the decode optimization cannot raise system throughput at this PD split. What
it bought instead:

- **per-user decode speed doubled** (intvty p50 17 -> 33 tok/s, ITL 60 -> 30 ms),
  which is the metric a user actually feels;
- **half the decode node is now idle capacity** — batch 41 -> 24 for the same
  offered load, index pool 91% -> 57%, and admission stopped deferring.

**Anything further on the decode side is worth ~0 in throughput until prefill is
addressed.** The levers are, in order: profile the prefill node (never done —
26.0K tok/s is an unexplained number), and rebalance the PD GPU split, since
decode now has roughly 2x the capacity this workload needs.

---

# The prefill prefix cache was the throughput lever (2026-08-08)

Prefill is the ceiling, and a prefill trace showed nothing structural to reclaim:
inside a common 4924 ms window all four PP stages ran 91-97% compute-busy with
idle near zero, and the 18,20,20,20 split left only a 6% compute spread. So the
only ways to serve more requests are to compute fewer tokens or buy more
hardware. Computing fewer tokens means a higher prefix-cache hit — and 81% of
prefill requests were evicting cached blocks.

`PREFILL_GPU_UTIL=0.93` (from 0.85) moves the prefill KV budget 244.79 -> 267.83
GB per GPU. One config value:

| | before (`gather_opt_c48`) | after (`prefill_util93_c48`) |
|---|---|---|
| **output tok/s** | 266.2 | **388.4 (+46%)** |
| total tok/s | 37,702 | 52,686 |
| requests served | 1109 | **1558** |
| **prefix-cache hit** | 30.8% | **53.5%** |
| prefill tokens actually computed | 94.2M | 88.6M |
| scheduled prefill batches | 24,000 | 18,983 |
| TTFT p50 | 158.6 s | 85.6 s |
| e2e p50 | 177.7 s | 104.8 s |
| failures | 0 | 0 |

The mechanism is exactly the one the earlier runs implied: prefill compute is a
fixed budget, and the hit rate decides how many requests that budget buys. Here
it bought 40% more requests off *less* compute. The hit moved well outside the
30.8-37.5% band two identically-configured runs had already drawn, so this is
not the run-to-run variance that band represents.

Against the session's starting point (`postfix_c48_m48_r14`, 288.6 tok/s) this
is +35%.

## Where the bottleneck sits now

Both sides are close to balanced:

- prefill's compute rate fell 26.0 -> 24.5K tok/s, so it is no longer pinned;
- decode came back under load — batch 40/48, index pool 91%, **host cold pool
  343,674/343,722 (99.99%)**, and admission deferred 5 times.

The host cold pool is the hard wall now, and RATIO cannot go much higher (16 is
already where startup pinning is known to hang). With host pages fixed, total
decode capacity is `host + gpu_cold` and gpu_cold is what the HBM budget buys —
so the symmetric next experiment is the decode side's own `GPU_UTIL`, and 83% of
prefill requests still evict, so its cache is not saturated either.

## Throughput on this workload is a function of the cache hit, and little else

Four c48 rounds, same trace, same duration:

| run | prefix hit | out tok/s | fit | residual |
|---|---|---|---|---|
| `postfix_c48_m48_r14` | 37.5% | 288.0 | 295.5 | -7.5 |
| `gather_opt_c48` | 30.8% | 265.6 | 259.4 | +6.2 |
| `prefill_util93_c48` | 53.5% | 387.6 | 382.5 | +5.1 |
| `both_util_c48` | 47.9% | 348.6 | 352.5 | -3.8 |

`out tok/s = 91 + 5.44 x hit%`, **R^2 = 0.985**, every residual inside +-2%.

Two things follow, and they are the most useful results here.

**Nothing that fails to move the hit rate moves throughput.** The gather kernel
work (decode step -18.5%, ITL halved) and the decode `GPU_UTIL` raise (index
pages +6.3%, GPU cold pages +63%) both land inside the residual. They are real
improvements to latency and to headroom — decode went from index 91% / 5
admission deferrals to 74% / 0 — but on this workload they are not throughput.

**The single-run resolution is +-15 tok/s.** Two rounds at the *same* prefill
setting drew 53.5% and 47.9%, so the hit rate itself carries +-2.8 points of
run-to-run variance. Any A/B expecting less than that needs repeats, and the
+16.5 points the prefill cache bought is the only change so far that clears it
comfortably.
