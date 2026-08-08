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
