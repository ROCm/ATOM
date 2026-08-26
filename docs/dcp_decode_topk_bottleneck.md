# GLM-5.2 DCP decode: where the step time goes, before and after the index-page relayout fix

**Re-measured 2026-08-26.** The 1.60 ms below was measured while the mooncake
DCP relayout was corrupting the DSA index cache, so the top-k ran on a plane
that was 5-7 % NaN -- a single radix bucket the kernel cannot split. On the
fixed relayout (`docs/dcp_index_cache_page_relayout.md`) the same kernel costs
**0.40 ms**. The numbers in this document are kept as the before side of that
A/B; the after side is the section at the end.

Measured 2026-08-25 on MI355X, GLM-5.2-MXFP4, PP4xTP1 prefill -> TP4xDCP4 decode
over mooncake PD, replaying the SemiAnalysis `cc-traces-weka-062126` agentic
corpus at concurrency 32 for 3600 s.

Artifacts: `results/glm52_pp4pd_dcp_replicate_aiperf_1h_20260825_1326/`
(1 h run + `traces_replicate1/` -- a 3-minute kernel-only kineto capture taken
20 minutes into the profiling phase, 14:02:53-14:05:53).

## What the decode step is actually made of

Rank 0, 3-minute window, 4.80 M GPU kernel events, 156.0 s of GPU time:

| bucket | GPU time | share |
|---|---|---|
| **DSA top-k select** | **79.26 s** | **50.8 %** |
| DSA indexer (fp8 paged mqa logits) | 19.02 s | 12.2 % |
| MoE | 17.73 s | 11.4 % |
| GEMM / linear | 14.03 s | 9.0 % |
| Collective (RCCL) | 9.53 s | 6.1 % |
| MLA attention | 5.75 s | 3.7 % |
| norm / elementwise | 3.87 s | 2.5 % |

One kernel, `aiter::ob::radix_topk_one_block_kernel`, is 50.2 % of all GPU time:
48 875 calls at **1.60 ms each**, one per full-IndexShare layer per decode step,
so ~33.6 ms of a p90 68.9 ms step. It is 47.7 % of the time inside `decode[...]`
spans (78.24 s of 164.1 s).

MLA attention is 3.7 %. The sparse indexer -- logits plus top-k -- is 63 %.

## Replicating the index cache quadruples that top-k

`ATOM_DCP_REPLICATE_INDEX_CACHE=1` exists to skip the indexer candidate
all-gather and global merge (`_dcp_index_comm_required`,
`deepseek_v2.py:1401`). What it costs is visible in the two decode branches:

| | logits plane | context handed to top-k |
|---|---|---|
| sharded (`=0`) | `l_max = ceil(max_model_len / W)` -- 1/W wide | `local_ctx = g_ctx / W` |
| replicated (`=1`) | `[num_rows, max_model_len]` -- full width | `context_lens`, the global length |

Every rank scans **W times as many logits** (`deepseek_v2.py:1457` vs `:1832`).
The all-gather it buys back is inside a 6.1 % bucket, of which
`ncclDevKernel_Generic` is 2.7 %.

**Trading a >=3x increase on a 50 % component for a <=3 % collective is a net
loss at this context length.** Expect the sharded path to cut the top-k to
roughly 8-9 ms/step and the step from ~69 ms to ~45 ms. Not yet measured --
the A/B capture is the obvious next experiment.

## What is *not* the bottleneck

- **KV capacity.** `atom:preemptions_total` = 0 over the run; resident 25-35 of
  `max_num_seqs` 512; the admission gate (`scheduler.py:1294`,
  `len(running) + parked >= max_num_seqs`) never fired. Contrast the DPA
  baseline (`results/glm52_pp4pd_dpa_aiperf_1h_20260814_0416/`), which pinned
  `kv_usage` at 1.00 with 882 backpressure samples -- DCP fixed that.
- **KV transfer.** 554 GiB over the steady hour = 158 MB/s average. TTFT is
  2.3 % of request latency (DPA: 35.7 %); TTFT p50 827 ms against DPA's 7152 ms.
- **Prefill compute.** 8.3 M tokens recomputed against 276 M served from cache
  (97.1 % hit) -- about 2.3 k tok/s of real prefill work.

Note that the `PD backpressure` log line is *not* a capacity signal: it prints
only every 100 scheduler ticks and counts requests in
`WAITING_FOR_REMOTE_KVS`, which every PD request passes through while its KV
arrives. It would be non-zero with an infinite KV pool.

## Why the headline throughput trails the DPA baseline

431 tok/s here against 797 tok/s for DPA, but the two runs are not the same
decode mode: **the DPA baseline ran MTP** (`--method mtp
--num-speculative-tokens 3 --spec-decode-acceptance-rate 0.6633`, ~3 tokens per
step), and `ATOM_DCP_REPLICATE_INDEX_CACHE` forbids speculative decode outright.
`concurrency / ITL` reproduces both numbers (DPA `12.51/0.01556 = 804`, DCP
`24.63/0.05342 = 461`), so the gap is tokens-per-step, not time-per-step: DCP
carries 2x the decode concurrency at 1.15x the step time.

MTP under DCP is blocked separately, by
`assert attn_metadata.max_seqlen_q == 1` in `_dcp_decode_candidate_exchange`
(`deepseek_v2.py:1441`).

## After the page-aware relayout fix (2026-08-26)

Same script, same corpus, same concurrency 32, same 3-minute kernel-only capture
20 minutes into the profiling phase. Artifacts:
`results/dcp_index_relayout_fix_20260826/traces_postfix/` (rank 0 analysis and
the re-run of the same analyzer over the 2026-08-25 trace, side by side).

The two windows do the same amount of work -- 48 875 vs 48 890 top-k calls,
174 545 vs 174 600 MoE calls -- so the times compare directly.

| | broken relayout | page-aware relayout |
|---|---|---|
| `radix_topk_one_block_kernel` per call | 1.601 ms | **0.400 ms** |
| that kernel's share of GPU time | 50.2 % | **19.7 %** |
| GPU time in the 3-min window | 156.0 s | **99.3 s** |
| decode step p90 | 68.87 ms | **45.08 ms** |
| decode step avg | 32.66 ms | **16.49 ms** |
| decode steps in the window | 5 024 | **6 691** |

| bucket | before | after |
|---|---|---|
| DSA top-k select | 79.26 s (50.8 %) | 20.55 s (20.7 %) |
| DSA indexer (fp8 paged mqa logits) | 19.02 s (12.2 %) | 16.85 s (17.0 %) |
| MoE | 17.73 s (11.4 %) | 20.65 s (20.8 %) |
| GEMM / linear | 14.03 s (9.0 %) | 16.18 s (16.3 %) |
| Collective (RCCL) | 9.53 s (6.1 %) | 8.79 s (8.9 %) |
| MLA attention | 5.75 s (3.7 %) | 5.72 s (5.8 %) |
| norm / elementwise | 3.87 s (2.5 %) | 3.80 s (3.8 %) |

Every other bucket holds its absolute time to within noise; only the top-k
moves. That is the signature of a data-dependent kernel, not of a slower
machine: the radix select was spending its time on a NaN bucket that the
transfer had manufactured.

**This retires the sharded-vs-replicated recommendation above.** The 3x logits
plane that replication costs is real, but it is now 3x of a 20 % component,
not of a 50 % one, and the decode step is MoE-and-GEMM-bound again. The
sharded A/B is worth running for its own sake, not as a fix for this.
