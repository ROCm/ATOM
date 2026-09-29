# MiniMax-M3: TP QKV and replicated o_proj

GPU experiments completed on 2026-09-29. At 128K input / 100 output tokens / C16,
v2 measured 44.88 output tokens/s, 19,511.88 ms Mean TTFT and 160.81 ms Mean ITL.
Output throughput was 9.17% above the saved TP4 + QuickReduce result and 3.51%
above saved SP4. The separate output=1 prefill result was 56,561.73 input tokens/s,
1.34% above saved TP4 and 5.60% below saved SP4. These are comparisons against
2026-09-24 results from an earlier code revision, not paired measurements of the
layout change alone. No baseline was rerun.

A follow-up diagnosis on 2026-09-29 found that the original output=1 run slowed
progressively in its second half: both SP4 and that run completed the first 80
requests in about 176 seconds, while the latter took about 20.76 seconds longer
over the remaining 80. An unchanged-implementation repeat on a fresh server,
with 16 warmups and 160 measured requests, delivered 62,925.21 input tokens/s
(+5.02% versus historical SP4) without that drift. Both measurements are retained;
neither percentage isolates the layout effect. The original run followed other
workloads and had no continuous hardware telemetry, so its slowdown's underlying
trigger is still unresolved. See the [recorded results](#recorded-results) below.

Full GSM8K (1319 samples, 5-shot) scored 94.24% strict match and 94.16% flexible
extract; 128K/C16 retrieval passed 16/16. A controlled INT4 experiment motivated
keeping the accumulated replicated residual outside reduction (the v2 layout
below). Detailed request logs, source snapshots and GPU traces are retained in
the experiment artifacts outside this source branch; their results and scope
are summarized here.

The only new environment variable is `ATOM_M3_TP_REPLICATED_O_PROJ` (default `0`).
Config snapshots it for worker propagation and includes the layout in the
compilation cache key. Existing TP4 and Ulysses SP4 retain their default behavior.

| Path | Environment variable | TP | SP |
| --- | --- | ---: | ---: |
| Existing TP4 | `ATOM_M3_TP_REPLICATED_O_PROJ=0` | 4 | 1 |
| Existing SP4 | `ATOM_M3_TP_REPLICATED_O_PROJ=0` | 1 | 4 |
| Experimental replicated o_proj | `ATOM_M3_TP_REPLICATED_O_PROJ=1` | 4 | 1 |

## Recorded results

Experiments used four gfx950 GPUs and MiniMax-M3-MXFP4 with BF16 activations,
per-token FP8 projection quantization (excluding MoE), FP8 KV/index cache,
block size 128, token budget 32768, maximum model length 132096,
maximum sequences 128, memory utilization 0.8, and index cache with
`index_topk_freq=4`. Prefix caching was disabled. QuickReduce used INT4 with
`AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=1`.

The implementation was tested on ATOM base
`2080080877d320331c06af82cfa135f0f3df7bc2` and AITER
`67405e7fd1f40bf8fcc3ef766f49e541e4cf45a4`. Historical performance measurements
used the earlier ATOM base `edad514142066d9957fb9f253dd8ccf49d8f83b2`.

Each performance run used 131072 input tokens, concurrency 16, 16 warmup
requests and 160 measured requests. All requests succeeded with exact configured
input and output lengths. The SP4 output=100 column combines two existing runs
by total output tokens / total elapsed time; TTFT and ITL are request-weighted.

| Metric | Historical TP4 + QuickReduce | Historical SP4 | Replicated o_proj v2 |
| --- | ---: | ---: | ---: |
| Output throughput, output=100 (tok/s) | 41.11 | 43.36 | 44.88 |
| Mean TTFT, output=100 (ms) | 21,375.55 | 20,024.10 | 19,511.88 |
| Mean ITL, output=100 (ms) | 174.87 | 168.21 | 160.81 |
| Input throughput, output=1, original run (tok/s) | 55,812.11 | 59,915.82 | 56,561.73 |
| Input throughput, output=1, diagnostic repeat (tok/s) | Not repeated | Not repeated | 62,925.21 |

Mean ITL is the serving metric, including prefill/decode scheduling effects;
it is not a standalone decode-kernel latency. No performance result in this
table was measured with the profiler enabled.

The original output=1 run took 370.772 seconds, compared with historical SP4's
350.016 seconds. Their first 80 completions both took approximately 176 seconds
(progress-log resolution: one second); the extra 20.76 seconds occurred mainly
in the second half. The diagnostic repeat took 333.277 seconds with no comparable
drift. Its final 32-request window took 66 seconds versus 67 seconds for its
first comparable window; the original run's windows took 80 versus 69 seconds.

The repeat started a fresh server and ran only output=1. The original server
had first run smoke generation, long retrieval and output=100 benchmarks.
Telemetry was collected only for the repeat: worker RSS and GPU memory remained
stable, peak GPU hotspot temperatures were 90/90/92/93 degrees C, and average
clock frequencies changed by about 1% or less across elapsed-time quarters.
This does not identify the original slowdown's cause or exclude effects of
prior workloads. Retain both measurements; neither establishes the isolated
effect of the layout against the historical SP4 baseline.

| GSM8K, 1319 questions / 5-shot | Strict match | Flexible extract |
| --- | ---: | ---: |
| Historical TP4 + QuickReduce | 94.54% | 94.47% |
| Historical SP4 | 94.16% | 94.09% |
| Replicated o_proj v2 | 94.24% (1243/1319) | 94.16% (1242/1319) |

GSM8K used temperature 0, seeds `0,1234,1234,1234`, 1024 maximum generated
tokens and 64 concurrent requests. Both v2 scores exceeded the existing 93%
check. Long-context retrieval passed 16/16 at 128K/C16; this checks whether the
target code appears in the answer, not general long-context accuracy.

Independent output=20 profiles contained the same 64 prefill chunks on each
rank. Their prefill GPU spans were about 34.972 seconds for historical SP4 and
34.416 seconds for v2. Rank 0 communication-kernel duration summed to
11.204 versus 9.013 seconds; hidden norm/quantization summed to 0.762 versus
1.338 seconds. The profiles did not capture the original output=1 slowdown.
Collective durations include peer waits, so these figures are diagnostic
accounting rather than a controlled decomposition of the throughput change.

## Per-layer dataflow

Let `T` be the token count seen by normal TP attention, `P = ceil(T/4)*4`,
and `L = P/4`. All shapes below are per rank.

| Stage | Shape / placement | Communication |
| --- | --- | --- |
| Input residual and input norm | `[T,6144]`, replicated tokens | None |
| QKV projection and attention | Normal TP4 weights and KV metadata; attention output `[T,2048]` | No QKV input exchange |
| Pad output and exchange heads | `[P,2048]` → `[L,8192]` | Head all-to-all over TP group |
| Replicated o_proj | `[L,8192]` → `[L,6144]` | None |
| Post-attention residual add and norm | Use this rank's contiguous residual slice, `[L,6144]` | None |
| Router and MoE input quantization | Local tokens | Gather inputs and routing fields over TP group |
| MoE GEMMs | TP-sharded experts produce `[P,6144]` partial output | None |
| Attention increment merge | Add local o_proj output only to this rank's owned rows | None |
| Replicated layer increment | Drop padding and all-reduce `[T,6144]` | One TP all-reduce, including QuickReduce when selected |

The layer returns the reduced increment plus the original replicated input
residual as separate tensors. The next input norm (or final model norm) adds
them without an all-reduce. The accumulated residual is never quantized by
QuickReduce.

The first dense FFN layers keep their TP gate/up/down weights. They gather the
local normalized input (FP8 and FP32 scales when supported), compute full-token
partials and use the same attention-increment merge/all-reduce. If shared experts cannot be
fused into the MoE, their separate TP MLP also gathers its input and contributes
before the common all-reduce.

Padding is introduced only after attention. Positions, KV writes, sparse indexer
metadata and model-runner input ownership stay on the existing TP path. Dummy
rows are discarded before the output all-reduce. MoE metadata capacity includes
the padding even when the scheduler's token budget is not divisible by four.

## Quantization and transport

On the existing gfx950 BF16/per-token-FP8 recipe, large batches reuse the SP
attention output quantizer and IPC head-exchange kernel with the TP communicator.
FP8 requires one small BF16 amax all-gather before the main head exchange so all
head shards use the same per-token scale. Small batches use the established BF16
head gather/exchange followed by local FP8 quantization. Other projection
quantizers use BF16 transport. Therefore the low-byte-volume claim applies to the
large-batch FP8 path; it is not a guarantee for every batch or quantization recipe.

MoE reuses the existing capability checks for local MXFP4 quantization and local
top-k, with explicit TP-group gathers. Inline/small-batch kernels can gather BF16
inputs and logits instead. The existing tiled sorter remains eligible for the
same supported expert geometry and distributions; this change adds no new sorting
algorithm or AITER kernel.

## Residual semantics and numerical validation

The layer's input residual `R` is already replicated. Rank `r` owns a token
interval of the o_proj output `A`; it evaluates its FFN partial using the local
post-attention norm of `R + A`. Only `A` needs to join the FFN all-reduce:

`increment[t] = sum_r(FFN_partial_r[t] + owner_r(t) * A[t])`.

`output[t] = R[t] + increment[t]` (fused into the next/final norm).

CPU tests verify single ownership, padding and residual preservation. With
ordinary arithmetic this has the intended sum; floating-point reduction order
can still differ. INT4 quantizes the joint attention/FFN increment instead of
the two separate TP reductions. The completed accuracy checks above cover this
v2 path, but do not establish bitwise equivalence or accuracy for every task.

The initial version also put `R` into the owner contribution. A controlled
8192-token test on 2026-09-29, with fixed random FFN/attention tensors, measured
relative RMS error of 12.15% versus 4.05% with `R` outside. Scaling `R` by 16
gave 8.72% versus 0.268%. This is an operator diagnostic, not model accuracy;
the original inputs, results and initial source are retained in the experiment
artifacts. The correction adds no communication and no new switch.

Replicating FP8 o_proj weights adds approximately 2.11 GiB per rank versus TP4
for 60 layers (excluding scales and temporary loading memory). KV-cache capacity
and serving admission may change under the same memory-utilization setting.

## Completed validation and scope

GPU transport tests covered T=1,3,4,16,8192,32768,32769, including ranks owning
only padding. BF16 exchange, FP8 values and global scales matched independent
references bitwise. Gather, ownership and changed-input graph replay at T=3/32768
passed. Both ordinary reduction and eligible INT4 reduction were exercised.
The initial operator suite preceded the residual orchestration correction; its
INT4 error numbers describe v1. The controlled residual comparison and full-model
run validated v2 separately.

Full-model weight loading, piecewise compilation, graph capture across batch
sizes 1–128, generation, GSM8K and chunked 128K prefill completed. Both performance
workloads had 16 warmups and 160 successful measured requests, with exact input
and output lengths verified. CPU tests also cover residual preservation and
final norm without an extra collective. No full-model unquantized layer-by-layer
equivalence run was performed.

The 128K / output=20 / C16 rank-0 trace contains 64 prefill chunks. Each of the
60 layers has one FP8 head exchange and one INT4 all-reduce: 3840 of each, plus
64 existing TP embedding all-reduces. QKV packing/exchange and MoE reduce-scatter
are absent. The 57 MoE layers use MXFP4/scale/top-k gathers and the existing tiled
sorter (histogram, prefix_tiles, prefix_experts, scatter). Small batches retain
the existing BF16 transport and ordinary reduction fallbacks.

The server and all four workers were stopped after the run, and GPU memory was
released. No PR was changed. The implementation remains behind its single flag;
the unresolved drift in the original prefill run and the cross-date/revision
comparison limit conclusions about replacing the existing SP4 default.
