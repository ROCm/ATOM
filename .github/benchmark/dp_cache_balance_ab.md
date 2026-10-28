# V4.1 DP cache balance A/B

Select `dp-cache-ab` in the ATOM Agentic Benchmark workflow. It creates two
independent cells, each using attention DP2 / MoE TP2 on two MI355X GPUs,
128 concurrent session slots and 3600 seconds of profiling arrivals by default.
The workflow's `duration_seconds` input can override the duration for both arms.
To run only ON, set `variants` to `-agentic-dspark5-dtp2-cache-on`.
Leaving `variants` empty retains the two-arm comparison. Unknown suffixes fail
matrix generation instead of silently starting extra jobs.

| Arm | DP policy | Legacy session affinity |
| --- | --- | --- |
| OFF | `least_requests` | `0` |
| ON | `cache_affinity` | `0` |

Both arms enable unified prefix caching and LMCache MP with separate PAGE/STATE
object groups, 256 GiB host L1, LRU eviction and identical recovery thresholds.
The ON arm uses the policy defaults, including a 5-second preparation timeout.
This isolates the routing/recovery policy; it does not compare LMCache on/off
or compare this feature branch with unmodified main.

The workload uses the main branch's `agentx` scenario with the SA 393-entry
trace, seed 42, 5 warmup requests per lane, no context truncation and real
DSpark5 acceptance. GPU memory utilization
is 0.9. Pin `image` and `atom_commit` at dispatch to reproduce a run. The image
must include the LMCache MP CLI with native PAGE/STATE object-group support.

The opt-in helper starts LMCache inside each benchmark container before ATOM
and waits for its engine health check. Container teardown also stops LMCache.
It passes `--max-gpu-workers 2` so the two DP GPU clients can use separate
STORE/RETRIEVE handler threads. Override
`ATOM_BENCHMARK_LMCACHE_GPU_WORKERS` with a positive integer to change that
count; CPU handler workers retain LMCache's default. This uses the dependency's
existing affinity pool and requires no LMCache source patch.
The usual benchmark bundles contain TTFT, ITL, token throughput and errors;
`atom-cache-diagnostics-*` adds the LMCache log/status and final DP router
counters. These final counters include warmup and are not profiling-only
deltas. The two cells may run on different dedicated CI hosts, so host effects
remain a limitation of a single A/B run.

## GPU handler workers and guard audit (2026-10-28 UTC)

A separate simultaneous 900-second run compared one versus two LMCache GPU
handler workers, with the strict decode guard unchanged. Both used attention
DP2 / MoE TP2, c128, fixed AL=3.51 with synthetic acceptance, memory 0.95,
main-compatible graph/replay settings, SA393/seed42 and the same source/runtime.
These settings differ from the earlier real-acceptance comparison below.

| Metric | 1 GPU worker | 2 GPU workers | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s/GPU | 1521.44 | 1522.01 | +0.04% |
| Total tokens/s/GPU | 177569.78 | 180238.44 | +1.50% |
| Mean TTFT | 1190.09 ms | 1113.24 ms | -6.46% |
| P99 TTFT | 9398.56 ms | 6322.01 ms | -32.73% |
| Mean ITL | 26.47 ms | 26.74 ms | +1.05% |

Throughput uses successful profiling-record tokens over the request span,
929.485/928.646 seconds, divided by two GPUs. Total includes cached input.
Both clients exited zero; successful profiling requests were 3347/3434, with
0/1 exported errors (one connection reset in the two-worker arm). Among3316
matched successful requests, mean TTFT improved4.71% and mean ITL increased1.80%.
This single run shows a TTFT benefit, not a measured decode-throughput gain.

All numeric decision-audit events were collected. The two-worker arm rejected
516 formal-window migration candidates:334 had a target no lighter in either
request count or initial-prompt-context proxy,122 traded more requests for less
context, and60 traded fewer requests for more context. Of the516,198 had a joint
score gap of at most5%;34 would increase neither peak load proxy by more than1%.
These subsets overlap the preceding categories. The34 had predicted TTFT gain
median0.392s and mean0.577s; those are uncalibrated predictions, not measured
savings. The earlier498 rejected candidates had no individual audit records.

The new1% per-dimension tolerance targets this narrow subset. Setting
`decode_balance_tolerance_ratio=0` restores the strict guard used for both arms
in this table. The tolerance's performance has not yet been measured. It changes
ATOM placement only; LMCache transfer code and prefix-recovery checks are unchanged.

## Incremental routing comparison (2026-10-27 UTC)

This local comparison differs from the workflow's OFF/ON preset: **both arms
used cache affinity**. A was `c4de6c6` plus shared recovery fixes; B added
completed-prefill work retirement, expected recovered-prefix accounting,
separate restore backlog, exact-prefix recovery-miss suppression and joint
request-count/context balancing. No active decode migration was added.

- V4.1-Flash, MI355X, attention DP2 / MoE TP2, real DSpark5, memory 0.9,
  AgentX 1.0.9, SA393/seed42, c128 and separate 256 GiB LMCache servers.
- A GPU4/5, B GPU2/3 on one host. Common post-warmup barrier at
  `2026-10-27 17:21:15.396 UTC`; client starts differed by 0.61 ms.
- 900 s of arrivals. InferenceX throughput divides successful profiling tokens
  by first request start to last successful request end, then by two GPUs:
  929.249 / 929.933 s including completed drain requests. Total includes cached
  input. AgentX's own throughput summary uses a different duration; do not mix
  the two denominators.
- Immutable image:
  `rocm/atom-dev@sha256:57477e7a4b5156fcc7b21bab082eba98577752465f58479ac2ead8514f882a61`.
  AITER `41d4d8d7ea831d35f4a60a6c44228d7a33ed2e27` was unchanged.

| Metric | A | B | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s/GPU | 1091.23 | 1136.58 | +4.16% |
| Total tokens/s/GPU | 135063.23 | 139129.66 | +3.01% |
| Mean / P90 inter-token latency | 49.58 / 76.44 ms | 45.71 / 70.54 ms | -7.81% / -7.71% |
| Mean / P90 TTFT | 2.266 / 4.032 s | 2.405 / 4.618 s | +6.12% / +14.54% |
| P99 TTFT | 9.492 s | 9.292 s | -2.10% |
| Actual scheduled prefill tokens | 14951699 | 14942640 | -0.06% |
| Sampled mean in-flight request-count spread | 8.90 | 5.21 | -41.40% |
| Migration fallbacks / starts | 194 / 604 | 55 / 231 | |

The B constraint rejected 498 otherwise time-profitable migration candidates;
51 repeated exact-prefix misses were avoided. Router counts bracket the formal
window by at most 4.30 s; they are event counts, not exact request cohorts.
Reduced migration can protect decode balance while keeping more prefill at a
busy owner. The increased P90 TTFT must remain visible alongside throughput.

Both clients exited 0. A/B had 2569/2600 successful profiling requests and 1/0
profiling errors. A's `ClientOSError` occurred at +848.34 s; the same source in
B produced 2372 output tokens. Adding those tokens to A without changing its
denominator leaves a +4.03% output-throughput difference. This is an accounting
sensitivity check, not a counterfactual replay of downstream scheduling. B had
one `InvalidInferenceResultError` during warmup. Among 2506 shared successful
requests, output-length distributions matched and mean ITL fell 7.58%; their
mean TTFT still increased. Finite-window completion mixes differ. No statistical
significance or isolated causal breakdown is claimed.

An earlier healthy comparison of work accounting without the joint guard had
output throughput -2.31% and total throughput -1.90% despite improved TTFT. It
is retained as a negative result, not combined with this comparison.

### Common recovery and overlap limits

Both arms include the same ATOM fatal runner-loop exit fix and the same narrow
[LMCache STORE error-reply overlay](patches/lmcache-store-pre-dma-recovery.patch).
The overlay is an experiment artifact; it is **not installed by the workflow
or Dockerfile**. It applies to LMCache `f22dec2832124ea4ffeb464553b819a6369d4a30`;
the original `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` SHA256
is `a159eda921b97a394621e2cb3bed1cf2ded08de81e2500048a734cb6f85c7124`.
After verifying that hash, apply from the LMCache source root with
`git apply --unidiff-zero /path/to/lmcache-store-pre-dma-recovery.patch`.

The overlay handles only STORE producer-event import/wait failure before any
PAGE/STATE read or cache reservation. It returns False with a completion event;
the device future must prove completion before source buffers are released.
Failure to record/export that event remains unprovable. The same patched source
was mounted into both model and MP server containers. No HIP, producer-event,
unprovable-transfer or fatal-worker error was logged in the completed comparison.

The earlier failed comparison is excluded. Its STORE event exception left a
future nonterminal until the existing 1200 s safety deadline. The low-level HIP
cause is unproven. Upstream [LMCache #5439](https://github.com/LMCache/LMCache/pull/5439)
retains server-imported IPC events and is relevant to this lifecycle, but the
pinned image and this comparison do not include it. This comparison therefore
does not validate the unmodified pinned dependency under all failure modes.

The final revision passed 266 tests and 6 subtests on the exact runtime image,
including DP routing/backpressure, completed work and preemption, joint balance,
RPC/shutdown, native worker ownership and the common recovery cases. Earlier
work-accounting validation passed 322 tests and 6 subtests. These overlapping
test selections should not be added together.

Transfer scoring retains `max(compute readiness, transfer readiness)`. The
global STATE-restore compute fence remains unchanged; removing it requires
slot-reuse, relocation and cancellation safety. GPU-forward device-event
histograms were absent from this run's scrapes, so there is no measured stall
breakdown. The MP "Retrieved ... seconds" log stops at enqueue, not device
completion; do not fit an end-to-end transfer rate from that timer.

The separate [ON-only 3600 s Action](https://github.com/ROCm/ATOM/actions/runs/40521263205)
succeeded on baseline `c4de6c6`: 10473 successful requests, 1424.27 output and
163120.85 total tokens/s/GPU. It used another host and duration and is not paired
with the local 900 s A/B.
