---
name: lumen-sdma
description: "Use when deciding whether SDMA or NCCL/RCCL is the right backend for a Lumen benchmark, training path, or collective operation. Covers schedule-specific backend rankings, crossover points, and how to interpret benchmark results without over-generalizing."
---

# Lumen SDMA Guide

Use this skill when a user asks any of the following:

- when `SDMA` should be enabled in Lumen
- whether `SDMA` or `NCCL`/`RCCL` is better for a benchmark or training path
- why `SDMA` wins in one benchmark but loses in another
- how to choose a backend for `AllGather`, `ReduceScatter`, fused pipeline, or deferred wgrad

This skill is based on the `main` branch benchmark result set under `benchmarks/results`,
especially:

- `bench_comm_overlap_2GPU_analysis.md`
- `bench_comm_overlap_8GPU_analysis.md`
- `bench_e2e_fusion_8GPU_analysis.md`
- `bench_fp8_param_allgather_8GPU_analysis.md`
- `bench_fused_pipeline_8GPU_analysis.md`
- `bench_wgrad_delay_8GPU_analysis.md`

If those result files change, re-validate the crossover points and defaults before
reusing this skill verbatim.

## Preconditions

These conclusions are specific to the measured AMD + MORI stack and should not be
treated as hardware-agnostic.

- `SDMA` in these benchmarks assumes the MORI path is available
- distributed SDMA tests typically require `MORI_ENABLE_SDMA=1`
- 2-GPU and 8-GPU results should be treated as different operating regimes

## Core Rule

Do **not** talk about `SDMA` as if it is globally faster or slower.

Always classify the question into one of these buckets first:

1. **Raw collective / overlap microbench**
2. **Fused end-to-end schedule**
3. **Deferred wgrad schedule**
4. **FP8 param gather pipeline**

The backend ranking is **schedule-specific**.

## Fast Decision Table

| Situation | Recommended backend | Why |
| --- | --- | --- |
| 2-GPU `AllGather` overlap microbench | `SDMA` | Best absolute latency across all measured `N` |
| 2-GPU `ReduceScatter` overlap microbench, `N=256` | `NCCL` | Small-message fixed overhead hurts SDMA |
| 2-GPU `ReduceScatter` overlap microbench, `N>=1024` | `SDMA` | Bandwidth advantage begins to amortize |
| 8-GPU `AllGather` overlap microbench, small/medium `N` | `NCCL` | SDMA control-plane overhead dominates |
| 8-GPU `AllGather` overlap microbench, very large `N` around `28672` | `SDMA` or tie-check | Only regime where SDMA catches up or narrowly wins |
| 8-GPU `ReduceScatter` overlap microbench | `NCCL` | NCCL wins at every measured size |
| 8-GPU `bench_e2e_fusion` fused backend | `SDMA` | Better fused backend than NCCL inside the fused schedule, even though naive still wins overall |
| 8-GPU `bench_fused_pipeline` column-heavy or longer paths | `SDMA` | Wins most phase/chunk points, especially `column_*` and `row_fwd_bwd` |
| 8-GPU `bench_fused_pipeline` short `row_fwd` | `NCCL`/`RCCL` first | SDMA loses at `chunks=1/2/4` and only nears parity at `chunks=8` |
| 8-GPU `bench_wgrad_delay` | `NCCL` | Deferred-wgrad overlap usually favors NCCL |
| FP8 param allgather raw collective | `chunks=2` best raw | Raw bandwidth win only |
| FP8 param allgather end-to-end | `chunks=1` best practical default | Extra chunking adds schedule overhead |

## What SDMA Is Good At

`SDMA` tends to be good when all of the following are true:

- the unit under test is dominated by transport/backend behavior, not Python-side or per-chunk orchestration overhead
- the message is large enough to amortize fixed setup cost
- the path is `AllGather`-like or column-parallel heavy
- the benchmark measures a full fused backend where SDMA's lower transport cost can improve end-to-end latency

In practice, that means:

- `2 GPU` overlap microbenches are often a good fit for `SDMA`
- `8 GPU` fused backend comparisons can still favor `SDMA`
- large-message `AllGather` is the most promising 8-GPU overlap-microbench case for `SDMA`

## What SDMA Is Bad At

`SDMA` is usually a bad default when any of these are true:

- the path is `ReduceScatter`-heavy at `8 GPU`
- the path is a short `row_fwd` phase with little compute to amortize setup
- the schedule contains explicit synchronization or control-plane overhead that collapses overlap
- the benchmark is `deferred wgrad`, where `NCCL` currently behaves more robustly

In practice, that means:

- prefer `NCCL` for 8-GPU `ReduceScatter` overlap microbenches
- prefer `NCCL` for `bench_wgrad_delay`
- be cautious with small tensors and fine chunking

## Benchmark-Specific Reading Rules

### 1. `bench_comm_overlap`

Use this benchmark to reason about **overlap microbenches** and simple comm+compute
patterns, not standalone collectives in total isolation.

Safe takeaways:

- `2 GPU AllGather`: `SDMA > NCCL`
- `2 GPU ReduceScatter`: small `N` favors `NCCL`, larger `N` favors `SDMA`
- `8 GPU AllGather`: usually `NCCL`, with SDMA only catching up at the largest tested size
- `8 GPU ReduceScatter`: `NCCL > SDMA` across the whole sweep

Phrase answers carefully: this benchmark varies the effective overlap workload and
should be described as an **overlap microbench**, not as a pure standalone
collective benchmark.

### 2. `bench_e2e_fusion`

Use this benchmark to reason about the **current fused single-layer pure-pipeline backend**.

Safe takeaway:

- inside this schedule, `SDMA > NCCL`
- but both fused backends still lose to `naive`
- there is no measured case where fused-over-naive is justified by memory alone
- SDMA vs NCCL is not the memory differentiator in this benchmark

Do **not** use this benchmark to claim every raw SDMA primitive is faster than NCCL.

### 3. `bench_fused_pipeline`

Use this benchmark to reason about **backend behavior inside a chunked fused schedule**.

Safe takeaway:

- `SDMA` wins most measured phase/chunk points
- `column_fwd`, `column_fwd_bwd`, and `row_fwd_bwd` are good SDMA cases
- `row_fwd` is the stress case where SDMA often loses
- more chunks improve SDMA's relative position in some cases, but still worsen absolute latency

### 4. `bench_wgrad_delay`

Use this benchmark to reason about **deferred wgrad overlap**, not generic SDMA value.
Also note the benchmark's design intent may sound more optimistic about SDMA than
the current measured result set; use the measured results, not the aspirational
story, as the final source of truth.

Safe takeaway:

- current evidence favors `NCCL`
- SDMA rarely shows a clear win
- large shapes only approach break-even; they do not yet show a stable SDMA crossover

### 5. `bench_fp8_param_allgather`

Use this benchmark to separate **raw comm wins** from **real end-to-end wins**.

Safe takeaway:

- raw FP8 allgather likes `chunks=2`
- end-to-end FP8 gather+compute prefers `chunks=1`
- `chunks=4` is consistently the worst practical choice in the measured 8-GPU runs

## Common Answer Pattern

When a user asks "Should I use SDMA here?", answer in this order:

1. State the **unit under test**: overlap microbench, fused backend, deferred wgrad, or FP8 pipeline
2. Give the **backend recommendation**
3. Explain the **amortization story**: whether fixed control/sync overhead is being hidden or exposed
4. Add the key caution: do not transfer conclusions across benchmarks with different schedules

## Recommended Defaults

Use these defaults unless the user asks for experimental tuning:

- `8 GPU AG/RS overlap microbench`: default to `NCCL`
- `8 GPU fused layer backend inside the current fused schedule`: default to `SDMA`
- `8 GPU deferred wgrad`: default to `NCCL`
- `FP8 param gather chunk count`: default to `chunks=1`
- `2 GPU AG`: default to `SDMA`

Do not reinterpret these as blanket full-training defaults without checking which
schedule the user is actually asking about.

## Boundary Cases Worth Re-Profiling

If the user wants to verify a decision rather than rely on the current results, profile these first:

- `8 GPU AllGather` near `N=28672`
- `row_fwd` in `bench_fused_pipeline`
- largest-shape `NCCL` cases in `bench_wgrad_delay`
- any case where a user claims a new firmware/runtime reduced SDMA control-plane cost

## Example Outputs

### Example 1

User: "Should I use SDMA for 8-GPU reduce-scatter overlap?"

Answer shape:

- This is a raw `ReduceScatter` question, so `bench_comm_overlap` is the right benchmark.
- On 8 GPUs, `NCCL` wins every measured `RS` size and the gap widens at larger `N`.
- Recommendation: use `NCCL`, not `SDMA`.

### Example 2

User: "Why does SDMA look good in e2e fusion but bad in wgrad delay?"

Answer shape:

- These are different schedules.
- `bench_e2e_fusion` measures a fused single-layer backend, where `SDMA` reduces schedule-level backend cost better than `NCCL`.
- `bench_wgrad_delay` measures deferred-wgrad overlap, where sync/control overhead makes `NCCL` more robust.
- So the correct conclusion is not "SDMA is inconsistent"; it is "backend preference depends on the schedule."

### Example 3

User: "What should I try first for SDMA optimization?"

Answer shape:

- focus on large-message `AllGather`
- focus on fused backend paths, not deferred wgrad first
- avoid spending time on `8 GPU` `ReduceScatter` overlap microbenches until SDMA control-plane overhead changes materially
