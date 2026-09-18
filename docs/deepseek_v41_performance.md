# DeepSeek-V4.1 P09: packed cache and graph execution

P09 adds optional native cache storage, GPU W4A8 expert dispatch and PIECEWISE
CUDAGraph execution to the accepted P05 runtime. Model arithmetic, cache
representation and execution policy have separate owners. Quality is compared
with P05 commit `83c85207f71ff717378c2911851b20a84665fa3f`; this does not change
the separately documented P04 differences from the mathematical reference.

## Current V4 primitive reuse (2026-09-15)

Target and draft now directly use `atom.model_ops.layernorm.RMSNorm` for
attention/FFN inputs, Q/KV, index keys and compressor output. The separate
`deepseek_v41/normalization.py` implementation is removed. The model's ordinary
TP initialization also serves these layers; standalone tests supply the TP
fixture rather than adding a separate production normalization class.

Forward RoPE uses AITER's `rope_cached_positions_fwd_inplace`, as V4 does.
The V4.1 adapter owns its batch/position interface, trailing rotary lanes and
FP32 YaRN cache. It converts forward positions to the cached kernel's int64
ABI and preserves unit stride even for length-one views. Inverse RoPE continues
to use the original V4 kernel. The V4 fused Q/K normalization cannot be called
verbatim: its weightless per-head Q normalization is absent from V4.1.

The grouped output projection calls V4's AITER `batched_gemm_bf16` for 2..32
rows, emitting token-major output for the following `wo_b`. Single-row native
GEMM was already as fast in the isolated test and stays native. Larger batches
use V4's native einsum path. The earlier 128-row output-projection padding is
removed. Production delayed mHC keeps its existing AITER stages; the old FP32
coefficient projection helper is only used by reference diagnostics.

A gfx950 operator comparison against `c7a33d834` uses identical inputs, rows
1/6/24/128, normalization widths 512/1280/5120 and TP4 wo_a geometry
G=2, N=1024, K=4096. Timings are CUDA graph replays outside the profiler, in
before/after/after/before order. RoPE timings include an identical input clone
on both arms and use int64 positions; an int32 caller also pays a cast.

| Operation | Before | After | Observed ratio |
|---|---:|---:|---:|
| RMSNorm, tested widths/rows | 18.4–37.9 us | 1.9–2.6 us | 8.2–18.4x |
| Forward RoPE, tested rows | 10.9–12.9 us | 6.3–8.9 us | 1.4–1.7x |
| wo_a, 6 rows | 17.69 us | 7.94 us | 2.23x |
| wo_a, 24 rows | 17.78 us | 11.55 us | 1.54x |
| wo_a, 1 / 128 rows | 7.21 / 12.68 us | 7.20 / 12.76 us | unchanged path |

A separate operator trace, attributed via HIP launch correlation IDs, confirms
8→1 normalization kernels, 4→1 rotation kernels, and 2→1 grouped GEMM kernels
at 24 rows. These counts exclude input-clone memcpy events. This is not an
end-to-end speedup claim. There were small BF16 rounding differences: maximum
normalized RMS error was 1.60e-5 for normalization and 2.80e-5 for wo_a; forward
RoPE was identical on these inputs. Historical eager-path quality scores do
not transfer to the new intermediate normalization boundaries.

168 targeted tests pass on GPU 3, including FP64/operator reference checks,
quantization, chunked Full/Reuse/Reindex composition, speculative state,
ragged positions and graph replay with changing int32/int64 positions. The
reference composition test's wo_a initialization now follows the current FP8
allocation / BF16 post-load conversion. It still substitutes an independently
checked attention result, so it is not a full-model accuracy test.

Evidence is under
`/app/logs_claude/atom_dsv41_flash_impl_20260912/p10_dspark/`:
`benchmark_v4_reuse220.py`, `v4_reuse220/{result.json,primitives.trace.json.gz,trace_summary.json}`,
and `v4_reuse221_gpu3_tests.log`. The diagnostic runner accepts
`--torch-profiler-dir` and uses ModelRunner's existing profiler lifecycle.
Profiled measurements are kept separate from throughput comparisons.

A first TP2 non-speculative runtime comparison on GPUs 0/1 uses the same three
workloads, 32 finalized tokens/request, one warmup and one measured repeat.
Baseline/current throughput was 11.31/11.05, 10.96/10.17 and 18.63/16.89 tok/s.
This pair does **not** demonstrate an end-to-end speedup. It motivated widening
the graph boundaries: Q/KV projections and output LoRA were still outside the
captured stages. `trace221_current` and `trace222_baseline` contain separate
16-token traces; their instrumented times are not the throughput comparison.
`perf221_*` and `v4_reuse221_comparison.json` retain the unprofiled results.
These TP2 measurements are neither TP4 DSpark acceptance nor task accuracy.

## Decode graph boundary and shared metadata (2026-09-15)

The required decode execution boundary is at most two graph replays per step,
keyed by `running_bs` and `running_tokens`. Bucket-specific instances are allowed;
per-layer graph splits do not satisfy this requirement. The experiment adding
input/output projection graphs to every layer was withdrawn before commit and
archived as `graph227_withdrawn.patch`. The existing four-stage-per-layer executor
is still present and must be replaced; full decode capture is not complete.

Request metadata now uses the same persistent `CpuGpuBuffer` storage and token
layout helpers as V4: `build_batch_ids`, `prefill_positions`, and `pack_rows`.
V4 and V4.1 share state-slot buffer allocation and the existing
`_populate_state_slot_mappings` / `_stage` publisher. V4.1 consumes
`v4_meta_state_slot_out`; it does not introduce a separate state-slot field.
Pool-slot conversion remains geometry-owned: V4's unified plane reverses the
slot axis, while V4.1's entry arena uses scheduler slot IDs directly.

The actual request/token counts are `scheduled_bs` / `scheduled_tokens`, and
execution capacities are `running_bs` / `running_tokens`. Block-table row stride
and buffer addresses stay fixed across steps. Token padding follows V4's `-1`
request-ID sentinel. Existing eager consumers still use active views; changing
these metadata buffers alone does not make the per-request compressor/indexer
loop capturable. Those consumers and their speculative state writes remain the
next integration work, using V4's batched plans and indexing interfaces.

The focused metadata/cache/DSpark-interface regression passes 59 tests,
including replay of the existing V4 window-write kernel after request reorder,
slot relocation and position changes without recapture. It also checks V4's
original slot conversion and empty-batch padding. Evidence:
`p10_dspark/metadata229_tests.log`. The final `metadata229_target` TP2 runtime
completed three fixed-output workloads (16 output tokens/request) on physical
GPUs 3/0, HIP 0/1. This is a runtime smoke test, not a quality or throughput gate.
`metadata228_dspark` stopped at the existing TP4 admission gate before model
loading and is not a passing DSpark run.

## Cache format and attention boundary

| Region | Values | Scales | Bytes per row |
|---|---|---|---:|
| Main KV, all 512 dimensions | E2M1 FP4 | E4M3, group 16 | 288 |
| Index keys, 128 dimensions | E2M1 FP4 | E8M0, group 32 | 68 |
| SWA, all 512 dimensions | E4M3 FP8 | E8M0, group 32 | 528 |

Values and scales are stored as interleaved byte rows; the existing quantizers
produce both without quantizing a previously rounded BF16 value again. Owners
2, 8 and 14 use ratio 2, and owner 20 uses ratio 1: the global payload is
`(288 + 68) * (3/2 + 1) = 890` bytes per original token per replica.

| Production allocation, block size 16 | P05 BF16 | P09 packed |
|---|---:|---:|
| PAGE, including alignment | 51,200 B | 15,104 B |
| STATE per request, including tails/cursor | 5,256,192 B | 2,715,904 B |

Packed PAGE storage is 70.5% smaller and STATE storage is 48.3% smaller. These
percentages describe cache capacity, not total model memory. FP32 compressor
tails and committed Engram history retain their precision and checkpoint
semantics. Layout identity distinguishes packed and BF16 images.

The shared V4 `paged_prefill.py` and `paged_decode.py` are unchanged.
`attentions/deepseek_v41/packed_rows.py` owns row encoding/decoding, and
`packed_attention.py` adapts selected CSR rows to the V4 BF16 interface. Main
and SWA addresses are tagged byte offsets internal to this backend; the V4
kernel receives ordinary int32 row indices and BF16 values.

Prefill decodes at most 32 queries' selected history into bounded scratch and
retains the current chunk as BF16. Decode keeps the original batch size and
split-K selection, with at most `(index_topk + sliding_window) * batch` scratch
rows. At 512 dimensions and top-k 512/window 128, prefill scratch is at most
20 MiB; decode uses at most 640 KiB per query. Index scoring reads only its
requested tile or candidate positions. No path materializes the entire
historical main KV pool.

A direct mixed-format `PACKED_KV` loader was tested and removed from the shared
V4 kernels. Prefill must preserve the selected V4 implementation: OPUS and
Triton differ at BF16 rounding boundaries, and substituting Triton caused a
real-model logit mismatch. Local decoding preserves OPUS and its dispatch.
Direct decode did not show a consistent advantage over local decoding across
batch sizes. A warm synthetic single-GPU measurement, 16 local heads and 640
rows per query, gave:

| Batch | Original BF16 | Inline packed candidate | Gather + original BF16 |
|---:|---:|---:|---:|
| 1 | 7.66 us | 12.02 us | 14.02 us |
| 8 | 10.66 us | 21.80 us | 20.45 us |
| 64 | 19.76 us | 47.57 us | 42.04 us |

All outputs were bitwise equal. This microbenchmark has warm, bounded pools;
it is not an end-to-end speedup claim or a long-context bandwidth measurement.

## Shared MoE kernels and graph ownership

`models/deepseek_v41/moe.py` is V4's `MoE`, subclassed only to flatten the
offline caller's batch dimension and to declare `bias_vl`. There is no second
expert backend and no HF option selecting one: the two models' routed experts
are the same layer. V4/FusedMoE owns the activation format, routing-weight
placement, GEMM schedule and expert-parallel exchange; these are not redefined
by V4.1. Actual gfx950 dispatch includes FP4-activation expert kernels.

The earlier P09 experiments -- a custom routed GEMM, then an adapter over
AITER's `fused_routing_from_topk` / `moe_gemm_a8w4` behind an
`expert_backend` switch -- have both been removed, along with the eager expert
loop they were alternatives to. Their measurements survive only as historical
diagnostics; the numbers below that compare expert backends describe code that
no longer exists.

A decode step is **two replays**: the draft's, and one whole target forward.
`cudagraph_mode=FULL` uses the runner's own capture -- `capture_cudagraph`
records `model(input_ids, positions)` into `self.graphs[(bs, max_q_len)]` and
`run_model` replays exactly one of them -- so there is no V4.1-specific graph
machinery and no per-stage entries. The per-stage `DenseGraphExecutor` that
preceded it keyed its entries on a bound method, giving every layer its own:
40 layers x 4 stages x 3 buckets, which trace230 measured as 120 launches per
step at 12.3% kernel coverage.

Measured on the DSpark performance workload at TP4, FP8 index plane, 12 decode
steps: **23 `hipGraphLaunch` in total, one per `decode[...]` scope and one per
`propose_dspark[...]`**, with 99.6% of the decode scope's kernel time inside
its graph and 72.9% across the whole run (the rest is prefill, which is eager
by design).

What that cost to make possible: the decode forward has to be pure tensor work
at a fixed width. The cursor write moved out of the model into
`prepare_model_inputs` (a replay runs no Python); the compression plan is cut
to a content-independent `running_bs * per-seq bound` with a sentinel tail; and
the step's tensors span the forward's own width rather than the scheduled
batch. A sentinel plan row keeps its `-1`, which makes `page * per_page +
offset` negative -- the row index V4's writers already skip, and what
`indexer_k_quant_and_cache` bails on. The one writer that cannot skip on its
own is torch advanced indexing, where a negative index is legal and lands on
somebody's live row, so the packed-main scatter filters by the plan's own
`batch_id >= 0` first.

`cudagraph_mode=PIECEWISE` remains available and records the compiled dense
pieces with attention eager between them. Startup capture binds the serving
allocation and uses STATE slots `[0, bs)`, and names **block 0 for every entry
of every synthetic request** -- V4's capture block table exactly. Naming a run
of distinct pages instead makes capture write that many, and those are the
pages the block pool hands out first: a request that later gets one reads
capture's rows wherever its own prefill has not reached yet, which surfaced as
a fault in the prefill scorer several hundred tokens later.
`validate_runtime --graph` asserts every page outside that bound is untouched.
Host Engram lookup stays outside either mode. `torch.compile` remains
unsupported.

## Accepted MoE numerical change

On 2026-09-13 the user accepted the measured A8W4 precision change and directed
integration to continue. This is explicit acceptance of a tradeoff; it does
not turn a failed independent NLL threshold into a pass.

The paired TP4 run covers 37 cases, 148 prefill/decode records, 7,258 labels
and 7,295 output positions, including the 2,049-token case. Router, shared
experts, quantization boundaries and reductions are held fixed. Every baseline
record exactly reproduces the previously frozen P05 NLL.

| Metric | Result |
|---|---:|
| Accepted P05 mean NLL | 0.6042607703 |
| AITER A8W4 candidate mean NLL | 0.6098483537 |
| Change versus P05 | +0.0055875834 |
| Mathematical reference mean NLL | 0.5959472320 |
| A8W4 difference from mathematical reference | +0.0139011217 |
| Original independent allowance | +0.01 (not met) |
| Top-1 agreement with P05 | 96.600411% |

The AITER backend also completed a matched GSM8K generation regression:
5-shot, greedy, 256 maximum generated tokens, the same 16 documents and seeds
as the accepted eager run. Strict and flexible exact match are both 12/16
(75%), equal to eager's aggregate score, with one gain and one loss. The paired
report verifies document, prompt and target hashes. This is a 16-item regression,
not the full 1,319-question GSM8K test set. The reproducible wrapper, launcher,
responses and paired report are retained in the phase evidence archive.

ARC, code and Chinese task accuracy have not been revalidated for this kernel
substitution. Their P04 results remain historical results for eager experts.
Runtime parity separately holds the selected MoE backend fixed and checks that
paging and graph execution add no error relative to uncaptured execution with
a private BF16 cache.

The final production adapter reproduces every field of all 148 numerical
records from the accepted candidate. The final TP4 runtime test has zero
unequal logits at 670 positions across 17 chunks. Forty scheduler batches
cover reorder, simultaneous prefix restores, missing-state replay,
preempt/resume, cancellation and slot reuse. It records 480 tensor-stage
graphs and 3,680 replays; startup capture leaves live cache bytes unchanged.
These are comparisons against the same AITER backend, not equality with P05.

The targeted regression passes 953 tests. Forty-seven parameter combinations
in `test_sub_pool_spec.py` are intentionally skipped because each budget is
covered by either its success test or its failure test. No V4.1 model/reference
test is skipped. The expert-adapter tests were removed with the adapter; the
routed experts are now covered by V4's own `FusedMoE` tests.

The dense native FP8 candidate remains disabled: it changed BF16 outputs and
had no consistent measured speedup (about 0.97–1.04x). Dense projections still
use group32 BF16 dot products and FP64 accumulation; the new MoE GEMMs use
AITER's native A8W4 arithmetic.

## Runtime benchmark

`tests/attentions/deepseek_v41/benchmark_runtime.py` runs ModelRunner and Scheduler,
including scheduling, host Engram preparation, model execution, sampling and
postprocessing. It excludes model loading, graph capture, HTTP serving and
arrival queues. Each case uses a fresh scheduler (no prefix hits), one warmup,
three measured repetitions and greedy generation of 32 tokens. Reports retain
all output tokens and their SHA256 hashes, and check repeat/rank agreement.
Per-case medians use the slowest rank's TTFT/TPOT and allocation measurements.

Run with the same pinned environment as the runtime validation:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.benchmark_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --label packed-aiter-graph \
  --cache-dtype fp4 --graph --expert-backend aiter \
  --cases 1:128,4:128,8:128,1:1024 --repeats 3 --output-tokens 32 \
  --output /tmp/v41-benchmark.json
```

For an independent P05 baseline, invoke this same script by absolute path with
`PYTHONPATH` pointing to an isolated checkout of the accepted P05 commit, using
`--cache-dtype bf16` and omitting the graph/grouped flags.

Measured TP4 medians (milliseconds), using packed cache and PIECEWISE graphs.
Taken on the removed AITER expert adapter, so they date the P09 comparison, not
the current `FusedMoE` path:

| Batch / prompt tokens | P05 TTFT | P09 TTFT | P05 TPOT | P09 TPOT | Decode speed ratio |
|---|---:|---:|---:|---:|---:|
| 1 / 128 | 1365.55 | 499.92 | 247.44 | 102.72 | 2.41x |
| 4 / 128 | 1835.09 | 622.30 | 302.77 | 127.47 | 2.38x |
| 8 / 128 | 2102.02 | 848.10 | 346.46 | 164.11 | 2.11x |
| 32 / 128 | 4914.40 | 2019.90 | 636.52 | 406.69 | 1.57x |
| 1 / 1024 | 6301.07 | 2136.67 | 250.20 | 101.50 | 2.46x |

TTFT improves by 2.43–2.95x and decode by 1.57–2.46x in these cases. Each
candidate's three repeats and all ranks agree on generated tokens. Candidate
generations differ from P05 in every cohort; both the numerical difference
and all output token sequences are retained in the evidence. This is a
fixed-prompt, fixed-output-length runtime comparison, not a same-token-path
operator comparison or a task-quality score.

Peak allocated memory is 72.39–72.83 GiB per rank in this bounded setup,
dominated by weights. The PAGE/STATE byte reductions above describe cache
capacity rather than a corresponding total-memory reduction. These results
cover TP4 text inputs through 1,024 prompt tokens and B32, excluding HTTP,
arrival queues, loading and capture; they make no long-context, vision or
larger-topology throughput claim.

Earlier measurements of packed eager, dense-only graphs and the removed
custom routed-GEMM prototype are retained in the archive as historical
experiments. They are not the performance of the final AITER backend.

Evidence, commands, numerical comparisons and the phase commit patch are
archived under `/app/logs_claude/atom_dsv41_flash_impl_20260912/p09_runtime/`
in `ljin_dev`.
