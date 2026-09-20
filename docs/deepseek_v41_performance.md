# DeepSeek-V4.1 cache format and graph execution

Native cache storage, GPU W4A8 expert dispatch and captured CUDAGraph execution,
on top of the BF16/eager runtime. Model arithmetic, cache representation and
execution policy have separate owners; quality is compared against the BF16
runtime at commit `83c85207f71ff717378c2911851b20a84665fa3f`, which does not
change the separately documented differences from the mathematical reference.

## Shared V4 primitives

Target and draft use `atom.model_ops.layernorm.RMSNorm` directly for attention
and FFN inputs, Q/KV, index keys and compressor output; V4.1 has no
normalization class of its own, and the model's ordinary TP initialization
serves these layers.

Forward RoPE uses AITER's `rope_cached_positions_fwd_inplace`, as V4 does. The
V4.1 adapter owns its batch/position interface, trailing rotary lanes and FP32
YaRN cache; it converts forward positions to the cached kernel's int64 ABI and
preserves unit stride even for length-one views. Inverse RoPE keeps the original
V4 kernel. V4's fused Q/K normalization cannot be called verbatim, because its
weightless per-head Q normalization is absent from V4.1.

The grouped output projection calls V4's AITER `batched_gemm_bf16` for 2..32
rows, emitting token-major output for the following `wo_b`. A single row was
already as fast through the native GEMM and stays there; larger batches use V4's
native einsum path. Production delayed mHC keeps its existing AITER stages, and
the FP32 coefficient projection helper is reference-diagnostic only.

Operator timings against the pre-reuse implementation, on gfx950 with identical
inputs: rows 1/6/24/128, normalization widths 512/1280/5120, TP4 `wo_a` geometry
G=2, N=1024, K=4096. Timings are CUDA graph replays taken outside the profiler
in before/after/after/before order. RoPE timings include an identical input
clone on both arms and use int64 positions; an int32 caller also pays a cast.

| Operation | Before | After | Observed ratio |
|---|---:|---:|---:|
| RMSNorm, tested widths/rows | 18.4–37.9 us | 1.9–2.6 us | 8.2–18.4x |
| Forward RoPE, tested rows | 10.9–12.9 us | 6.3–8.9 us | 1.4–1.7x |
| wo_a, 6 rows | 17.69 us | 7.94 us | 2.23x |
| wo_a, 24 rows | 17.78 us | 11.55 us | 1.54x |
| wo_a, 1 / 128 rows | 7.21 / 12.68 us | 7.20 / 12.76 us | unchanged path |

An operator trace attributed through HIP launch correlation IDs confirms 8→1
normalization kernels, 4→1 rotation kernels and 2→1 grouped GEMM kernels at 24
rows, excluding input-clone memcpy events. None of this is an end-to-end speedup
claim. BF16 rounding does move: maximum normalized RMS error is 1.60e-5 for
normalization and 2.80e-5 for `wo_a`, and forward RoPE was identical on these
inputs. Quality scores taken before the reuse do not transfer across the new
intermediate normalization boundaries.

## Request metadata

Request metadata uses the same persistent `CpuGpuBuffer` storage and token
layout helpers as V4 — `build_batch_ids`, `prefill_positions` and `pack_rows` —
and the two models share state-slot buffer allocation and the existing
`_populate_state_slot_mappings` / `_stage` publisher. V4.1 consumes
`v4_meta_state_slot_out` rather than introducing a state-slot field of its own.

Pool-slot conversion stays geometry-owned: V4's unified plane reverses the slot
axis, while V4.1's entry arena uses scheduler slot IDs directly.

The actual request and token counts are `scheduled_bs` / `scheduled_tokens`;
execution capacities are `running_bs` / `running_tokens`. Block-table row stride
and buffer addresses stay fixed across steps, and token padding follows V4's
`-1` request-ID sentinel.

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

| Production allocation, block size 16 | BF16 | Packed |
|---|---:|---:|
| PAGE, including alignment | 51,200 B | 15,104 B |
| STATE per request, including tails/cursor | 5,256,192 B | 2,715,904 B |

Packed PAGE storage is 70.5% smaller and STATE storage is 48.3% smaller. These
percentages describe cache capacity, not total model memory. FP32 compressor
tails and committed Engram history retain their precision and checkpoint
semantics, and layout identity distinguishes packed from BF16 images.

The shared V4 `paged_prefill.py` and `paged_decode.py` are unchanged.
`attentions/deepseek_v41/packed_rows.py` owns row encoding and decoding, and
`packed_attention.py` adapts selected CSR rows to the V4 BF16 interface. Main
and SWA addresses are tagged byte offsets internal to this backend; the V4
kernel receives ordinary int32 row indices and BF16 values.

Prefill decodes at most 32 queries' selected history into bounded scratch and
retains the current chunk as BF16. Decode keeps the original batch size and
split-K selection, with at most `(index_topk + sliding_window) * batch` scratch
rows. At 512 dimensions and top-k 512 / window 128, prefill scratch is at most
20 MiB and decode uses at most 640 KiB per query. Index scoring reads only its
requested tile or candidate positions. No path materializes the entire
historical main KV pool.

A direct mixed-format `PACKED_KV` loader was tested and is not used: prefill
must preserve the selected V4 implementation, because OPUS and Triton differ at
BF16 rounding boundaries and substituting Triton caused a real-model logit
mismatch. Local decoding preserves OPUS and its dispatch. Direct decode also
showed no consistent advantage — a warm synthetic single-GPU measurement, 16
local heads and 640 rows per query:

| Batch | Original BF16 | Inline packed candidate | Gather + original BF16 |
|---:|---:|---:|---:|
| 1 | 7.66 us | 12.02 us | 14.02 us |
| 8 | 10.66 us | 21.80 us | 20.45 us |
| 64 | 19.76 us | 47.57 us | 42.04 us |

All outputs were bitwise equal. This microbenchmark has warm, bounded pools; it
is not an end-to-end speedup claim or a long-context bandwidth measurement.

## Shared MoE kernels and graph ownership

`models/deepseek_v41/moe.py` is V4's `MoE`, subclassed only to flatten the
offline caller's batch dimension and to declare `bias_vl`. There is no second
expert backend and no HF option selecting one: the two models' routed experts
are the same layer. V4/FusedMoE owns the activation format, routing-weight
placement, GEMM schedule and expert-parallel exchange; V4.1 does not redefine
them. Actual gfx950 dispatch includes FP4-activation expert kernels.

A decode step is **two replays**: the draft's, and one whole target forward.
`cudagraph_mode=FULL` uses the runner's own capture — `capture_cudagraph`
records `model(input_ids, positions)` into `self.graphs[(bs, max_q_len)]` and
`run_model` replays exactly one of them — so there is no V4.1-specific graph
machinery and no per-stage entries. The per-stage `DenseGraphExecutor` that
preceded it keyed its entries on a bound method, giving every layer its own: 40
layers x 4 stages x 3 buckets, measured at 120 launches per step and 12.3%
kernel coverage.

Measured on the DSpark performance workload at TP4, FP8 index plane, 12 decode
steps: **23 `hipGraphLaunch` in total, one per `decode[...]` scope and one per
`propose_dspark[...]`**, with 99.6% of the decode scope's kernel time inside its
graph and 72.9% across the whole run — the rest is prefill, which is eager by
design.

What that cost to make possible: the decode forward has to be pure tensor work
at a fixed width. The cursor write moved out of the model into
`prepare_model_inputs` (a replay runs no Python); the compression plan is cut to
a content-independent `running_bs * per-seq bound` with a sentinel tail; and the
step's tensors span the forward's own width rather than the scheduled batch. A
sentinel plan row keeps its `-1`, which makes `page * per_page + offset`
negative — the row index V4's writers already skip, and what
`indexer_k_quant_and_cache` bails on. The one writer that cannot skip on its own
is torch advanced indexing, where a negative index is legal and lands on
somebody's live row, so the packed-main scatter filters by the plan's own
`batch_id >= 0` first.

`cudagraph_mode=PIECEWISE` remains available and records the compiled dense
pieces with attention eager between them. Startup capture binds the serving
allocation, uses STATE slots `[0, bs)`, and names **block 0 for every entry of
every synthetic request** — V4's capture block table exactly. Naming a run of
distinct pages instead makes capture write that many, and those are the pages
the block pool hands out first: a request that later gets one reads capture's
rows wherever its own prefill has not reached yet, which surfaced as a fault in
the prefill scorer several hundred tokens later. `validate_runtime --graph`
asserts every page outside that bound is untouched. Host Engram lookup stays
outside either mode, and `torch.compile` remains unsupported.

## Accepted MoE numerical change

The A8W4 precision change below was measured, presented and accepted as a
tradeoff. It does not turn a failed independent NLL threshold into a pass.

The paired TP4 run covers 37 cases, 148 prefill/decode records, 7,258 labels and
7,295 output positions, including the 2,049-token case. Router, shared experts,
quantization boundaries and reductions are held fixed, and every baseline record
exactly reproduces the frozen BF16-runtime NLL.

| Metric | Result |
|---|---:|
| Accepted BF16-runtime mean NLL | 0.6042607703 |
| AITER A8W4 candidate mean NLL | 0.6098483537 |
| Change versus BF16 runtime | +0.0055875834 |
| Mathematical reference mean NLL | 0.5959472320 |
| A8W4 difference from mathematical reference | +0.0139011217 |
| Original independent allowance | +0.01 (not met) |
| Top-1 agreement with the BF16 runtime | 96.600411% |

The AITER backend also completed a matched GSM8K generation regression: 5-shot,
greedy, 256 maximum generated tokens, the same 16 documents and seeds as the
accepted eager run. Strict and flexible exact match are both 12/16 (75%), equal
to eager's aggregate score, with one gain and one loss; the paired report
verifies document, prompt and target hashes. This is a 16-item regression, not
the full 1,319-question GSM8K test set.

ARC, code and Chinese task accuracy have not been revalidated for this kernel
substitution; their earlier results are for eager experts. Runtime parity
separately holds the selected MoE backend fixed and checks that paging and graph
execution add no error relative to uncaptured execution with a private BF16
cache.

The production adapter reproduces every field of all 148 numerical records from
the accepted candidate. The TP4 runtime test has zero unequal logits at 670
positions across 17 chunks; forty scheduler batches cover reorder, simultaneous
prefix restores, missing-state replay, preempt/resume, cancellation and slot
reuse, recording 480 tensor-stage graphs and 3,680 replays, with startup capture
leaving live cache bytes unchanged. These are comparisons against the same AITER
backend, not equality with the BF16 runtime.

The dense native FP8 candidate remains disabled: it changed BF16 outputs and had
no consistent measured speedup (about 0.97–1.04x). Dense projections still use
group32 BF16 dot products and FP64 accumulation; the MoE GEMMs use AITER's
native A8W4 arithmetic.

## Measuring a runtime change

Peak allocated memory is 72.39–72.83 GiB per rank at TP4 through 1,024 prompt
tokens and batch 32, dominated by weights. The PAGE/STATE byte reductions above
describe cache capacity rather than a corresponding total-memory reduction.

For an end-to-end number, serve the model and use the standard benchmark in
[the serving guide](serving_benchmarking_guide.md); the launch recipe is in
[recipes/DeepSeek-V4.1-Flash.md](../recipes/DeepSeek-V4.1-Flash.md). Compare
arms by alternating them on one host rather than running one arm to completion
and then the other — this model's run-to-run spread is wide enough to invert a
sequential comparison.
