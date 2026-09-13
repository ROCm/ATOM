# DeepSeek-V4.1 P09: packed cache and graph execution

P09 adds optional native cache storage, GPU W4A8 expert dispatch and PIECEWISE
CUDAGraph execution to the accepted P05 runtime. Model arithmetic, cache
representation and execution policy have separate owners. Quality is compared
with P05 commit `83c85207f71ff717378c2911851b20a84665fa3f`; this does not change
the separately documented P04 differences from the mathematical reference.

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

`model_ops/deepseek_v41/moe_aiter.py` adapts the existing AITER
`fused_routing_from_topk` and `moe_gemm_a8w4` entry points. P09 adds no MoE device
kernel. At loading time it assembles native FP4/E8M0 arenas and rebinds the
original expert parameters to views of the same storage. Weights are not
permanently duplicated or expanded to BF16. Remote experts sort into a final
sentinel bin that is excluded from the local GEMM schedule. Up/gate share one
GEMM, followed by one down GEMM; up to 512 tokens are processed per chunk to
bound intermediates and the existing sorter's 4,096-route limit.

The adapter keeps the V4.1 boundaries: BF16 gate/up results, FP32 asymmetric
clamp and weighted SwiGLU, BF16 rounding before group32 A8 quantization, and
ascending expert-ID accumulation of BF16 down results into FP32. Router math,
shared expert TP partials and RCCL remain in their existing model owners.

The HF option `expert_backend="aiter"` selects these kernels for both prefill
and decode. `expert_backend="eager"` retains the accepted P05 arithmetic and
remains the default. This is independent of packed cache storage. V4's default
FP4-activation path and its gfx1250 preshuffled decode wrapper are not suitable
configurations for this checkpoint on MI355X; the existing raw-layout A8W4
GEMMs run on gfx950 without modifying AITER or the V4 model.

The earlier P09 custom routed GEMM has been removed. Its exact-arithmetic
prototype and measurements are retained as historical diagnostics only.
The first EP-sort probe exposed a compile error in the pinned AITER EP wrapper;
production uses the existing top-k sorter instead, with per-call metadata and
no shared persistent routing scratch.

`models/deepseek_v41/execution.py` owns stable inputs, outputs and a distinct
CUDA graph allocation pool per block stage and token bucket. Startup capture
uses private dummy PAGE/STATE storage. Live request state is never captured or
modified by warmup. Replay copies current inputs, clears padding and returns
only live rows; an uncaptured bucket falls back to normal execution.

Captured stages are attention preparation, FFN preparation and residual
completion. With AITER experts, decode FFN and its collectives are also
captured. Attention/cache/index/compressor work, request positions, Engram CPU
lookup and committed history stay outside these graphs. FULL graphs and
`torch.compile` remain unsupported.

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
test is skipped. AITER expert tests include empty/all-remote routes, native
weight view preservation, 512-token chunk boundaries and refreshed routes
under graph replay.

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

Measured TP4 medians (milliseconds), using packed cache, AITER experts and
PIECEWISE graphs:

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
