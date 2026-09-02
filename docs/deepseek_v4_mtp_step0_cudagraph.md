# DeepSeek-V4 MTP step-0 CUDA graph optimization

> Status: experimental, validated on DeepSeek-V4-Pro with MI355X TP8.
>
> This note records the motivation, implementation boundary, trace evidence,
> and current safety gates for the serial-MTP step-0 CUDA graph fast path.

## Summary

Before this change, `EagleProposer` captured serial-MTP steps `1..k-1`, but
ran step 0 eagerly. For DeepSeek-V4-Pro at concurrency 1 and `mtp_k=3`, step 0
contains three small `[4, 7168]` TP custom AllReduces plus the collectives in
`compute_draft_ids`. The arithmetic in the three AllReduces is short, but the
eager launch sequence lets TP ranks reach each rendezvous at different times.

The new fast path captures the DeepSeek-V4 step-0 backbone and draft-id
epilogue as one `DraftGraph`. Replay is confirmed on all eight ranks, while
prefill and unsupported layouts continue to use the eager fallback.

A corrected controlled comparison, with NUMA binding enabled on both arms and
without RPC/forward debug markers, measured rank-0 step-0 P50 at `461.9 us`
eager and `479.6 us` graphed. The graph was `17.7 us` (`3.8%`) slower in that
sample, and the accompanying end-to-end comparison showed no attributable net
gain. The implementation is therefore experimental graph-path infrastructure,
not a demonstrated performance improvement.

## Workload and measurement

The optimized and baseline traces used the following decode configuration:

| Item | Value |
|---|---|
| Model | DeepSeek-V4-Pro |
| Accelerator | 8 x AMD Instinct MI355X |
| Parallelism | TP8, DP=1, DCP=1, PCP=1 |
| Concurrency | 1 |
| Speculative method | serial MTP |
| Speculative tokens | 3 |
| Full verify width | 4 tokens per sequence |
| CUDA graph mode | `FULL` |
| Captured batch sizes | 1 and 2 |
| Draft graph switch | `ATOM_DRAFT_CUDAGRAPH=1` |
| NUMA placement | `ATOM_NUMA_BIND=1` |

The baseline was ATOM commit `4c2870b7`, where steps 1 and 2 already replayed
complete backbone-plus-epilogue graphs. Only step 0 remained eager.

The controlled trace comparison uses the P50 duration of the rank-0
`propose_eagle[0/3 tok=4 ...]` GPU annotation. Both arms enable NUMA binding,
and neither contains the extra RPC/forward debug markers that perturb CPU
submission and overlap.

## Trace result

| Metric | Baseline | Step-0 graph | Change |
|---|---:|---:|---:|
| Rank-0 step-0 GPU annotation P50 | `461.9 us` | `479.6 us` | `+17.7 us` (`+3.8%`) |

An earlier diagnostic comparison reported `3.130 ms` eager versus `0.479 ms`
graphed and much smaller cross-rank AllReduce envelopes. That eager trace used
additional RPC/forward instrumentation which slowed CPU execution and changed
overlap, so those numbers are superseded by the controlled result above and
must not be interpreted as graph speedup.

Every rank in the optimized trace reports:

```text
propose_eagle[0/3 tok=4 bs=1/1 graph]
```

The prefill in the same trace reports an eager label:

```text
propose_eagle[0/3 tok=92 bs=1]
```

This confirms both sides of the dispatch decision: the rectangular decode
path replays the recording, while the dynamic prefill path falls back to the
existing eager implementation.

## Why the graph path is still useful

Step 0 is the only serial-MTP pass that previously had no graph representation.
Capturing it establishes fixed staging, replay, and fallback behavior for the
full verification rectangle, and makes later launch-overhead experiments
possible without changing model semantics. The corrected measurement shows
that capture alone does not improve this workload; any follow-up optimization
must reduce graph-external staging or improve end-to-end overlap, and must be
validated with an uninstrumented end-to-end comparison.

## Implementation

### Per-pass token width

`DraftGraph` capture sizes are expressed in sequences, while forward context
and MoE communication sizes are expressed in token rows. A serial-MTP drafter
now declares the conversion on each pass through `tokens_per_seq`:

- DeepSeek-V4 step 0: `mtp_k + 1` rows per sequence;
- serial-MTP steps 1 and later: 1 row per sequence;
- DSpark block pass: its existing block draft width.

This avoids applying one drafter-wide width to passes with different shapes.

### Fixed step-0 inputs

The step-0 graph owns fixed-address staging buffers with these logical shapes:

| Input | Shape per captured batch | Dtype |
|---|---|---|
| Token IDs | `[batch, mtp_k + 1]` | `int32` |
| Positions | `[batch, mtp_k + 1]` | `int64` |
| mHC hidden states | `[batch, mtp_k + 1, hc_mult, hidden_size]` | model dtype |
| Last-token indices | `[batch]` | `int32` |

The forward flattens the first two axes and invokes the existing V4 MTP model.
The captured epilogue performs the dynamic `index_select` with staged
last-token indices and then calls `compute_draft_ids`.

Only the selected `[batch, hc_mult, hidden_size]` rows and draft IDs leave the
graph. The full `batch * (mtp_k + 1)` hidden tensor stays in graph-owned
storage, avoiding an unnecessary full-output copy after replay.

### Attention metadata

DeepSeek-V4's target CUDA-graph capture builder already synthesizes the exact
full-width decode rectangle needed by step 0. Its attention metadata points to
persistent `forward_vars` buffers, and runtime decode refills the same
addresses. Step-0 warmup therefore reuses this target-built decode context and
sets the pass token count to `batch * (mtp_k + 1)`.

The post-step-0 rewrite to one-row-per-sequence metadata remains outside this
graph. Steps 1 and later continue to use the existing mid-step graph.

## Runtime safety gates

The step-0 graph is declared only when all of the following are true:

- the draft model is `DeepseekV4MTP`;
- DP, DCP, and PCP sizes are all 1;
- expert parallelism and TBO are disabled;
- MRoPE is disabled;
- MTP index sharing is disabled;
- the V4 configuration exposes `hc_mult`.

Even when a recording exists, a serving step replays it only when all runtime
conditions match:

- the step is decode, not prefill;
- `scheduled_bs == running_bs`;
- the token stream is exactly `batch * (mtp_k + 1)` rows;
- `attn_metadata.max_seqlen_q == mtp_k + 1`;
- token IDs, positions, hidden states, and last-token indices have the captured
  dtype, rank, and contiguous layout;
- a recording exists for the exact batch size.

Any mismatch takes the pre-existing eager path. There is no runtime capture.

## Validation

The implementation passed:

- Python bytecode compilation for all modified modules;
- `tests/test_draft_graph.py` and `tests/test_dspark.py`: 92 tests passed;
- TP8 startup capture for batch sizes 1 and 2;
- 11/11 warmup requests and 8/8 profiled requests with zero request errors;
- 475 traced decode cycles with the step-0 graph label on all eight ranks;
- server-log checks for GPU faults, illegal accesses, assertions, and
  tracebacks.

The short integration run used forced speculative acceptance length `2.49`,
so it validates execution, shapes, token counts, and collective ordering. It
does not replace a non-forced semantic-output comparison.

The extra startup work was small in this configuration: draft-pass warmup
increased from `0.39 s` to `0.50 s`, and the logged CUDA graph pool allocation
did not show a measurable increase at GiB precision.

## Reproduction outline

Start ATOM from this branch with the fixed TP8 configuration:

```bash
ATOM_DRAFT_CUDAGRAPH=1 \
ATOM_NUMA_BIND=1 \
python3 -u -m atom.entrypoints.openai_server \
  --model /models/DeepSeek-V4-Pro \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --cudagraph-mode FULL \
  --method mtp \
  --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --max-num-seqs 2 \
  --torch-profiler-dir /profiles
```

Capture a steady-state window through the profiling endpoints:

```bash
curl -X POST http://127.0.0.1:8000/start_profile
# Run steady decode traffic for approximately 20 seconds.
curl -X POST http://127.0.0.1:8000/stop_profile
```

Check every rank trace for the exact step-0 graph label and align collective
kernels by launch order. For captured custom AllReduce kernels, Kineto omits
the launch grid, while the baseline eager step-0 calls expose grid x = 56.

## Remaining work

1. Run a paired 900-second end-to-end A/B with the same request trajectory and
   compare ITL, output throughput, and acceptance; the current shorter
   controlled comparison shows no attributable net gain.
2. Run without forced acceptance and compare generated token IDs against the
   eager path.
3. Add and validate DP, DCP, PCP, EP, and TBO support before relaxing the
   declaration gates.
4. Continue with the independent target-side bottleneck: 123 TP AllReduces per
   target graph. This change intentionally does not alter that protocol.
