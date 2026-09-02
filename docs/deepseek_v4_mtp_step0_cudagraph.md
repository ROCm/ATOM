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
epilogue as one `DraftGraph`. On the measured TP8 workload, the median step-0
GPU annotation fell from `3.130 ms` to `0.479 ms`. The per-step sum of the
three AllReduce cross-rank envelopes fell from `2.337 ms` to `48.9 us`.

This result is an operator-trace result, not a 900-second end-to-end benchmark
claim. The implementation is intentionally restricted to the configuration
that was validated.

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

The trace comparison uses two measurements:

1. **Step-0 GPU annotation:** the P50 duration of
   `propose_eagle[0/3 tok=4 ...]` on each rank, followed by the median of the
   eight per-rank P50 values.
2. **Collective envelope:** for each corresponding AllReduce call, align the
   eight ranks by launch order and measure from the earliest rank's kernel
   entry to the latest rank's kernel completion. Sum the three values for each
   step, then report P50 across steps.

The baseline trace contains 390 complete decode cycles. The optimized trace
contains 475 complete decode cycles.

## Trace result

| Metric | Baseline | Step-0 graph | Change |
|---|---:|---:|---:|
| Step-0 GPU annotation P50 | `3.130 ms` | `0.479 ms` | `-2.651 ms` (`-84.7%`) |
| Three-AR envelope sum P50 | `2.337 ms` | `48.9 us` | `-97.9%` |
| Three-AR entry-skew sum P50 | `2.314 ms` | `28.1 us` | `-98.8%` |
| All captured draft-AR envelope sum P50 | 6 calls: `0.572 ms` | 9 calls: `0.241 ms` | More calls, lower total wait |

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

## Why the eager path was expensive

The three baseline `[4, 7168]` AllReduce calls had a combined minimum-rank
kernel duration of only about `20.8 us` per step. Their combined cross-rank
envelope was `2.337 ms`. Most of the observed duration was therefore not the
reduction arithmetic; it was time spent inside collective synchronization
waiting for another rank to arrive.

Capturing the backbone and epilogue gives all ranks one replay launch for the
same fixed operation sequence. The three optimized AllReduces have only
`28.1 us` of combined entry skew at P50, which is why changing the graph
boundary is much more effective here than tuning the 56-CTA reduction kernel.

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
   compare ITL, output throughput, and acceptance.
2. Run without forced acceptance and compare generated token IDs against the
   eager path.
3. Add and validate DP, DCP, PCP, EP, and TBO support before relaxing the
   declaration gates.
4. Continue with the independent target-side bottleneck: 123 TP AllReduces per
   target graph. This change intentionally does not alter that protocol.
