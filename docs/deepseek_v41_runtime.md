# DeepSeek-V4.1 native paged runtime

Native ATOM text execution runs through ModelRunner and Scheduler with chunked
prefill, ragged batches, continuous decode and complete request-state recovery.
The existing V4 BF16 sparse attention and inverse RoPE kernels are unchanged.
Arithmetic is validated against the published model and generation quality
through end-to-end `lm_eval` evaluation.

## Ownership and execution boundaries

- `models/deepseek_v41/model.py` and `attention.py` own model arithmetic.
  `runtime.py` adapts its input/output contract to ModelRunner. Q/KV and output
  projections run over the flat token batch, and compression and index selection
  take every boundary in that batch in one call rather than looping per request.
- `model_ops/attentions/deepseek_v41/` owns metadata, addresses, PAGE/STATE views
  and checkpoint copies. Geometry is declared separately in
  `pool_layout/v41_pool_geometry.py` without model or scheduler imports.
- `model_loader/deepseek_v41.py` owns native checkpoint loading and mapped-table
  lifetime. Native post-load processing runs exactly once, preserving W4A8/QAT.
- `model_ops/engram/` prepares Engram rows after final GPU token IDs and restored
  state are available. There is no separate committed history map. Its host half
  (`mapping`, `tables`, `host`) imports without Triton; `device/` does not.
- Scheduler consumes the existing generic `StateTransfer.copy` capability.
  The only scheduling change fixes cancellation of requests with no sampled
  output, including a middle prefill chunk and the first deferred step.

Only Full owners 2, 8, 14 and 20 allocate global main/index storage. Reuse and
Reindex layers read those owner regions. Every request has all 40 SWA rings,
three ratio-2 FP32 compressor tails, the latest three compressed Engram IDs and
a committed position in its STATE entry. Prefill retains the old rings until
all queries have consumed their causal prefixes, including chunks wider than
the ring. Index reads gather a tile or candidate positions, never a full copy
of the historical main KV.

Allocation and checkpoints use the same `V41PoolGeometry` declarations.
Positions and Engram history advance after the model forward; checkpoints
carry every state field and padding byte. Images are versioned by geometry,
including the index plane format. See
[cache format and graph execution](deepseek_v41_performance.md#cache-format-and-attention-boundary)
for the storage formats.

An exact prefix hit restores the entire state before preparing Engram inputs.
Without a matching image, the generic scheduler replays from a recoverable
boundary. Checkpoint relocation/fork and restored tentative suffixes cannot
retain stale compressor tails, window rows or Engram history.

## Scheduler acceptance

`tests/attentions/deepseek_v41/validate_runtime.py` drives the real TP4 engine
through chunked and reordered batches, prefix forks, replay from a missing
image, preemption and cancellation, and asserts each finishes with the tokens
an uninterrupted scheduled run produced. It carries no separate oracle.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
torchrun --standalone --nproc_per_node=4 \
  -m tests.attentions.deepseek_v41.validate_runtime \
  --model /mnt/DeepSeek-V4.1-Flash --output /tmp/v41-runtime.json
```

## Current execution scope

The architecture is `DeepseekV41ForCausalLM` and the cache block size must be
even. Routed experts take either arrangement: `enable_expert_parallel=True`
gives each rank whole experts, and leaving it off shards every expert's
intermediate dimension across TP instead, which is the path `FusedMoE` takes on
its own. Both arrangements are supported at TP4; whole-expert EP is the primary
deployment configuration. Validate quality and throughput for the selected
arrangement and workload.

`index_cache_dtype="fp8"` is the only index plane, and the runtime refuses any
other before loading weights. The main pool is independent of it and takes
`kv_cache_dtype="bf16"` or `"fp4"`; under FP4, main rows use their own format
and SWA uses FP8. Both use the original V4 BF16 attention kernels and inverse
RoPE; no V4 file is modified to serve V4.1.

One plane means one scorer, for every shape. It reads the plane in place and
gives each query row its own bound and its own tile list, so a prefill token, a
decode token and a drafted token are one shape to it and a ragged batch is not
a case. The scorer consumes the existing paged index plane during prefill,
decode and speculative verification.

Because a block id names 16 index rows and a ratio-2 owner halves the PAGE
before that count is taken, the PAGE token count has a floor of 32; production
rounds it to 256 for block-table reasons.

The index query is left on the grid its reader rounds it to: the paged scorer
quantizes the query itself. The published model instead FP4-rounds query and
key alike whatever it holds (`inference/model.py`, `fp4_act_quant` on both
sides of the index score).

`enforce_eager=True` remains the baseline. Two graph modes are accepted, both
with `enforce_eager=False` and `CompilationConfig(level=0, ...)`:

- `cudagraph_mode=CUDAGraphMode.FULL` captures the whole decode forward, one
  graph per `(batch size, query bucket)`, and a decode step is one replay of
  it. The scorer reads its bounds off the device, so there is nothing here for
  a capture to freeze. Prefill stays eager -- the runner only ever captures
  decode shapes.
- `cudagraph_mode=CUDAGraphMode.PIECEWISE` records the compiled dense pieces
  and leaves attention eager between them.

Under either mode a decode forward runs the width the step declares --
`running_bs` requests and `running_tokens` rows -- rather than the scheduled
batch, because a replay runs the width it was captured at whatever the batch
turns out to be. The padding carries V4's own sentinels: a padding token's
batch id is `-1` and a padding request is zero-length in `cu_seqlens_q`, and
every scatter bails on one or the other, so those rows read and write nothing.

The FFN and its collective reductions are captured with the rest: the routed experts
are V4's `FusedMoE`, which is capturable at every shape, so there is no expert
backend to select and no capture exclusion.

See [cache format and graph execution](deepseek_v41_performance.md) for cache
formats, graph ownership and memory limits. Native five-token DSpark admits
TP sizes that satisfy model dimension divisibility, with BF16 caches and
optional target graphs; its draft
windows, accepted-prefix state, calibration profile, supported scope and **quality
limitations** are in [the DSpark guide](deepseek_v41_dspark.md). Packed
speculative caches and multimodal speculation are rejected, as are
PP/CP, DP without DP attention, decode TBO, KV transfer, plugin execution and
EPLB — all before loading. DP attention supports text requests and prefill TBO.
Compilation level 3 is admitted with FULL graphs or eager execution; the
[AgentX recipe](../recipes/DeepSeek-V4.1-Flash-Agentic.md) records TP2/TP4
no-EP GPU benchmark results with fixed acceptance length 3.51.

The [chat and tool protocol](deepseek_v41_protocol.md) and
[vision and multimodal chunking](deepseek_v41_vision.md) are enabled
independently of speculation. Host Engram lookup still reads final GPU IDs on
the CPU; moving the lookup to HBM and fusing it further is future work.


## Data-parallel attention

`--tensor-parallel-size 4 --enable-dp-attention` starts four attention ranks,
each with TP1 attention, embeddings and output head. Tensor-sharded routed
experts exchange activations through the existing DP gather/scatter transport.
For five-token DSpark, keep BF16 KV and the FP8 index plane:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python -m atom.entrypoints.openai_server \
  --model /mnt/DeepSeek-V4.1-Flash \
  --tensor-parallel-size 4 --enable-dp-attention \
  --kv-cache-dtype bf16 --index-cache-dtype fp8 \
  --method dspark --num-speculative-tokens 5 \
  --level 3 --cudagraph-mode FULL \
  --max-num-seqs 64 --max-num-batched-tokens 16384 \
  --attn-prefill-chunk-size 16384
```

Idle ranks participate in collectives using empty request metadata in the same
pool buffers that graph capture used. Padding rows carry batch ID -1 and
zero-length requests, so they neither read nor write state. Startup warmup,
before pool allocation, continues to use its private scratch cache.

Engram follows the attention TP group: under DPA each rank reads all hash
heads for its own requests. Each Flash rank registers about 183 GiB of mapped
host table storage, so startup registration takes longer than TP4.

## Prefill two-batch overlap

Add `--enable-tbo prefill` to the DPA command above, leaving EP disabled.
Microbatches execute eagerly; decode keeps its compiled/CUDA Graph path.
The existing `ATOM_TBO_PREFILL_MIN_TOKENS` threshold applies, and
`ATOM_TBO_PREFILL_TOKEN_SPLIT=1` permits splitting within a request. Eligibility
is agreed across DP ranks; an idle or incompatible peer selects ordinary execution.

Parent preparation and TBO children both preserve the explicit prefill phase.
A one-token prefill therefore uses prefill indptrs and bounded tile tables even
when DSpark is enabled; tentative verification retains decode semantics.

Each microbatch preserves absolute token positions, request state slots and
page tables, with separate compression plans, attention indptrs and cross-layer
selection state. Ragged prefill retains local row counts and reuses the
per-microbatch counts exchanged by ForwardMode.decide for variable-size MoE
collectives. Callers without that count table retain the CPU-collective fallback.

Engram snapshots the parent's n-gram history before its cursor advances.
After metadata construction, the parent starts one side-stream lookup around
both microbatches. Each consumes its token slice and waits at its Engram layer;
the parent joins the lookup after both workers finish, including failure.
Already-staged rows are sliced identically when Engram overlap is disabled.

V4.1 requests communication priority -1 through `tbo_comm_stream_priority`;
models without a declaration retain priority 0. MoE keeps the existing
compute-to-communication yield and event order. V4.1 records the compute
consumer of the fallback routed output at the dispatch return, before
shared-expert combine and mHC consume the result. This prevents a partner
microbatch from reusing its storage before those consumers finish. The existing
`create_comm_fused_moe_backend` factory rejects TBO and DP > 1, so TBO cannot
enter the communication-fused backend. A `complete=True` return already includes
shared-expert combine and TP reduction; a marker after that return can protect
only downstream consumers, not backend-internal allocations or combine. If
comm-fused TBO support is added, its internal consumer boundary must be audited
separately. A custom-op boundary
keeps the thread-local TBO query at runtime and retains the lifetime marker
in compiled execution. Microbatches use the normal model call and honor the
configured compilation level. TP vision requests pass token-aligned image
embeddings and masks through the split. DPA remains text-only.

Child metadata reuse first queries the prior completion event. If it has
completed, storage is reused without host or device waits. Otherwise, pinned
staging waits only if its previous H2D is still pending, while the upload stream
waits for prior GPU consumers before overwriting device storage. Storage is
released with the KV pools. When the private Engram TP collective is unavailable,
the parent materializes one
fallback gather per layer before launching workers; child views only wait
for and slice these rows.

Eager prefill expands index tile tables only through the largest request end,
including cached prefixes. Paged scoring bounds each logits band by both its
addressing limit and `ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB` (default 2048 MiB).
Decode retains fixed-width metadata for graph replay. TBO uses the existing
single warmup and KV sizing calculation, with no additional budget policy.

Correctness and overlap must be checked together: serial execution can hide
cross-stream reuse errors, while overlapping kernels alone do not demonstrate
an end-to-end throughput improvement. The GPU regressions exercise fallback
dispatch and a synthetic `complete=True` return under delayed consumption and
partner allocation pressure. The latter checks downstream output lifetime and
completion-flag preservation, not real comm-fused backend internals. The CPU
comm-fused integration tests verify that its factory rejects TBO and DP > 1.
