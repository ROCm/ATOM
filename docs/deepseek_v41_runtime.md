# DeepSeek-V4.1 native paged runtime

Native ATOM text execution runs through ModelRunner and Scheduler with chunked
prefill, ragged batches, continuous decode and complete request-state recovery.
The existing V4 BF16 sparse attention and inverse RoPE kernels are unchanged.
The engine is the only implementation: the offline eager cache it used to be
compared against has been removed, and arithmetic is judged against the
published model and end-to-end `lm_eval` scores instead.

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

With BF16 production geometry and block size 16, one PAGE costs 51,200 bytes
and one STATE entry costs 5,256,192 bytes. The complete image occupies 103 PAGE
units. The optional packed layout uses 15,104 bytes per PAGE and 2,715,904 bytes
per STATE (180 smaller PAGE units). Allocation and checkpoints use these same
declarations. Positions and Engram history advance after the model forward;
checkpoints carry every state field and padding byte. Images are versioned by
geometry, the index plane's format included.

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
its own. Whole-expert EP is the more heavily exercised of the two — most of the
validation below was run that way — but TP-only is not refused: at TP4 it scores
0.9204 on the 1,319-question GSM8K set, inside the band six same-code EP runs
span. Their relative throughput has not been measured.

`index_cache_dtype="fp8"` is the only index plane, and the runtime refuses any
other before loading weights. The main pool is independent of it and takes
`kv_cache_dtype="bf16"` or `"fp4"`; under FP4, main rows use their own format
and SWA uses FP8. Both use the original V4 BF16 attention kernels and inverse
RoPE; no V4 file is modified to serve V4.1.

One plane means one scorer, for every shape. It reads the plane in place and
gives each query row its own bound and its own tile list, so a prefill token, a
decode token and a drafted token are one shape to it and a ragged batch is not
a case. DeepSeek-V4 scores its own prefill by concatenating the batch's keys
instead; both arrangements were measured on the production geometry, and the
paged one holds `1/batch` of the logits at equal speed.

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
formats, graph ownership and measured limits. Native five-token DSpark supports
TP4 text requests with BF16 caches and optional target graphs; its draft
windows, accepted-prefix state, calibration profile, validated scope and **open
quality regression** are in [the DSpark guide](deepseek_v41_dspark.md). Packed
speculative caches and multimodal speculation are rejected, as are
torch.compile, PP/CP/DP, TBO, KV transfer, plugin execution and EPLB — all
before loading.

The [chat and tool protocol](deepseek_v41_protocol.md) and
[vision and multimodal chunking](deepseek_v41_vision.md) are enabled
independently of speculation. Host Engram lookup still reads final GPU IDs on
the CPU; moving the lookup to HBM and fusing it further is future work.
