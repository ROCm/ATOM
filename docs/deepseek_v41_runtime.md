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
- `runtime.py` compiles the forward as one graph (`_Backbone`), or as two
  around the last KV-source layer (`_EarlyLayers`, `_LateLayers`) under decoder
  SWA bounded replay; `bounded_replay.py` lays out the rows each layer from 20
  on runs (below).
- Scheduler consumes the existing generic `StateTransfer.copy` capability, or `StateTransfer.replay` under `--encoder-swa-bounded-replay` (the default).
  Bounded replay adds where a prefill forward starts (`BlockManager.replay_start`,
  below); cancellation of requests with no sampled output is fixed too,
  including a middle prefill chunk and the first deferred step.

Only Full owners 2, 8, 14 and 20 allocate global main/index storage. Reuse and
Reindex layers read those owner regions. Every request has all 40 SWA rings,
three ratio-2 FP32 compressor tails, the latest three compressed Engram IDs and
a committed position in its STATE entry. Prefill retains the old rings until
all queries have consumed their causal prefixes, including chunks wider than
the ring. Index reads gather a tile or candidate positions, never a full copy
of the historical main KV.

Allocation and checkpoints use the same `V41PoolGeometry` declarations.
Positions and Engram history advance after the model forward; checkpoints
carry the SWA rings and the cursor (position and Engram history), but not the
FP32 compressor tails: CSA2 pools without overlap and a checkpoint sits on a
PAGE boundary, a multiple of every ratio, so the first group a resumer reads is
one its own forward writes (as DeepSeek-V4 leaves out its HCA tails). Images
are versioned by geometry, including the index plane format. See
[cache format and graph execution](deepseek_v41_performance.md#cache-format-and-attention-boundary)
for the storage formats.

An exact prefix hit restores the entire state before preparing Engram inputs.
Without a matching image, the generic scheduler replays from a recoverable
boundary. Checkpoint relocation/fork and restored tentative suffixes cannot
retain stale window rows or Engram history; a restore leaves the slot's own
compressor tails, which the resumed forward overwrites before reading.

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

The index plane is `index_cache_dtype="fp8"` (default) or `"fp4"`, and the
runtime refuses any other before loading weights. FP4 is the official
arithmetic (E2M1, one E8M0 per 32 dims) in the row-group scorer's page-8
layout, values and scales in two planes (`index_plane.py`); one aiter kernel
norms, rotates, quantizes and stores a key, bit for bit the FP8 path's key
chain before its quantization. The main pool is independent of it and takes
`kv_cache_dtype="bf16"` or `"fp4"`; under FP4, main rows use their own format
and SWA uses FP8. Both use the original V4 BF16 attention kernels and inverse
RoPE; no V4 file is modified to serve V4.1.

Each plane format has one scorer, for every shape. It reads the plane in place and
gives each query row its own bound and its own tile list, so a prefill token, a
decode token and a drafted token are one shape to it and a ragged batch is not
a case. The scorer consumes the existing paged index plane during prefill,
decode and speculative verification.

Because a block id names 16 index rows and a ratio-2 owner halves the PAGE
before that count is taken, the PAGE token count has a floor of 32; production
uses 128, coarse enough to keep the block table small and no coarser than the
window, since a PAGE is also the prefix cache's hash block (the granularity of
prefix reuse and of what encoder bounded replay recomputes below a hit).

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
PP/CP/DP, TBO, KV transfer other than `lmcache_mp`, RapidServe, plugin
execution and EPLB — all before loading. `lmcache_mp` offloads PAGE units and
restores STATE through the same PAGE-backed checkpoint images the local prefix
cache uses (`get_kv_transfer_tensors` publishes each plane of a unit, in
`StateCopies` order).
Compilation level 3 is admitted with FULL graphs or eager execution; the
[AgentX recipe](../recipes/DeepSeek-V4.1-Flash-Agentic.md) records TP2/TP4
no-EP GPU benchmark results with fixed acceptance length 3.51.

The [chat and tool protocol](deepseek_v41_protocol.md) and
[vision and multimodal chunking](deepseek_v41_vision.md) are enabled
independently of speculation. Host Engram lookup still reads final GPU IDs on
the CPU; moving the lookup to HBM and fusing it further is future work.

## Decoder SWA bounded replay

Decoder SWA bounded replay is on by default; `--no-decoder-swa-bounded-replay`
turns it off. It follows the V4.1 report's SWA Bounded Replay (section 3.2.2).
V4.1 is a causal encoder (layers 0..19) and a decoder (layers 20..39) whose
global KV layer 20 projects from the encoder's output; past that, the decoder
layers own only their sliding-window rings, so a prefill computes only what
anything after it reads (`models/deepseek_v41/bounded_replay.py`). With `W`
the window (128), the forward that finishes a prompt runs, per request:

- every row: the encoder, and layer 20's compressed KV and index keys;
- its last `W` rows, the tail: layers 20..38, and layer 39's KV;
- its last row: layer 39's query, attention and FFN, the row the LM head reads.

Every decoder layer's window starts at the tail (`swa_replay_start`): a tail
row at position `i` reads `[max(s, i - W + 1), i]`, `s` the tail's first
position. A forward that does not finish its prompt
(`ScheduledBatch.is_final_chunk`) runs nothing past the encoder but layer 20's
compression: no kernel of the decoder's attention or FFN. Layer 39's queries
are fewer than its KV rows (`QueryRows`): the rows are gathered in one launch
(`token_layout.gather_rows`), the index lists built over all KV rows with every
other row's batch id -1, so the queries' lists come out adjacent, and attention
runs on the queries alone. The DSpark draft's window is built from layers
37..39, so it follows the decoder: a replayed forward leaves its plan on the
step (`BatchStep.replay`), and the draft writes its context on the tail through
the tail's step, and nothing in a forward that ran no decoder.

The forward finishing a prompt has to hold its last `W` rows. One that resumes
closer to the end -- a short last chunk, or a prefix hit with a short suffix --
starts a whole hash block earlier (`BlockManager.replay_start`,
`Config.min_final_prefill_rows`), recomputing rows the cache holds without
changing them: their global KV is not written again
(`make_compress_plans(write_from=...)`), the encoder's rings keep the rows they
hold (`swa_write(cached_until=...)`), and the rows from where the forward
resumed read those rings below that point (`swa_cached_until`), as a forward
resuming there would. The slot's state stays; its cursor steps back to the
forward's start. Only what the recomputed rows feed the decoder is new. The
forward starts on the hash-block grid, however its resume was cut. Its
long-prefill cap counts from where it resumed, and one the budget cannot carry
past that point waits for a step that can: a chunk ending there would recompute
cached rows alone and hand the next pass the same back-off.

With `--no-decoder-swa-bounded-replay` the runtime model compiles its forward
as one graph (`runtime._Backbone`), as before. Otherwise layers 0..19 and
21..38 are two graphs (`_EarlyLayers`, `_LateLayers`, each with its own
compile-cache key) and layers 20 and 39 run between and after them, where a
replay cuts their rows. Decode, warmup, draft, TBO and padded steps run every
layer on every row. After a replay only each finishing request's last row of
the model output is defined; the LM head reads only those.

Accuracy: this is not a full prefill. In every decoder layer the tail's rows
see a window truncated at the tail start, so from layer 21 on even the last
token's window is made of rows computed from truncated windows, and the last
token's logits and every ring row decode reads differ from a full prefill's.
The encoder and the global path (the compressed KV and the top-k selections
the decoder reuses) are exact. DeepSeek post-trains on this truncation.

## Encoder SWA bounded replay

`--encoder-swa-bounded-replay` (on by default; `--no-encoder-swa-bounded-replay`
keeps state checkpoints instead) resumes a prefix hit without a
state checkpoint. The backend declares a `replay` state transfer
(`StateTransfer.replay()`): the engine keeps no checkpoint (so no prefill chunk
is cut for one, and no PAGE unit holds a STATE image), accepts a hit at any
hash-block boundary, and starts the hit's forward a whole hash block below it
(`BlockManager.replay_start`; 128 rows at V4.1's block size, the report's
`n_win`, which also leaves the decoder its `W` rows however short the suffix).
That forward rebuilds the sliding-window state in a fresh slot: it recomputes
the rows below the hit without writing what the prefix cache holds
(`make_compress_plans(write_from=...)` drops every compressed group ending
below the hit), its windows start at its first row (`swa_replay_start`), and
its Engram history is the compressed tokens before that row, an image position
DEAD (`Sequence.replay`, `Replay.history_ids`; `Config.replay_history_tokens`
of them). The hit stays the reported hit. `validate_runtime_config` refuses a
PAGE shorter than the window - 1, which one PAGE of rebuild would not cover.
TorchSpec hidden-state export reads every prefill row, so it needs
`--no-decoder-swa-bounded-replay` at launch.

Accuracy: the encoder's rebuilt rows see windows truncated at the replay
start, so a resumed request's logits differ from a cold prefill's the way decoder replay's
do; the cached global KV it reads is exact.
