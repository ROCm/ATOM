# DeepSeek-V4.1 piecewise prefill and FULL decode

V4.1 text prefills can replay token-bucket CUDA graphs while preserving
decoder SWA bounded replay. Opt in with either of these options:

```bash
--level 3 --cudagraph-mode PIECEWISE
--level 3 --cudagraph-mode AF_PIECEWISE
```

FULL remains the default. PIECEWISE captures the dense work around attention
and Engram, including mHC and FFN. AF_PIECEWISE also captures each layer's
whole attention call. Both modes use the existing attention kernels.

## FULL_AND_PIECEWISE

Use hybrid mode for AF piecewise prefill and whole-forward FULL decode:

```bash
--level 3 --cudagraph-mode FULL_AND_PIECEWISE
```

In this mode, the capture ladder is measured in **tokens**. Omitting
`--cudagraph-capture-sizes` defaults to 91 prefill token sizes through 2048:
`[1, 2, 4] + range(8, 256, 8) + range(256, 1025, 16) + range(1152, 2049, 128)`.
FULL decode and its derived draft batch sizes retain the original 83-size
ladder through 1024. The Python configuration uses the same defaults. A single
`CompilationConfig.cuda_graph_sizes` value expands to `[1, 2, 4]`, multiples
of 8 below 256, then multiples of 16 through that maximum. An explicit
`--cudagraph-capture-sizes '[1, 2, 4, 8, 16, 32, 64, 128, 256, 448, 512, 704, 768, 1024]'`
uses those token candidates for both phases, overriding the defaults. Internal
full-window tail buckets can still supplement the prefill list. In particular,
`'[1024]'` selects just that candidate rather than expanding the ladder.
Existing FULL, PIECEWISE and AF_PIECEWISE options retain their default
batch-size lists and units.

`CompilationConfig.cudagraph_prefill_capture_sizes` can supply an independent
prefill list without changing FULL/draft decode. `Config.prefill_capture_sizes`
drives prefill capture, its memory estimate and boundary-buffer capacity;
`Config.capture_sizes` continues to drive decode.

Prefill captures the token ladder independently of request count. FULL decode
rounds each candidate up to a multiple of Q, derives BS=T/Q, deduplicates and
also includes every BS from 1 through 32, and filters by the token ceiling and
scheduler capacity. The dense small-BS coverage keeps decode and mono draft
from padding BS=5 to 6 or BS=9 to 10, regardless of prefill bucket spacing.
Padding selects the BS
ladder for the DP-agreed Q. With DSpark 5, Q=6: a 600-token prefill uses the
608 bucket in the default ladder; BS=100 decode uses FULL at BS=102,
612 tokens. Out-of-range decode runs eagerly, not through AF piecewise.

Hybrid mode permits mono target decode and mono DSpark draft whenever their
other deployment and per-step conditions hold. Mono eligibility checks the
decode graph mode; piecewise prefill does not disable it. Target prefills
continue through the original runtime model, and PIECEWISE/AF_PIECEWISE
decode remains ineligible for mono.

PIECEWISE is captured first, then FULL, then draft passes. FULL capture runs
the compiled pieces without inner graph replay/capture. Draft warmup covers
the union of batch sizes reachable at every decode Q. Bounded-replay tail
buckets are added internally when missing. The existing per-bucket piecewise
memory pools are retained; a shared token ladder does not merge graph objects.
Hybrid mode allocates each layer's attention/Engram output slot and the late
stage's inputs once at the maximum token capacity, then supplies prefix views
to each bucket. These model-owned buffers have stable addresses across
serial forwards. Compiled intermediates retain their per-bucket graph pools.

The final capture summary reports total reserved and allocated deltas.
These startup measurements exclude AF graphs first captured by live prefills.
The pre-capture memory-budget estimate remains conservative and still uses
the sum of token sizes; it does not reclaim the boundary-buffer savings for KV.

## Graph boundaries

Attention, Engram and begin/end-forward are explicit splitting operations.
Dense pieces are keyed by total token count. In AF mode, each attention
boundary owns a separate graph cache; ordinary, FULL/REINDEX and subset-query
attention paths are supported.

An attention graph includes QKV/norm, query gathering when required, RoPE,
CSR construction, sparse attention, ordered window writes and output
projection/TP reduction. FULL/REINDEX paths also include their compression,
index writes, scoring and top-k. Bounded replay's layer-20 precompression
and row selection remain outside these attention calls.

Begin/end-forward maintain the live BatchStep dictionaries on every call.
Engram staging starts and joins before dense capture to close its stream
boundary. FULL and eager retain their existing Engram scheduling. Image
prefills execute the stage's Python body so a text graph cannot freeze an
image routing mask.

## Prefill dispatch and bounded replay

Only the V4.1 backend opts into prefill token padding. A single-DP-rank text
prefill selects the smallest captured total-token bucket that holds its
live rows, independently of BS and the decode query-length bucket. FULL
graph dispatch remains disabled for prefill. Dummy runs, multimodal batches,
TBO, eager-required steps, DP prefills and rows beyond the captured ladder
retain their existing dispatch behavior.

The runner uses persistent input/position buffers, zeroes padding and returns
only scheduled rows for sampling and draft KV processing. Metadata retains
the real request spans and cu_seqlens; padded batch IDs are -1. State
relocations and restores still run before model execution.

With bounded replay, early layers compute the full padded token bucket and
layer 20 compresses the live rows. Finishing requests run the late layers on
their window tails, with layer 39 querying their last rows. Request spans
determine the replay plan independently of token padding.

The late stage copies its five mHC state tensors into persistent inputs and
uses its own exact token count. Its graph can replay even when the early
stage exceeds the capture ladder. Unseen tail sizes run compiled pieces
without a CUDA graph; live prefills do not capture new dense tail buckets.

Startup includes missing full-window tail sizes within the configured BS/Q
bounds, with the same shapes included in graph memory budgeting. For the
128-token window and TP2/DSpark validation ladder, added sizes are 128, 256,
512 and 640; 384 and 768 already exist. A 600-token prefill uses early bucket
640 and late bucket 128 in that configuration.

## AF buffers and metadata

[`@piecewise_core`](piecewise_core.md) owns live AF capture and replay.
V4.1 follows V4's builder-owned metadata policy: the fixed buffers published
by the attention backend are read directly. `Block.attention_forward` only
uses the decorator, without a metadata schema, adapter or graph subclass.

The backend publishes a common `PiecewiseState` once per forward/view. Its
key covers host kernel choices and the addresses/layouts of the existing
metadata. The common `ReplayCache` handles dependencies and Python results
for selections, candidates and derived index tables. No per-layer metadata
copy, reconstruction, reset/bind declaration or publication callback is needed.

Warmup and recording see the same cache entry state. Only the entries the
core actually reads are dependencies, and replay updates only its writes.
Query-index views retain their original shared dictionaries. Selected rows
and candidates use builder-owned result slots so eager, newly captured and
replayed producers supply the same downstream addresses.

AF reads the upstream dense graph's stable hidden tensor and writes the
persistent output slot bound by the downstream graph. Bounded replay's
standalone layers 20 and 39 still stage temporary hidden inputs. Subset
metadata is stabilized when its view is constructed, before its layers run.
Parent and TBO child storage has distinct namespaces; existing indptr fills,
publication fences and view-exit invalidation are preserved.

The first eligible invocation warms once, records without replaying, and
returns the warmup result. Later invocations replay once. This preserves
exactly-once KV updates, including prefill's read-before-window-write order.

## Graph reuse limits

Attention graph keys include tensor layouts, decode/prefill mode, token
count, BS capacity, maximum query length, packed-scoring host band boundaries,
KV pool address and directly captured hidden addresses. Positions, slots,
PAGE IDs and index contents remain dynamic tensor data. A FULL layer without
compression plans has a distinct signature.

Each layer retains at most 32 attention graphs. After reaching that limit,
new shapes use eager attention; existing keys still replay. First-shape
capture adds latency and retained graph memory, so profile after warmup.
Graph storage is released through the existing graph-holder lifecycle.

## Validation

Run the regression suite in an ATOM ROCm environment:

```bash
python -m pytest -q \
  tests/test_piecewise_af.py \
  tests/test_hybrid_cudagraph.py \
  tests/test_forward_mode.py \
  tests/models/deepseek_v41/test_piecewise.py \
  tests/models/deepseek_v41/test_af_attention.py \
  tests/models/deepseek_v41/test_attention_streams.py \
  tests/models/deepseek_v41/test_bounded_replay.py \
  tests/attentions/deepseek_v41/test_runtime_contract.py
```

Coverage includes token padding across request layouts, independent early/
late buckets, state maintenance, live Engram boundaries, exception restoration,
image bypass and AF capture/replay. GPU tests compare real compression, FP4
scoring, top-k, CSR, sparse attention and KV bytes against eager with projection
stubs. Producer/consumer graph tests cover fixed output addresses, changed
hidden addresses and metadata refresh on a reused BatchStep.

On 2026-10-14, 168 related tests passed. Full 1,319-question, 3-shot GSM8K at
TP2, concurrency 64 and native DSpark 5 scored 1217/1319 (92.27%) for AF and
1208/1319 (91.58%) for FULL. This run showed no GSM8K regression; it does not
establish bit-exact whole-model generation parity.

The warmed synthetic prefill used BS=1, 600 scheduled tokens, 74,200 total
context and 73,600 cached-prefix tokens. Both ranks replayed 40 attention
and 43 dense graphs, with no target capture/instantiate calls. On TP0, shared
buffers reduced graph-external memcpy calls from 455 to 62, CPU forward time
from 16.47 ms to 9.93 ms and the main GPU interval from 29.10 ms to 27.07 ms.
These are profiler measurements; clean agentic TTFT needs a separate benchmark.

For comparisons, hold image, capture buckets, KV/index precision, prefix cache,
bounded replay and speculative verification settings constant. Use native
speculative acceptance for correctness and accuracy runs.
