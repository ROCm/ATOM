# DeepSeek-V4.1 PIECEWISE and AF_PIECEWISE

V4.1 text prefills can replay token-bucket CUDA graphs while preserving
decoder SWA bounded replay. Opt in with either of these options:

```bash
--level 3 --cudagraph-mode PIECEWISE
--level 3 --cudagraph-mode AF_PIECEWISE
```

FULL remains the default. PIECEWISE captures the dense work around attention
and Engram, including mHC and FFN. AF_PIECEWISE also captures each layer's
whole attention call. Both modes use the existing attention kernels.

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

AF follows V4's producer/consumer buffer policy. It directly reads a stable
upstream dense-graph hidden tensor and writes the persistent output slot
already bound by the downstream dense graph. The producer address is part
of the key to prevent a changed input allocation from replaying stale data.
With bounded replay, standalone layers 20 and 39 still stage hidden inputs.
Basic PIECEWISE and AF fallback paths copy their eager outputs into these
same downstream slots.

Request metadata is staged in shared fixed-address buffers. Matching inputs
are reused across layers within one live BatchStep. The memo is reset by
begin_forward, including when the step and source allocations are reused.
Early/late metadata, subset-query layouts and incoming selected/candidate
rows are supported. Graph holders retain and release the input storage.

Each recording starts with fresh Python memo dictionaries so warmup cannot
omit index-building work from the recorded graph. Replay publishes selected
rows, candidates and CSR/window results into the current step.

The first eligible invocation of a shape warms the computation once, then
records without replaying and returns the warmup result. Subsequent calls
refresh shared inputs and replay once. This preserves exactly-once KV
updates, including prefill's read-before-window-write ordering.

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
