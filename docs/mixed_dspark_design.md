# Mixed prefill and decode on DeepSeek V4

`--enable-mixed-prefill-decode` packs a batch as `[prefill | decode]`.
The feature requires chunked prefill. DSpark and TBO are supported;
PCP, DCP, EPLB, TR_HYBRID graph modes, other speculative methods, and
speculative decoding with pipeline parallelism are rejected at startup.
Expert parallelism alone does not enable EPLB.

## Scheduling

Prefill and decode share `max_num_batched_tokens`. A decode row costs
`decode_spec_width + 1` tokens, including speculative verification. The
scheduler reserves that demand for running decodes while retaining at least
one eighth of the token budget for prefill progress. At high decode load,
some decodes wait for a later step.

New-prompt admission also reserves the eligible running decodes' append-page
demand. A new prompt that would consume those pages stays in the waiting
queue, rather than forcing an older decode to discard its KV and recompute.
This reservation does not allocate pages and includes speculative lookahead.

## Attention execution

Dense projection, QK normalization, RoPE, and the output projection run on
the combined token tensor. The attention tail dispatches the prefill and
decode segments to their respective compressors, indexer, and sparse kernels.
The combined QK/RoPE kernel fuses decode SWA stores. Prefill tokens carry a
negative store mask, so their window is written only after prefill attention
has consumed the preceding chunk's KV.

`ScheduledBatch.slice_rows` owns the distinction between request fields,
token fields, request-keyed maps, and filtered state/page rows. It rejects
unclassified fields rather than passing parent data through to a segment.
The V4 builder passes resolved rejection counts explicitly to decode
preparation; decode code needs no proxy type or parent-row offset.

V4 owns separate publication banks for prefill metadata and decode metadata.
Each bank has its own host sources, device destinations, and H2D ownership.
They are allocated before memory profiling and follow forward-slot reuse.
The common attention builder retains whole-batch query-span semantics;
only V4 publishes decode-local spans for its mixed dispatch.

TBO splits mixed batches only when the cut lies inside prefill. Its first
microbatch is pure prefill and the second holds the remaining prefill tokens
and every decode token. Both use the same carrier construction and SWA-mask
rules as an unsplit mixed batch. Segment state is checked when that carrier
is built, rather than repeatedly in every model layer.

## Startup memory profiling

Dummy warmup does not exercise V4 attention because KV has not been bound.
Mixed startup therefore binds a temporary KV pool and runs real forwards
before the serving pool is sized. It samples prefill-heavy, balanced, and
decode-heavy shapes within the configured token and sequence limits, at
short and maximum valid contexts. DSpark covers both full-width verification
and q=1, retaining the full draft width in both cases. The configured TBO
path executes as part of these forwards.

Old history can share a read-only zero page during this allocation profile;
current writes use private pages and state slots. Temporary KV bindings are
released before allocating the serving pool. The budget uses the measured
allocation/cache high-water mark with temporary KV subtracted, along with
the existing graph, non-PyTorch, and safety allowances. There is no fixed
5% mixed reserve. The sampled peak is configuration-dependent; it does not
prove that every future allocator fragmentation pattern is bounded.

See also the [configuration guide](configuration_guide.md) and
[scheduling and KV cache guide](scheduling_kv_cache_guide.md).
