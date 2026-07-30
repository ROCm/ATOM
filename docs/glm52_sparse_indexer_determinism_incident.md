# GLM-5.2 sparse-indexer corruption incident

Date: 2026-07-14

## Scope

GLM-5.2-MXFP4 served with TP=8 produced intermittent long-context corruption:
reasoning/tool tags leaked into normal text, digits were inserted into words,
and greedy (`temperature=0`) responses could differ on the same cached prefix.

The repro uses the locally recorded request `1e2dfe2b680a4b1ca7c15162`:

1. Full prefill: 60,753 input tokens.
2. Exact prefix-cache replay: 1 input token plus 60,752 cache-read tokens.
3. A second identical replay.

The request log includes private conversation content and remains local under
`logs/glm_requests.jsonl`; this document contains no prompt or response text.

## Root cause

The immediate corruption source is the AITER decode primitive
`top_k_per_row_decode`, not MLA KV/indexer-cache writes.

The indexer is replicated on every tensor-parallel rank. Its selected sparse
KV positions must consequently be identical on every rank before sparse MLA
and tensor-parallel all-reduce. In the failing configuration, the following
were byte-identical on all eight ranks and all three replays:

- indexer Q, K, weights, and paged DeepGEMM logits;
- the 132 logical bytes written for the current indexer entry (128 FP8 K bytes
  and its 4-byte scale);
- the complete 16-token physical indexer-cache tile;
- the current main MLA cache row (576 FP8 bytes).

Despite identical logits, `top_k_per_row_decode` returned different local
token positions on each rank. The following global-index conversion and sparse
MLA therefore consumed different KV subsets per rank. Its all-reduce combined
incompatible attention results, which amplified through later layers into text
corruption.

This is consistent with an undefined parallel tie behaviour in the native
top-k path. FP8 quantisation and the indexer's per-head ReLU create many equal
scores; a parallel top-k that compares only score and has no token-position
tie-break may legally choose different members of the tied boundary set on
different launches/ranks. The observations prove that the primitive violates
the replicated-indexer determinism requirement; whether its implementation is
an intentionally unspecified tie policy or an internal reduction race does not
change the required fix.

## Cache-layout investigation

The initial cache-write/132-vs-144 concern was investigated directly.

- Main MLA and indexer caches are separately allocated tensors, not adjacent
  views of one slab.
- Runtime main MLA layout is `[963840, 1, 576]`.
- Runtime indexer layout is `[60240, 16, 144]`, with strides `(2304, 144, 1)`.
- The current token slot was valid (for example `60753`), and well within both
  allocations.
- The AITER indexer write kernel receives `cache_stride=144`, writes K into
  the first 128 bytes, and writes the scale at bytes 128--131. Its
  `preshuffle=True` layout tile-shuffles K across the 16 token slots; it does
  not advance token rows by the logical 132-byte size.

Raw write-after hashes confirmed there was no cache-write drift in the
controlled replay. The diagnostic reconstructs the logical 132-byte value
using the same preshuffle address formula as the kernel, and separately hashes
the full physical tile.

## Changes

### Deterministic sparse indexer decode top-k

`atom/models/deepseek_v2.py` now supports this process-start environment
variable:

```bash
ATOM_DETERMINISTIC_GLM_INDEXER_TOPK=1 python -m atom.entrypoints.openai_server ...
```

The local `serve_glm.sh` convenience launcher maps
`DETERMINISTIC_INDEXER_TOPK=1` to that environment variable.

When enabled, the decode indexer uses
`_deterministic_top_k_per_row_decode` instead of AITER
`top_k_per_row_decode`.

For every row it:

1. obtains the top-k threshold value with `torch.topk`;
2. retains every position strictly above that threshold;
3. fills the remaining positions from values equal to the threshold in
   ascending logical-token order.

Thus its effective comparison key is `(score descending, token position
ascending)`. It avoids a full stable sort while making the selected *set*
rank-independent. It is intentionally eager-only: it reads each row's dynamic
context length on the host and is not yet suitable for level-3 CUDAGraph
capture.

### Forensic probes

`GLM_DIAG_HASH=1` enables opt-in eager-only diagnostics. They record BLAKE2
hashes, never tensor values, for:

- layer-0 main MLA cache writes;
- layer-0 indexer logical/cache-tile writes;
- paged indexer logits, native/deterministic local top-k, block table;
- sparse MLA selected KV, indices, and output.

Use only with local request logs; GPU-to-host hashing is intentionally too
expensive for normal serving.

### Independent BF16 projection issue

An earlier, independent AITER BF16 M=1 projection drift was also observed in
the layer-0 QKV projection. `BF16_QKV_FALLBACK=1` uses the existing BF16
correctness fallback. It was enabled for the determinism investigation so the
top-k diagnosis had byte-identical upstream inputs. It remains part of the
current safe serving configuration.

## Verification

With `BF16_QKV_FALLBACK=1`, `DETERMINISTIC_INDEXER_TOPK=1`, eager mode, and
the recorded 60k-token request:

- all eight TP ranks produced the same deterministic local top-k hash;
- all eight TP ranks produced the same converted global-index and selected-KV
  hashes;
- full prefill and two exact prefix-cache replays had matching logits, local
  top-k, selected KV, and main MLA output hashes;
- two additional exact replays produced the same response-content hash at
  `temperature=0` (response IDs are intentionally excluded).

The currently running safe service configuration is:

```text
DETERMINISTIC_INDEXER_TOPK=1
BF16_QKV_FALLBACK=1
SAFE_INDEXER=0
TRITON_DSA=0
ENFORCE_EAGER=1
COMPILATION_LEVEL=0
CUDAGRAPH_MODE=NONE
REQUEST_LOG=1
```

### Final service retest (2026-07-14)

After restarting without `GLM_DIAG_HASH`, the current resident service was
tested again with `temperature=0` and `max_tokens=1`:

| Run | Input tokens | Cache-read tokens | Response-content BLAKE2 |
| --- | ---: | ---: | --- |
| Full prefill | 60,753 | 0 | `fd6e182a083c512646525d6cafcf9c92` |
| Prefix replay 1 | 1 | 60,752 | `fd6e182a083c512646525d6cafcf9c92` |
| Prefix replay 2 | 1 | 60,752 | `fd6e182a083c512646525d6cafcf9c92` |

The digest covers only the Anthropic response `content`, excluding the request
ID and other intentionally varying response metadata.

## Follow-up

The production-quality solution is an AITER-side deterministic
`top_k_per_row_decode` implementation, ideally comparing `(score, position)`
inside the parallel reduction and remaining trace/CUDAGraph-safe. That would
remove the eager fallback and permit level-3 CUDAGraph serving again.

## Addendum: residual dense-MLA replay drift (2026-07-14)

After the native AITER top-k implementation was changed to return the full
canonical order `(score descending, logical token position ascending)`, a
separate failure remained in a newer locally logged Claude-GLM request. This
addendum records that second investigation. As above, it deliberately contains
only hashes, shapes, positions and configuration; request/response text stays
in the local request log.

### Reproduction

- Input: a locally captured 48,175-token Anthropic request.
- Serving: GLM-5.2-MXFP4, TP=8, prefix caching enabled.
- Probe: the exact same request body, with `stream=false`, `temperature=0`,
  and `max_tokens=64`, replayed twice.
- Result: completion-text BLAKE2 and byte length differed between replays.

The drift reproduces in eager mode as well as FULL CUDAGraph mode, so graph
capture is not the common cause. A one-token replay succeeds but is too short
to reliably exercise the failure.

### Eliminated causes

For the native deterministic top-k kernel, all eight ranks had byte-identical
indexer logits, complete `topk_local` ordering, stable-reference top-k,
selected members and block table. This removes the earlier tied-top-k issue
from this reproduction.

The following dense-MLA layer-1 tensors were also byte-identical across two
independent replay runs at the first divergent decode position:

- layer-0 output / layer-1 input;
- layer-1 Q/K projection outputs;
- layer-1 fused RoPE/cache output (`q_out`);
- the just-written layer-1 KV cache row.

The layer-1 `self_attn` result then differed. GLM-5.2 configures
`first_k_dense_replace=3`, so layer 1 is a dense MLA layer, not a DSA sparse
indexer layer. This explains why enabling Triton DSA did not remove this
failure: it did not replace this dense layer's native decode path.

The remaining boundary is:

```text
identical q_out + identical KV cache
             |
             v
native dense MLA decode (mla_decode_fwd)
             |
             +--> raw attention output o        [not yet compared cross-run]
             |
             v
V-up/O projection and TP reduction              [not yet compared cross-run]
```

An experimental run with the native page-size-1 persistent decode workspace
disabled still diverged, so that workspace is not the root cause. The current
non-shuffle Triton dense-MLA fallback stalled on this 48k request during
decode, so it is not an operational replacement.

### Independent greedy-sampling correctness fix

`Sampler.forward` now honours `all_greedy` before entering either sampling
path. Previously a request with `temperature=0` and no top-k/top-p filter
could reach epsilon-temperature Gumbel sampling instead of strict argmax.
`tests/test_sampler.py` covers this condition and passed in the serving
container. The fix removes a separate source of output variation but did not
eliminate the dense-MLA drift above.

### Next diagnostic and fix criterion

Compare raw `o` immediately after `mla_decode_fwd` across two replays, then
compare the result after V-up/O projection. If raw `o` differs, fix or bypass
the native dense MLA decode kernel. If raw `o` matches and projected output
differs, isolate the V-up/O projection or its TP reduction instead. Do not
submit a performance fallback before this boundary is established.

## Final correction: sparse-prefill top-k ordering (2026-07-14)

The preceding addendum's ``dense MLA`` hypothesis was disproved by a later
replay at a narrower boundary. The actual remaining fault is the **ordering**
returned by AITER's `top_k_per_row_prefill`, not KV-cache corruption and not
an MLA cache writer overflow.

### Evidence from the private 48,175-token replay

For each of two prefix-cache hits (`48,160` cached tokens plus a `15`-token
suffix), the layer-0 probe recorded only BLAKE2 digests and shapes:

- `q`, sparse CSR lengths, worker/reduction metadata, and the *sorted* set of
  2,048 logical top-k members were identical on all eight TP ranks;
- `topk_local` (the native return order) differed on every TP rank and between
  the two hits;
- physical `block_table` digests differed between hits, as expected when the
  suffix receives a different physical cache block; this is not an error;
- the resulting selected-KV byte sequence and sparse MLA output differed only
  because that same logical member set was traversed in a different order;
- a second AITER MLA call in the **same forward**, with the same physical KV
  addresses and metadata, was byte-identical. The MLA kernel itself is
  deterministic for fixed ordered inputs.

The critical observation was:

```text
same 2,048 token members
        +
different native top-k permutation
        |
        v
different FP8 MLA reduction traversal per TP rank
        |
        v
different low-precision local attention result / greedy boundary
        |
        v
Claude-GLM completion can diverge
```

### Fix and replay result

When `ATOM_DETERMINISTIC_GLM_INDEXER_TOPK=1`, ATOM now keeps AITER's prefill
top-k member selection but canonicalises its selected 2,048 entries by:

```text
(score descending, logical token_id ascending)
```

It first sorts selected token IDs ascending, then performs a stable descending
score sort; hence equal scores retain ascending token ID order. This is an
`O(k log k)` reorder of the selected `k=2048` entries, not a full-context
sort. `ATOM_DETERMINISTIC_PREFILL_TOPK_MAX_ROWS=32` was used for the forensic
replay so the canonicalisation applied only to the reproduced short
prefix-cache suffix, not its 48k initial prefill.

Under that configuration, both suffix replays had the same complete
`topk_local` digest on all eight TP ranks and the same completion semantic
BLAKE2 at `temperature=0`, `max_tokens=1`. The raw HTTP-body digest may still
differ because request IDs are API bookkeeping.

This closes the second incident boundary. A production AITER follow-up should
make `top_k_per_row_prefill` itself return this canonical ordering so ATOM can
remove the optional post-ordering step and use it for arbitrary prefill sizes.

### AITER kernel implementation (same day)

The canonicalisation was subsequently moved into the locally deployed AITER
`module_top_k_per_row` implementation. After native radix selection completes,
a GPU bitonic kernel sorts the selected 2,048 `(score, token_id)` pairs by
score descending and token ID ascending. ATOM's reorder was disabled for this
test (`DETERMINISTIC_INDEXER_TOPK=0`).

This kernel patch canonicalises the returned survivor **order**. That is the
observed prefill failure: the survivor member-set digest was already identical.
If a future workload shows different members at an exact score boundary, the
radix-selection stage itself will additionally need a token-ID tie-break.

To avoid applying a 2,048-item bitonic sort to every row of an enormous first
prefill, the kernel currently reads
`AITER_CANONICAL_PREFILL_TOPK_MAX_ROWS` (default `32`). It canonicalises the
short prefix-cache suffix path that reproduced the incident; setting the value
to `0` removes that row cap for broader benchmarking.

The recompiled AITER module was validated with another private prefix-cache
replay (33,045-token prefix, 5-token suffix): all eight ranks and both cache
hits had an identical complete `topk_local` digest, and the two response
semantic BLAKE2 values were identical.
