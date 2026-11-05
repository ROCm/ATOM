# Chat token logprobs

The Python Chat Completions endpoint returns target-model token probabilities
for ordinary and speculative decoding, including DSpark.

```json
{
  "model": "<served-model>",
  "messages": [{"role": "user", "content": "What is 17 + 23?"}],
  "logprobs": true,
  "top_logprobs": 5
}
```

`top_logprobs` accepts 0–20 and requires `logprobs=true`. Both JSON and SSE
responses attach token entries to the corresponding choice's `logprobs.content`
or `logprobs.reasoning_content`. Entries include token text, UTF-8 bytes, the
chosen token's logprob, and optional top candidates. Bytes preserve tokens that
contain only part of a multibyte character. A token below rank 20 uses the
DeepSeek-compatible `-9999.0` wire sentinel; internal probabilities remain exact.
Equal-logit tokens share a rank, so ties at the boundary are retained.

## Probability definition

For each committed token, use its **target-model** logit minus the full-vocabulary
logsumexp at that position. This is the raw model distribution before temperature,
top-k, top-p sampling adjustments, and grammar masking, consistent with ATOM's existing offline
logprob computation. Candidate probabilities use the same normalizer. These are
token conditional probabilities, not calibrated confidence in answer correctness.

In speculative steps, each accepted token and the correction or bonus token is
mapped to its target verification row. Rejected drafts and padding are removed
with their token IDs. This requires no additional target-model forward or change
to the acceptance algorithm. Probabilities travel with the same async batch as
token IDs and are trimmed with max-token/stop boundaries. Offline scalar
logprob arrays contain committed tokens only, not speculative placeholders.

## Supported combinations and limitations

- ByteLevel tokenizers are supported; other decoder types return an explicit
  error rather than inventing UTF-8 bytes. The intended V4.1 tokenizer is ByteLevel.
- Text and thinking output, streaming/non-streaming, and multiple choices are
  included. Normal tool requests are supported; probability fields describe
  visible content/reasoning, not tool arguments. Multimodal input and P/D KV
  transfer with logprobs are rejected. Their existing behavior is retained when
  logprobs is disabled. RapidServe requests with logprobs are also rejected before
  inference: its prefill handoff transfers the first token without its probability.
  No image implementation is added.
- Forced synthetic acceptance is rejected. It cannot validate target-model
  semantics or token probabilities.
- JSON Schema with logprobs is supported without speculative decoding. Scores
  describe the raw pre-grammar distribution; they are not renormalized over the
  schema-allowed tokens. Schema + speculative decoding + logprobs is rejected
  before admission. Mixed batches may still contain separate Schema-only and
  logprob-only requests.
- This does not add logprob fields to Responses or Anthropic Messages.
- Full-model teacher-forcing comparisons, broad deployment-topology validation,
  and performance qualification remain required before declaring production
  readiness. Numerical unit tests and API smoke tests are separate evidence.

## Cost and implementation notes

Disabled batches retain the original sampling path. A cached `any_logprobs`
batch flag gates top-K metadata, requested-row staging, scoring and transport;
requested sequence indices are built once with the batch. Plain tool-parser
text uses the native string join without source-span processing. Streaming
requests select logprob handling once, and disabled chunks skip logprob error
lookups and attribution. These are fast-path guarantees, not a claim of literally
zero Python bookkeeping or a measured zero-percent overhead. In mixed batches, only
requesting sequences have their vocabulary rows scored. Enabled requests add
FP32 normalization, chosen-token ranking, optional top-K, and result serialization.
Every participating rank computes these results. With deferred output and TP > 1,
the complete compact payload uses one broadcast from TP rank 0 and one asynchronous
device-to-host copy: chosen/top probabilities, candidate IDs and ranks stay aligned.
Integer fields are packed by reinterpreting their int32 bits, without a lossy
conversion to floating-point numbers. The completed CPU buffer is unpacked using
its own top-K width and the corresponding deferred batch's request IDs and mask.
Existing token-ID synchronization is unchanged. This retains the original TP
probability synchronization rather than assuming only the output-owning rank
needs probability data.

All logprob scoring uses native PyTorch operations on both CPU and GPU,
including `top_logprobs=0`. The scorer has no `torch.compile` wrapper or
compilation cache. Optional snapshots and ragged row mapping are resolved
before scoring; disabled requests do not invoke the scorer. Snapshot capture and
scoring share one row plan. The runner builds requesting/model/sample row indices
from settled CPU query lengths during preparation, after any DSpark q-bucket
adjustment. All three arrays share one staged buffer and the existing sampling
H2D publication; disabled batches omit it. Postprocess uses these device views
without launching row-mapping kernels or reading GPU metadata back to the CPU.
The staging slot's existing completion event protects buffer reuse.

The implementation avoids explicitly retaining a full log-softmax matrix, but
the native selected-logit gather/conversion still allocates vocabulary-sized workspace.
Padded positions within requesting sequences are safely masked at delivery;
avoiding their computation is a later optimization opportunity.

Only selected probabilities and candidate IDs/scores are copied to the host,
not full vocabularies. Records live in request-local memory, with no temporary
files or per-record disk I/O. The completed payload's structured NumPy view
produces flat tuples directly, without constructing and unpacking per-token
objects. Offline scalar logprob arrays are retained for `generate()` callers.
Text slices share a character-to-byte index built once per decoded span; ASCII
spans use arithmetic offsets. Malformed UTF-8 keeps its original byte mapping.
Slicing still materializes Python strings, but does not re-encode source prefixes.
Resolved records are retired after parsing and serialization; zero-byte records
at the current source boundary remain until that boundary advances. Unresolved tool regions retain their records until classified;
there is no constant-memory guarantee for an arbitrarily long unfinished region.
This is not a zero-overhead claim: throughput, tail latency, temporary device
memory and CPU/SSE cost require on/off comparisons at representative loads.

This follows the approach of [vLLM's logprob rank reduction](https://github.com/vllm-project/vllm/blob/0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c/vllm/v1/sample/ops/logprobs.py)
for comparisons and reductions, and its
[compact sampled/top-K output](https://github.com/vllm-project/vllm/blob/0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c/vllm/v1/sample/sampler.py).
ATOM retains its existing strictly-greater rank/tie semantics and raw-target
probability definition. This is not a claim of equivalent vLLM throughput.

## Text attribution and delivery

Tokenizers that enable whitespace cleanup are rejected for logprob requests;
the request never changes the tokenizer's decoding policy. Only a standalone
ByteLevel decoder is supported; decoder chains and TikToken tokenizers are not
currently supported. Kimi replay tests exercise parser attribution only and do
not constitute real Kimi tokenizer support. Parsing
keeps the original byte ranges through reasoning and tool regions, so visible
text joined across hidden regions retains the correct token probabilities even
when the same text also occurs inside tool arguments. SSE attaches probabilities
before serialization; it does not search or reparse serialized text. A token
crossing a hidden marker or channel boundary belongs once to its first visible
source span. Its token bytes remain intact, so they can include hidden marker
bytes; probability entries cannot be split into partial tokens.

Logprob streams decode once through their source-aware byte decoder, after raw
chunks coalesce. They do not construct the ordinary incremental detokenizer.
The service lazily owns one TokenBytes object for its current tokenizer; no
process-global LRU cache retains previous tokenizers.

When the tool parser holds earlier text, enabled streams hold later deltas until
that region can be classified. Attribution then follows source-byte order, so a
token shared by content and reasoning is assigned consistently in JSON and SSE.
This can delay enabled SSE deltas for unfinished tool-like text. Special added
tokens are skipped just as in normal tokenizer decoding. A padded-vocabulary
candidate with no tokenizer entry has empty text and bytes, retaining its score.

A missing or misaligned engine payload terminates only its request. Streaming
errors include a choice index; other choices continue and the stream ends with
`[DONE]`. Internal attribution failures are server errors, not invalid-input errors.

Mixed batches compact probability payloads to requesting sequences before CPU
transfer and Python conversion. Deferred delivery uses the previous batch's
request IDs and logprob mask, preserving token/probability alignment. Ordinary
requests do not create source trackers or probability stores.

## Code organization

`atom/entrypoints/openai/logprobs.py` owns byte decoding, source positions,
request-local probability storage and processing. `LogprobsProcessor` converts
coalesced engine deltas into traced text and assigns each token once to visible
fields. Unresolved spans remain in request-local memory until classified.

`atom/entrypoints/openai/serving_chat.py` owns JSON/SSE response assembly and
stream cleanup. Both response modes use the same source-order attribution.
GPU scoring remains in `atom/model_engine/token_logprobs.py`.

This follows vLLM's separation between its logprob processor and Chat serving
methods, rather than adding separate public modules for each helper. ATOM's
source-span tracking additionally handles separate visible reasoning/content
fields and tool-parser buffering; consolidating files does not remove that work.
