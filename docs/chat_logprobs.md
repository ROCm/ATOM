# Chat token logprobs

This change connects target-model token probabilities to the Python Chat
Completions endpoint for ordinary and speculative decoding, including DSpark.
It is separate from the frontend compatibility work in #2844.

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

Disabled requests retain the original sampling path. In mixed batches, only
requesting sequences have their vocabulary rows scored. Enabled requests add
FP32 normalization, chosen-token ranking, optional top-K, and result serialization.
Every participating rank computes these results. With deferred output and TP > 1,
the complete compact payload is broadcast from TP rank 0 before asynchronous
output transfer: chosen/top probabilities, candidate IDs and ranks stay aligned.
Existing token-ID synchronization is unchanged. This retains the original TP
probability synchronization rather than assuming only the output-owning rank
needs probability data.

All logprob scoring uses native PyTorch operations on both CPU and GPU,
including `top_logprobs=0`. The scorer has no `torch.compile` wrapper or
compilation cache. Optional snapshots and ragged row mapping are resolved
before scoring; disabled requests do not invoke the scorer.

The implementation avoids explicitly retaining a full log-softmax matrix, but
the native selected-logit gather/conversion still allocates vocabulary-sized workspace.
Padded positions within requesting sequences are safely masked at delivery;
avoiding their computation is a later optimization opportunity.

Only selected probabilities and candidate IDs/scores are copied to the host,
not full vocabularies. Streaming probability records remain in the request-owned
spool until close; reading a visible source span marks intersecting records as claimed.
The spool bounds its RAM buffer and spills to disk. This is not a zero-overhead
claim; throughput, tail latency, temporary device memory, and CPU/SSE costs need
on/off comparisons at representative loads.

This follows the approach of [vLLM's logprob rank reduction](https://github.com/vllm-project/vllm/blob/0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c/vllm/v1/sample/ops/logprobs.py)
for comparisons and reductions, and its
[compact sampled/top-K output](https://github.com/vllm-project/vllm/blob/0e5b2ba118cfb3763a3640942e02dd8d93e8ed6c/vllm/v1/sample/sampler.py).
ATOM retains its existing strictly-greater rank/tie semantics and raw-target
probability definition. This is not a claim of equivalent vLLM throughput.

## Review follow-up: alignment and bounded transfer

Logprob-enabled requests explicitly disable tokenizer whitespace cleanup. Parsing
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

Unclassified tool-like regions may become visible text at end of stream. Their
fixed-width probability records are kept in a request-owned spool with a 64 KiB
in-memory limit, spilling to a temporary file beyond that limit. The file is
closed on completion or cancellation. This bounds retained probability RAM but
adds file I/O for long enabled requests; it is not a zero-overhead feature.

Mixed batches compact probability payloads to requesting sequences before CPU
transfer and Python conversion. Deferred delivery uses the previous batch's
request IDs and logprob mask, preserving token/probability alignment. Ordinary
requests do not create source trackers or probability spools.

## Code organization

`atom/entrypoints/openai/logprobs.py` owns byte decoding, source positions,
request-local probability storage and processing. `LogprobsProcessor` converts
coalesced engine deltas into traced text and assigns each token once to visible
fields. Its private spool keeps unresolved parser regions bounded in RAM.

`atom/entrypoints/openai/serving_chat.py` owns JSON/SSE response assembly and
stream cleanup. Both response modes use the same source-order attribution.
GPU scoring remains in `atom/model_engine/token_logprobs.py`.

This follows vLLM's separation between its logprob processor and Chat serving
methods, rather than adding separate public modules for each helper. ATOM's
source-span tracking additionally handles separate visible reasoning/content
fields and tool-parser buffering; consolidating files does not remove that work.
