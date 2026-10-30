# Chat token logprobs (draft)

This draft connects target-model token probabilities to the Python Chat
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
top-k, and top-p sampling adjustments, consistent with ATOM's existing offline
logprob computation. Candidate probabilities use the same normalizer. These are
token conditional probabilities, not calibrated confidence in answer correctness.

In speculative steps, each accepted token and the correction or bonus token is
mapped to its target verification row. Rejected drafts and padding are removed
with their token IDs. This requires no additional target-model forward or change
to the acceptance algorithm. Probabilities travel with the same async batch as
token IDs and are trimmed with max-token/stop boundaries. Offline scalar
logprob arrays contain committed tokens only, not speculative placeholders.

## Current draft boundaries

- ByteLevel tokenizers are supported; other decoder types return an explicit
  error rather than inventing UTF-8 bytes. The intended V4.1 tokenizer is ByteLevel.
- Text and thinking output, streaming/non-streaming, and multiple choices are
  included. Tools, multimodal input, and P/D KV transfer with logprobs are
  explicitly rejected in this initial draft. Their existing behavior is retained
  when logprobs is disabled. No image implementation is added.
- Forced synthetic acceptance is rejected. It cannot validate target-model
  semantics or token probabilities.
- This does not add logprob fields to Responses or Anthropic Messages.
- Full-model teacher-forcing comparisons, broad deployment-topology validation,
  and performance qualification remain required before declaring production
  readiness. Numerical unit tests and API smoke tests are separate evidence.

## Cost and implementation notes

Disabled requests retain the original sampling path. In mixed batches, only
requesting sequences have their vocabulary rows scored. Enabled requests add
FP32 normalization, chosen-token ranking, optional top-K, and result serialization.
The implementation avoids explicitly retaining a full log-softmax matrix, but
the selected-logit gather/conversion still allocates vocabulary-sized workspace.
Padded positions within requesting sequences are safely masked at delivery;
avoiding their computation is a later optimization opportunity.

Only selected probabilities and candidate IDs/scores are copied to the host,
not full vocabularies. Streaming byte trackers release consumed records. This
is not a zero-overhead claim; throughput, tail latency, temporary device memory,
and CPU/SSE costs need on/off comparisons at representative loads.
