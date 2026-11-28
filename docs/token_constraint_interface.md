# Token constraints

Tool output parsing is done by the shared ToolCallParser stream engine. A model
format declares its wire syntax and constraint rules. K3 declares its channel
rule through its `required_call_channel` hook, which compiles it with
`parse_channel_rule(rule, tokenizer)` to the native RequiredChannel, and its
grammar through `tool_call_grammar`.

Importing `atom.constraint_parser` (`ChannelRule`, `parse_channel_rule`)
requires no torch, Triton, AITER or llguidance. Format modules such as the
Kimi-K3 tool parser import the tool parser and registry stack, which does
require torch. The Triton kernels are imported when the channel runs.

## Sampling boundary

A request carries at most one rule, a grammar (JSON Schema or strict tool
arguments) or a required channel; a `presence_penalty` may come with either.
Request admission
(`InputOutputProcessor.preprocess_fanout`, which `LLMEngine.add_request`, the
API server and atomesh call) rejects a request with both a grammar and a
required channel. The OpenAI server
already clears `required_channel` whenever it configures a grammar.

`TokenConstraints` in `atom/model_engine/token_constraints.py` is the one
boundary the runner calls; each step it handles a batch's grammar, channel and
presence-penalty rows:

- `prepare_token_constraints`: after deferred input assembly and before
  forward, snapshots the grammar's draft ids so their D2H overlaps the forward.
- `TokenConstraints.apply`: the presence penalty and then the grammar edit the
  source logits; the step then gathers the bonus and target rows (speculative
  decode) and the channel masks those, in place. Unconstrained batches return
  None before any allocation.
- `needs_constraint_snapshot`: whether logprob capture must copy the logits
  before they are masked. Grammar masks and the presence penalty edit the
  source; a plain channel decode masks aliased source logits, while speculative
  channel masks work on copies.
- `TokenConstraints.commit`: after the token broadcast, receives the accepted
  output ids (never proposed drafts). The channel advances first and, for a
  deferred step, returns its states as the GPU carry; then the grammar commits
  and the presence penalty records the emitted ids.
- `constraint_errors`: the grammar's request errors, delivered with the step
  output.
- `release_token_constraints`: releases the grammar's per-request state and the
  presence penalty's table row; the channel's device tables are server
  resources and its carried states belong to `tokenIDProcessor`'s output/batch
  lifecycle. Release requests are sent for structured (grammar) requests and
  requests with a presence penalty.

A verify head's token ids (`VerifyHead.samples_step`) and the gfx1250 step override decline when
`has_constraints(batch)` is true. RapidServe prefill applies the channel
directly (`apply_required_channel`); structured output is refused there.

The presence penalty (`atom/model_ops/presence_penalty.py`, kernels in
`triton_presence_penalty.py`) follows vLLM: a request's logits lose the penalty
on every token it has generated, not its prompt, and a verification row also
counts the drafts before it. Each request with a penalty holds a row of a bit
table from its first sampled step to its release or preemption; a resumed
request marks its earlier output again (`ScheduledBatch.presence_resumed`).
The table has two rows per `max_num_seqs` and is allocated before the memory
profile. `prepare_sample` stages each logits row's table row, penalty and
draft count in the `sampling` publication.
Admission (`require_local_sampling` in `sampling_params.py`, shared with
structured output) refuses RapidServe and P/D engines and requests, whose first
token never reaches this engine's `postprocess`, and pipeline parallelism; a KV
offload tier (LMCache) keeps the penalty.

Grammar release uses asynchronous copy buffer leases: a release must not clear
or reuse a still-in-flight buffer.
