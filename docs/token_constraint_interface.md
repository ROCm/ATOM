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
arguments) or a required channel. Request admission
(`InputOutputProcessor.preprocess_fanout`, which `LLMEngine.add_request`, the
API server and atomesh call) rejects a request with both. The OpenAI server
already clears `required_channel` whenever it configures a grammar.

`TokenConstraints` in `atom/model_engine/token_constraints.py` is the one
boundary the runner calls; each step it handles a batch's grammar rows and
channel rows:

- `prepare_token_constraints`: after deferred input assembly and before
  forward, snapshots the grammar's draft ids so their D2H overlaps the forward.
- `TokenConstraints.apply`: the grammar masks the source logits; the step then
  gathers the bonus and target rows (speculative decode) and the channel masks
  those, in place. Unconstrained batches return None before any allocation.
- `needs_constraint_snapshot`: whether logprob capture must copy the logits
  before they are masked. Grammar masks the source; a plain channel decode masks
  aliased source logits, while speculative channel masks work on copies.
- `TokenConstraints.commit`: after the token broadcast, receives the accepted
  output ids (never proposed drafts). The channel advances first and, for a
  deferred step, returns its states as the GPU carry; then the grammar commits.
- `constraint_errors`: the grammar's request errors, delivered with the step
  output.
- `release_token_constraints`: releases the grammar's per-request state; the
  channel's device tables are server resources and its carried states belong to
  `tokenIDProcessor`'s output/batch lifecycle. Release requests are sent for
  structured (grammar) requests only.

Head-ID bypass (`_head_ids_serve`) and the gfx1250 step override decline when
`has_constraints(batch)` is true. RapidServe prefill applies the channel
directly (`apply_required_channel`); structured output is refused there.

Grammar release uses asynchronous copy buffer leases: a release must not clear
or reuse a still-in-flight buffer.
