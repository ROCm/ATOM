# Constraint parser and execution adapter contract

Tool output parsing is done by the shared ToolCallParser stream engine. A model
format declares its wire syntax and constraint rules. K3 declares its channel
rule through its `required_call_channel` hook, which compiles it with
`parse_channel_rule(rule, tokenizer)` to the native RequiredChannel, and its
grammar through `tool_call_grammar`.

Provider classes and state never cross Sequence IPC; adapters read the native
scheduled fields at the engine boundary.
Importing `atom.constraint_parser` (`ChannelRule`, `parse_channel_rule`)
requires no torch, Triton, AITER or llguidance. Format modules such as the Kimi-K3 tool parser import the tool parser
and registry stack, which does require torch. Backend imports happen when the
corresponding adapter runs.

## Runtime adapter contract

Register adapters with `register_constraint_provider`. An adapter declares:

- `backend`: unique name.
- `enabled(batch)`: cheap host query, without allocations or backend imports.
- `mask_stage`: source logits = 0; gathered sampler logits = 1. The registry
  orders masks by this stage, preserving registration order within a stage.
  A source adapter must finish before any sampler adapter can gather rows.
- `masks_source(metadata)`: whether logprob capture needs the original source
  distribution. This is independent of mask_stage; plain channel decode masks
  alias source logits, while speculative channel copies do not.
- `prepare(runner, batch, metadata)`: deferred input work before forward.
- `apply(step)`: apply the native backend mask to the declared stage.
  Source adapters use step.source; sampler adapters call step.gather and use
  step.logits/step.target. Adapters preserve existing masks.
- `commit_order`: explicit lifecycle order independent of mask stage. The
  channel GPU advance runs at 0 and grammar commit at 10.
- `commit(step, tokens, batch, deferred)`: receives synchronized
  accepted output tokens after broadcasts, including rejection sentinels. It
  never commits merely proposed draft IDs. The current GPU carry envelope allows
  one adapter to return a native carry tensor; all other adapters return None.
- `errors(runner)`: backend request errors for output delivery, or None.
- `release(runner, request_ids)`: release only owned request resources. Release
  requests are sent for structured (grammar) requests only; stateless channels
  acquire no cleanup RPC.
- `release` failures are isolated per adapter: `release_token_constraints`
  releases every registered adapter, then re-raises the first error.

Preparation, application, accepted commit, release, logprob capture, error output
and sampling-bypass eligibility all query this registry. Unconstrained requests
return before steps or provider instances are allocated. A new adapter
must not require a ModelRunner special case to disable head IDs or step override:
both decline when has_constraints(batch) is true.

A request carries at most one rule: `LLMEngine` admission rejects a request with
both a JSON Schema (or strict tool grammar) and a `required_channel`. The OpenAI
server already clears `required_channel` whenever it configures a grammar.

Grammar release uses asynchronous copy buffer leases. Channel
immutable device tables are server resources; tokenIDProcessor owns GPU carried
request states through its output/batch lifecycle. Adapter release must
not clear another component's buffers or reuse a still-in-flight buffer.
