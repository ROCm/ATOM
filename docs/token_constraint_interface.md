# Constraint parser and execution adapter contract

Tool output parsing is done by the shared ToolCallParser stream engine. A model
format declares its wire syntax and constraint rules; the common
ConstraintParser compiles those declarations through registered rule adapters.
K3 declares its channel rule through its `required_call_channel` hook and its
grammar through `tool_call_grammar`. The compiled values are the native
RequiredChannel and grammar objects.

`ConstraintParser.register(rule_type, adapter)` adds a rule without changing
parser dispatch. `parse(rule, context)` returns the native compiled value.
Provider classes and state never cross Sequence IPC; adapters read the native
scheduled fields at the engine boundary.
Importing `atom.constraint_parser` (rule declarations, `ConstraintParser`)
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
- `hard_requests(batch)`: distinct affected request indices in the
  scheduled batch, not source-logit indices, request IDs, or packed draft rows.
  This declares hard-mask participation for per-request intersection. An adapter
  must not claim unrelated requests. The channel adapter resolves active states
  from GPU carry and adds its participation during application.
- `masks_source(metadata)`: whether logprob capture needs the original source
  distribution. This is independent of mask_stage; plain channel decode masks
  alias source logits, while speculative channel copies do not.
- `prepare(runner, batch, metadata)`: deferred input work before forward.
- `apply(step)`: apply the native backend mask to the declared stage.
  Source adapters use step.source; sampler adapters call step.gather and use
  step.logits/step.target. Adapters preserve existing hard masks. They may report
  failure tensors; the executor combines failures by OR, never overwrites them.
- `commit_order`: explicit lifecycle order independent of mask stage. The
  channel GPU advance runs at 0 and grammar commit at 10.
- `commit(step, tokens, batch, deferred, failure)`: receives synchronized
  accepted output tokens after broadcasts, including rejection sentinels. It
  never commits merely proposed draft IDs. The current GPU carry envelope allows
  one adapter to return a native carry tensor; all other adapters return None.
- `errors(runner)`: backend request errors for output delivery, or None.
- `release(runner, request_ids)`: release only owned request resources. Release
  requests are sent for structured (grammar) requests only; stateless channels
  acquire no cleanup RPC.
- `release` failures are isolated per adapter: `release_token_constraints`
  releases every registered adapter, then re-raises the first error.

The executor counts declared hard-mask participation per request. The channel
FSM preserves a forbidden forced token when that same request has another hard
rule, even if the other adapter was registered later. Non-intersection channel
requests use the forced-inf repair. After all masks, intersections
are checked without relaxing either rule. The safe ID is GPU-only and failure
travels through the output-copy pipeline; safe IDs are retained for GPU
carry, and error sentinels are only applied to delivered output.

For speculative decoding, request participation is expanded using inclusive
cu_num_draft_tokens; zero/ragged segments, target rows and bonus rows follow the
same per-request mapping as the verifier. A rejected future conflicting draft
must not fail the current accepted output; this holds for both target-row and
bonus-row failure reports (a bonus-row failure only applies to the bonus position
and only when that position was emitted). Source and sampler row domains must
not be interchanged.

Preparation, application, accepted commit, release, logprob capture, error output
and sampling-bypass eligibility all query this registry. Unconstrained requests
return before steps or provider instances are allocated. A new adapter
must not require a ModelRunner special case to disable head IDs or step override.
ModelRunner centrally rejects those bypasses when has_constraints(batch) is true.

A single request carrying both a forced K3 channel token and a
JSON Schema/grammar rule has its hard masks intersected without relaxing
either rule. If the forced token is forbidden by the grammar, the request fails
immediately with "no valid continuation". The OpenAI server clears
`required_channel` whenever it configures a grammar, so only direct engine API
use can reach this combination.

Grammar release uses asynchronous copy buffer leases. Channel
immutable device tables are server resources; tokenIDProcessor owns GPU carried
request states through its output/batch lifecycle. Adapter release must
not clear another component's buffers or reuse a still-in-flight buffer.
