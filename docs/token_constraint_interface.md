# Constraint parser and execution adapter contract

Tool output parsing remains in the shared ToolCallParser stream engine. A model
format declares its original wire syntax and constraint rules; the common
ConstraintParser compiles those declarations through registered rule adapters.
K3 declares its channel rule and optional tool grammar, while JSON Schema uses a
GrammarRule. The compiled native RequiredChannel and grammar object retain their
existing serialization, stop IDs, fields and backend semantics.

`ConstraintParser.register(rule_type, backend, adapter)` adds a rule without
changing parser dispatch. `parse(rule, context)` returns the native compiled
value. `bind(rule, context)` tags that value with its backend. Bindings are rebuilt
at the engine boundary from native scheduled fields; provider classes and state
never cross Sequence IPC. Rule declaration imports require no torch, Triton,
AITER or llguidance. Backend imports happen when the corresponding adapter runs.

## Runtime adapter contract

Register adapters with `register_constraint_provider`. An adapter declares:

- `backend`: unique name.
- `enabled(batch)`: cheap host query, without allocations or backend imports.
- `bind(batch)`: an enabled adapter returns a ConstraintBinding with native
  payload. Disabled adapters are not bound. Enabled-but-unbound is an error.
- `mask_stage`: source logits = 0; gathered sampler logits = 1. The registry
  orders masks by this stage, preserving registration order within a stage.
  A source adapter must finish before any sampler adapter can gather rows.
- `hard_requests(binding, batch)`: distinct affected request indices in the
  scheduled batch, not source-logit indices, request IDs, or packed draft rows.
  This declares hard-mask participation for per-request intersection. An adapter
  must not claim unrelated requests. The channel adapter resolves active states
  from GPU carry and adds its participation during application.
- `masks_source(metadata)`: whether logprob capture needs the original source
  distribution. This is independent of mask_stage; plain channel decode masks
  alias source logits, while speculative channel copies do not.
- `prepare(runner, batch, metadata, binding)`: deferred input work before forward.
- `apply(step, binding)`: apply the native backend mask to the declared stage.
  Source adapters use step.source; sampler adapters call step.gather and use
  step.logits/step.target. Adapters preserve existing hard masks. They may report
  failure tensors; the executor combines failures by OR, never overwrites them.
- `commit_order`: explicit lifecycle order independent of mask stage. The
  original channel GPU advance runs at 0 and grammar commit at 10.
- `commit(step, tokens, batch, deferred, failure, binding)`: receives synchronized
  accepted output tokens after broadcasts, including rejection sentinels. It
  never commits merely proposed draft IDs. The current GPU carry envelope allows
  one adapter to return a native carry tensor; all other adapters return None.
- `release(runner, request_ids)`: release only owned request resources. Optional
  `errors(runner)` exposes backend request errors for output delivery.

The executor counts declared hard-mask participation per request. The channel
FSM preserves a forbidden forced token when that same request has another hard
rule, even if the other adapter was registered later. Non-intersection channel
requests keep their original forced-inf repair. After all masks, intersections
are checked without relaxing either rule. The safe ID is GPU-only and failure
travels through the existing output-copy pipeline; safe IDs are retained for GPU
carry, and error sentinels are only applied to delivered output.

For speculative decoding, request participation is expanded using inclusive
cu_num_draft_tokens; zero/ragged segments, target rows and bonus rows follow the
same per-request mapping as the verifier. A rejected future conflicting draft
must not fail the current accepted output. Source and sampler row domains must
not be interchanged.

Preparation, application, accepted commit, release, logprob capture, error output
and sampling-bypass eligibility all query this registry. Unconstrained requests
return before bindings, steps or provider instances are allocated. A new adapter
must not require a ModelRunner special case to disable head IDs or step override.
ModelRunner centrally rejects those bypasses when has_constraints(batch) is true.

Grammar release keeps its original asynchronous copy buffer leases. Channel
immutable device tables are server resources; tokenIDProcessor owns GPU carried
request states through its existing output/batch lifecycle. Adapter release must
not clear another component's buffers or reuse a still-in-flight buffer.
