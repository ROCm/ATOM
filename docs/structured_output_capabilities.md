# Structured output and deployment discovery

The native Chat and Responses endpoints support JSON output constraints through
the optional `atom[structured-output]` dependency. This is separate from the
SGLang plugin's structured-output implementation.

## Code organization

The structured-output implementation lives in `atom/structured_output/`, following the
manager/backend separation used by vLLM's `vllm/v1/structured_output/` package.
ATOM currently has one backend, so it does not introduce a backend registry or
multi-backend interfaces.

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Lightweight entry points used by the API preflight and model runner. |
| `manager.py` | Worker admission, cached compilation, committed and speculative grammar state, and prepare/apply/commit ordering. |
| `backend_guidance.py` | llguidance compiler and matcher adapter. |
| `dsml_grammar.py` | DSML tool-call grammar for beta strict function tools. |
| `schema.py` | Offline reference resolution, bounded RE2 validation, exact constraint lowering and shared expansion budgets. |
| `validation.py` | Chat/Responses format and final answer validation, tool/reasoning adapters, and admission-time worker RPC/cache. |
| `utils.py` | Reusable packed mask buffers, H2D kernel dispatch, and owned asynchronous token D2H snapshots. |

API handlers and the model runner import from this single package. Protocol
adapters import the existing tool/reasoning parsers and SSE helpers only inside
the functions that use them; the execution manager and backend do not import the
API adapters. Importing the package does not load llguidance or allocate
mask/transfer resources. The GPU kernel remains in
`atom/model_ops/triton_structured_output.py`.

## Request formats

Chat accepts `response_format: {"type": "json_object"}` or:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "schema": {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": false
      }
    }
  }
}
```

Responses uses the same schema under
`text.format: {"type": "json_schema", "name": "answer", "schema": {...}}`.
`text.format: {"type": "json_object"}` is also supported.

Schemas are checked before enqueue and compiled with llguidance 1.9.1 in strict mode.
Admission uses a worker RPC: each execution rank checks the API tokenizer/stop-token
fingerprint and configured vocabulary against its target LM head, then compiles
and caches the grammar used by decoding. Any rank failure rejects the request
before generation (HTTP 400). Supported deployments need collective worker RPC
and a `ParallelLMHead` target projection; unsupported topologies/projections are
rejected at admission. Cold compilation runs in a worker background executor, polled by admission RPCs.
The ready-cache lock is never held across RPCs. Before enqueue, each request gets
its own matcher, which retains the compiled grammar even after cache eviction.
Compilation and tokenizer setup still consume CPU/GIL time; background execution
does not imply zero interference. Repeated schema/tool/reasoning configurations
use bounded compilation caches. Ordinary requests do not
initialize this path or perform these RPCs.

Compilation-only preflight does not allocate request matchers. Binding happens at
submission after the API has validated the tokenized context budget. Admission,
completion, cancellation and failed submission share request ownership in the
core manager. Cleanup reaches every DP replica used by collective preparation,
including replicas that did not generate the request. Completion cleanup uses a
fire-and-forget utility command so output receiver threads never wait for their
own RPC replies. A cancelled API submission is settled before aborting its IDs,
preventing an executor task from enqueueing after cancellation cleanup.


Only local references are allowed, including references reached transitively
through a local pointer into an otherwise literal `const` or `enum` value.
Every runtime validator also disables external resource retrieval. Schema regexes
use the non-backtracking RE2 evaluator from the optional `google-re2` dependency
for candidate filtering and final validation, including property-name matching.
Each compiled regex has a 2 MiB memory limit; patterns are limited to 262,144
characters and the cache retains at most 32 compiled expressions. Unsupported
RE2 syntax, such as backreferences or lookarounds, is rejected before enqueue.
Nested dialect declarations cannot switch validation back to Python's
backtracking regex engine. Only JSON Schema draft 2020-12 is accepted.
Unsupported grammar constructs cause a request error;
compilation is not a claim of complete JSON Schema vocabulary support. A final
JSON Schema validator additionally checks the visible answer before a successful
completion is returned. A validation failure becomes an error, never a successful
invalid JSON result. Streaming deltas are provisional until the terminal event.
An output token cap can still produce partial JSON: Chat reports `length` and
Responses reports `incomplete` in that case.

Supported intersections and numeric `multipleOf` constraints are compiled into
the grammar. An exact normalization pass additionally supports numeric/Boolean/null
negation, literal string negation, length/cardinality bounds, required/property
negation, and conditions whose guards can be complemented exactly. Those guards
do not require a finite discriminator domain. More complex guards may use finite
enumeration. `uniqueItems` supports finite item and prefix-item domains, including
prefix values outside the trailing `items` domain. Finite expansions are capped
by one shared budget per schema: 256 generated branch/enum/array entries, 4,096
finite-domain candidate evaluations, and 262,144 generated regex characters.
Nested and sibling schemas share these limits; capacity is reserved before
constructing each expansion, and exhaustion cannot enter a conditional fallback.
Existing input branches are not charged as generated entries. Literal-string
negation additionally remains capped at 256 characters before constructing its
prefix regex. General regex negation, unbounded `uniqueItems`, and
other predicates without an exact lowering remain unsupported and fail before
generation. A 400 for
an unsupported schema is not positive feature support. Client `x-guidance`
annotations cannot enable lenient compilation or change JSON separators.

The initial XGrammar-based candidate silently ignored some constraints and
returned 500 only at final validation. This revision replaces that behavior;
it does not claim complete coverage of the JSON Schema standard.

Constraints apply to the final answer. Grammar activation uses the same rendered
prompt or token-ID prefix and reasoning-channel state as response parsing, rather
than assuming a missing reasoning toggle means thinking is enabled. If generation
starts inside an inline reasoning block, it remains unconstrained until its
configured end-marker text; the grammar recognizes that text across alternative
tokenizations and inside speculative prefixes. Otherwise JSON is constrained immediately. Framed/multiple-
transition reasoning dialects such as Kimi-K3 are rejected before worker admission
until their complete channel framing is supported. Target logits are masked before sampling. For speculative
decoding each verification row uses the grammar state after its proposed draft
prefix; rejected drafts never advance the committed state. The implementation
bridges deferred output delivery with persistent request state and incremental
acknowledgements. Replay rolls back only tokens not yet acknowledged by the scheduler. No target-model forward or AITER operator is added.

Once the answer grammar is active, inline reasoning markers inside JSON strings
remain literal answer text. The grammar tokenizer also preserves HF added tokens
marked `special=False` as visible bytes: llguidance 1.9.1 otherwise classifies
tokens such as `<think>` as specials and can disagree between masking and token
consumption. This adapter preserves model token IDs and EOS and is built once
when a worker first initializes structured output; the API/model tokenizer is
unchanged. Tool-only Responses results skip answer validation
after conversion to `function_call`, `custom_tool_call`, or `tool_search_call`;
any accompanying visible answer is still validated. Structured requests are
rejected at both API and worker admission when `ATOM_ENABLE_RELAXED_MTP=1`, since
relaxed acceptance can otherwise accept a draft with zero grammar-masked target
probability. Unconstrained requests retain their existing behavior.

Structured output preserves the non-speculative engine's scalar logprob
semantics: normalization uses the raw model distribution before grammar and
required-channel masks. Chat token-logprob responses, top candidates, and
DSpark speculative logprobs are described in [Chat token logprobs](chat_logprobs.md).
The combined Schema + speculative decoding + logprobs mode is not supported and
is rejected before enqueue. Disable speculative decoding or logprobs when using Schema.
The ordinary path does not copy logits for this feature.

With a DSML parser, normal function tools can accompany Chat `response_format`
or Responses `text.format`. The grammar admits the parser's canonical function
envelope or a schema-conforming answer. Required/named choices select the tool
branch; `tool_choice="none"` selects JSON text. Function-only turns have no
response text to validate; subsequent text answers remain constrained. Other
tool parser formats, pipeline parallelism, P/D handoff,
multimodal input, custom stop strings, and `ignore_eos` remain unsupported with
constraints. Synthetic forced acceptance is also rejected. No image
implementation is included.

Cleanup follows request ownership rather than a shared frontend ID registry.
Finished sequences carry their Schema metadata, so the output path releases
request state on every replica. Waiting/running cancellations use that same
terminal path; a submission failure releases its locally held sequence list.
Worker release is idempotent, and ordinary requests send no grammar-cleanup
message. There is no additional request-registry lock; the existing transport
synchronization remains in place.

Committed grammar states live until request completion, rejection, or cancellation,
including across interleaved prefills and ordinary batches. The scheduler sends
schema configuration once and then only newly acknowledged output tokens, excluding
deferred placeholders. Speculative mask construction temporarily consumes draft
tokens and rolls them back in a `finally` block; it does not clone full matcher
history every step. Terminated/unreachable branches reuse a precomputed stop mask.
A grammar failure terminates the affected request with a structured-output error;
other requests retain their matcher state. Ordinary batches do not initialize
this machinery or discard state belonging to requests scheduled later.

Constrained requests fill reusable CPU bitmask rows directly through llguidance.
On GPU, a Triton kernel reads packed 32-bit words and writes negative infinity
only to forbidden logits in the selected target/bonus rows. It does not expand
a vocabulary-sized Boolean tensor or gather and scatter complete logit rows.
CPU execution retains a reference implementation. Pinned host and device mask
buffers grow on demand and are reused; events protect host writes from an
unfinished copy and device reuse from an unfinished kernel, including when the
runner changes streams. No buffers or kernel imports are needed for a request
batch without constraints. Speculative rows within a request remain sequential because each depends on its
draft prefix. This implementation does not parallelize independent CPU matchers.

Deferred output remains asynchronous. Target/bonus row indices use their CPU
mirrors, not GPU `tolist()` calls. After device-side input assembly and before
target forward, the runner snapshots only constrained requests' draft tokens and
copies them to pinned host memory on a separate stream. It consumes the snapshot
after submitting the forward. Sampled tokens are likewise copied asynchronously;
the next step advances committed grammar state after submitting its own forward.
Neither publication waits for D2H. Consumption waits only for the corresponding
copy event, so residual waits are possible if that copy has not finished.
Owned snapshots and request IDs protect against input-buffer reuse, batch
reordering, cancellation, and replay. Ordinary output delivery keeps its existing
deferred queue. The scheduler-to-worker protocol carries token deltas rather than
copying the full generated history every step.

Enabled requests still copy masks and committed tokens between CPU and GPU and
perform host grammar work. There is no zero-overhead claim. Performance results
must distinguish enabled-request overhead from disabled-request regression and
compare matching generated outputs. Production and broader-topology qualification
remain separate tasks.

## Strict function tools (beta)

On `/beta/chat/completions`, a function with `"strict": true` has its
arguments constrained by its `parameters` schema, with or without
`response_format`. `/v1/chat/completions` keeps ignoring `strict`.

Strict tools are constrained decoding, so a request with them is refused with
HTTP 400 wherever `response_format` is (see above): with a tool parser other
than DSML, pipeline parallelism, P/D handoff, images, custom stop strings,
`ignore_eos`, relaxed MTP, or logprobs under speculative decoding. Without
`response_format`, the error starts with `strict function tools:`, so the
client can tell which part of its request asked for the grammar.

The schema is checked before enqueue and refused with HTTP 400 where it breaks
DeepSeek's rules or asks for what the grammar cannot hold:

- every object lists all its properties in `required` and sets
  `additionalProperties: false`, and `patternProperties` is refused, so the
  arguments carry exactly the declared keys;
- every schema gives its type (or a `$ref`, `anyOf`, `oneOf`, `allOf`, `enum`
  or `const`), and every array its `items`;
- `minLength`, `maxLength`, `minItems` and `maxItems` are refused, and a
  `format` must be one of DeepSeek's five: `email`, `hostname`, `ipv4`, `ipv6`,
  `uuid`;
- the top of `parameters` holds only its properties, definitions and
  annotations, and neither the tool's name nor a parameter's contains `"` or
  `<`: the DSML tags carry them unescaped, so the parser would read a name up
  to its quote, or a tag inside it;
- definitions live at the top, under DeepSeek's `$def` (read as `$defs`),
  `$defs` or `definitions`, and every `$ref` points into them, at a schema
  held to these same rules (not into a `default`, say). Each parameter
  is compiled on its own, where `#` would name the parameter; `$id`, `$anchor`
  and dynamic references are refused for the same reason. A draft-07
  `$schema`, as OpenAI's zod helpers write, is read as 2020-12.

A strict tool is checked off the event loop. The last 256 distinct tools under
64 KiB are remembered, so an agent resending its tools is not checked again.

The grammar is the DSML envelope itself (`dsml_grammar.py`): parameters appear
once each, in declared order. A string is written raw (`string="true"`), as
the encoder writes it: a plain string, a string `enum`/`const` (behind a
`$ref` too), and the string side of a nullable one. Every other value,
patterned and formatted strings included, is a `string="false"` JSON value
under its own JSON grammar, carrying the definitions it references. The answer
and every parameter share one expansion budget, and the grammar may not grow
past twice the size of the tool schemas (or 1 MiB). Whitespace around the
answer and the calls is free; inside a call it is allowed only where the
encoder writes line breaks. Neither the free text before the calls nor a raw
value may contain what the tool parser reads as the start of a call, the end
of a value or another call, with or without the DSML marker, so the parser
reads them back as written. A non-strict function's body is free but for
another call or the call's end, in either spelling. A string inside a JSON
value is not held to this: the parser cuts the value short at such a tag, as
it does on `/v1`. The cut value usually fails the validation below; one its
schema still takes (a nullable string written as JSON, say) goes out cut
short.

Parsed arguments are validated against their schema again before success is
reported (formats are the grammar's to enforce), and every call must name one
of the request's functions. They must be standard JSON too: the parser reads
a number past a double's range as infinity, which no JSON can carry. A
streamed reply holds its frames from the first tool call on until then, so a
call that fails is never sent: the stream ends with a
`structured_output_error`, as a non-streamed reply fails. Nor is a call whose
choice never finished, so was never checked: the stream ended in an error
first, and only that error goes out. The reply is read literally, as under
`response_format`. The thinking-mode refusal of
required/named `tool_choice` still applies.

## Model discovery

`/v1/models` reports deployment limits rather than copying a hosted model card:

- `context_window` is bounded by configured context and checkpoint context.
- `max_output_tokens` reflects the accepted deployment output limit; a request's
  prompt must still fit within the total context budget.
- DSpark advertises text input, even if the checkpoint also declares vision.
- Text output and the configured model's supported effort levels are exposed.
- Unknown capabilities are omitted rather than inferred from a served alias.

The V4.1 adapter advertises `low`, `high`, and `max` thinking effort. The output
cap is also bounded by its 393,216-token API limit. Returning model metadata does
not qualify inference at that context length; context retrieval and capacity
need their own acceptance tests.

## Flash and web search

The official Responses contract currently documents hosted `web_search` for
DeepSeek V4 Pro. The Flash compatibility test requires that the unsupported
built-in tool be ignored while an ordinary visible answer is returned. It does
not require executing a search. ATOM does not add an external search provider,
search credentials, fabricated search results, or citations in this change.

References: [Responses contract](https://api-docs.deepseek.com/api/create-response/)
and [model discovery](https://api-docs.deepseek.com/api/list-models/).

When an OpenAI output cap is omitted (or null), its default is bounded by the
deployment limit and, after tokenization, the remaining prompt context budget.
Messages uses the same behavior for an omitted cap. An explicitly supplied cap
is validated without silently reducing it. A prompt that leaves no room for
output is rejected. Model-specific discovery values and effort levels come
from the same API contract as request adaptation.
