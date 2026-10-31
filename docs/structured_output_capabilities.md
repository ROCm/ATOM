# Structured output and deployment discovery

The native Chat and Responses endpoints support JSON output constraints through
the optional `atom[structured-output]` dependency. This is separate from the
SGLang plugin's structured-output implementation.

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

Schemas are checked before enqueue and compiled with llguidance 1.9.1 in strict mode. Only local
references are allowed. Unsupported grammar constructs cause a request error;
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
at 256 branches or arrays. General regex negation, unbounded `uniqueItems`, and
other predicates without an exact lowering remain unsupported and fail before
generation. A 400 for
an unsupported schema is not positive feature support. Client `x-guidance`
annotations cannot enable lenient compilation or change JSON separators.

The initial XGrammar-based candidate silently ignored some constraints and
returned 500 only at final validation. This revision replaces that behavior;
it does not claim complete coverage of the JSON Schema standard.

Constraints apply to the final answer; reasoning remains unconstrained until its
configured end marker. Target logits are masked before sampling. For speculative
decoding each verification row uses the grammar state after its proposed draft
prefix; rejected drafts never advance the committed state. The implementation
bridges deferred output delivery and reconstructs grammar state from committed
history on replay. No target-model forward or AITER operator is added.

Chat can combine constraints with `logprobs` and `top_logprobs`. Probabilities
still describe the raw target distribution: only requested target/bonus logit
rows are preserved before grammar masking. This extra snapshot is made only
when constraints and probabilities are both enabled. The ordinary path does not
copy logits for this feature.

With a DSML parser, normal function tools can accompany Chat `response_format`
or Responses `text.format`. The grammar admits the parser's canonical function
envelope or a schema-conforming answer. Required/named choices select the tool
branch; `tool_choice="none"` selects JSON text. Function-only turns have no
response text to validate; subsequent text answers remain constrained. This is
not beta strict-tool argument validation. Other tool parser formats, P/D handoff,
multimodal input, custom stop strings, and `ignore_eos` remain unsupported with
constraints. Synthetic forced acceptance is also rejected. No image
implementation is included.

Committed grammar states are retained for the immediately preceding batch and
advanced only by newly committed tokens. Each speculative request clones that
state once and advances its draft branch incrementally; rejected drafts cannot
mutate the committed state. Prefill/replay, history mismatches, or requests absent
from the preceding batch reconstruct from the scheduler's authoritative history.
An ordinary batch skips grammar processing and releases any prior states once;
it does not allocate masks or add CPU/GPU synchronization for this feature.

Constrained requests fill reusable CPU bitmask rows directly through llguidance.
On GPU, a Triton kernel reads packed 32-bit words and writes negative infinity
only to forbidden logits in the selected target/bonus rows. It does not expand
a vocabulary-sized Boolean tensor or gather and scatter complete logit rows.
CPU execution retains a reference implementation. Pinned host and device mask
buffers grow on demand and are reused; events protect host writes from an
unfinished copy and device reuse from an unfinished kernel, including when the
runner changes streams. No buffers or kernel imports are needed for a request
batch without constraints. Speculative matcher cloning and committed-state
semantics are unchanged by this mask optimization.

Enabled requests still copy masks and committed tokens between CPU and GPU and
perform host grammar work. There is no zero-overhead claim. Performance results
must distinguish enabled-request overhead from disabled-request regression and
compare matching generated outputs. Production and broader-topology qualification
remain separate tasks.

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
