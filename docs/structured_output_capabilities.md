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
negation, negation over finite enums, conditions on finite object discriminators,
and `uniqueItems` over finite item domains. Finite expansions are capped at 256
branches or arrays. General negation, unbounded discriminator domains and
unbounded `uniqueItems`, and `uniqueItems` combined with `prefixItems` remain unsupported and fail before generation. A 400 for
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

This initial draft rejects structured output combined with tools, P/D handoff,
multimodal input, logprobs, custom stop strings, or `ignore_eos`. Synthetic forced
acceptance is also rejected. No image implementation is included.

The initial implementation replays grammar history and copies masks and committed
tokens between CPU and GPU. This adds latency and host work; long outputs can
accumulate substantial grammar replay cost. There is no zero-overhead claim.
Performance qualification and broader topology coverage remain separate tasks.

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
