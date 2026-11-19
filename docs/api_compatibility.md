# Python frontend compatibility rules

ATOM's Chat, Responses, and Anthropic Messages endpoints share model control
resolution but retain their own request and streaming contracts. This document
describes the text API changes; image handling and beta endpoints are outside
this change.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| Endpoint | Read protocol fields and format protocol-specific errors and events |
| `reasoning_controls.py` | Resolve explicit controls and merge them into a copy of template settings |
| Encoder adapter | Supply model effort normalization and activation rules; render the prompt using the resolved template toggle |
| `api_compatibility.py` | Select frontend usage framing, Chat tool-choice restrictions, and Responses ignored parameters |
| Responses emitter | Emit incremental content and exactly one completed, incomplete, or failed terminal event |

The frontend currently selects DeepSeek compatibility for the checkpoint-selected
`encoding_dsv41` adapter. That selection is centralized, does not use the served
model alias, and does not put HTTP behavior inside the encoder. Other adapters
retain the existing ATOM frontend policy. This is not a claim that one provider's
contract applies to every model or every OpenAI-compatible service.

## Reasoning controls

Chat reads `thinking` and `reasoning_effort`; Responses reads `reasoning.effort`.
The DeepSeek-compatible Messages endpoint reads `thinking` and
`output_config.effort`. Its `thinking.budget_tokens` remains ignored.

For the V4.1 adapter, `minimal` maps to `low`, and `medium`/`xhigh` map to `high`.
Supported string values are `none`, `low`, `high`, and `max`; invalid values fail
before generation. ATOM also retains its integer 1–100 extension.

| Explicit controls | V4.1 Chat / Responses behavior |
| --- | --- |
| No controls | Preserve template defaults |
| Integer effort only | Preserve thinking toggle; set effort |
| `low` / `high` / `max` or their aliases | Enable thinking unless explicitly disabled |
| `none` | Disable thinking and remove inherited effort |
| Chat `thinking.type=disabled` with `high` | Disable thinking and remove effort |
| Chat `thinking.type=enabled` with `none` | Disable thinking and remove effort |
| Chat `thinking.effort` and `reasoning_effort` both supplied | Nested effort wins the effort value; top-level `none` still disables thinking |

The conflict precedence above preserves ATOM's existing explicit-disable behavior;
it is an intentional compatibility choice, not a claim that the hosted API
documents every conflicting-input combination. Explicit controls are applied
after server defaults and Chat's `chat_template_kwargs`. Integer efforts never
implicitly enable thinking on any of these paths.

Messages uses its separate thinking switch: output effort adjusts the budget
without implicitly opting into thinking. An explicit disabled state, including
the supported `none` effort extension, removes effort. With no explicit switch,
the existing template default and Messages reasoning-visibility policy remain.

Other adapters retain the previous behavior: Chat does not enable thinking solely
because an effort is present, and non-adapted Responses continues to pass its
effort value through to the template. Their supported values are not rewritten
using V4.1 aliases.

## Wire contracts

- Responses validates a supplied model name against the served alias before
  rendering or enqueueing, using the same rule as Chat (including trailing-slash
  normalization). Unknown names return HTTP 400 with an OpenAI `error` object
  instead of silently using the loaded model; omitted/null model keeps the
  existing single-model default.
- Chat validates `stream_options` together with `stream=true` and a boolean
  `include_usage`. DeepSeek-compatible Chat includes usage on the last finish
  chunk; other adapters retain the separate usage-only chunk. Chat retains
  `[DONE]` in both cases.
- The DeepSeek Chat contract rejects `required` or named tool selection when
  effective thinking is enabled. It is a Chat restriction; it is not propagated
  to Responses or Messages.
- DeepSeek-compatible Responses strips the explicitly documented unsupported
  top-level parameters before schema validation, including metadata and stateful
  parameters. It returns `store=false`, `previous_response_id=null`, and
  `parallel_tool_calls=true`. Unknown extension fields are not blanket-stripped.
  Other adapters retain the existing errors for unsupported stateful requests.
- Python Responses streams emit reasoning deltas immediately. Normal completion,
  token-limit truncation, and failure terminate with `response.completed`,
  `response.incomplete`, and `response.failed`, respectively, without `[DONE]`.
  This event lifecycle is shared independently of the model adapter.

## Render endpoints

`POST /v1/chat/completions/render` and `POST /v1/completions/render` take the
body of `/v1/chat/completions` and `/v1/completions` and return the prompt
token IDs those routes run for it, without running the model. A router that
places requests by their prompt, such as atomesh's `cache_aware --kv-events`,
reads the engine's tokens there instead of rendering chats itself. They follow
vLLM's [render endpoints](https://docs.vllm.ai/en/stable/serving/online_serving/renderer/),
with a subset of its response's fields:

- chat render returns one object, `{"request_id", "model", "token_ids"}`;
- completions render returns a list of them, one per prompt. ATOM takes one
  prompt per request, so the list has one item.

Chat render resolves the request's template kwargs and renders its prompt with
the chat route's own code, then tokenizes the prompt as the engine does
(`encode_prompt`). A request's `prompt_token_ids` come back as sent.

Each endpoint makes the checks its inference route makes before it hands a
request to the engine, but for the two below, with the route's code and in its
order, and refuses what they refuse with the route's 400, in the same
`{"error": ...}` form and logged as the route logs it. That includes:

- one for another model;
- a chat with no messages, an untyped tool, a tool choice its reasoning mode
  does not allow, or a malformed or unsafe `response_format`;
- fields the route takes only together or apart, such as `top_logprobs`
  without `logprobs`, `stream_options` without `stream`, or `return_token_ids`
  with `stream`;
- a `max_tokens` below 1 or above the deployment's output cap;
- chat `logprobs` on a tokenizer they cannot be read from: one whose decoder is
  not ByteLevel, or that cleans up whitespace in decoding, as Llama-3's does;
- a `response_format` the deployment cannot hold the output to: one sent with
  `stop` or `ignore_eos`, with tools its tool parser does not write as DSML, or
  under a framed reasoning dialect such as Kimi-K3's, and any on a deployment
  that cannot run a grammar (pipeline parallelism, P/D, rapidserve, relaxed MTP,
  synthetic acceptance, or speculative decoding with `logprobs`);
- a prompt that does not fit `max_model_len` with the request's `max_tokens`,
  or leaves no room for any output when `max_tokens` is left to the server.

Render leaves two refusals to the route. It takes a prompt larger than the KV
cache can hold: a worker sizes its cache from the memory free once its weights
load, so the workers serving a model can differ in it, and a router may render
a request on another worker than the one that serves it. And it prepares no
grammar, so what preparing one refuses gets past it: a `response_format` whose
grammar does not compile, and any on a deployment that cannot prepare grammars,
such as one without the `atom[structured-output]` extra, with a tokenizer that
cannot be serialized, or whose workers lack the preparation RPC, differ in
their structured-output configuration, or run out of time preparing it. Without
the extra, a schema that uses `pattern` or `patternProperties`, directly or
through a `$ref`, still fails render with the route's 500: checking its
expressions needs `re2`. Image input is not rendered yet and gets a 501; the
`/beta` routes have no render endpoint.

## Validation

`tests/entrypoints/test_reasoning_controls.py` records the precedence table,
template-toggle differences, invalid values, and non-mutating merges.
`tests/entrypoints/test_api_compatibility.py` tests actual ASGI routes with a mock
engine, including cross-endpoint and non-V4.1 controls, usage framing, stream
cleanup, and terminal events. `tests/entrypoints/test_render_endpoints.py`
checks that the render endpoints return the IDs their inference routes give
the engine, through the V4.1 adapter and through a Jinja template, and refuse
what those routes refuse. The existing V4.1 protocol suite can also use the
official encoder by setting `ATOM_DSV41_MODEL` to a local checkpoint directory;
the reference fixture verifies the encoder hash. These are CPU protocol tests,
not GPU generation or model-quality qualification.

Contract references:

- [DeepSeek Chat](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek Responses compatibility](https://api-docs.deepseek.com/guides/responses_api)
- [DeepSeek Messages compatibility](https://api-docs.deepseek.com/guides/anthropic_api/)
