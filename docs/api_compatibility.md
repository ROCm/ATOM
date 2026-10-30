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

## Validation

`tests/entrypoints/test_reasoning_controls.py` records the precedence table,
template-toggle differences, invalid values, and non-mutating merges.
`tests/entrypoints/test_api_compatibility.py` tests actual ASGI routes with a mock
engine, including cross-endpoint and non-V4.1 controls, usage framing, stream
cleanup, and terminal events. The existing V4.1 protocol suite can also use the
official encoder by setting `ATOM_DSV41_MODEL` to a local checkpoint directory;
the reference fixture verifies the encoder hash. These are CPU protocol tests,
not GPU generation or model-quality qualification.

Contract references:

- [DeepSeek Chat](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek Responses compatibility](https://api-docs.deepseek.com/guides/responses_api)
- [DeepSeek Messages compatibility](https://api-docs.deepseek.com/guides/anthropic_api/)
