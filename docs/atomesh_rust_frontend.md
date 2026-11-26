# ATOM standalone with a Rust frontend

ATOM standalone (`USE_ATOMESH_ENTRYPOINTS=1`) serves ATOM's OpenAI API from
the mesh router (Rust) and stops at **LLMEngine**: Rust renders the chat template, tokenizes,
detokenizes, runs the reasoning and tool-call parsers and writes the SSE;
Python takes the prompt's token ids into `LLMEngine.io_processor` and
`CoreManager` exactly as the OpenAI server does, and hands each engine step's
tokens back. LLMEngine, CoreManager and the EngineCores are unchanged.

```bash
USE_ATOMESH_ENTRYPOINTS=1 python -m atom.entrypoints.openai_server \
    --model <model> --kv-cache-dtype fp8 -tp 8 --port 8000
```

It replaces the Python standalone service (`AtomStandaloneService`), which
served the API in Python behind the same HTTP server. This page is the plan
for the frontend (scheme A in the design discussion), what is implemented
now, and what is left.

## Why

The OpenAI server's per-request work grows with the context: JSON and
pydantic, the chat template, tokenizing, detokenizing, SSE. It runs on one
event loop and one GIL (`frontend_workers.py` exists because of it). Moving
that work to Rust removes the bulk of the frontend's cost while keeping
everything below the frontend -- DP routing, back-pressure, the schedulers --
as it is. What a request carries to it that the bridge does not pass yet (P/D
transfers, grammars) is refused, not dropped (see Status).

Two ways to do it were weighed:

| | A: Rust -> LLMEngine / CoreManager (this page) | B: Rust -> EngineCore (`zmq_protocol/messages.py`) |
|---|---|---|
| Python on the data path | one admission per request (`Sequence`, pickle), one output pass per step | none |
| CoreManager's features (DP router, prefix_spread / share_cache, session affinity, back-pressure, PP, multi-node, metrics) | all of them, unchanged; P/D's `kv_transfer_params` and structured output are not carried yet (P2, P4) | re-implemented in Rust, or refused |
| Python changes | a ~200-line adapter | the Core / CoreManager protocol (pickle -> msgspec) |
| Effect on the OpenAI server | none | it shares the new protocol |
| Ceiling | the frontend-workers hub's: Python admission and output threads | the EngineCores' |

A is taken first: it gets the frontend's gain at a fraction of the change and
risk. Its Rust half is written so that B can replace only the backend (see
[Moving to B](#moving-to-b)), once measurements show the remaining Python is
the bottleneck.

## How it works

```
 client ─HTTP─► mesh ingress ─► AtomBridgeRouter
                                  │ prepare (template, tokenize)         Rust
                                  │ BridgeDispatcher.dispatch ───────────────────────┐
                                  │                                                  │ spawn_blocking
                                  │                                                  ▼ + GIL
                                  │                        TokenBridge.submit(key, token_ids, params)
                                  │                          io_processor.preprocess_fanout(token_ids)
                                  │                          CoreManager.add_request ──► EngineCores
                                  │                                                         │
                                  │                        per-seq callback: buffer (key, tokens, ...)
                                  │                        _flush_stream_batch_fn: StepSink.push(step)
                                  ▼                                                  ◄──────┘ one call per step
 client ◄─SSE── render (detokenize, parsers) ◄─ SeqEventSource ◄─ EventRegistry[key]
```

* **Keys, not sequence ids.** Rust registers a request under a key before it
  calls `submit`; Python tags every output with that key. No output can arrive
  for a request Rust does not know yet, and Python's sequence ids never cross.
* **One push per step.** A sequence's callback only appends to a thread-local
  buffer; CoreManager's `_flush_stream_batch_fn` hook pushes the whole step in
  one PyO3 call, as the frontend-workers hub does. Each output thread (one per
  DP rank) flushes its own buffer.
* **Abort on drop.** `Admission` stops a request whose client went away,
  whichever side of the submit that happens on: dropped after the submit, it
  calls `TokenBridge.abort`; dropped while the submit is still running, the
  submit aborts it as it returns. A request that finished is never aborted.
* **Errors keep their meaning.** A `ValueError` from Python (sampling values,
  `max_tokens` over the deployment cap, a context or KV pool overflow) is a
  400; anything else is a 503.
* **Admission as the OpenAI server's.** `TokenBridge` builds `SamplingParams`
  with the same defaults (`protocol.py`), caps `max_tokens` by
  `deployment_metadata`, shrinks a defaulted `max_tokens` to the context left
  and refuses what `_validate_context_length` / `_validate_pool_capacity`
  refuse. These are copies of the OpenAI server's admission for now; a later
  PR moves that admission out of FastAPI so the OpenAI routes, the gRPC
  servicer (#3673) and `TokenBridge` call one implementation.
* **Output read as the OpenAI server reads it** (`pipeline::AtomOutput`).
  Stop strings go to the engine, which ends the output where their ids
  match; like the OpenAI server, the text is not cut at them again. The
  model's end-of-sequence ids are left out of the text (the tiktoken
  decoder ignores `skip_special_tokens`).
* **One model.** The router loads the model's tokenizer (`--model-path` is
  added unless `--model-path` / `--tokenizer-path` is given) under the served
  model name, and prepares every request for it whatever `model` the client
  sent, as the OpenAI server does.
* **Parsers as ATOM picks them.** The launcher reads the tool-call format and
  the reasoning dialect off the model as ATOM's server does and hands the
  router the same choice (`--tool-call-parser`, `--reasoning-parser`, in the
  router's names: `MESH_TOOL_CALL_PARSERS`), unless the launch named one. A
  prompt that ends by opening a channel (`<think>`, K3's think or response)
  has the parser read that opener first (`ResponseContext.prompt_opener`),
  on this router only (`Pipeline::reading_prompt_opener`): gRPC routing keeps
  its parsers' own start.
* **Python templates.** A model whose chat template the router cannot render
  -- a Python message encoder (DeepSeek-V4), or a tokenizer whose template is
  code (Kimi-K3) -- has each chat prompt rendered by `TokenBridge.render_chat`
  (ATOM's `apply_chat_template`, encoded as the engine encodes a prompt); the
  router then runs the rest. `src/reasoning` adds the `kimi_k3` reasoning
  parser the `reasoning-parser` crate lacks.

### Code

| Where | What |
|---|---|
| `atom/entrypoints/atomesh/token_bridge.py` | `TokenBridge`: `submit`, `abort`, `is_healthy`, `render_chat`, the step flush; `ChatRenderer` |
| `atom/entrypoints/atomesh/server.py` | builds the bridge, picks the router's parsers, hands both to `launch_mesh` |
| `atom/mesh/src/routers/atom_bridge/mod.rs` | `AtomBridgeRouter` (`RouterTrait`), `AtomBridgeRuntime`, the one-engine planner |
| `atom/mesh/src/routers/atom_bridge/events.rs` | backend-neutral: `EventRegistry`, `SeqEventSource` (events -> `TokenChunk`), `Admission` |
| `atom/mesh/src/routers/atom_bridge/py_bridge.rs` | `StepSink` (PyO3), `BridgeDispatcher`, what a request may carry (`BridgeRequest`) |
| `atom/mesh/src/python.rs` | `launch_mesh(token_bridge=..., served_model_name=..., default_chat_template_kwargs=...)`, exports `StepSink` |
| `atom/mesh/src/reasoning/` | reasoning parsers the `reasoning-parser` crate lacks (`kimi_k3`) |

The router is the gRPC router's pipeline (`routers/grpc/pipeline.rs`) with
its `Dispatcher` and planner injected, so prepare and render are the code the
mesh's gRPC routing already runs.

## Status

### Implemented

* `/v1/chat/completions`, `/v1/completions`, `/generate`, streaming and not.
* `/v1/models`, `/get_model_info`, `/get_server_info`; `/health_generate`
  reports `CoreManager.is_healthy()`.
* Sampling: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`,
  `ignore_eos`, `n` (not streamed; greedy sampling collapses it to 1, as on
  the OpenAI server). Penalties and `min_p` are ignored, as the OpenAI
  server ignores them.
* `data_parallel_rank` in the body, `--default-chat-template-kwargs` under a
  request's `chat_template_kwargs`.
* DP session affinity: with `ATOM_DP_SESSION_AFFINITY`, the session and its
  parent come from the headers the OpenAI server reads
  (`x-dynamo-session-id` / `x-session-id` / `x-correlation-id`,
  `x-dynamo-parent-session-id`) and reach CoreManager's `SessionPolicy`.
* `tool_choice`: `required`, a named function and `allowed_tools` are read
  with the model's own tool parser, as the Python standalone service read
  them; `required` holds the output to the format's tool-call channel where
  it has one (Kimi-K3's `required_channel`).
* Reasoning read from a prompt that opened it; a non-streamed reply over the
  parser's 64 KiB buffer is split rather than refused.
* Models with a Python chat template (DeepSeek-V4 / V4.1, Kimi-K3), with
  `thinking` / `reasoning_effort` resolved by the encoder's own rules
  (V4.1's effort aliases and budgets) and a string `tool_choice` passed to
  the template, as `_chat_completions` does.
* Usage with `cached_tokens` (the engine's prefix-cache hit); a chat's also
  with `prompt_tokens_details.cached_tokens` and, not streamed, `ttft_s` /
  `tpot_s` / `latency_s` (`num_choices` past one), timed as `api_server.py`
  times them (`render/atom_usage.rs`).
* Client disconnect aborts the request.

Refused with a 400 until they are built: image, video or audio content
(the engine is handed token ids only, so it would be dropped silently),
P/D requests (`kv_transfer_params`, or `/generate`'s bootstrap fields: served
plainly, the peer would wait for a transfer that never comes), a chat whose
tool calls the OpenAI server would hold to their schemas (Kimi-K3's
tool-call grammar, with `ATOM_TOOL_CALL_GRAMMAR` and the structured-output
backend; unconstrained, K3 can copy an empty call out of its history),
streaming with `n > 1`, logprobs,
`stop_token_ids`, structured output (`response_format` `json_object` /
`json_schema`, `regex`, `ebnf`, `json_schema`), `return_hidden_states`, and
an empty prompt. The structured-output refusal sits in the router, not the
bridge: the shared prepare step drops those fields before a payload exists,
so passing them through would answer unconstrained without saying so.

Checked: GSM8K (3-shot, `local-completions`) on Qwen3-30B-A3B, TP 1, against
the Python standalone service on the same build: 0.6876 / 0.8901
(flexible / strict) against 0.6808 / 0.8825. Models whose prompt ATOM
renders are checked token for token against the OpenAI server's chat render
(`_chat_completions`'s steps), on the body as the router serializes it: 13
requests (thinking on and off through `thinking`, `reasoning_effort` and
`chat_template_kwargs`, content lists, tools with `tool_choice` auto and
required, multi-turn tool calls with `reasoning_content`) match on
DeepSeek-V4.1-Flash, DeepSeek-V4-Flash and Kimi-K3. These models still need
a GPU run: V4.1 on gfx942 needs the FP4PTPC checkpoint.

Tests: `tests/entrypoints/test_token_bridge.py` (admission, submit, the
per-step push, abort, a lost step, the rendered prompt's encoding, session
affinity, the K3 tool-call refusal); `events.rs` (a request's steps as its
`TokenChunk` stream, siblings in index order, the abort on either side of the
submit, the finish-reason mapping); `reasoning/kimi_k3.rs` (the K3 channels);
`render/chat_aggregator.rs` (`split_reasoning`); the rest end to end on a GPU.

### Not yet

| Phase | Work |
|---|---|
| P1 | `/health` and `/metrics` with the engine's state and series (`get_metrics_statistics`, same metric names); `/start_profile`, `/stop_profile`, `/debug/mtp_stats`, `/debug/cache_stats` through the bridge; wait for the tokenizer before reporting ready; `ttft_s` / `tpot_s` / `latency_s` in a completion's `usage` (its render is the generate one, re-wrapped); ATOM's reasoning controls as `_chat_completions` resolves them (`thinking`, `reasoning_effort`) |
| P2 | logprobs and streaming `n > 1` (the streamed chat render writes one choice); structured output (`json_schema` to `SamplingParams`), and with it Kimi-K3's tool-call grammar (`configure_tool_call_grammar` -> `SamplingParams.structured_tool_format`), refused until then; OpenAI error bodies as the OpenAI server writes them; batching submits (one GIL entry for many requests) if admission shows up in profiles |
| P3 | `/v1/messages` (Anthropic), `/v1/responses` checked against the OpenAI server, FIM and prefix completion |
| P4 | multimodal (images to `preprocess_fanout(multimodal_data=...)`), `kv_transfer_params` for P/D (refused until then) |

Each model served this way is checked against the OpenAI server: the prompt
tokens (`atom/mesh/scripts/atom_token_parity.py`, #3596), the SSE and JSON of
every endpoint, GSM8K, and a concurrency sweep.

### Known differences from the Python standalone service

* Chat rendering (Jinja models) and all output parsing are the mesh's
  (`routers/prepare`, `reasoning_parser`, `tool_parser`), not ATOM's Python
  ones; a model whose template or parser differs between the two can render
  or parse differently.
* Streaming with `n > 1` is refused; the Python service streamed it.
* P/D requests are refused (P4): neither frontend carried `kv_transfer_params`;
  the Python service served them as plain requests.
* With `ATOM_TOOL_CALL_GRAMMAR`, a Kimi-K3 chat with tools is refused (P2):
  the OpenAI server holds its calls to their schemas, the Python service did
  not.
* `/v1/completions` requires `model` in the body, as the mesh's typed
  request does.
* `thinking` / `reasoning_effort` reach only a model whose prompt ATOM
  renders; a Jinja-rendered model (Qwen3, ...) takes its switch through
  `chat_template_kwargs`. `reasoning_effort` is a string in the mesh's typed
  request, so an integer budget goes through `thinking.effort`.
* The mesh's typed request refuses what the OpenAI server accepts: an
  integer `reasoning_effort`, `top_k: 0`, a tool without `parameters`.
* Kimi-K3 output parsing: a reply that skips the think channel is read as
  reasoning with no content; a literal `<|open|>think<|sep|>` inside the
  answer re-enters reasoning; a thinking-off reply over 64 KiB keeps
  `<|close|>response<|sep|>` in its content.
* `/health` is the router's liveness, not the engine's (P1).

## Moving to B

Only the backend changes. `events.rs` and the router know nothing of Python,
and B's Rust half is in place: `atom/mesh/src/atom_engine` is the client of
every EngineCore over the msgspec protocol (`zmq_protocol/messages.py`), and
`routers/atom_bridge/direct` its `Dispatcher`. Each DP rank is a worker that
mesh's own policies place on -- `round_robin`, `least_requests` and
`least_tokens` (`policies/dp_load.rs`, CoreManager's ledger), `dp_sticky`
under session affinity -- unless the request pins `data_parallel_rank`. The
dispatcher admits the request as `TokenBridge` does (`SamplingParams` values,
`max_tokens` against the cap and the context, the KV pool, stop strings to
token sequences) and submits it to its rank's Core.

`launch_mesh(engine_spec=...)` serves that way, and the standalone launcher
takes it with `--engine-dispatch direct` (the default stays `bridge`). The
EngineCores speak the msgspec protocol to any client; with an external one,
CoreManager spawns and reaps them but never connects
(`Config.external_engine_client`), and `atom/entrypoints/atomesh/engine_spec.py`
writes their `EngineSpec`: the deployment's output cap, the required tool-call
channel and the ids that end an output, decided as `TokenBridge` decides them.
Direct dispatch has no Python to render in, so it refuses at startup, before
the weights load, a model whose chat prompt is rendered in Python
(`ChatRenderer.needed`: DeepSeek-V4's message encoder, Kimi-K3's code
template), a P/D worker and a DP placement Rust does not have (`share_cache`,
`prefix_spread`); CoreManager refuses PP, multi-node and RapidServe.
