# ATOM standalone with a Rust frontend

`--frontend rust` serves ATOM's OpenAI API from the mesh router (Rust) and
stops at **LLMEngine**: Rust renders the chat template, tokenizes,
detokenizes, runs the reasoning and tool-call parsers and writes the SSE;
Python takes the prompt's token ids into `LLMEngine.io_processor` and
`CoreManager` exactly as the OpenAI server does, and hands each engine step's
tokens back. LLMEngine, CoreManager and the EngineCores are unchanged.

```bash
python -m atom.entrypoints.atomesh.server \
    --model <model> --kv-cache-dtype fp8 -tp 8 --port 8000 \
    --frontend rust            # default: python (AtomStandaloneService)
```

This page is the plan for that frontend (scheme A in the design discussion),
what is implemented now, and what is left.

## Why

The OpenAI server's per-request work grows with the context: JSON and
pydantic, the chat template, tokenizing, detokenizing, SSE. It runs on one
event loop and one GIL (`frontend_workers.py` exists because of it). Moving
that work to Rust removes the bulk of the frontend's cost while keeping
everything below the frontend -- DP routing, P/D, structured output, the
schedulers -- as it is.

Two ways to do it were weighed:

| | A: Rust -> LLMEngine / CoreManager (this page) | B: Rust -> EngineCore (`engine_v1`) |
|---|---|---|
| Python on the data path | one admission per request (`Sequence`, pickle), one output pass per step | none |
| CoreManager's features (DP router, prefix_spread / share_cache, back-pressure, P/D, PP, multi-node, metrics) | all of them, unchanged | re-implemented in Rust, or refused |
| Python changes | a ~200-line adapter | the Core / CoreManager protocol (pickle -> `engine_v1`) |
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
  refuse. Stop strings go to the engine, which matches them in token space;
  render trims them from the text.
* **One model.** The router loads the model's tokenizer (`--model-path` is
  added unless `--model-path` / `--tokenizer-path` is given) under the served
  model name, and prepares every request for it whatever `model` the client
  sent, as the OpenAI server does.

### Code

| Where | What |
|---|---|
| `atom/entrypoints/atomesh/token_bridge.py` | `TokenBridge`: `submit`, `abort`, `is_healthy`, the step flush |
| `atom/entrypoints/atomesh/server.py` | `--frontend {python,rust}`; builds the bridge and hands it to `launch_mesh` |
| `atom/mesh/src/routers/atom_bridge/mod.rs` | `AtomBridgeRouter` (`RouterTrait`), `AtomBridgeRuntime`, the one-engine planner |
| `atom/mesh/src/routers/atom_bridge/events.rs` | backend-neutral: `EventRegistry`, `SeqEventSource` (events -> `TokenChunk`), `Admission` |
| `atom/mesh/src/routers/atom_bridge/py_bridge.rs` | `StepSink` (PyO3), `BridgeDispatcher`, what a request may carry (`BridgeRequest`) |
| `atom/mesh/src/python.rs` | `launch_mesh(token_bridge=..., served_model_name=...)`, exports `StepSink` |

The router is the gRPC router's pipeline (`routers/grpc/pipeline.rs`) with
its `Dispatcher` and planner injected, so prepare and render are the code the
mesh's gRPC routing already runs.

## Status

### Implemented (P0)

* `/v1/chat/completions`, `/v1/completions`, `/generate`, streaming and not.
* `/v1/models`, `/get_model_info`, `/get_server_info`; `/health_generate`
  reports `CoreManager.is_healthy()`.
* Sampling: `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`,
  `ignore_eos`. Penalties and `min_p` are ignored, as the OpenAI server
  ignores them.
* Usage with `cached_tokens` (the engine's prefix-cache hit).
* Client disconnect aborts the request.

Refused with a 400 until they are built: `n > 1`, logprobs, `stop_token_ids`,
`tool_choice` that constrains the output (`required` or a named function),
structured output (`response_format` `json_object` / `json_schema`, `regex`,
`ebnf`, `json_schema`), `return_hidden_states`, and an empty prompt. The
structured-output refusal sits in the router, not the bridge: the shared
prepare step drops those fields before a payload exists, so passing them
through would answer unconstrained without saying so.

Tests: `events.rs` checks a request's steps becoming its `TokenChunk`
stream and the abort on either side of the submit; the rest is checked end to
end on a GPU (GSM8K through `--frontend rust`).

### Not yet

| Phase | Work |
|---|---|
| P1 | `/health` and `/metrics` with the engine's state and series (`get_metrics_statistics`, same metric names); `/start_profile`, `/stop_profile`, `/debug/mtp_stats`, `/debug/cache_stats` through the bridge; wait for the tokenizer before reporting ready; ATOM's chat semantics from `mesh-engine-v1-switch` (`atom_engine/chat.rs`, `request.rs`, `render/atom_usage.rs`: reasoning controls as `_chat_completions` resolves them, `ttft_s` / `tpot_s` / `latency_s` in `usage`) |
| P2 | logprobs and streaming `n > 1` (siblings carry an index through the sink); structured output and `tool_choice="required"` (`json_schema`, `required_channel` to `SamplingParams`); `--default-chat-template-kwargs`; OpenAI error bodies as the OpenAI server writes them; batching submits (one GIL entry for many requests) if admission shows up in profiles |
| P3 | `/v1/messages` (Anthropic), `/v1/responses` checked against the OpenAI server, FIM and prefix completion |
| P4 | multimodal (images to `preprocess_fanout(multimodal_data=...)`), models with a custom message encoder (DeepSeek-V4.1 after #3597; others ported or rendered in Python), `kv_transfer_params` for P/D; then make `rust` the default |

Before `rust` becomes the default, each model is checked against the OpenAI
server: the prompt tokens (`atom/mesh/scripts/atom_token_parity.py`, #3596),
the SSE and JSON of every endpoint, GSM8K, and a concurrency sweep.

### Known differences from the Python frontend

* Chat rendering and parsing are the mesh's (`routers/prepare`,
  `reasoning_parser`, `tool_parser`), not ATOM's Python ones; a model whose
  template or parser differs between the two can render or parse
  differently. DeepSeek-V4.1's reasoning controls are covered by #3597.
* `--tool-call-parser` takes the mesh router's names here; ATOM-only names
  (`dsml`, `kimi_k3`, ...) reach the router, which reads the model's format
  itself.
* `/health` is the router's liveness, not the engine's (P1).

## Moving to B

Only the backend changes. `events.rs` and the router know nothing of Python:
a dispatcher that sends `engine_v1` `SubmitBatch` to the EngineCores
(`atom/mesh/src/atom_engine`) turns their stream and completion frames into
`SeqEvent`s, aborts through an `engine_v1` abort frame (`Aborter`), and
replaces `BridgeDispatcher`. What then has to move from Python into Rust is
what `TokenBridge` delegates today: `SamplingParams` validation, the context
and pool checks, stop strings to token sequences, the prompt hash chain, and
CoreManager's DP routing.
