# gRPC Router Engine Extraction — Design

**Date**: 2026-05-19 (revised 2026-05-19)
**Scope**:
- `atom/mesh/src/routers/grpc/` (refactor; internally reorganized into top-level pipeline + HTTP wiring and a `grpc/engine/` subdir holding everything that imports `mesh_grpc::*`)
- `atom/mesh/src/routers/prepare/` (**new**, transport-neutral request preparation — HTTP req → `(GenerationPayload, ResponseContext)`)
- `atom/mesh/src/routers/render/` (**new**, transport-neutral response rendering — `Stream<TokenChunk>` + `ResponseContext` → `axum::Response`)
- `atom/mesh/src/routers/worker_stream/` (**new**, transport-neutral engine boundary types — `TokenChunk`, `WorkerStream`, `EngineError`)
- `atom/mesh/src/routers/openai/responses/` (**new**, relocated from `grpc/{common,regular}/responses/`; namespaced under `openai/` because the folder name `responses` comes from the OpenAI spec and would otherwise read like a generic sibling of `render/`)

Note: `routers/shared/` already exists in the tree (currently 3 files: `mod.rs`, `metrics_utils.rs`, `placement_response.rs`) and is left untouched by this refactor. The transport-neutral logic extracted here does NOT go under `shared/` — `shared` is an umbrella noun (utils/common anti-pattern) and the new code lives in job-named peer folders. See §3.1 for rationale.

Naming guideposts applied here (each verified by §6 acceptance criteria):
- **Verb pair for the pipeline data path**: `prepare/` (build request) → `render/` (build response). Each folder is a single verb that says what its files do.
- **Symmetric noun pair for the in-folder split**: `chat_streaming.rs` / `chat_aggregator.rs` (and the parallel generate pair). `aggregator` describes the non-streaming role positively ("collapse a stream of chunks into one response") instead of negatively as `non_streaming`. Inspiration: `dynamo/lib/llm/src/protocols/openai/chat_completions/aggregator.rs`.
- **Protocol namespace, not a flat sibling**: OpenAI-spec endpoint folders live under `routers/openai/` (mirroring `dynamo/lib/llm/src/protocols/openai/`), so the path itself says "this is the OpenAI Responses API" rather than colliding visually with `render/`.

**Relation to other docs**:
- Supersedes `2026-05-19-grpc-refactor-detailed-plan.md` (file-level slicing without addressing coupling).
- Companion: `2026-05-19-grpc-pd-merge-spec.md` — PD stream merge state machine. **Required reading before implementing `grpc/engine/pd_stream_merge.rs`.**

---

## 1. Why a Different Design

The prior plan slices large files (`utils.rs` 1214, `streaming.rs` 1326, `processor.rs` 465) into smaller files within `grpc/`. It explicitly acknowledges that `streaming.rs` and `processor.rs` cannot be moved out of `grpc/` because they import `mesh_grpc::sglang_proto::*` types at signature level.

But that import coupling is the actual problem, not file size. Once a transport-neutral chunk type exists, those files become independent of gRPC and belong outside `routers/grpc/` — both physically and by import path — at `routers/prepare/` and `routers/render/`. Their logic (SSE encoding, stop decoding, tool/reasoning parsing, response rendering) has no gRPC concern, so consumers (e.g. a future HTTP-native backend) can use them without going through `crate::routers::grpc::*`. With them out, the gRPC folder shrinks to its true responsibility: wire transport, concentrated in `grpc/engine/`.

In addition, `grpc/{common,regular}/responses/` (2059 lines, **0** references to `mesh_grpc::*`, **0** PD-mode code — PD router returns 501 for `/v1/responses`) is also misplaced. It is the OpenAI Responses API implemented on top of `Pipeline::execute_chat*()` and belongs at `routers/openai/responses/`, namespaced under an `openai/` folder rather than as a bare sibling of `routers/grpc/`. The original placement was inertia from when `Pipeline` lived under `grpc/`.

This design introduces the minimum new types required to break the gRPC coupling, and relocates the two misplaced subtrees without re-architecting them.

---

## 2. What `grpc/` Actually Contains Today

`grpc/` total: **9280 lines** (verified via `wc -l $(find grpc -name '*.rs')`), classified by what each function actually does:

| Concern | Lines | Where it lives today |
|---|---|---|
| **gRPC transport** | ~1100 | `client.rs`, `proto_wrapper.rs`, ~250 in `utils.rs` (`get_grpc_client_from_worker`, `collect_stream_responses`, logprob proto adapters) |
| **Protocol / decoding (transport-neutral)** | ~2500 | `utils.rs` (chat template, tool constraints, parser factories, stop decoder, finish reason — ~700 lines), `regular/streaming.rs` (1326), `regular/processor.rs` (465) |
| **Pipeline orchestration** | ~1500 | `context.rs`, `pipeline.rs`, `common/stages/*`, `regular/stages/*` |
| **Responses API (zero gRPC code, zero PD-specific code)** | ~2059 | `common/responses/*`, `regular/responses/*` |
| **HTTP entry / wiring** | ~600 | `router.rs`, `pd_router.rs`, `completion_adapter.rs` |
| **Misc / tests / mod files** | rest | scattered |

Less than 15% of the `grpc/` tree is genuinely about gRPC. The rest happens to consume `ProtoStream` or sit under a `pipeline` that historically lived here, and got placed in `grpc/` by inertia.

---

## 3. Target Architecture

### 3.1 File layout

Five top-level moves under `routers/`, all peers of `grpc/`:

```
routers/
├── grpc/                              # transport-coupled: HTTP wiring + pipeline + gRPC engine
│   ├── mod.rs
│   ├── http_router.rs                 # was router.rs       (regular-mode axum handler)
│   ├── http_router_pd.rs              # was pd_router.rs    (PD-mode axum handler)
│   ├── completion_adapter.rs          # unchanged
│   ├── pipeline.rs                    # 4-step orchestrator: prepare → engine → render
│   └── engine/                        # the ONLY place that imports mesh_grpc::*
│       ├── mod.rs                     # GrpcEngine + dispatch + dispatch_one
│       ├── worker_client_cache.rs     # was client.rs
│       ├── proto_stream_wrapper.rs    # was proto_wrapper.rs
│       ├── payload_to_proto.rs        # GenerationPayload → tonic request (sglang + vllm)
│       ├── proto_to_chunk.rs          # tonic Streaming → TokenChunk (sglang + vllm, logprob collapse)
│       └── pd_stream_merge.rs         # merge_pd_streams state machine + T1–T7 tests (private fn)
│
├── prepare/                           # transport-neutral: HTTP req → (GenerationPayload, ResponseContext)
│   ├── mod.rs                         # prepare_chat / prepare_generate (return tuple)
│   ├── generation_payload.rs          # GenerationPayload + SamplingParams + StopConfig + LogprobConfig + PdMetadata
│   ├── response_context.rs            # ResponseContext + ProtocolRequest (constructed here, consumed by render/)
│   ├── chat_template.rs               # extracted from utils.rs
│   ├── tool_constraints.rs            # extracted from utils.rs
│   ├── stop_sequence_decoder.rs       # extracted from utils.rs
│   └── parser_factory_lookup.rs       # tool / reasoning parser selection
│
├── worker_stream/                     # transport-neutral: engine boundary types (produced by grpc/engine/, consumed by render/)
│   ├── mod.rs
│   ├── token_chunk.rs                 # TokenChunk + FinishReason + MatchedStop + TokenLogprobs
│   ├── worker_stream.rs               # WorkerStream<TokenChunk>
│   └── engine_error.rs                # EngineError (§5.2 of PD merge spec, 7 variants)
│
├── render/                            # transport-neutral: Stream<TokenChunk> + ResponseContext → axum::Response
│   ├── mod.rs
│   ├── chat_streaming.rs              # split from regular/streaming.rs (chat branch, SSE)
│   ├── chat_aggregator.rs             # split from regular/processor.rs (chat branch, collapse stream → one response)
│   ├── generate_streaming.rs          # split from regular/streaming.rs (generate branch, SSE)
│   ├── generate_aggregator.rs         # split from regular/processor.rs (generate branch, collapse stream → one response)
│   └── finish_reason_mapping.rs       # extracted from utils.rs
│
└── openai/                            # OpenAI-protocol endpoint subtrees that sit ABOVE the pipeline
    └── responses/                     # /v1/responses implementation; depends on grpc::Pipeline (concrete) — see §3.4
        └── ...
```

**Naming rationale at a glance** (each rule backed by §6 acceptance criteria):

| Pair / convention | Why this shape |
|---|---|
| `prepare/` ↔ `render/` (verb pair) | Single-verb folders read top-to-bottom as a sentence: "prepare the request, render the response." Each folder name says what its files do. |
| `chat_streaming.rs` ↔ `chat_aggregator.rs` (noun pair) | Both names describe the role positively. `aggregator` (verb-derived noun) beats `non_streaming` (defined negatively) and matches dynamo's `chat_completions/aggregator.rs` convention. |
| `openai/responses/` (namespaced) | The folder name `responses` comes from the OpenAI spec; placing it under `openai/` makes that origin visible in the path and removes any visual collision with `render/`. Extensible: a future `openai/messages/` or `anthropic/` slots in at the same level. |
| `worker_stream/` (peer folder, named by the boundary it owns) | Holds the three types that cross the engine boundary (`TokenChunk`, `WorkerStream`, `EngineError`). Owning these in a third folder keeps `prepare/` and `render/` from reaching into each other's namespace. |
| No `shared/`, no `utils.rs`, no `common.rs` | Umbrella nouns hide what the file does. Each helper lives in a folder named after the job (`chat_template.rs`, `stop_sequence_decoder.rs`, `parser_factory_lookup.rs`). |

**Why peer folders, not `routers/shared/`.** The pre-existing `routers/shared/` is reserved for genuine cross-router utilities (`metrics_utils`, `placement_response`). `shared` is an umbrella noun (same anti-pattern as `utils`/`common`) and violates the project's name-matches-function rule: it tells the reader "someone else uses this" but not "this does X". Each transport-neutral concern gets its own job-named folder at the same level as `grpc/`, mirroring the existing convention of `routers/parse/`, `routers/tokenize/`, `routers/conversations/`.

**Why the new folders are peers of `grpc/`, not under it.** Goal #1 of this refactor is that protocol parsing and response rendering are independently consumable by a future non-gRPC backend. Co-locating them inside `routers/grpc/` would satisfy the import-coupling check (§6.1) but not the physical-coupling check: `use crate::routers::grpc::prepare::*` still names `grpc` in the path. Peer placement makes the independence visible at the import path.

**Layering rules** (enforced by acceptance criteria §6.11–§6.13):
- `grpc/` may depend on `prepare/`, `render/`, `worker_stream/` — these are below it in the data flow.
- `grpc/` may NOT depend on `openai/responses/`.
- `openai/responses/` may depend on `grpc::Pipeline` (concrete type) only.
- `openai/responses/` does NOT import `mesh_grpc::*`.
- `prepare/`, `render/`, `worker_stream/` may NOT depend on `grpc/` (verified by grep — §6.2).

### 3.2 Core types

Five new things, all in their final home from day one. No traits. Two of them (`GenerationPayload`, `TokenChunk`) sit on opposite sides of the engine and together form a **symmetric transport-neutral boundary**: protocol logic on both the request-construction and response-consumption sides becomes independently unit-testable, without any `mesh_grpc::*` in scope.

**`routers/worker_stream/token_chunk.rs`** — transport-neutral worker output:

```rust
pub enum TokenChunk {
    Partial {
        token_ids: Vec<u32>,
        logprobs: Option<TokenLogprobs>,
    },
    Complete {
        token_ids: Vec<u32>,
        finish_reason: FinishReason,
        matched_stop: Option<MatchedStop>,
        usage: Usage,
        logprobs: Option<TokenLogprobs>,
        /// **Source semantics**: in single-worker mode, filled by the worker's Complete.
        /// In PD mode, decode's Complete carries None; `merge_pd_streams` injects the
        /// value from prefill's Complete. See `2026-05-19-grpc-pd-merge-spec.md` §2.
        input_logprobs: Option<InputLogprobs>,
        meta: WorkerMeta,                    // request_id, weight_version, cached_tokens
    },
}

pub enum FinishReason { Stop, Length, ContentFilter, ToolCalls, Abort, Other(String) }
pub enum MatchedStop  { Str(String), TokenId(u32) }

pub struct TokenLogprobs { pub items: Vec<TokenLogprob> }
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
    pub decoded_text: Option<String>,
    pub top: Vec<(u32, f32, Option<String>)>,
}

/// impl Stream<Item = Result<TokenChunk, EngineError>>.
///
/// In single mode, holds one underlying tonic Streaming<T>; Drop closes the H2
/// stream (tonic-native cancellation — no separate AbortHandle needed).
/// In PD mode, holds both prefill and decode upstreams as owned values; Drop
/// propagates to both.
pub struct WorkerStream { /* inner stream(s) */ }

/// Full variant set. State-machine sources (first 5) come from `merge_pd_streams`
/// and proto stream consumption; the last 2 come from `dispatch_one` before any
/// stream exists. See companion spec §5.2 for transition-level semantics.
pub enum EngineError {
    Transport(tonic::Status),       // any tonic-layer failure
    Prefill(ProtoErrorMessage),     // prefill yielded a proto Error
    DecodeError(ProtoErrorMessage), // decode yielded a proto Error during Streaming
    PrefillEarlyClose,              // prefill stream closed without Complete or Error
    DecodeIncomplete,               // decode stream closed without Complete or Error
    ConnectionAcquireFailed(String),
    RequestBuildFailed(String),
}
```

`TokenLogprobs` is the OpenAI-compatible shape minus the formatting. SGLang and vLLM proto logprob structures collapse into this in `grpc/engine/proto_to_chunk.rs`.

`WorkerStream` and `EngineError` live in `routers/worker_stream/worker_stream.rs` and `routers/worker_stream/engine_error.rs` respectively. All three types (`TokenChunk`, `WorkerStream`, `EngineError`) are produced by `grpc/engine/` and consumed by `routers/render/`; they live at the boundary, in their own peer folder, so neither side reaches into the other's namespace.

**`routers/prepare/generation_payload.rs`** — transport-neutral dispatch input:

```rust
/// Everything the engine needs to issue one generation, with zero mesh_grpc dependency.
/// Built by `routers::prepare::{prepare_chat, prepare_generate}`; consumed by `GrpcEngine::dispatch`.
pub struct GenerationPayload {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopConfig,
    pub logprob: LogprobConfig,              // includes `input_logprobs: bool`
    pub tool_constraints: Option<(String, String)>,
    pub pd_metadata: Option<PdMetadata>,     // bootstrap host/port, kv slot, etc.
}
```

`GenerationPayload` is the mirror image of `TokenChunk`: together they form a symmetric boundary across the engine. The engine becomes a pure wire-format adapter — `to_sglang_proto(&payload)` and `to_vllm_proto(&payload)` are intended as field copies, not logic. All protocol logic (sampling param mapping, stop sequence assembly, tool-constraint encoding, PD metadata injection) lives in `routers/prepare/` and can be unit-tested without `mesh_grpc::*` in scope. The symmetric counterpart on the response side lives in `routers/render/` and consumes `Stream<TokenChunk>` + `ResponseContext`.

This subsumes today's `build_generate_request_from_chat` / `build_generate_request_from_generate` helpers (currently methods on `SglangSchedulerClient` / `VllmEngineClient` from the **external `smg-grpc-client` crate**, called from `regular/stages/*/request_building.rs`) into one transport-neutral type plus two adapters. **Risk note**: because those methods live in an external crate, byte-identical replacement requires either (a) re-implementing the mapping in our adapters or (b) upstreaming a `to_proto(payload)` API in `smg-grpc-client`. This is validated by the Step 2a spike in §7 before any bulk migration.

`logprob.input_logprobs: bool` is read by `merge_pd_streams` to decide whether to drain prefill before yielding decode (see PD merge spec §1).

**`prepare_chat()` / `prepare_generate()` — return tuple, not struct.**

The replacement for today's `ProcessingState` is two independent values returned together. There is no umbrella struct; the split mirrors the two downstream consumers exactly.

```rust
pub fn prepare_chat(
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &SharedComponents,
) -> Result<(GenerationPayload, ResponseContext), Response> { ... }

pub fn prepare_generate(...) -> Result<(GenerationPayload, ResponseContext), Response> { ... }

/// Everything the render side needs. Constructed atomically by prepare_*;
/// consumed by render::{chat_streaming, chat_aggregator, ...}. Zero mesh_grpc::*.
///
/// Name choice: `ResponseContext` rather than `DecodeContext` to avoid collision
/// with LLM-serving "prefill/decode" terminology — this struct is about HTTP
/// response rendering, not token decoding.
pub struct ResponseContext {
    pub original: ProtocolRequest,
    pub model_id: Option<String>,
    pub headers: Option<HeaderMap>,
    pub original_text: Option<String>,
    pub processed_messages: Option<ProcessedMessages>,  // chat-only; None for generate
    pub tokenizer: Arc<dyn Tokenizer>,
    pub stop_decoder: StopSequenceDecoder,
}

pub enum ProtocolRequest {
    Chat(Arc<ChatCompletionRequest>),
    Generate(Arc<GenerateRequest>),
}
```

The tuple split is explicit and physical: `GenerationPayload` goes down to the engine, `ResponseContext` goes sideways to the render layer. Pipeline uses destructuring at the call site:

```rust
let (payload, resp_ctx) = prepare_chat(req, ...)?;
let stream = engine.dispatch(&placement, &payload).await?;
render::chat_aggregator::process(stream, resp_ctx).await
```

No umbrella struct means no god-object risk, no "junction box" hand-wave, no deferred split decision. `Option` in `ResponseContext` only marks data that is semantically optional (chat-only fields when handling generate, headers when not supplied) — not "stage hasn't filled this in yet."

**Engine internals — split into the `grpc/engine/` subdirectory.** The down-side and up-side adapters are kept in separate files so the symmetric boundary is visible in the file names themselves (`payload_to_proto` ↔ `proto_to_chunk`).

| File | Lines (est.) | Contents |
|---|---|---|
| `grpc/engine/mod.rs` | ~100 | `GrpcEngine::dispatch` + `dispatch_one`. Single vs PD branching, Sglang vs vLLM branching, error mapping to `EngineError::{ConnectionAcquireFailed, RequestBuildFailed, Transport}`. |
| `grpc/engine/payload_to_proto.rs` | ~200 | `to_sglang_proto(&payload)` and `to_vllm_proto(&payload)` (down-side, pure field copy). |
| `grpc/engine/proto_to_chunk.rs` | ~200 | `proto → TokenChunk` converters for both backends (up-side, includes logprob shape collapse). Pure functions, no state. |
| `grpc/engine/pd_stream_merge.rs` | ~250 + tests | `merge_pd_streams` state machine. Implementation MUST follow `2026-05-19-grpc-pd-merge-spec.md` §3. Test obligations T1–T7 in the same file. Private `fn`, not exposed at module boundary. |
| `grpc/engine/worker_client_cache.rs` | ~200 | Renamed from `client.rs`. Per-worker `GrpcClient` cache (`get_grpc_client_from_worker` etc.). |
| `grpc/engine/proto_stream_wrapper.rs` | ~500 | Renamed from `proto_wrapper.rs`. `ProtoStream` enum + `AbortOnDropStream`. |

```rust
// Dispatch file
pub struct GrpcEngine {
    client_registry: Arc<ClientRegistry>,
}

impl GrpcEngine {
    pub async fn dispatch(
        &self,
        placement: &PlacementPlan,
        payload: &GenerationPayload,
    ) -> Result<WorkerStream, EngineError> {
        match placement {
            PlacementPlan::Single { worker, .. } => self.dispatch_one(worker, payload).await,
            PlacementPlan::Pair { prefill, decode, .. } => {
                let p = self.dispatch_one(prefill, payload).await?;
                let d = self.dispatch_one(decode, payload).await?;
                Ok(merge_pd_streams(p, d, payload.logprob.input_logprobs))
            }
        }
    }

    async fn dispatch_one(&self, worker: &Arc<dyn Worker>, payload: &GenerationPayload)
        -> Result<WorkerStream, EngineError>
    {
        let client = self.client_registry.get(worker)
            .map_err(|e| EngineError::ConnectionAcquireFailed(e.to_string()))?;
        match client {
            GrpcClient::Sglang(mut c) => {
                let proto = to_sglang_proto(payload);   // from proto adapter file
                let stream = c.generate(proto).await
                    .map_err(EngineError::Transport)?;
                Ok(WorkerStream::from_sglang(stream))   // converter in proto adapter file
            }
            GrpcClient::Vllm(mut c) => { /* symmetric */ }
        }
    }
}
```

Engine never sees `ResponseContext` — it consumes only `&GenerationPayload`. Single/PD and Sglang/vLLM are `match` arms, not trait polymorphism. There is one engine implementation today and for the foreseeable future, so an `Engine` or `Dispatcher` trait would be single-impl indirection. Adding a third backend means writing one more `to_X_proto` + one more match arm, not refactoring.

**`grpc/pipeline.rs`** — 4-step explicit function body with a 3-way fork at step 4:

```rust
pub struct Pipeline {
    planner: Arc<dyn PdPlanner>,           // existing
    engine: GrpcEngine,
    components: Arc<SharedComponents>,
    backend_label: &'static str,           // metrics label
}

impl Pipeline {
    pub async fn execute_chat(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
    ) -> Response {
        // 1. prepare
        let (payload, resp_ctx) = match prepare::prepare_chat(req, headers, model_id, &self.components) {
            Ok(t) => t,
            Err(e) => return e,
        };
        // 2. select
        let placement = match self.planner.plan(&payload.descriptor()).await {
            Ok(p) => p,
            Err(e) => return placement_err_to_response(e, resp_ctx.model_id.as_deref()),
        };
        let _guards = LoadGuards::new(&placement, resp_ctx.headers.as_ref());
        // 3. dispatch (engine sees only the transport-neutral payload)
        let stream = match self.engine.dispatch(&placement, &payload).await {
            Ok(s) => s,
            Err(e) => return engine_err_to_response(e),
        };
        // 4. render response (2-way fork: streaming / aggregator)
        if resp_ctx.original.is_streaming() {
            render::chat_streaming::process(stream, resp_ctx, self.backend_label)
        } else {
            render::chat_aggregator::process(stream, resp_ctx).await
        }
    }

    pub async fn execute_generate(&self, req: Arc<GenerateRequest>, ...) -> Response {
        // same four steps with generate-specific prepare/render
    }

    /// Used by /v1/responses (non-streaming path only — Responses streaming
    /// transforms the SSE output of execute_chat). Returns typed value instead
    /// of axum Response so the openai/responses layer can convert + persist.
    pub async fn execute_chat_for_responses(...)
        -> Result<ChatCompletionResponse, Response>
    { /* same four steps, no streaming branch */ }
}
```

Chat and generate are two explicit code paths. There is no runtime `match RequestType` fork inside a "stage" — the language already splits them at the call site. The seven wrapper stages in today's `regular/stages/` collapse into two functions per endpoint (`prepare_*` and `render::*`).

### 3.3 Data flow per request

```
ChatCompletionRequest
        │
        ▼
routers::prepare::prepare_chat ──► (GenerationPayload, ResponseContext)  ← tuple, not struct
        │                                  │             │
        │                                  │             └── consumed only by routers::render
        ▼                                  ▼ (payload)
planner.plan ──► Placement     grpc::engine::dispatch  ◄── sees ONLY GenerationPayload,
        │              │                  │                no ResponseContext, no mesh_grpc
        └──────┬───────┘                  │                leaking back upward
               ▼                          ▼
        routers::worker_stream::WorkerStream<TokenChunk>
                                          │
                                          ▼
                          routers::render::{chat_aggregator, chat_streaming}
                                          │   (uses ResponseContext fields)
                                          ▼
                                       axum::Response

         ▲
         │ (HTTP entry; sits ABOVE pipeline, not inside)
         │
  ResponsesRequest ──► routers/openai/responses ──► grpc::Pipeline::execute_chat[_for_responses]
                              │                              │
                              │                              └── returns SSE Body or typed response
                              ▼
                       conversions + storage + SSE event re-emission
                              │
                              ▼
                       ResponsesResponse / SSE
```

Every arrow inside the pipeline is a typed value returned from one function and consumed by the next. No shared mutable state. The two transport-neutral boundary types — `GenerationPayload` going down, `TokenChunk` coming back up — together ensure that **`mesh_grpc::*` is imported ONLY inside `grpc/engine/`** (six files). Everything in `routers/{prepare,render,worker_stream,openai/responses}/` is free of `mesh_grpc::` references and physically independent of the `grpc/` namespace.

The OpenAI Responses API layer is *above* the pipeline: it consumes the pipeline's HTTP-level outputs (axum `Body` for streaming, typed `ChatCompletionResponse` for non-streaming) and re-encodes them in OpenAI Responses-API format. It has no view into pipeline internals.

### 3.4 OpenAI Responses API extraction

The OpenAI Responses API subtree moves from `grpc/{common,regular}/responses/` (2059 lines) to `routers/openai/responses/` with no architectural change — file renames and one merger, business logic untouched.

**Why it moves**:
- Zero `mesh_grpc::*` imports anywhere in the subtree.
- Zero PD-specific code (`GrpcPDRouter` returns 501 for `/v1/responses` via the trait default).
- The `grpc/regular/responses/` path label is misleading: it's not "regular-mode responses", it's "the only responses implementation" — placement of `Pipeline` under `grpc/` historically pulled it in.

**Why the new path is `routers/openai/responses/` (not bare `routers/responses/`)**:
- The folder name `responses` is **the OpenAI endpoint name** (`POST /v1/responses`), not a generic English word for HTTP responses. Hoisting it to a top-level sibling of `render/` would invite future readers to misread it as a generic response folder paired with `render/` — the same singular/plural ambiguity that motivated renaming `response/` → `render/` in the first place.
- An `openai/` namespace makes the spec origin explicit at the import path: `crate::routers::openai::responses::*`. Inspiration: `dynamo/lib/llm/src/protocols/openai/responses/`.
- It is extensible. If we later add another protocol-family endpoint (an `openai/messages/` for the audio Realtime API, or an `anthropic/messages/` if we ever serve Anthropic-shape requests), each slots in next to `responses/` without further folder-naming bikeshedding.

**Why the layout consolidates a few files**:
- `grpc/common/responses/streaming.rs` (638 lines: `ResponseStreamEventEmitter`) has exactly one caller: `grpc/regular/responses/streaming.rs` (383 lines: SSE body transformer). Two files exist because of the historic `common/` vs `regular/` split, not because of any reuse boundary. Merged into `routers/openai/responses/streaming.rs`.
- `grpc/regular/responses/common.rs` → `conversation.rs` (file name should describe content; "common" is reserved for genuinely shared infra).
- `grpc/common/responses/utils.rs` → `persistence.rs` (single responsibility; "utils" is a reverse-anti-pattern name).
- `grpc/common/responses/handlers.rs` → `retrieve.rs` (avoids name collision with `routers/openai/responses/handlers.rs`; only contains GET/cancel storage operations).

**Concrete-type coupling, not trait abstraction**:

`ResponsesContext.pipeline` keeps the concrete type `Arc<crate::routers::grpc::Pipeline>`. No `ChatPipeline` trait — there is exactly one production caller today, so it would be a single-implementation indirection. Extracting a trait when a second implementation arrives is a mechanical refactor.

The dependency `routers::openai::responses → routers::grpc::Pipeline` is permitted and checked by §6.11.

---

## 4. What Gets Deleted / Moved

After full migration, the following are gone or relocated:

| File / type | Replaced by |
|---|---|
| `grpc/context.rs::ProcessingState` (god-bag) | `(GenerationPayload, ResponseContext)` tuple returned from `routers::prepare::prepare_*` |
| `grpc/context.rs::{PreparationOutput, ResponseState, LoadGuards, ClientSelection, WorkerSelection, ExecutionResult, FinalResponse}` | `GenerationPayload` + `ResponseContext` fields + `Placement` (existing) + `WorkerStream` |
| `grpc/pipeline.rs::RequestPipeline` (7-stage `Vec<Box<dyn PipelineStage>>`) | `grpc/pipeline.rs::Pipeline` (4 explicit steps) |
| `grpc/common/stages/*` (5 files) | inlined into `Pipeline` body |
| `grpc/regular/stages/*` (8 files incl. 3 wrapper stages) | inlined into `Pipeline` + redistributed to `routers/prepare/` and `routers/render/` |
| `grpc/regular/stages/*/request_building.rs` proto construction (sglang + vllm branches) | `GenerationPayload` + `grpc/engine/payload_to_proto.rs` |
| `grpc/regular/streaming.rs` (1326) | split into `routers/render/chat_streaming.rs` + `routers/render/generate_streaming.rs` |
| `grpc/regular/processor.rs` (465) | split into `routers/render/chat_aggregator.rs` + `routers/render/generate_aggregator.rs` |
| `grpc/common/{response_collection,response_formatting}.rs` | folded into `routers/render/*` files |
| `grpc/client.rs` | → `grpc/engine/worker_client_cache.rs` (rename) |
| `grpc/proto_wrapper.rs` | → `grpc/engine/proto_stream_wrapper.rs` (rename) |
| `grpc/router.rs` | → `grpc/http_router.rs` (rename) |
| `grpc/pd_router.rs` | → `grpc/http_router_pd.rs` (rename) |
| `grpc/utils.rs` (1214 lines, umbrella name) | **deleted**. ~700 transport-neutral lines (chat template, tool constraints, stop decoder, parser factory lookup, finish-reason mapping) redistributed to `routers/prepare/*.rs` and `routers/render/finish_reason_mapping.rs`; ~250 transport-bound lines (`get_grpc_client_from_worker`, `collect_stream_responses`, logprob proto adapters) absorbed into `grpc/engine/worker_client_cache.rs` and `grpc/engine/proto_to_chunk.rs`; ~250 helper lines (`resolve_tokenizer`, `error_type_from_status`) move to `grpc/mod.rs` or their nearest single caller. |
| `PipelineStage` trait | not needed; `Pipeline::execute_*` is a function body |
| `grpc/common/responses/context.rs` | → `routers/openai/responses/context.rs` |
| `grpc/common/responses/streaming.rs` (`ResponseStreamEventEmitter`, 638 lines) | MERGED → `routers/openai/responses/streaming.rs` |
| `grpc/common/responses/handlers.rs` (GET/cancel, 71 lines) | → `routers/openai/responses/retrieve.rs` |
| `grpc/common/responses/utils.rs` (persist + extract_tools) | → `routers/openai/responses/persistence.rs` |
| `grpc/regular/responses/handlers.rs` (entry dispatch) | → `routers/openai/responses/handlers.rs` |
| `grpc/regular/responses/non_streaming.rs` | → `routers/openai/responses/non_streaming.rs` |
| `grpc/regular/responses/streaming.rs` (SSE body transform, 383 lines) | MERGED into `routers/openai/responses/streaming.rs` |
| `grpc/regular/responses/common.rs` (load_conversation_history) | → `routers/openai/responses/conversation.rs` |
| `grpc/regular/responses/conversions.rs` (Responses ⇄ Chat conversions, 425 lines) | → `routers/openai/responses/conversions.rs` |

End state line counts (estimate, before unit tests):
- `routers/grpc/`: ~9280 → ~1800 (HTTP wiring + pipeline + `engine/` subdir)
- `routers/prepare/`: ~900 (extracted from utils.rs and stages/)
- `routers/worker_stream/`: ~150 (three small type files)
- `routers/render/`: ~2100 (split from streaming.rs + processor.rs + utils finish-reason)
- `routers/openai/responses/`: ~2059 (relocated, business logic untouched)

---

## 5. Explicit Non-Goals

| Out of scope | Reason |
|---|---|
| `Engine` trait abstraction | One implementation (`GrpcEngine`); a trait with one impl is a single-impl indirection, not polymorphism. If a second engine ever appears, extracting a trait then is a mechanical change. |
| `Dispatcher` trait + `SingleDispatcher` / `PdDispatcher` types | Single vs PD is a `match Placement` arm, not a polymorphism axis. Wrapping it in a trait scatters two lines of logic across three files. |
| `ChatPipeline` trait abstraction over `Pipeline` for Responses to consume | One pipeline implementation today (gRPC). HTTP and PD routers do not implement `/v1/responses`. Single-impl trait — extract later if a second implementation appears. Decision rationale: §3.4 last paragraph. |
| Moving `routers/prepare/`, `response/`, `worker_stream/` higher (e.g. `mesh/src/` top level) | They are router-layer concerns: every consumer today and in the foreseeable HTTP-pipeline future lives under `routers/`. Promoting them above `routers/` would imply they serve non-router code, which they don't. |
| Trait-based generalization of `Pipeline` over engine type | `Pipeline` holds `engine: GrpcEngine` concretely. When a second engine appears (e.g. HTTP-native), extract the trait then; today it would be single-impl indirection. Same logic as `ChatPipeline` non-goal above. |
| `proto_wrapper.rs` enum → trait | Working as designed. Trait-based polymorphism would propagate generic parameters through ~15 files for marginal type-safety gain. |
| `completion_adapter.rs` rewrite | Pure adapter, orthogonal to this work. *(Note: 475 lines is large for an adapter; quick audit recommended during step 7 to confirm no hidden protocol logic should migrate out.)* |
| HTTP / gRPC code unification | Different RFC; HTTP path today is proxy, not pipeline. |
| Responses subtree re-architecture | Step 7 is a relocation, not a rewrite. The Responses API has its own design problems (e.g., the streaming event emitter is a 638-line state machine); they are out of scope here. |

---

## 6. Acceptance Criteria

1. **Zero `mesh_grpc::*` imports outside `routers/grpc/engine/`.** Verified by `grep -rl 'mesh_grpc::' routers/` — every hit must be under `routers/grpc/engine/`. `GenerationPayload` and `TokenChunk` are the boundary types.
2. **`routers/{prepare,render,worker_stream,openai/responses}/` do not depend on `routers/grpc/`.** Verified by `grep -rE 'crate::routers::grpc' routers/{prepare,render,worker_stream}/` must be empty, and `routers/openai/responses/` follows §6.11. This is the physical check that backs Goal #1 (independence).
3. **No god-bag struct of `Option<T>` fields representing pipeline-stage state.** `prepare_*` returns a tuple `(GenerationPayload, ResponseContext)`; each member is atomically constructed. `Option` only marks data that is semantically optional (chat-only fields when handling generate, tool_constraints when no tools).
4. **`Pipeline::execute_chat` and `Pipeline::execute_generate` read top-to-bottom as four function calls.** No `Vec<Box<dyn PipelineStage>>`, no runtime stage indexing.
5. **Every file in `routers/{grpc,prepare,render,worker_stream,openai/responses}/` has a one-sentence purpose stated at the top of the file, and contains nothing outside that purpose.** Line count is a leading indicator, not the goal — if a file is long but every function clearly serves its stated purpose, that is acceptable; if a file is short but mixes two concerns, it must still be split.
6. **No file named `utils.rs`, `common.rs`, `helpers.rs`, or `chunk.rs` / `payload.rs` exists inside the refactored tree.** Verified by `find routers/{grpc,prepare,render,worker_stream,openai} -name 'utils.rs' -o -name 'common.rs' -o -name 'helpers.rs' -o -name 'chunk.rs' -o -name 'payload.rs'` must be empty. Each transport-bound helper goes to its specific functional file under `grpc/engine/`; each transport-neutral helper goes to a job-named file under `routers/prepare/` or `routers/render/`.
7. **Protocol logic on both sides of the engine is unit-testable without `mesh_grpc::*` in scope.** Concretely: `routers::prepare::prepare_chat` can be tested against `(GenerationPayload, ResponseContext)` outputs, and `routers::render::*` functions can be tested against synthetic `Vec<TokenChunk>` / `Stream<TokenChunk>` inputs — neither test needs tonic or a worker.
8. **Proto snapshot tests** verify `to_sglang_proto(payload)` and `to_vllm_proto(payload)` produce byte-identical proto serializations to today's `SglangSchedulerClient::build_generate_request_from_chat(...)` / `VllmEngineClient::build_generate_request_from_chat(...)` (defined in external crate `smg-grpc-client`).

    **Scenarios** — chosen for **field-coverage orthogonality**, not surface variety. Removing any one scenario must leave a class of fields untested:

    | # | Scenario | Field classes locked by this snapshot |
    |---|---|---|
    | A | chat + tools + logprobs + non-default sampling | full sampling params (`temperature`, `top_p`, `top_k`, `repetition_penalty`), tool grammar/regex constraint, `output_logprobs`, `top_logprobs` |
    | B | generate (raw prompt) + stop array + `input_logprobs` | raw-generate path, stop sequence encoding (array vs join), `input_logprob=true` (PD-critical), `max_new_tokens` |
    | C | PD pair + bootstrap metadata + `n>1` (SGLang) | full `DisaggregatedParams` (`bootstrap_host`, `bootstrap_room`, `kv_port`), n-way fan-out (vLLM scenarios drop to n=1) |
    | D | vLLM-specific sampling | vLLM-only fields not present on SGLang (`min_p`, `length_penalty`, `seed`, `ignore_eos`, etc.) |

    **Self-check rule**: if any 3 scenarios already cover what the 4th covers, replace the 4th with a different field class. **This is the central design hypothesis; if any scenario fails byte-equality, the boundary is wrong and Step 2b must not proceed until the divergence is resolved (see §7 Step 2a spike for early-warning).**
9. **`merge_pd_streams` implementation and tests conform to `2026-05-19-grpc-pd-merge-spec.md`.** Test obligations T1–T7 from that doc must all pass.
10. **`WorkerStream::Drop` propagates cancellation to all underlying gRPC streams.** Verified by single-mode test (drop mid-flight, observe H2 RST_STREAM) and PD-mode test (drop mid-flight, observe BOTH prefill and decode cancelled).
11. **`routers/openai/responses/` may reference `routers::grpc` only via the concrete `Pipeline` type.** Verified by `grep -rE 'crate::routers::grpc::[a-z_]+' routers/openai/responses/`: every hit must be `grpc::Pipeline` or `grpc::pipeline::Pipeline`. Any other `grpc::` reference violates layering.
12. **`routers/grpc/` does not reference `routers/openai/`.** Verified by `grep -r 'crate::routers::openai' routers/grpc/` must be empty.
13. **`routers/openai/responses/` does not import `mesh_grpc::*`.** Verified by `grep -r 'mesh_grpc::' routers/openai/responses/` must be empty.
14. **Chat SSE bytes are identical before and after the refactor.** Snapshot tests record `pipeline.execute_chat(streaming=true)` output bytes from main branch, replay against new pipeline, must match byte-for-byte. This is the precondition that allows the Responses streaming layer to move without changes.
15. **All existing tests pass; full `/mesh-e2e-test` matrix passes** (sglang/vllm × regular/PD × chat/generate × streaming/non-streaming, plus `/v1/responses` smoke).

---

## 7. Migration Posture

Strangler-fig, not big-bang. Each step is independently shippable and gated by e2e. Double-wired states are the safety mechanism; they end within the same milestone.

| Step | Content | Gate |
|---|---|---|
| 1 | Create empty `routers/{prepare,render,worker_stream}/` (`mod.rs` only). Move transport-neutral helpers from `grpc/utils.rs` into their final files under `routers/prepare/` (`chat_template.rs`, `tool_constraints.rs`, `stop_sequence_decoder.rs`, `parser_factory_lookup.rs`) and `routers/render/finish_reason_mapping.rs`. Update import sites under `grpc/`. **No new types.** | Existing tests pass. **Cheapest reality check on the premise**: if these helpers can't move out cleanly, the architecture is wrong before any new types are introduced. |
| **2a (spike)** | **Single-scenario byte-equality spike.** Write **one** fixture (chat + tools + logprobs, drawn from §6.8 scenario A). Implement a minimal `GenerationPayload` (in `routers/prepare/generation_payload.rs`) + `to_sglang_proto` prototype (in `grpc/engine/payload_to_proto.rs` — at this point `grpc/engine/` is a new subdir holding only this file plus `mod.rs`) and compare its `prost::Message::encode_to_vec` output to `SglangSchedulerClient::build_generate_request_from_chat(...)` from `smg-grpc-client`. Decide the path forward **before** writing the remaining 3 scenarios or any production code. | One of three outcomes documented in PR: (i) **byte-equal** → proceed to Step 2b unchanged; (ii) **small deterministic diff** → align field-by-field, document each alignment as a `// match upstream default` comment for future `smg-grpc-client` upgrade audits; (iii) **diff from external-crate internals** → choose either (a) upstream a `to_proto(payload)` API to `smg-grpc-client` (cleanest, requires version coordination) or (b) keep `to_sglang_proto` as a thin shim that internally calls `build_generate_request_from_chat` (preserves single source of truth at the cost of "pure field copy" purity). Either way: re-revise the relevant §3.2 paragraph before Step 2b. |
| **2b** | Bulk-add full `GenerationPayload` and both `to_sglang_proto` / `to_vllm_proto` adapters in `grpc/engine/payload_to_proto.rs`. Replace scattered `build_*_request` paths with `GenerationPayload` construction in `routers::prepare::prepare_*`. | **§6.8 proto snapshot tests pass for all 4 scenarios** (matrix locked by orthogonality rule). |
| 3 | Add `routers/worker_stream/{token_chunk,worker_stream,engine_error}.rs`, add `routers/prepare/response_context.rs`, add `grpc/engine/{mod.rs (dispatch), proto_to_chunk.rs}`. Move `grpc/client.rs` → `grpc/engine/worker_client_cache.rs`; move `grpc/proto_wrapper.rs` → `grpc/engine/proto_stream_wrapper.rs`. Not wired into Pipeline yet. Implement `grpc/engine/pd_stream_merge.rs` per PD merge spec, with T1–T7 passing. | Unit tests + §6.9 PD merge tests pass. |
| 4 | Build `routers/render/{chat_streaming, chat_aggregator, generate_streaming, generate_aggregator}.rs` consuming `Vec<TokenChunk>` / `Stream<TokenChunk>`. Not wired into Pipeline yet. | Unit tests pass against synthetic chunk inputs. |
| 5 | Add new `grpc/pipeline.rs::Pipeline` (4 explicit steps) alongside old `RequestPipeline`. Switch `grpc/http_router.rs` (regular) over first. | `/mesh-e2e-test` regular matrix passes. |
| 6 | Switch `grpc/http_router_pd.rs` and `/v1/responses` over to new pipeline. **Record chat SSE byte snapshots from main branch BEFORE switching, replay AFTER switching — §6.14 must pass.** This is the precondition for step 7. | PD e2e passes + SSE byte snapshots match. |
| 7 | **Relocate OpenAI Responses subtree**: create `routers/openai/mod.rs`, move `grpc/{common,regular}/responses/` → `routers/openai/responses/`, apply renames and the one streaming-file merge per §3.4. Update `grpc/http_router.rs` import (`crate::routers::openai::responses::*`). Rename `grpc/router.rs` → `grpc/http_router.rs` and `grpc/pd_router.rs` → `grpc/http_router_pd.rs` (deferred to this step to minimize churn in earlier steps). **No business-logic changes — pure relocation + rename.** | §6.11 / §6.12 / §6.13 layering checks pass; `/v1/responses` smoke tests unchanged. |
| 8 | Delete old `RequestPipeline`, `PipelineStage` trait, `grpc/common/stages/`, `grpc/regular/stages/`, `grpc/regular/streaming.rs`, `grpc/regular/processor.rs`, `grpc/context.rs::ProcessingState`, `grpc/utils.rs` (now empty after Steps 1+3 drained its transport-neutral and transport-bound contents respectively), and the now-empty `grpc/common/` and `grpc/regular/` directories. | Compile + full e2e + `cargo-machete` / `cargo udeps` show no orphans + `find routers/grpc -name 'utils.rs' -o -name 'common.rs'` empty (§6.6). |
