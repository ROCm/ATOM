# gRPC Router Refactoring — Detailed Plan

**Date**: 2026-05-19
**Scope**: `atom/mesh/src/routers/grpc/`
**Supersedes**: `docs/2025-05-19-grpc-refactor-design.md` (kept as historical record)

---

## 1. Why Revise the 2025-05-19 Design

After re-reading every file in `routers/grpc/`, four claims in the prior design no longer match reality:

1. **File size**: `regular/streaming.rs` is **1326 lines / 57 KB**, not "57K lines". Every file in `grpc/` already satisfies the "no file > 1500 lines" success criterion.
2. **Stage decomposition**: `regular/stages/{chat,generate}/` and `common/stages/` already exist; the pipeline has already been broken down past what the prior design proposed.
3. **`ResponseProcessor` move**: `regular/processor.rs` imports `mesh_grpc::sglang_proto::generate_complete::MatchedStop`, `ProtoGenerateComplete`, `ExecutionResult`, `common::{response_collection, response_formatting}`. Moving it to `routers/shared/` is not a "pure file move" — it requires first abstracting the proto-completion shape into a trait.
4. **Phase B rationale**: the prior design justified `ProtocolProcessor` trait extraction by "a future HTTP router can reuse it". The current `http_router.rs` is a `reqwest::Client` HTTP-to-HTTP proxy that does not tokenize, apply chat templates, or parse tool calls — the HTTP backend does. The parallel proposal `http_grpc_mesh_router_plan.md` explicitly keeps HTTP and gRPC execution paths protocol-specific. The reuse premise does not hold.

This revision drops `ResponseProcessor` extraction and `ProtocolProcessor` trait, narrows the file-move scope to genuinely proto-free helpers, and adds two architectural improvements (Phase D + E) that address the real coupling.

---

## 2. Real Problems To Solve

| # | Problem | Where it bites |
|---|---------|----------------|
| P1 | Three files mix unrelated responsibilities | `utils.rs` (1214 lines), `streaming.rs` (1326 lines), `processor.rs` (465 lines) |
| P2 | `ProcessingState` is a god-bag of 7 `Option<T>` fields with implicit dependencies | Every stage opens with 8–12 lines of `.ok_or_else(...)` boilerplate; stage dependencies are not visible in signatures |
| P3 | The "unified 7-stage pipeline" is a fiction — 3 of the 7 stages internally re-dispatch on `RequestType` | `PreparationStage`, `RequestBuildingStage`, `ResponseProcessingStage` each contain a hidden chat/generate fork |

Three things the prior design also called out but that we intentionally do NOT touch:

- `ProtocolProcessor` trait extraction (single-implementation trait, no second consumer)
- `ResponseProcessor` move to `routers/shared/` (proto-coupled at type level; speculative-generality trait abstraction not justified by current scope)
- The streaming SSE side-channel pattern (where `process_streaming_response` spawns a background task and returns `Some(response)` from the pipeline — orthogonal concern, deserves its own RFC)

---

## 3. The Five Phases

| Phase | Goal | Files touched | Risk | Depends on |
|-------|------|---------------|------|------------|
| **A** | Decompose `utils.rs`; lift truly proto-free helpers to `routers/shared/protocol/` | ~7 new + 1 modified | Low | — |
| **B** | Decompose `streaming.rs` into 5 responsibility-scoped files | ~5 new + 1 directory conversion | Medium | A complete |
| **C** | Decompose `processor.rs` into `processor/{mod,chat,generate}.rs` | 3 new (replaces 1) | Low–Medium | A complete; parallel with B |
| **D** | Replace `ProcessingState`'s 7 `Option` fields with 4 stage-grouped outcome structs | `context.rs` + ~10 stage files | Medium | A, B, C complete |
| **E** | Split unified `RequestPipeline` into `ChatPipeline` + `GeneratePipeline`; delete 3 wrapper stages that only re-dispatch | `pipeline.rs`, `router.rs`, `pd_router.rs`, 3 stage files deleted | Medium–High | D complete |

Total: **10 PRs**, sequenced so each is independently shippable and reversible.

---

## 4. Phase A — `utils.rs` Decomposition

### 4.1 Function-level classification

Every function in `utils.rs` was inspected. Classification (P = proto-agnostic, lift to `shared/`; G = gRPC-bound, keep in `grpc/`):

| Function | Class | Reason | Target file |
|----------|:-:|--------|-------------|
| `process_chat_messages` | P | `ChatCompletionRequest` + `Tokenizer` only | `shared/protocol/chat_template.rs` |
| `process_content_format` | P | `ChatMessage` + `serde_json` | `shared/protocol/chat_template.rs` |
| `process_tool_call_arguments` (priv) | P | `serde_json` only | `shared/protocol/chat_template.rs` |
| `transform_content_field` (priv) | P | `serde_json` only | `shared/protocol/chat_template.rs` |
| `generate_tool_constraints` | P | `Tool` + `ToolChoice` + JSON | `shared/protocol/tool_constraints.rs` |
| `build_required_array_schema` (priv) | P | `Tool` + JSON | `shared/protocol/tool_constraints.rs` |
| `filter_tools_by_tool_choice` | P | `Tool` + `ToolChoice` | `shared/protocol/tool_constraints.rs` |
| `filter_chat_request_by_tool_choice` | P | `ChatCompletionRequest` | `shared/protocol/tool_constraints.rs` |
| `parse_json_schema_response` | P | text + `ToolChoice` + `ToolCall` | `shared/protocol/tool_constraints.rs` |
| `create_stop_decoder` | P | `Tokenizer` + `StringOrArray` | `shared/protocol/stop_decoder.rs` |
| `check_reasoning_parser_availability` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `check_tool_parser_availability` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `get_reasoning_parser` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `create_reasoning_parser` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `get_tool_parser` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `create_tool_parser` | P | `ParserFactory` | `shared/protocol/parsers.rs` |
| `parse_finish_reason` | P | `GenerateFinishReason` + serde | `shared/protocol/finish_reason.rs` |
| `generate_tool_call_id` | P | `String` + uuid | `shared/protocol/finish_reason.rs` |
| `get_history_tool_calls_count` | P | `ChatCompletionRequest` | `shared/protocol/finish_reason.rs` |
| `convert_proto_to_openai_logprobs` | G | imports `mesh_grpc::sglang_proto::OutputLogProbs` | `grpc/logprobs.rs` |
| `convert_generate_output_logprobs` | G | imports `OutputLogProbs` | `grpc/logprobs.rs` |
| `convert_generate_input_logprobs` | G | imports `InputLogProbs` | `grpc/logprobs.rs` |
| `resolve_tokenizer` | G | depends on `RequestContext` | `grpc/utils.rs` (kept) |
| `get_grpc_client_from_worker` | G | depends on `GrpcClient` | `grpc/utils.rs` (kept) |
| `collect_stream_responses` | G | depends on `ProtoStream` | `grpc/utils.rs` (kept) |
| `error_type_from_status` re-export | G | cross-module re-export | `grpc/utils.rs` (kept) |

### 4.2 Target layout

```
routers/
├── shared/
│   └── protocol/                    ← new
│       ├── mod.rs
│       ├── chat_template.rs         ~250 lines (6 existing tests move here)
│       ├── tool_constraints.rs      ~210 lines
│       ├── stop_decoder.rs          ~50 lines
│       ├── parsers.rs               ~130 lines
│       └── finish_reason.rs         ~80 lines
└── grpc/
    ├── utils.rs                     ~250 lines (down from 1214)
    └── logprobs.rs                  ~110 lines (new)
```

### 4.3 PR breakdown

| PR | Scope | Estimated diff |
|----|-------|----------------|
| **A1** | Create `shared/protocol/` directory with all 5 files; delete migrated functions from `utils.rs`; update all caller imports in one pass | ~700 lines |
| **A2** | Extract `logprobs.rs` (intra-`grpc/` move); update callers | ~200 lines |
| **A3** | Add module-level docs to `utils.rs` explaining what stays and why | ~20 lines |

### 4.4 Verification

After each PR:
1. `cargo test -p mesh` (note: 6 tests in old `utils.rs` `test_transform_messages_*` migrate to `chat_template.rs`)
2. `cargo clippy -p mesh --all-targets -- -D warnings`
3. After A1: `/mesh-e2e-test` sglang regular chat (non-streaming + streaming)

---

## 5. Phase B — `streaming.rs` Decomposition

### 5.1 Section analysis

| Section | Current lines | Size | Role |
|---------|---------------|------|------|
| ① `StreamingProcessor` struct + `new()` + `process_streaming_response` entry | L40–174 | 135 | Public API + background task dispatch |
| ② Chat main loop `process_streaming_chunks` | L177–591 | 415 | 5-phase chat SSE loop |
| ③ PD chat `process_dual_streaming_chunks` | L595–643 | 49 | Prefill consume + delegate to ② |
| ④ Generate streaming entry + single/dual variants | L651–1022 | 372 | Generate SSE (with logprobs variant) |
| ⑤ Helper methods (chunk_tokens, reasoning_stream, specific_function_stream, tool_calls_stream) | L1024–1290 | 267 | Three classes of sub-stream processing |
| ⑥ `format_sse_chunk_into` + `build_sse_response` | L1292–1326 | 35 | SSE encoding infrastructure |

### 5.2 Target layout

```
regular/
└── streaming/                       ← streaming.rs becomes a directory
    ├── mod.rs                       ~150 lines (StreamingProcessor struct + entry methods)
    ├── sse.rs                       ~50 lines (build_sse_response + format_sse_chunk_into)
    ├── chat.rs                      ~470 lines (chat single + dual)
    ├── generate.rs                  ~400 lines (generate single + dual + with-logprobs variant + GenerateStreamContext + metrics)
    └── helpers.rs                   ~270 lines (chunk_tokens + reasoning + specific_function + tool_calls)
```

### 5.3 Technical decisions (locked)

- Extracted methods become `pub(super) fn` **free functions** that take `&StreamingProcessor` as their first parameter. Rationale: function signatures make dependencies explicit; per-file readability does not require opening `mod.rs`.
- `StreamingProcessor`'s public API (`process_streaming_response`, `process_streaming_generate`) stays in `mod.rs`. External callers (`pipeline.rs`, `stages/chat`, `stages/generate`) see no change.
- `chat.rs` at ~470 lines is **not** further split into init/loop/finalize — additional fragmentation would scatter intent without proportional gain.

### 5.4 PR breakdown

| PR | Scope | Estimated diff |
|----|-------|----------------|
| **B1** | Create `streaming/` directory; extract `sse.rs` + `helpers.rs` (simplest pure-function moves) | ~350 lines |
| **B2** | Extract `chat.rs` (largest, most complex) | ~500 lines |
| **B3** | Extract `generate.rs`; convert remaining `streaming.rs` to `streaming/mod.rs` | ~450 lines |

### 5.5 Verification

After each PR (mandatory because SSE output is user-visible):
1. `cargo test -p mesh`
2. sglang regular chat **streaming** e2e
3. sglang PD chat **streaming** e2e
4. vllm regular generate **streaming** e2e
5. sglang PD generate **streaming** e2e (after B3)

---

## 6. Phase C — `processor.rs` Internal Decomposition

### 6.1 Section analysis

| Section | Current lines | Size | Role |
|---------|---------------|------|------|
| ① `ResponseProcessor` struct + `new()` | L37–57 | 21 | Public API |
| ② `process_single_choice` | L60–209 | 150 | Single choice: decode + reasoning + tool + finish_reason + logprobs + assemble |
| ③ `process_non_streaming_chat_response` | L212–301 | 90 | Chat entry: collect + per-choice loop |
| ④ `parse_tool_calls` | L304–358 | 55 | Tool parser invocation |
| ⑤ `process_non_streaming_generate_response` | L361–464 | 104 | Generate entry: collect + per-completion decode + assemble |

### 6.2 Target layout

```
regular/
└── processor/                       ← processor.rs becomes a directory
    ├── mod.rs                       ~80 lines (ResponseProcessor struct + entry methods routing to chat/generate)
    ├── chat.rs                      ~250 lines (process_single_choice + process_non_streaming_chat + parse_tool_calls)
    └── generate.rs                  ~110 lines (process_non_streaming_generate)
```

`parse_tool_calls` ships with `chat.rs` because it is only called from chat flow.

### 6.3 Technical decision (locked)
Same convention as Phase B: extracted functions become `pub(super) fn` taking `&ResponseProcessor`; public API stays in `mod.rs`.

### 6.4 PR breakdown

| PR | Scope | Estimated diff |
|----|-------|----------------|
| **C1** | Single PR: convert `processor.rs` to `processor/{mod,chat,generate}.rs` | ~500 lines |

C1 can be developed in parallel with B (no shared file).

### 6.5 Verification
1. `cargo test -p mesh`
2. sglang regular chat **non-streaming** e2e
3. sglang regular generate **non-streaming** e2e
4. sglang PD chat **non-streaming** e2e

---

## 7. Phase D — `ProcessingState` Regrouping

### 7.1 Current vs target

**Current** (`context.rs`):
```rust
pub(crate) struct ProcessingState {
    pub preparation: Option<PreparationOutput>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub workers: Option<WorkerSelection>,
    pub clients: Option<ClientSelection>,
    pub proto_request: Option<ProtoRequest>,
    pub dispatch: Option<DispatchMetadata>,
    pub load_guards: Option<LoadGuards>,
    pub response: ResponseState,
}
```

**Target**:
```rust
pub(crate) struct ProcessingState {
    /// Written by PreparationStage, read by RequestBuildingStage + ResponseProcessingStage
    pub prepared: Option<PreparedRequest>,

    /// Written by WorkerSelection + ClientAcquisition, read by RequestBuilding + RequestExecution
    pub placement: Option<Placement>,

    /// Written by RequestBuilding + DispatchMetadata, read by RequestExecution
    pub dispatch: Option<Dispatch>,

    /// Written by RequestExecution, consumed by ResponseProcessing
    pub execution: Option<Execution>,

    /// Written by ResponseProcessing, consumed by pipeline orchestrator
    pub final_response: Option<FinalResponse>,
}

pub(crate) struct PreparedRequest {
    pub original_text: Option<String>,
    pub token_ids: Vec<u32>,
    pub processed_messages: Option<ProcessedMessages>,
    pub tool_constraints: Option<(String, String)>,
    pub filtered_request: Option<ChatCompletionRequest>,
    pub tokenizer: Arc<dyn Tokenizer>,
    pub stop_decoder: StopSequenceDecoder,
}

pub(crate) struct Placement {
    pub workers: WorkerSelection,
    pub clients: ClientSelection,
}

pub(crate) struct Dispatch {
    pub proto_request: ProtoRequest,
    pub metadata: DispatchMetadata,
}

pub(crate) struct Execution {
    pub result: ExecutionResult,
    pub load_guards: LoadGuards,
}
```

Key changes:
- `tokenizer` and `stop_decoder` move into `PreparedRequest` (their lifecycle matches preparation)
- `WorkerSelection` + `ClientSelection` merge into `Placement` (always set together by adjacent stages)
- `proto_request` + `DispatchMetadata` merge into `Dispatch`
- `ExecutionResult` + `LoadGuards` merge into `Execution`
- `ResponseState`'s `stop_decoder` field is gone (moved to `PreparedRequest`); `execution_result` moves to `Execution`; remaining `final_response` is promoted to top-level

### 7.2 Stage boilerplate reduction

**Before** (e.g. `ChatRequestBuildingStage::execute`):
```rust
let prep = ctx.state.preparation.as_ref().ok_or_else(|| ...)?;
let clients = ctx.state.clients.as_ref().ok_or_else(|| ...)?;
// workers fetched later via ctx.state.workers.as_ref().unwrap() inside the body
```

**After**:
```rust
let prepared = ctx.state.prepared.as_ref().ok_or_else(|| ...)?;
let placement = ctx.state.placement.as_ref().ok_or_else(|| ...)?;
// placement.workers, placement.clients — one fetch, both available
```

### 7.3 PR breakdown

| PR | Scope | Estimated diff |
|----|-------|----------------|
| **D1** | Single PR: rewrite `context.rs` `ProcessingState`; update all stage files and `pipeline.rs` final-response read | ~600 lines across ~12 files |

Single PR is required — type-level breaking change cannot be incrementally migrated.

### 7.4 Verification
1. `cargo test -p mesh`
2. `cargo clippy -p mesh --all-targets -- -D warnings`
3. Full `/mesh-e2e-test` matrix: sglang regular chat / generate / streaming, sglang PD chat / generate / streaming, vllm equivalents

---

## 8. Phase E — `ChatPipeline` + `GeneratePipeline` Split

### 8.1 The wrapper-stage problem

`PreparationStage`, `RequestBuildingStage`, `ResponseProcessingStage` each look like a pipeline stage but their `execute()` is purely:

```rust
match ctx.input.request_type {
    RequestType::Chat(_) => self.chat_stage.execute(ctx).await,
    RequestType::Generate(_) => self.generate_stage.execute(ctx).await,
}
```

This is a runtime dispatch where compile-time selection would suffice — `execute_chat()` and `execute_generate()` already know which variant they need.

### 8.2 Target shape

```rust
// pipeline.rs

pub(crate) struct ChatPipeline {
    // Shared stages (cloned at construction)
    worker_selection: WorkerSelectionStage,
    client_acquisition: ClientAcquisitionStage,
    dispatch_metadata: DispatchMetadataStage,
    request_execution: RequestExecutionStage,
    // Chat-specific stages (held directly, no wrapper)
    preparation: ChatPreparationStage,
    request_building: ChatRequestBuildingStage,
    response_processing: ChatResponseProcessingStage,
    backend_type: &'static str,
}

pub(crate) struct GeneratePipeline {
    worker_selection: WorkerSelectionStage,
    client_acquisition: ClientAcquisitionStage,
    dispatch_metadata: DispatchMetadataStage,
    request_execution: RequestExecutionStage,
    preparation: GeneratePreparationStage,
    request_building: GenerateRequestBuildingStage,
    response_processing: GenerateResponseProcessingStage,
    backend_type: &'static str,
}
```

Each pipeline has a constructor `new_regular(...)` and `new_pd(...)` that pre-configures shared and specific stages.

### 8.3 Execute implementation (locked: option β)

Retain `PipelineStage` trait. Each pipeline's `execute()` builds a local `[&dyn PipelineStage; 7]` from its fields and runs the existing generic loop:

```rust
impl ChatPipeline {
    pub async fn execute(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let stages: [&dyn PipelineStage; 7] = [
            &self.preparation,
            &self.worker_selection,
            &self.client_acquisition,
            &self.request_building,
            &self.dispatch_metadata,
            &self.request_execution,
            &self.response_processing,
        ];
        // Generic loop (same as today's RequestPipeline) — metrics, error handling, streaming early-exit
    }
}
```

The metrics/error-handling loop body is identical to today's; only the stage collection construction changes.

### 8.4 What gets deleted

- `regular/stages/preparation.rs` — wrapper, replaced by direct use of `ChatPreparationStage`/`GeneratePreparationStage`
- `regular/stages/request_building.rs` — wrapper, replaced
- `regular/stages/response_processing.rs` — wrapper, replaced
- Old `RequestPipeline` struct in `pipeline.rs` — replaced by `ChatPipeline` + `GeneratePipeline`

### 8.5 PR breakdown

| PR | Scope | Estimated diff |
|----|-------|----------------|
| **E1** | Add `ChatPipeline` + `GeneratePipeline` alongside existing `RequestPipeline` (parallel implementations); wire `factory.rs` to expose both | ~400 lines |
| **E2** | Switch `router.rs` + `pd_router.rs` + responses-endpoint to use new pipelines; delete `RequestPipeline` and the 3 wrapper stages | ~150 lines (mostly deletions) |

### 8.6 Verification

After E2 (full pipeline replacement):
1. `cargo test -p mesh`
2. Full `/mesh-e2e-test` matrix (8 combinations)
3. `/v1/responses` non-streaming endpoint smoke test (uses `execute_chat_for_responses` path)

---

## 9. Out of Scope (Explicit Non-Goals)

| Item | Reason |
|------|--------|
| `ProtocolProcessor` trait | Single-implementation trait without a second consumer — speculative generality |
| `ResponseProcessor` extraction to `routers/shared/` | Requires first abstracting `ProtoGenerateComplete` into a trait; HTTP router does not need it (it proxies, not processes) |
| `proto_wrapper.rs` enum-wrapping | Working as designed; trait-based polymorphism would cost ~15 files of generic propagation for marginal type-safety gain |
| `regular/responses/` (the `/v1/responses` endpoint) | Independent endpoint, separate concern |
| `completion_adapter.rs` | Completion-proto adapter, not in the refactor scope |
| Streaming SSE side-channel pattern | Real architectural problem (`process_streaming_response` spawns background task and exits pipeline early) but orthogonal to file decomposition; deserves its own RFC |
| Reuse of gRPC protocol code by HTTP router | Belongs to `http_grpc_mesh_router_plan.md` (PD placement unification effort) |

---

## 10. Risk Register

| Risk | Affected phase | Mitigation |
|------|----------------|------------|
| Import-path churn breaks downstream crates | A1 | Run `cargo build` on the full workspace, not just `mesh` crate |
| SSE output regression invisible to unit tests | B1, B2, B3 | Each PR runs at least 2 streaming e2e tests; output bytes compared against baseline |
| `ProcessingState` field migration touches every stage in one PR | D1 | Use IDE rename refactor for `ctx.state.workers` → `ctx.state.placement.workers` etc.; type checker catches all sites |
| New pipelines have subtly different ordering from old | E1 | Each pipeline runs the same 7 stages in the same order; diff against `pipeline.rs:78-86` and `pipeline.rs:123-131` for parity |
| `/v1/responses` endpoint not migrated correctly | E2 | Add explicit smoke test for `execute_chat_for_responses` path before merging |
| Merge conflicts accumulate across long-running phase branches | All | Each PR is independently mergeable; do not stack PRs on long-lived branches |

---

## 11. Success Criteria

1. No file in `routers/grpc/` exceeds **400 lines** (current `streaming.rs` 1326, `utils.rs` 1214, `processor.rs` 465 → all reduced)
2. `routers/grpc/utils.rs` contains only gRPC-bound utilities (~250 lines)
3. `routers/shared/protocol/` exists and is importable without depending on `mesh_grpc::*`
4. `ProcessingState` field types document their producer-stage and consumer-stage in code (not in this doc)
5. `pipeline.rs` reading top-to-bottom shows the actual shape of `ChatPipeline` and `GeneratePipeline` without requiring further file dives
6. All existing tests pass; full `/mesh-e2e-test` matrix passes
7. The 3 wrapper stages (`PreparationStage`, `RequestBuildingStage`, `ResponseProcessingStage`) are deleted

---

## 12. PR Sequence Summary

| # | PR | Phase | Risk | Depends on |
|---|----|-------|------|------------|
| 1 | A1 — shared/protocol/ + utils slim | A | Low | — |
| 2 | A2 — logprobs.rs extraction | A | Low | A1 |
| 3 | A3 — utils.rs module docs | A | Trivial | A2 |
| 4 | B1 — streaming/sse + streaming/helpers | B | Medium | A3 |
| 5 | B2 — streaming/chat | B | Medium | B1 |
| 6 | B3 — streaming/generate + finalize | B | Medium | B2 |
| 7 | C1 — processor/{chat,generate} | C | Low–Med | A3 (parallel with B) |
| 8 | D1 — ProcessingState regrouping | D | Medium | B3, C1 |
| 9 | E1 — ChatPipeline + GeneratePipeline (parallel impl) | E | Med–High | D1 |
| 10 | E2 — switch + delete wrappers | E | Medium | E1 |

**Estimated total**: 10 PRs over 3–4 weeks (A: week 1; B + C parallel: week 2; D: week 3; E: week 4).
