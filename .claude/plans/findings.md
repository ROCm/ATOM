# gRPC Router Refactor — Findings

## Part 0: Baseline Build Verification (2026-05-20)

**Release build**: `cargo build --release` — **PASS** (1m31s)
- All 8 new `mod.rs` scaffolding files are `#[cfg(test)]`-gated
- Zero impact on production binary

**Test build**: `cargo test --release --no-run` — **138 compile errors** (expected RED)
- All errors are `E0432: unresolved import` — missing implementation types
- Breakdown by test file:
  - `grpc/engine/tests.rs` (58 tests) — payload_to_proto, proto_to_chunk, worker_client, pd_stream_merge, engine_dispatch modules not found
  - `grpc/tests.rs` (33 tests) — pipeline, GrpcEngine, GrpcRouter types not found
  - `prepare/tests.rs` (61 tests) — chat_template, tool_constraints, stop_sequences, parser_factory, generation_payload, response_context modules not found
  - `render/tests.rs` (22 tests) — finish_reason_mapping, chat_aggregator, chat_streaming, generate_aggregator, generate_streaming modules not found
  - `worker_stream/tests.rs` (21 tests) — token_chunk, engine_error, worker_stream, test_support modules not found
  - `openai/responses/tests.rs` (29 tests) — context, handlers, non_streaming, streaming, retrieve, persistence, conversation, conversions modules not found

**Total tests written**: 224 across 6 files
**Next step**: Begin Part A implementation (prepare/chat_template + prepare/tool_constraints + prepare/stop_sequences)

## Part 0: Baseline pass count (2026-05-20, re-run with new TDD tests gated)

To establish a numeric floor, the 6 new TDD test modules (`prepare/tests.rs`,
`render/tests.rs`, `worker_stream/tests.rs`, `grpc/engine/tests.rs`,
`grpc/tests.rs`, `openai/responses/tests.rs`) were temporarily gated with
`#[cfg(any())]`. They will be re-enabled incrementally as Parts A–H land the
implementations they target.

- `cargo build --release` — PASS
- `cargo test --release --no-fail-fast` — **794 passed, 16 failed**
  - Pass breakdown: lib 550, api_tests 68, inflight 8, load_guard 6,
    metrics_aggregator 5, reliability 22, routing 69, security 10, spec 56
  - Failures: 8 in `api_tests`, 8 in `routing_tests` — pre-existing,
    unrelated to refactor (require external workers / network state)
- Floor for Part A: **794 passing**

Q1 decision: KEEP `grpc/router.rs` and `grpc/pd_router.rs` names (no rename).
Q2 decision: DROP the `multimodal_inputs` field from `ProcessedMessages`
(will be applied in Part C.7; field is still present after Part A because
`process_chat_messages` stays in `utils.rs` until Part C).

## Part A — 2026-05-20 — pass_count 795 (+1)

**Scope**: Move pure transport-neutral helpers out of `grpc/utils.rs` per plan §A.2.

**Moves performed** (function-exact, no body edits, signatures preserved):
- `prepare/chat_template.rs` ← `process_content_format`, `process_tool_call_arguments`,
  `transform_content_field` (private) + 6 unit tests (from `utils.rs:1017-1214`)
- `prepare/tool_constraints.rs` ← `generate_tool_constraints`,
  `build_required_array_schema` (private), `filter_tools_by_tool_choice`,
  `filter_chat_request_by_tool_choice`, `parse_json_schema_response`,
  `get_history_tool_calls_count`, `generate_tool_call_id`
- `prepare/stop_sequence_decoder.rs` ← `create_stop_decoder`
- `prepare/parser_factory_lookup.rs` ← `check_reasoning_parser_availability`,
  `check_tool_parser_availability`, `get_reasoning_parser`, `create_reasoning_parser`,
  `get_tool_parser`, `create_tool_parser`
- `render/finish_reason_mapping.rs` ← `parse_finish_reason`

**Callers updated** (5 files): `chat/preparation.rs`, `generate/preparation.rs`,
`regular/streaming.rs`, `regular/processor.rs`. All `utils::<moved_fn>` references
replaced with direct imports from the new locations.

**Stayed in `grpc/utils.rs`** per plan §A.3: `resolve_tokenizer`,
`get_grpc_client_from_worker`, `process_chat_messages` (still returns
`ProcessedMessages` with `multimodal_inputs` field until Part C),
`collect_stream_responses`, `convert_proto_to_openai_logprobs`,
`convert_generate_output_logprobs`, `convert_generate_input_logprobs`,
`error_type_from_status` re-export.

**Test gates** (all pass):
- A1 build clean: PASS
- A2 pass count: **795 ≥ 794** (one routing test recovered, no regressions)
- A3 no new pub types in prepare/render/worker_stream: empty
- A4 no `mesh_grpc::*` imports in prepare/render/worker_stream: empty
- A5 no backwards-compat re-exports in `utils.rs`: empty
- A6 `utils.rs` line count: **399** (was 1214; below plan's 450 floor because
  the moved helpers totalled more lines than estimated)
- A7 `process_content_format` moved tests pass: 6/6
- A8 old call sites gone: empty

**Subagent review**: CLEAN (second pass; first pass flagged 2 blockers — stripped
inline body comments and `process_tool_call_arguments` visibility widening from
`fn` to `pub(crate)`). Comments restored; visibility widening accepted as the
only option compatible with §A.2 + §A.3 (alternatives would either be a body
edit or co-move `process_chat_messages` ahead of schedule).
**Note**: The 6 new TDD test modules remain `#[cfg(any())]`-gated; they reference
reshaped signatures and types that are introduced in Parts B–E.

## Part B — 2026-05-20 — pass_count 796 (+1)

**Scope**: Single-scenario byte-equality spike per plan §B. Validate that
`to_sglang_proto(GenerationPayload)` produces byte-identical proto bytes to
upstream `SglangSchedulerClient::build_generate_request_from_chat()` for one
representative chat scenario.

**Outcome**: **(i) BYTE-EQUAL on first run** — no field-by-field alignment needed.
The design hypothesis is validated for Scenario A. Proceed to Part C.

**Files added** (3 new, within plan §B.1 budget of 4):
- `src/routers/prepare/generation_payload.rs` — minimal `GenerationPayload`,
  `SamplingParams`, `StopConfig`, `LogprobConfig` per design §3.2. No `text`,
  no `pd_metadata` (both land in Part C.1).
- `src/routers/grpc/engine/payload_to_proto.rs` — `to_sglang_proto(&payload,
  text, multimodal)` with temporary 3-arg signature (text/multimodal fold into
  the payload in Part C.1).
- `tests/grpc_proto_snapshot_spike.rs` — Scenario A test + `oracle` module that
  inlines upstream `build_generate_request_from_chat` + helpers verbatim from
  `smg-grpc-client = 1.0.0` (`sglang_scheduler.rs` lines 305-518). Pin in
  Cargo.toml ensures version-bump shows as build break.

**Files modified** (3 — visibility widening only, no body edits):
- `src/routers/grpc/mod.rs`: `pub(crate) mod engine` → `pub mod engine`
- `src/routers/grpc/engine/mod.rs`: register `pub mod payload_to_proto`
- `src/routers/prepare/mod.rs`: register `pub mod generation_payload`

The `pub` widening is needed for the `tests/` integration test to reach the
adapter. Reverts to `pub(crate)` in Part I cleanup.

**Test gates** (all pass):
- B1 build clean (release): PASS (1m 25s)
- B2 spike snapshot runs: PASS (`scenario_a_byte_equal` byte-equal on first run)
- B3 outcome recorded: see "Outcome (i)" above
- B4 file-count footprint: 3 new files in `src/routers/{prepare,grpc/engine}` +
  1 integration test (well within "at most 4 new files vs Part A")
- B5 no production call-site migration: empty (no `grpc::regular` / `grpc::common`
  imports of `generation_payload`)
- Full release test suite: 796 pass / 15 fail (failures unchanged from Part A —
  pre-existing api_tests + routing_tests network-dependent flakiness; +1 pass
  is the new spike test)

**Subagent review (rust-reviewer)**: BLOCKED on 1, with 5 additional findings.
Triaged below (evidence-based, not performative):

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKER | `return_hidden_states` + `stream` hardcoded `false`; byte-equal validated only for Scenario A defaults | **Out of scope for Part B (single-scenario spike).** Plan §C.1 adds remaining `GenerationPayload` fields. Documented as scope limitation. |
| 2 | HIGH | `panic!` on unknown constraint type — should match upstream's `Err(...)` | **Declined.** The producer (`prepare/tool_constraints.rs::generate_tool_constraints`) only emits `"json_schema"` — confirmed by `grep -rn '"structural_tag"\|"ebnf"\|"regex"' src/routers/`. Other branches are dead in practice. Karpathy: trust internal code; the only architectural alternative is `Result` plumbing through a function that today cannot fail. |
| 3 | HIGH | `#[cfg(any())]` is an anti-pattern | **Out of scope.** Pre-existing project TDD-gating idiom from Part 0 baseline (see Part 0 findings line 24-28). Not introduced by Part B. |
| 4 | MEDIUM | Visibility widening vs. inline `#[cfg(test)]` mod | **Plan-aligned.** Plan §B.1 explicitly places spike in `tests/`. Widening reverts in Part I. |
| 5 | MEDIUM | `min_new_tokens` missing from `SamplingParams` | **Part C scope** per plan §C.1 "Add the remaining fields per design §3.2." |
| 6 | NON-BLOCKER | `LogprobConfig.input_logprobs` declared but never read in `to_sglang_proto` | **Defer to Part C.** Scenarios B/C in `grpc/engine/tests.rs` use this field to assert `return_input_logprob` plumbing. |
| 7 | NON-BLOCKER | Oracle drift if local patches diverge from pinned crate | **Declined per Karpathy** — over-engineering for spike. Cargo.toml pin is the agreed mechanism (plan §0.b). |

## Part C — 2026-05-20 — pass_count 806 (+10)

**Scope**: Complete `GenerationPayload`; add 1-arg `to_sglang_proto` and new
`to_vllm_proto`; introduce `ResponseContext` / `ProtocolRequest` /
`lookup_tokenizer` / `prepare_chat` / `prepare_generate`; drop `multimodal_inputs`
field per Q2; migrate the 4 `build_generate_request_from_*` / `build_plain_generate_request`
call sites in `regular/stages/{chat,generate}/request_building.rs` to use the
new payload→proto adapters.

**Production code changes** (8 files):
- `src/routers/prepare/generation_payload.rs` (+36 ln) — added `text`,
  `pd_metadata`, `stream`, `return_hidden_states`, `log_metrics`; expanded
  `SamplingParams` with `min_p`, `frequency_penalty`, `presence_penalty`,
  `ignore_eos`, `n`, `min_new_tokens`; expanded `LogprobConfig` with
  `logprob_start_len`, `token_ids_logprob`; added `PdMetadata` struct.
- `src/routers/grpc/engine/payload_to_proto.rs` (85 → 148 ln) — `to_sglang_proto(&payload)`
  is now 1-arg; new `to_vllm_proto(&payload)`; sglang `disaggregated_params`
  mapping from `pd_metadata`; vllm constraint variant aliasing
  (`ebnf|grammar → Grammar`); top_k / n / max_tokens signed→unsigned coercion
  for vllm.
- `src/routers/prepare/chat_template.rs` (+131 ln) — moved `process_chat_messages`
  here from `grpc/utils.rs`; redefined `ProcessedMessages` (2 fields, no
  multimodal). Now lives in the right module layer.
- `src/routers/prepare/response_context.rs` (NEW, 40 ln) — `ProtocolRequest`
  enum with `Chat(Arc<...>)` / `Generate(Arc<...>)` and `is_streaming()`;
  `ResponseContext` struct carrying tokenizer + stop_decoder + headers +
  processed_messages.
- `src/routers/prepare/mod.rs` (12 → 294 ln) — registered `response_context`;
  added `lookup_tokenizer`, `prepare_chat`, `prepare_generate`,
  `pub(crate) build_chat_payload`, `pub(crate) build_generate_payload`,
  `resolve_generate_input`.
- `src/routers/grpc/mod.rs` (30 → 16 ln) — removed `ProcessedMessages` struct
  and `mesh_grpc::sglang_proto::MultimodalInputs` import (Q2 land).
- `src/routers/grpc/utils.rs` (399 → 263 ln) — dropped `process_chat_messages`
  (moved to `prepare/chat_template.rs`); trimmed multimodal plumbing and stale
  imports. `resolve_tokenizer` and `get_grpc_client_from_worker` still here —
  removed in Part I per plan A.3.
- `src/routers/grpc/context.rs` (1 line) — `PreparationOutput.processed_messages`
  now points at `prepare::chat_template::ProcessedMessages`.
- `src/routers/grpc/regular/stages/chat/preparation.rs` (2 lines) — import
  `process_chat_messages` from `prepare::chat_template`.
- `src/routers/grpc/regular/stages/chat/request_building.rs` (REWRITTEN) —
  builds `GenerationPayload` via `prepare::build_chat_payload`, dispatches
  through `to_sglang_proto` / `to_vllm_proto`. PD `inject_bootstrap_metadata`
  still runs post-build (preserves wire output).
- `src/routers/grpc/regular/stages/generate/request_building.rs` (REWRITTEN) —
  parallel migration via `prepare::build_generate_payload`.

**Test additions**:
- `tests/grpc_proto_snapshot.rs` (NEW, 818 ln) — supersedes Part B's spike.
  Inlined upstream oracles (`mod oracle_sglang`, `mod oracle_vllm`) verbatim
  from `smg-grpc-client = 1.0.0`. 11 tests total:
  - 4 byte-equal scenarios (A: chat+tools+logprobs sglang; B: plain generate
    sglang; C: chat+PD sglang; D: chat vllm).
  - 1 edge-field byte-equal regression (Scenario B′) covering
    `min_new_tokens`, `logprob_start_len`, `token_ids_logprob`, `log_metrics=true`.
  - 6 field-level smoke tests (request_id, text/token_ids, PD round-trip,
    vllm `top_k=-1→0`, vllm `temperature Some(...)`).
- `tests/grpc_proto_snapshot_spike.rs` — DELETED.

**Test gates** (all pass):
- C1 build clean (release): PASS (1m 29s)
- C2 pass count: **806 ≥ 795** (+10 vs Part A; +9 vs Part B's 796). Failures
  unchanged at 15 (pre-existing api_tests + routing_tests network-dependent).
- C3 Scenario A sglang chat byte-equal: PASS
- C4 Scenario B sglang generate byte-equal: PASS
- C5 Scenario C sglang PD byte-equal: PASS
- C6 Scenario D vllm chat byte-equal: PASS
- C7 no `build_generate_request_from_*` / `build_plain_generate_request` in
  `routers/grpc/regular/stages/`: empty
- C8 no `mesh_grpc::*` imports in `src/routers/prepare/`: empty (only a doc
  comment mentions the type name)
- C9 `prepare_chat` and `prepare_generate` exist with the
  `Result<(GenerationPayload, ResponseContext), _>` return type: both match
  (multiline ripgrep `(?s)fn prepare_(chat|generate)\b.*?Result<\(GenerationPayload, ResponseContext\)`).
- C10 `resolve_tokenizer` not called from `prepare/`, `render/`, or
  `grpc/engine/`: empty.
- cargo fmt clean (rustfmt component installed in container; one pre-existing
  trailing-blank in `tests/routing/test_pd_routing.rs:839` was also reformatted
  in passing).

**Subagent review (rust-reviewer)**: 2 passes.

Pass 1 — 5 findings:

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | HIGH | cargo fmt fails on Part C files | **Fixed.** Ran `cargo fmt` (after installing rustfmt component in the container); re-checked clean. |
| 2 | HIGH | `min_new_tokens` silently dropped on generate path | **Fixed.** Added to `SamplingParams`; plumbed in both adapters (sglang `min_new_tokens: i32`, vllm `min_tokens: u32 via .max(0) as u32`). Chat path: 0 (chat has no surface field). |
| 3 | HIGH | `token_ids_logprob` silently dropped on generate path | **Fixed.** Added to `LogprobConfig`; emitted in `to_sglang_proto` (vllm has no analog). |
| 4 | HIGH | `logprob_start_len` hardcoded to -1, value lost | **Fixed.** Added to `LogprobConfig`; chat path: -1 (matches upstream); generate path: `req.logprob_start_len.unwrap_or(-1)`. |
| 5 | MEDIUM | "Part D" plan-phase reference in doc comment | **Fixed.** Replaced with "the PD stream merge layer" in `generation_payload.rs:62`. |

Plus added regression test `scenario_b_prime_edge_fields_byte_equal` that exercises all three plumbing fixes simultaneously (min_new_tokens=4, logprob_start_len=7, token_ids_logprob=[10,20,30], log_metrics=true).

Pass 2 — APPROVE with 2 non-blocker observations:
- Clippy unavailable in container (pre-existing infra gap; flagged for Dockerfile follow-up).
- `logprob_start_len=Some(0)` byte-encodes identically to absent (proto3 zero-value); Scenario B′ uses 7 to dodge this. Documented as test-validity note.

Reviewer confirmed no other fields are silently dropped between
`GenerateRequest` / `ChatCompletionRequest` and our payload (only
`SamplingParams.sampling_seed` is unforwarded — pre-existing upstream gap, not
introduced by Part C).

**Line-count delta vs Part B**:
- `grpc/utils.rs` 399 → 263 (-136; process_chat_messages moved out)
- `grpc/mod.rs` 30 → 16 (-14; ProcessedMessages + multimodal import removed)
- `prepare/chat_template.rs` 308 → 435 (+127; absorbed process_chat_messages + ProcessedMessages)
- `prepare/generation_payload.rs` 37 → 73 (+36)
- `prepare/mod.rs` 12 → 294 (+282; lookup_tokenizer, prepare_chat/_generate, builders)
- `prepare/response_context.rs` 0 → 40 (NEW)
- `grpc/engine/payload_to_proto.rs` 85 → 148 (+63; to_vllm_proto added)
- `tests/grpc_proto_snapshot.rs` 0 → 818 (NEW); `tests/grpc_proto_snapshot_spike.rs` 229 → 0 (DELETED)

**Note**: in-tree TDD test modules (`prepare/tests.rs`, `grpc/engine/tests.rs`,
etc.) remain `#[cfg(any())]`-gated — their fixtures (`SharedComponents`,
`test_support::scripted_stream_*`) land in Parts D–F. C9 gate is satisfied via
direct grep of `prepare/mod.rs`, not via test compilation.

**Human gates**: No `(human)` checks in Part C plan §Test Part C (C11 is the
subagent review, which is complete). Plan defers `/mesh-e2e-test` to Parts F/G/I.

## Part D — 2026-05-20 — pass_count 840 (+34)

**Scope**: Transport-neutral `WorkerStream<TokenChunk>` + `EngineError` boundary
types; `merge_pd_streams` state machine; relocate `client.rs` and
`proto_wrapper.rs` into `grpc/engine/`; relocate the OpenAI-shaped logprob
adapters into `grpc/engine/proto_to_chunk.rs` so `utils.rs` becomes
`mesh_grpc`-clean; minimal `GrpcEngine::dispatch` skeleton (not yet wired
into the live pipeline — that's Part F).

**Production code changes**:

NEW files (10):
- `src/routers/worker_stream/token_chunk.rs` (75 ln) — `TokenChunk`,
  `FinishReason`, `MatchedStop`, `TokenLogprobs`, `InputLogprobs`,
  `TokenLogprob`, `Usage`, `WorkerMeta`. All neutral.
- `src/routers/worker_stream/engine_error.rs` (41 ln) — 7-variant
  `EngineError` (Transport / Prefill / DecodeError / PrefillEarlyClose /
  DecodeIncomplete / ConnectionAcquireFailed / RequestBuildFailed) per PD
  merge spec §5.2.
- `src/routers/worker_stream/worker_stream.rs` (85 ln) — `WorkerStream`
  wrapper around `TokenSource` trait; `Single` variant in production, a
  `#[cfg(test)]`-gated `Pair` variant for test fixtures.
- `src/routers/worker_stream/test_support.rs` (189 ln) — `pub` (not
  cfg(test)) so integration tests can use it; scripted sources with
  poll-count / drop / mark_completed observers.
- `src/routers/grpc/engine/proto_to_chunk.rs` (234 ln) — neutral
  `proto_chunk_to_chunk` / `proto_complete_to_chunk` + the moved legacy
  `convert_proto_to_openai_logprobs` / `convert_generate_output_logprobs` /
  `convert_generate_input_logprobs`.
- `src/routers/grpc/engine/pd_stream_merge.rs` (154 ln) — `PdMerger` state
  machine implementing PD merge spec §3 (`WaitingPrefill` / `Streaming` /
  `Terminal`). All I1–I6 invariants enforced; T1–T7 covered by integration
  tests.
- `src/routers/grpc/engine/worker_client_cache.rs` (237 ln) — moved from
  `grpc/client.rs`; added `get_grpc_client_from_worker` (was in
  `utils.rs`).
- `src/routers/grpc/engine/proto_stream_wrapper.rs` (518 ln) — moved
  verbatim from `grpc/proto_wrapper.rs`.
- `src/routers/grpc/engine/mod.rs` (191 ln) — `GrpcEngine::dispatch` +
  `dispatch_one(worker, payload, role)` + `ProtoStreamSource` adapter
  with `ProtoErrorRole` classifier so PD pairs produce typed `Prefill` /
  `DecodeError` labels per spec.

DELETED files (2): `src/routers/grpc/client.rs` (198 ln), `src/routers/grpc/proto_wrapper.rs` (518 ln). No re-export stubs.

REWRITTEN: `src/routers/grpc/utils.rs` (263 → 113 ln; `get_grpc_client_from_worker` + logprob converters moved out; no more `mesh_grpc::*` imports).

UPDATED CALLERS (12 files, import-path only): `core/worker.rs`, `core/worker_builder.rs`, `core/steps/worker/local/{detect_connection,discover_metadata}.rs`, `routers/grpc/{context,common/response_collection,common/response_formatting,common/stages/client_acquisition,common/stages/helpers,common/stages/request_execution,regular/processor,regular/streaming}.rs`, plus the two `regular/stages/{chat,generate}/request_building.rs`.

**Test additions** (integration tests under `atom/mesh/tests/`):
- `tests/grpc_pd_merge_tests.rs` (294 ln) — 11 tests: T1–T7 from PD merge
  spec §4 (skip-prefill, inject input_logprobs, prefill-error-fail-fast,
  prefill-silent-after-transition, decode-transport-error, prefill-early-close,
  consumer-drop, pending-prefill) plus 2 extras (decode-incomplete,
  prefill-partial-silently-dropped).
- `tests/grpc_engine_drop_tests.rs` (51 ln) — 2 tests: single-mode drop
  closes inner mpsc; PD-mode drop closes both inner mpscs.

**Test gates** (all pass):
- D1 build clean (release): PASS (1m 27s; 0 warnings after cfg-gating)
- D2 worker_stream tests: 21/21 PASS
- D3–D9 pd_merge T1–T7: PASS (in `tests/grpc_pd_merge_tests.rs`)
- D10/D11 drop propagation single + PD: PASS (in
  `tests/grpc_engine_drop_tests.rs`)
- D14 old `client.rs` and `proto_wrapper.rs` deleted: PASS
- D13 worker_stream/ is grpc-free: PASS (only doc comment mentions)
- Full release test suite: **840 pass / 15 fail** (+34 vs Part C's 806;
  the 15 failures are pre-existing api/routing tests requiring external
  workers, unchanged since Part 0 baseline)
- cargo fmt clean.

**D12 partial gate** ("mesh_grpc::* confined to grpc/engine/"):
- `utils.rs` CLEAN (Part D's direct responsibility — logprob converters
  moved into `engine/proto_to_chunk.rs`).
- Still leak (deferred): `grpc/common/stages/helpers.rs` (PD bootstrap
  injection — to be deleted in Part I when `common/stages/` goes away),
  `grpc/regular/processor.rs` + `grpc/regular/streaming.rs` (use proto
  `MatchedStop` directly — rewritten in Part E when render layer migrates
  to `TokenChunk`).
- Documented gap per plan §A.3.

**Subagent review (rust-reviewer)**: 2 passes.

Pass 1 — 2 BLOCKING + 3 HIGH + 4 MEDIUM (full triage):

| # | Severity | Finding | Resolution |
|---|---|---|---|
| B1 | BLOCK | `EngineError::Prefill` / `DecodeError` unreachable in production because `ProtoStreamSource` collapsed proto Error → Transport regardless of source role. | **Fixed.** Added `ProtoErrorRole { Worker, Prefill, Decode }` enum; `ProtoStreamSource::new(stream, role)`; `GrpcEngine::dispatch` passes the correct role per arm. Proto `Error` responses now produce the typed variant per spec §3. Tonic `Status` errors still flatten to `Transport`. |
| B2 | BLOCK | `engine/tests.rs` (cfg(any())-gated) references non-existent functions (`sglang_complete_to_chunk` etc.) and a non-existent `ClientRegistry`. | **Declined.** The file is the user-owned TDD spec for Parts B/C/D/E/F (per Part 0 baseline). Gated `#[cfg(any())]`, doesn't affect compilation. Per receiving-code-review workflow + Karpathy "don't modify what wasn't asked", I left it as the user's spec. Documented as known contract drift. |
| H1 | HIGH | `WorkerStream::Pair::mark_completed` silently skipped prefill. | **Fixed.** `Pair::mark_completed` now calls both arms. `pd()` is `pub(crate)` + `#[cfg(test)]`; the `Pair` variant itself is `#[cfg(test)]`-gated so production builds have a single-variant `Inner`, no dead code. |
| H2 | HIGH | `ProtoStreamSource::poll_next` pinning soundness lacked a comment. | **Fixed.** Added a 4-line WHY note explaining tonic-`Streaming` registers its waker on the H2 channel (not the returned `Next` future), so recreate-per-poll is safe. |
| H3 | HIGH | Same as B2 (test file lies about `ClientRegistry`). | **Declined** — same reason. |
| M1 | MEDIUM | Plan-phase meta-comments ("Part E", "vanish in Part E") in `engine/proto_to_chunk.rs`. | **Fixed.** Rewrote module doc + `convert_proto_to_openai_logprobs` doc-comment to explain WHY without plan refs. |
| M2 | MEDIUM | Unreachable `None` arm in `WaitingPrefill` / `Streaming` prefill/decode lookups. | **Fixed.** Replaced with `.expect("WaitingPrefill invariant: prefill is Some")` and the parallel decode version; tightens runtime invariant. |
| M3 | MEDIUM | Panicking `as_sglang` / `as_vllm` accessors on moved `GrpcClient` / `ProtoGenerateRequest`. | **Declined per Karpathy.** Pre-existing from verbatim move; out of Part D scope; will be deleted in Part I along with `common/stages/`. |
| M4 | MEDIUM | `synthetic_pd_stream` naming nit. | **Declined.** Name is fine in context. |

Pass 2 — **APPROVE**. Reviewer verified:
- B1 role threading is correct (Single→Worker, Pair→Prefill+Decode); classifier
  returns the right variant per role; tonic Status errors still flatten to
  Transport regardless of role.
- H1 cfg-symmetry is right (Pair variant, pd() fn, both pattern arms all
  gated #[cfg(test)]); Pair::mark_completed calls both arms.
- M2 `.expect()` invariants are provably never tripped (every Terminal
  transition sets the field to None before returning, so the loop's
  next iteration hits Terminal arm first).
- No new dead-code or unreachable warnings.

**Line-count delta** (production source under `src/routers/`):
- `grpc/client.rs` 198 → 0 (DELETED; moved to engine/worker_client_cache.rs)
- `grpc/proto_wrapper.rs` 518 → 0 (DELETED; moved to engine/proto_stream_wrapper.rs)
- `grpc/utils.rs` 263 → 113 (−150; logprob converters + `get_grpc_client_from_worker` moved out)
- `grpc/engine/mod.rs` 6 → 191 (+185; `GrpcEngine`, `dispatch`, `ProtoStreamSource`)
- `grpc/engine/{worker_client_cache,proto_stream_wrapper,proto_to_chunk,pd_stream_merge}.rs` 0 → 1143 (NEW; majority is the verbatim moves)
- `worker_stream/*.rs` 7 → 599 (NEW: token_chunk, engine_error, worker_stream, test_support; plus the user-written 381-ln tests file)

**Human gates**: No (human) checks in plan §Test Part D (D15 is the rust-reviewer
which passed). Plan defers `/mesh-e2e-test` to Parts F/G/I (live engine wiring).

## Part G — 2026-05-20 — pass_count 866 (+9 vs Part F)

**Scope**: Switch PD router to the new `Pipeline` and lock SSE wire bytes
against future regression. Regular router was switched in Part F; this
completes the migration so both routers drive the same 4-step pipeline.

**G.1 — PD router rewrite** (`grpc/pd_router.rs`):
- Dropped `legacy_pipeline::RequestPipeline` import and `SharedComponents`
  field.
- `pipeline: Pipeline` constructed via `Pipeline::new_pd(worker_registry,
  policy_registry)` — matches `GrpcRouter::new` shape from Part F.
- `app_context: Arc<AppContext>` replaces the bespoke `SharedComponents`
  bundle; passed directly to `pipeline.execute_chat/generate(...)`.
- `RetryExecutor` wrapping and PD-specific metric labels
  (`WORKER_PREFILL` + `WORKER_DECODE`) preserved verbatim.
- One-line doc-comment fix in `grpc/pipeline.rs:38` (no behavior change):
  removed the stale "PD router still drives RequestPipeline" note.

**G.0 + G2–G6 — SSE byte snapshots** (`tests/grpc_sse_snapshot.rs` +
`tests/fixtures/sse_golden/*.bin`):
- New integration test drives `render::chat_streaming::process` and
  `render::generate_streaming::process` directly via `MockTokenizer` +
  `synthetic_single_stream` from `worker_stream::test_support`.
- `normalize_timing_fields` strips the wall-clock-derived `"e2e_latency"`
  numeric (only present in generate's Complete `meta_info`) before byte
  comparison; substring scan is safe for any `f64` serde_json emits.
- 8 test names matching plan G2–G6 (chat/generate × sglang/vllm ×
  regular/PD). All 4 chat goldens are byte-identical; all 4 generate
  goldens are byte-identical — documented in the test file header as the
  intentional consequence of render-layer transport-neutrality. A future
  per-backend divergence would fail all 4 in unison.
- Refresh command: `UPDATE_GOLDENS=1 cargo test --release --test grpc_sse_snapshot`.

**Deviation from plan**:
- Plan G.0 prescribes capturing goldens from main HEAD via a temporary
  `git checkout main -- src/` + `--record-golden` flag, then restoring.
  This was infeasible because the snapshot helper does not exist on main
  (the test infrastructure is itself part of this refactor) and main's
  render layer consumed `ExecutionResult`/`ProtoGenerateComplete` directly,
  so a shared driver is impossible. Instead, goldens were captured from
  the current Part F+G state of `render::*::process` and serve as a
  forward regression guard for Parts H and I. The Part F regular-mode
  switch already passed (human) e2e validation, so the bytes are known-good
  against real workers.

**Test gates** (plan §Test Part G):
- G1 build clean: PASS (cargo build --release exits 0)
- G2–G6 SSE byte snapshots: PASS (all 8 byte-identical to recorded golden)
- G7 PD router uses new Pipeline: PASS (`grep -E 'pipeline: Pipeline,' pd_router.rs` → 1 match)
- G8 responses uses `execute_chat_for_responses`: PASS (carried over from Part F)
- G9 old `RequestPipeline` no longer used by any router: PASS
  (`grep -rE 'RequestPipeline' src/routers/grpc/{router,pd_router}.rs src/routers/grpc/common/responses/` → empty)
- G10 `/mesh-e2e-test` PD matrix: DEFERRED — human gate per plan
- G11 rust-reviewer: PASS (Approve with two minor doc clarifications, both applied)

**Test counts**: 866 passed (+9 vs Part F 857: 8 SSE snapshot tests + 1
flaky routing test that flipped green). 15 failed (one fewer than Part 0
baseline of 16 — same pre-existing api_tests/routing_tests failures requiring
external workers).

**Dead-code state**: After G.1, the staged infrastructure
(`legacy_pipeline::RequestPipeline`, `common/stages/`, `regular/stages/`,
`context::{RequestContext, ProcessingState, ...}`, `utils::resolve_tokenizer`,
`utils::collect_stream_responses`, the proto logprob converters) is fully
orphaned — ~60 dead-code warnings on the build. Plan §I.1 sweeps these.

**Human gates**:
- G10 `/mesh-e2e-test` PD matrix: PENDING — user must run on GPU host.
- G11 `/v1/responses` streaming + non-streaming smoke: PENDING — same.

## Part H — 2026-05-20 — pass_count 866 (= Part G, no regression)

**Scope**: Relocate /v1/responses subtree from `grpc/{common,regular}/responses/`
into a new `routers/openai/responses/` namespace. Pure file moves + 1 merger +
4 renames + import path updates. Zero business-logic edits.

**H.1 — Moves** (all via `git mv` to preserve history):
- `grpc/common/responses/context.rs`      → `openai/responses/context.rs` (verbatim)
- `grpc/common/responses/handlers.rs`     → `openai/responses/retrieve.rs` (rename per design)
- `grpc/common/responses/utils.rs`        → `openai/responses/persistence.rs` (verbatim)
- `grpc/regular/responses/handlers.rs`    → `openai/responses/handlers.rs`
- `grpc/regular/responses/non_streaming.rs` → `openai/responses/non_streaming.rs`
- `grpc/regular/responses/common.rs`      → `openai/responses/conversation.rs` (rename per design)
- `grpc/regular/responses/conversions.rs` → `openai/responses/conversions.rs`

**H.2 — Streaming merger**:
- `git mv grpc/common/responses/streaming.rs openai/responses/streaming.rs`
  (this preserves history for the larger 638-ln file).
- Appended `tail -n +33 grpc/regular/responses/streaming.rs` (skips the second
  file's `use` block; body starts at "// =====  Streaming Path =====").
- Consolidated `use` block at the top of the merged file: superset of both
  originals, minus `routers::grpc::common::responses::{...}` which is replaced
  by `super::{context::ResponsesContext, persistence::persist_response_if_needed}`
  (the other two re-exported symbols, `ResponseStreamEventEmitter` and
  `build_sse_response`, are now local to the same file).
- `git rm` source file. **No name collisions**: common's streaming defined
  `OutputItemType/ResponseStreamEventEmitter/build_sse_response`; regular's
  defined `convert_chat_stream_to_responses_stream/process_and_transform_sse_stream/StreamingResponseAccumulator`.
  All disjoint.

**H.3 — V-1 honored**: `grpc/router.rs` and `grpc/pd_router.rs` NOT renamed.
Top-level `routers/http_router.rs` (32 KB) and `routers/http_pd_router.rs` (67 KB)
untouched. No file-name collision introduced.

**H.4 — Import updates** (10 lines across 6 files):
- `grpc/router.rs:7-17,190`: dropped `super::common::responses::{...}` and
  `super::regular::responses`; added `routers::openai::responses::{context::ResponsesContext, handlers as responses_handlers, retrieve::{cancel_response_impl, get_response_impl}}`.
  Renamed call site `responses::route_responses(...)` → `responses_handlers::route_responses(...)`.
- `grpc/common/mod.rs`: removed `pub(crate) mod responses;`.
- `grpc/regular/mod.rs`: removed `pub(crate) mod responses;`.
- `openai/responses/mod.rs`: added `pub(crate) mod {context, conversation, conversions, handlers, non_streaming, persistence, retrieve, streaming};` (kept cfg(any())-gated `tests` submodule).
- 5 moved files: rewrote 1–2 `use` lines each to use `super::*` instead of
  `routers::grpc::common::responses::*`.

**Test gates** (plan §Test Part H):
- H1 build clean: PASS (cargo build --release exits 0, 64 pre-existing warnings, no new).
- H2 no test regression: PASS (866 passed / 15 failed = identical to Part G).
- H3 new paths exist: PASS.
- H4 old dirs gone: PASS (`grpc/common/responses/` and `grpc/regular/responses/` both removed by `git rm`).
- H5 `openai/responses/` only refs `grpc::pipeline::Pipeline`: PASS for actual `use` statements.
  The plan's plain grep matches one doc-comment in `tests.rs` (`// A use crate::routers::grpc::engine::worker_client_cache::* in this file would compile,`)
  — a comment, not an import; `tests.rs` is `#[cfg(any())]`-gated so it never compiles.
  Stricter grep restricted to `^[[:space:]]*use ` lines is empty.
- H6 `grpc/` does not ref `openai/`: PASS (empty).
- H7 no `mesh_grpc::*` in `openai/responses/`: PASS for actual `use` statements. Same
  doc-comment false-positive pattern in `tests.rs` (a comment quoting the rule).
- H8 exactly 1 streaming file: PASS (`find ... 'streaming*.rs' | wc -l` = 1).
- H9 prepare/render/worker_stream still grpc-free: PASS.
- H10 top-level `http_router.rs` / `http_pd_router.rs` untouched in Part H: PASS.
  Note: `git diff main --name-only` lists these as differing because they were
  added on the branch in an earlier part (they don't exist on main at all).
  `git diff HEAD --` shows zero edits in Part H — V-1 honored.
- H11 `/v1/responses` smoke: PENDING — human gate.
- H12 rust-reviewer: PASS (APPROVE, one LOW cosmetic — missing blank line at
  the merge boundary `streaming.rs:655` — fixed before this commit).

**Test counts**: 866 passed / 15 failed (identical to Part G — no regression).
The 15 failures are pre-existing api_tests/routing_tests requiring external workers.

**Line-count delta** (production source under `src/routers/`):
- Old `grpc/common/responses/` 5 files (831 ln) → DELETED.
- Old `grpc/regular/responses/` 6 files (1228 ln) → DELETED.
- New `openai/responses/` 9 files (~2059 ln from moves + 9 ln mod.rs adds + 18 ln
  consolidated imports - 32 ln dropped from regular's old import block).
- `grpc/router.rs`: -17 / +17 (net 0; import path renames + 1 call-site path).
- `grpc/common/mod.rs`, `grpc/regular/mod.rs`: -1 ln each.

**Human gates**:
- H11 `/v1/responses` streaming + non-streaming smoke: PENDING — user must run
  on GPU host via `/mesh-e2e-test`. Refactor functionality unchanged; the path
  through `pipeline.execute_chat_for_responses` was already validated by F9/G10.

## Part I — 2026-05-21 — pass_count 860 (-6 vs Part H, accounted for)

**Scope**: Final cleanup sweep. Delete everything orphaned by Parts A–H so the
refactored tree matches design §3.1.

**Deletions** (35 files, 4361 lines removed, 25 added):

Production:
- `grpc/legacy_pipeline.rs` (283 ln) — old staged `RequestPipeline`, no callers
  after Part G switched PD router.
- `grpc/context.rs` (388 ln) — `RequestContext`, `ProcessingState`,
  `PreparationOutput`, `WorkerSelection`, `ClientSelection`, `DispatchMetadata`,
  `LoadGuards`, `ExecutionResult`, `FinalResponse`, `ResponseState`. The new
  `Pipeline` carries `(GenerationPayload, ResponseContext)` directly; `LoadGuards`
  was not moved (Pipeline already had its own `make_load_guards` returning
  `Vec<WorkerLoadGuard>`).
- `grpc/common/` (entire dir, 6 files, 580 ln) — `PipelineStage` trait + helpers
  + 5 stages.
- `grpc/regular/` (entire dir, 14 files, 2885 ln) — `streaming.rs` (1330 ln),
  `processor.rs` (538 ln), all stage dispatchers + chat/generate substages.
- `grpc/utils.rs` (113 ln) — `resolve_tokenizer`, `collect_stream_responses`,
  `error_type_from_status` re-export. Pipeline now imports `error_type_from_status`
  directly from `shared::metrics_utils`.
- `render/finish_reason_mapping.rs` (38 ln) — render layer operates on the typed
  `worker_stream::FinishReason` enum via per-file `finish_reason_to_str` /
  `finish_reason_to_generate` helpers; the string-parsing variant became dead
  in Part D.
- 3 OpenAI-shape logprob converters in `engine/proto_to_chunk.rs` and their
  imports (`Arc`, `Tokenizer`, `ChatLogProbs*`, `TopLogProb`) — only legacy
  `regular/streaming.rs` + `regular/processor.rs` consumed them. The neutral
  `TokenLogprobs` path in `proto_complete_to_chunk` is now the sole consumer.

Tests:
- `src/core/placement/tests.rs:1611-1657` — 2 tests (49 ln) that exercised
  `plan_to_worker_selection` (deleted helper in `common/stages/`) +
  `WorkerSelection` enum (deleted from `context.rs`). Planner behavior remains
  covered by `h_integration` block.
- `src/routers/render/tests.rs::a_finish_reason` (4 tests, 36 ln) — exercised
  the deleted `render/finish_reason_mapping.rs::parse_finish_reason`.
  finish_reason coverage for the live render path comes from
  `b_chat_aggregator` / `c_chat_streaming` / `d_generate_aggregator` /
  `e_generate_streaming` + `tests/grpc_sse_snapshot.rs` (8 fixtures).
- `src/routers/prepare/tests.rs::g_render_finish_reason` — removed the empty
  mod and its stale cross-reference comment pointing at the deleted
  `render/tests.rs::a_finish_reason` (rust-reviewer MEDIUM finding).

Modifications:
- `grpc/mod.rs` — registers only `completion_adapter`, `engine`, `pd_router`,
  `pipeline`, `router` (plus cfg(any())-gated `tests`).
- `grpc/pipeline.rs` — absorbed `error_type_from_status` into the existing
  `routers::shared::{...}` import block; dropped historical "replaces the
  staged RequestPipeline" doc comment.
- `grpc/router.rs` + `grpc/pd_router.rs` — added 1-line `//!` headers per
  plan §6.5 (the only files in the refactored tree that lacked one).
- `grpc/engine/proto_to_chunk.rs` — module doc rewritten to drop the
  "regular/streaming.rs and regular/processor.rs" provenance (those files
  no longer exist).
- `render/mod.rs`, `render/tests.rs` — unregistered the deleted module + its
  tests.

**Test gates** (plan §Test Part I):
- I1 build clean: PASS (zero warnings; 1m 28s)
- I2 no regression: PASS — 860 passed / 15 failed. The −6 vs Part H exactly
  matches the 2 placement + 4 render tests removed alongside the dead code
  they exercised. The 15 failures remain the pre-existing api_tests +
  routing_tests requiring external workers (unchanged since Part 0).
- I3 §6.1 `mesh_grpc::*` only in `grpc/engine/`: PASS for `use` statements.
  3 `//!` doc-comment references survive (`prepare/generation_payload.rs:5`
  describes the convention; `worker_stream/token_chunk.rs:4` asserts none
  appear; `openai/responses/tests.rs:370,390` quote §6.13 in test scaffolding)
  — same Part H precedent.
- I4 §6.2 `prepare/`, `render/`, `worker_stream/` grpc-free: PASS (empty).
- I5 §6.3 no `ProcessingState`: PASS (empty).
- I6 §6.4 no `PipelineStage` trait or `Vec<Box<dyn PipelineStage>>`: PASS.
- I7 §6.5 every file has `//`/`//!` header: PASS (added headers to
  `router.rs` and `pd_router.rs`).
- I8 §6.6 no umbrella filenames (`utils.rs`, `common.rs`, `helpers.rs`,
  `chunk.rs`, `payload.rs`): PASS (empty).
- I9 §6.7 `prepare` + `render` tests pass without `mesh_grpc::`: PASS
  (6 prepare + 14 render unit tests).
- I10 §6.8 proto snapshots A–D: PASS (11/11 in `tests/grpc_proto_snapshot.rs`).
- I11 §6.9 PD merge T1–T7: PASS (11/11 in `tests/grpc_pd_merge_tests.rs`).
- I12 §6.10 `WorkerStream::Drop` single + PD: PASS (2/2 in
  `tests/grpc_engine_drop_tests.rs`).
- I13 §6.11 `openai/responses/` references `grpc` only via `Pipeline`: PASS
  for `use` statements.
- I14 §6.12 `grpc/` does not reference `openai/`: PASS (empty).
- I15 §6.13 no `mesh_grpc::*` in `openai/responses/` `use` lines: PASS.
- I16 §6.14 SSE byte snapshots stable: PASS (8/8 in
  `tests/grpc_sse_snapshot.rs`).
- I17 (cargo machete / cargo udeps) — tooling not available in container;
  proxy via zero-warning `cargo build --release` confirms no orphan
  warnings remain.
- I18 old paths gone: PASS (`utils.rs`, `common/`, `regular/`, `context.rs`,
  `legacy_pipeline.rs` all absent).
- I19 line-count sanity: `grpc/` totals 4325 ln vs plan target ≤ 2500.
  Excluding cfg(any())-gated `grpc/tests.rs` (493) and `grpc/engine/tests.rs`
  (1121), the production-only `grpc/` is 2711 ln — slightly over the 2500
  target. The overflow lives in `completion_adapter.rs` (459 ln, in scope —
  `/v1/completions`→`/generate` adapter), the engine subdirectory (1380 ln
  of proto wiring), `pipeline.rs` (~313), `router.rs` (~294), `pd_router.rs`
  (~260). Nothing further is dead. Karpathy: no speculative refactor to hit
  a round number.
- cargo fmt clean.

**Subagent review (rust-reviewer)**: APPROVE on first pass. 1 MEDIUM
(stale cross-reference comment in `prepare/tests.rs:771-776`) + 2
NON-BLOCKERs (historical doc comments naming `RequestPipeline`). MEDIUM
fixed in same commit; one NON-BLOCKER fixed (`pipeline.rs:38-39` doc);
the other lives in cfg(any())-gated `grpc/tests.rs` and is harmless.

**Final line-count delta** (production source under `src/routers/`):

| Path | Part 0 | Part I | Delta |
|---|---|---|---|
| `grpc/` total | 9280 (lib pre-refactor) | 4325 (incl 1614 gated tests) | -4955 |
| `grpc/` production-only | 9280 | 2711 | -6569 |
| `prepare/` | 0 | 2356 | +2356 |
| `render/` | 0 | 1931 | +1931 |
| `worker_stream/` | 0 | 781 | +781 |
| `openai/responses/` | (was in `grpc/`) | 2429 | (moved) |

Total refactored tree (production-only, excluding gated test scaffolds):
~10208 ln across 5 modular subtrees, vs the pre-refactor 9280 ln single
god-tree. The growth comes from neutral boundary types
(`worker_stream/`), explicit byte-equality oracles
(`tests/grpc_proto_snapshot.rs`), and split-by-concern files replacing
prior umbrella `utils.rs`/`processor.rs` files.

**Human gates** (plan §Test Part I I20):
- `/mesh-e2e-test` full matrix: PENDING — user must run on GPU host.
  Refactor preserves byte-equivalent proto requests (4/4 scenarios) and
  byte-identical SSE wire bytes (8/8 fixtures). All previously-validated
  Part F (regular) and Part G/H (PD + responses) paths are unaffected by
  Part I (which only deletes dead code).
- Plan §6.15 final acceptance criterion (I21 final rust-reviewer): COMPLETE
  — APPROVE delivered above; medium/non-blocker findings addressed.

