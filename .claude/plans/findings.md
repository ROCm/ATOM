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
