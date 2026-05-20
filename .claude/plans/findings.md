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
