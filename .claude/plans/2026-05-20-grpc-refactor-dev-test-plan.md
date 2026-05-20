# gRPC Router Refactor — Detailed Dev & Test Plan (v2, code-verified)

**Date**: 2026-05-20
**Status**: Code-verified against `atom/mesh/src/routers/grpc/` HEAD (commit `0d0f8071`).
**Executor**: AI refactoring agent (autonomous on cargo+grep+snapshot gates; defers e2e to human).
**Companion design**: `docs/2026-05-19-grpc-engine-extraction-design.md` (authoritative for target architecture)
**Companion spec**: `docs/2026-05-19-grpc-pd-merge-spec.md` (required reading before Part D)
**Target tree root**: `/it-share/yajizhan/code/ATOM/atom/mesh/src/routers/`

---

## 0. Verification findings (what changed vs the design doc after reading the actual code)

Direct reads of `utils.rs` (1214 ln), `pipeline.rs` (412 ln), `context.rs` (390 ln), `router.rs` (312 ln), `pd_router.rs` (282 ln), `regular/stages/chat/{preparation,request_building}.rs`, `common/responses/{context,handlers}.rs`, `common/response_collection.rs`, and `grpc/mod.rs` reveal **5 things the design doc does not address** and **2 things the design doc gets wrong**. Each is reflected in the revised parts below. The verifier MUST read this section before starting Part A.

### V-1 (BLOCKING). The `router.rs` / `pd_router.rs` rename in design §7 Step 7 collides with existing top-level files.

- Top-level `routers/http_router.rs` already exists (32 KB) and is the actual HTTP server entry behind `RouterTrait`.
- Top-level `routers/http_pd_router.rs` already exists (67 KB).
- Design §7 Step 7 proposes `grpc/router.rs → grpc/http_router.rs` and `grpc/pd_router.rs → grpc/http_router_pd.rs`. This creates **two files named `http_router.rs`** in the same module tree (one at `routers/`, one at `routers/grpc/`). Confusing and against the project's name-matches-function rule.

**Decision**: Do **NOT** apply this rename. The files contain `pub struct GrpcRouter` and `pub struct GrpcPDRouter` respectively. **Keep the names as-is**; the path `routers/grpc/router.rs` is already unambiguous given the parent directory. If a future rename is desired, use `grpc_router.rs` / `grpc_pd_router.rs` (matches the type names). The dev plan removes this rename from Part H.

### V-2 (BLOCKING). `ProcessedMessages` is in `grpc/mod.rs`, not `utils.rs`, and contains a `mesh_grpc::*` type.

```rust
// grpc/mod.rs:14, 18-23
use mesh_grpc::sglang_proto::MultimodalInputs;
pub(crate) struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<MultimodalInputs>,    // ← mesh_grpc type leak
    pub stop_sequences: Option<StringOrArray>,
}
```

`process_chat_messages()` in `utils.rs:508` always assigns `multimodal_inputs = None` today, but `request_building.rs:74-78` plumbs it into the Sglang `build_generate_request_from_chat()` call (vLLM path does not consume it). So the field is "wired but always None", not unused.

**Decision** (must be explicit in Part C):
- Move `ProcessedMessages` to `prepare/chat_template.rs`.
- Type its `multimodal_inputs` field as `Option<crate::routers::prepare::multimodal::MultimodalInputs>` — a small neutral struct in `prepare/multimodal.rs` (only the fields actually plumbed today; if nothing is plumbed, make it a unit struct or drop the field).
- Conversion to `mesh_grpc::sglang_proto::MultimodalInputs` happens inside `engine/payload_to_proto.rs::to_sglang_proto`.
- vLLM path remains: no multimodal.

### V-3 (BLOCKING). The render-layer split in Part E is NOT a "pure split" — it changes the consumer signature.

`regular/processor.rs:9` and `regular/streaming.rs:11` both `use mesh_grpc::sglang_proto::generate_complete::MatchedStop;` and consume `ProtoStream` / `ProtoGenerateComplete`. Splitting them into `render/chat_*` and `render/generate_*` requires:
1. Rewriting them to consume `WorkerStream<TokenChunk>` (not `ProtoStream`) and `TokenChunk::Complete` (not `ProtoGenerateComplete`).
2. Replacing every `mesh_grpc::*::MatchedStop` pattern match with the neutral `worker_stream::MatchedStop` enum.
3. Deleting `grpc/common/response_collection.rs` (91 ln) and `grpc/common/response_formatting.rs` (29 ln) — both exist solely to consume `ExecutionResult` / `ProtoGenerateComplete`, which both vanish.

This makes Part E **dependent on Part D**, and **dependent on Part C** (for `ResponseContext` to exist as the render input). Part E and the render-layer wiring in Part F are the largest piece of work in the refactor.

### V-4. `resolve_tokenizer()` in `utils.rs:50` takes `&mut RequestContext` — cannot be moved cleanly to `prepare/`.

It writes `ctx.state.tokenizer = Some(...)` (line 82) so the response stage can re-use the tokenizer Arc. In the new design, `prepare_chat()` returns `ResponseContext { tokenizer: Arc<dyn Tokenizer>, ... }`, so the caching happens via the returned tuple — no shared mutable state needed.

**Decision** (Part C):
- Replace `resolve_tokenizer(&mut ctx, ...)` with `lookup_tokenizer(model_id: &str, registry: &TokenizerRegistry) -> Result<Arc<dyn Tokenizer>, Response>` in `prepare/mod.rs`.
- The caller (the new `prepare_chat` body) holds the resulting `Arc` and threads it into `ResponseContext`.
- Old `resolve_tokenizer` is deleted in Part I.

### V-5. The external proto crate is imported as `mesh_grpc`, package name is `smg-grpc-client = 1.0.0`.

From `atom/mesh/Cargo.toml`:
```
mesh_grpc = { package = "smg-grpc-client", version = "=1.0.0" }
```

So design doc references to both names are correct; the **import** is always `use mesh_grpc::...`, the **dependency line** uses `smg-grpc-client`. All grep checks in this plan use `mesh_grpc::` because that is what appears in code.

### V-6. The current `RequestPipeline` has 7 stages, NOT 5/6 — design table in §4 row 4 said "5 files" / "8 files". Actual:

```
common/stages/   = 5 files (client_acquisition, dispatch_metadata, helpers, request_execution, worker_selection)
regular/stages/  = 8 files total: 3 in chat/, 3 in generate/, plus preparation.rs, request_building.rs, response_processing.rs at top level
```
Both `regular/stages/preparation.rs` (45 ln) and `regular/stages/request_building.rs` (39 ln) etc. are **dispatchers** that match on `RequestType` and forward to `chat::` or `generate::` substages. The new `Pipeline::execute_chat` / `execute_generate` split eliminates these dispatchers entirely. Confirmed.

### V-7. `RetryExecutor` wraps `pipeline.execute_*` in both `router.rs` and `pd_router.rs`.

```rust
// router.rs:119-152, pd_router.rs has parallel
RetryExecutor::execute_response_with_retry(
    &self.retry_config,
    |_attempt| { /* clones, then `pipeline.execute_chat(...).await` */ },
    |res, _attempt| is_retryable_status(res.status()),
    ...
)
```

The new `Pipeline::execute_*` signature must remain `(Arc<Request>, Option<HeaderMap>, Option<String>, Arc<SharedComponents>) -> Response` so the retry wrapper is untouched. This is what the design's §3.2 code block already implies — confirmed safe. Part F does not need to alter the retry wrapping.

---

## 1. User decisions (RESOLVED 2026-05-20)

| # | Decision | Resolution |
|---|---|---|
| Q1 | Naming of `grpc/router.rs` / `grpc/pd_router.rs` (V-1). | **RESOLVED — KEEP CURRENT NAMES.** No rename in Part H. The path `routers/grpc/router.rs` is unambiguous because of the parent directory; the top-level `routers/http_router.rs` (32 KB) and `routers/http_pd_router.rs` (67 KB) stay untouched. Any reference in the plan to renaming these two files is dead text. |
| Q2 | `ProcessedMessages.multimodal_inputs` field (V-2). | **RESOLVED — DROP THE FIELD.** Per CLAUDE.md "no hypothetical features": the field is always `None` today, so it adds zero behavior and one `mesh_grpc::*` import. Part C deletes it; `to_sglang_proto` calls the external builder with the 5-arg signature (no multimodal). If multimodal lands later, re-introduce as a neutral type at that time. Update Part C.1 / C.5 / C.7 accordingly; remove the `prepare/multimodal.rs` option from C.7. |

These are now hard facts for the agent. Do not re-litigate during execution.

---

## 2. How to use this plan

1. Work through Parts 0 → I in order. **No parallelism within a part.** Each part's invariants are assumed by the next.
2. After each Dev Part, execute every **(auto)** check in the matching Test Part. ALL must pass. Pass count must monotonically increase or stay equal across baseline → A → … → I.
3. After every Dev Part that touches production Rust (B–I), launch the `rust-reviewer` subagent on the diff before marking the part done. Per user memory rule: subagent review is mandatory.
4. **(human)** checks are flagged for the user to run on a GPU host via `/mesh-e2e-test`. They do NOT gate the agent — record their status in `findings.md` and request user confirmation before treating the part as complete.
5. If any **(auto)** test fails: STOP. Diagnose. Do not patch the test to pass; patch the code or escalate.
6. **Stop and escalate** on: structural surprises (target file already exists with different content), Part B outcome (iii), or any SSE byte snapshot mismatch that does not collapse to a render-layer bug in one debug pass.
7. After each part, append a `findings.md` entry: `## Part X — <date> — pass_count <N>` with anomalies and human-gate status.

---

## 3. Coding principles

All code changes must follow the **Karpathy Guidelines** — load `/karpathy-guidelines` before starting any dev part.

Additional hard rules for this project:

- **No meta-comments in source files**: Default to zero comments; only add one when the WHY is non-obvious. Never put plan references, phase labels ("P1 will implement"), "covers §X", design-rationale paragraphs, or provenance notes ("from pd_router.rs:1199") in code — those belong in the plan, PR description, or commit message.
- **TDD + Subagent review**: After writing production code, always launch the `rust-reviewer` subagent for independent review. Do not self-approve.

## 4. Development environment

- Edit code on the **Host**; compile and run inside the **Container** (`atom_sglang_mesh`).
- **NEVER edit or create files inside the Container** — root ownership breaks Host user file permissions.
- Build and test commands:
  ```bash
  docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo build --release
  docker exec -w /it-share/yajizhan/code/ATOM/atom/mesh atom_sglang_mesh cargo test --package atom-mesh
  ```
- All `cargo build` / `cargo test` invocations go through `docker exec`. File edits use Edit/Write tools on the Host only.

---

## Part 0 — Pre-flight discovery (≤ 1 hour of agent time)

**Goal**: Confirm starting state matches plan assumptions; create scaffolding files used by every later part.

### 0.a Baseline build + test count
```bash
cd /it-share/yajizhan/code/ATOM/atom/mesh
cargo build --release 2>&1 | tee /tmp/grpc_refactor/baseline_build.log
cargo test  --release 2>&1 | tee /tmp/grpc_refactor/baseline_test.log
grep -E '^test result:' /tmp/grpc_refactor/baseline_test.log | tee /tmp/grpc_refactor/baseline_pass_count.txt
```
Record the integer pass-count. Use it as the floor for every subsequent part.

### 0.b Verify external crate path
```bash
grep 'mesh_grpc' atom/mesh/Cargo.toml          # must show: mesh_grpc = { package = "smg-grpc-client", version = "=1.0.0" }
```
If the package/version differs from `smg-grpc-client = "=1.0.0"`, STOP — the byte-snapshot strategy assumes the version is pinned.

### 0.c Inventory the call sites that will move
Run and save the output of:
```bash
grep -rln 'mesh_grpc::'              src/routers/grpc/    # expect 7 files (matches V-3)
grep -rln 'utils::'                  src/routers/grpc/    # callers of utils.rs (15-20 files)
grep -rn  'build_generate_request_from_chat\|build_generate_request_from_generate' src/   # exactly 4 call sites
grep -rn  'ProtoStream\|ProtoGenerateComplete\|ExecutionResult' src/routers/grpc/ | wc -l   # callers that must convert
```
Save to `findings.md` so later parts can spot churn.

### 0.d Create scaffolding directories (empty)
- `mkdir -p atom/mesh/tests/fixtures/sse_golden/` — Part G stores main-branch SSE captures here.
- `mkdir -p atom/mesh/tests/fixtures/proto_snapshots/` — Part C stores byte-snapshots here.
- `touch atom/mesh/.claude/plans/findings.md` — running log.

### 0.e Q1/Q2 decisions (already resolved — see §1)
- Q1 = KEEP `grpc/router.rs` and `grpc/pd_router.rs` names.
- Q2 = DROP the `multimodal_inputs` field.

Record these two lines verbatim in `findings.md` under Part 0 for traceability. No further user input needed before Part A.

**Done when**: baseline_pass_count.txt exists; inventory captured; Q1/Q2 lines logged in findings.md.

---

# DEV PARTS A–I

## Dev Part A — Skeleton + move pure transport-neutral helpers

**Maps to design §7 Step 1.** Goal: peer folders exist and contain helpers that have ZERO `mesh_grpc::` imports and ZERO `RequestContext` parameter. No new types, no signature changes.

### A.1 Create empty mod files (compile-clean, re-exports only)
- `atom/mesh/src/routers/prepare/mod.rs`
- `atom/mesh/src/routers/prepare/chat_template.rs`
- `atom/mesh/src/routers/prepare/tool_constraints.rs`
- `atom/mesh/src/routers/prepare/stop_sequence_decoder.rs`
- `atom/mesh/src/routers/prepare/parser_factory_lookup.rs`
- `atom/mesh/src/routers/render/mod.rs`
- `atom/mesh/src/routers/render/finish_reason_mapping.rs`
- `atom/mesh/src/routers/worker_stream/mod.rs` (empty for now; types arrive in Part D)

Register `prepare`, `render`, `worker_stream` in `atom/mesh/src/routers/mod.rs` (alphabetically with existing `parse`, `shared`, `tokenize`).

### A.2 Move helpers (function-exact map; no body edits)

Source: `atom/mesh/src/routers/grpc/utils.rs` (line numbers from the read above).

| Source lines | Function(s) | Destination file |
|---|---|---|
| 122-157 | `process_tool_call_arguments` (private) | `prepare/chat_template.rs` |
| 160-178 | `process_chat_messages` consumes this — see Part C; for Part A, move with `process_chat_messages`'s helpers only | — |
| 160-178 | `process_content_format` | `prepare/chat_template.rs` |
| 182-226 | `transform_content_field` (private) | `prepare/chat_template.rs` |
| 230-277 | `generate_tool_constraints` | `prepare/tool_constraints.rs` |
| 281-340 | `build_required_array_schema` (private) | `prepare/tool_constraints.rs` |
| 346-371 | `filter_tools_by_tool_choice` | `prepare/tool_constraints.rs` |
| 380-393 | `filter_chat_request_by_tool_choice` | `prepare/tool_constraints.rs` |
| 518-557 | `create_stop_decoder` | `prepare/stop_sequence_decoder.rs` |
| 560-635 | `parse_json_schema_response` | `prepare/tool_constraints.rs` |
| 694-706 | `get_history_tool_calls_count` | `prepare/tool_constraints.rs` |
| 718-737 | `generate_tool_call_id` | `prepare/tool_constraints.rs` |
| 740-752 | `check_reasoning_parser_availability` | `prepare/parser_factory_lookup.rs` |
| 755-765 | `check_tool_parser_availability` | `prepare/parser_factory_lookup.rs` |
| 772-793 | `get_reasoning_parser` | `prepare/parser_factory_lookup.rs` |
| 796-817 | `create_reasoning_parser` | `prepare/parser_factory_lookup.rs` |
| 824-845 | `get_tool_parser` | `prepare/parser_factory_lookup.rs` |
| 848-869 | `create_tool_parser` | `prepare/parser_factory_lookup.rs` |
| 992-1013 | `parse_finish_reason` | `render/finish_reason_mapping.rs` |
| 1017-1214 | tests (unit tests of `process_content_format`) | move with `chat_template.rs` |

### A.3 Helpers that STAY in `grpc/utils.rs` for now (transport-bound or context-bound)
| Source lines | Function | Why it stays |
|---|---|---|
| 50-85 | `resolve_tokenizer` | takes `&mut RequestContext`; refactored away in Part C (V-4) |
| 88-118 | `get_grpc_client_from_worker` | returns `GrpcClient` (mesh_grpc type); → `engine/worker_client_cache.rs` in Part D |
| 397-515 | `process_chat_messages` | returns `ProcessedMessages` which still references `mesh_grpc::sglang_proto::MultimodalInputs`. STAY until Part C resolves Q2. |
| 649-690 | `collect_stream_responses` | takes `ProtoStream`; → `engine/proto_to_chunk.rs` in Part D (replaced by render-side iteration over `WorkerStream`) |
| 875-946 | `convert_proto_to_openai_logprobs` | proto type input; → `engine/proto_to_chunk.rs` in Part D |
| 952-961 | `convert_generate_output_logprobs` | proto type input; → `engine/proto_to_chunk.rs` in Part D |
| 967-980 | `convert_generate_input_logprobs` | proto type input; → `engine/proto_to_chunk.rs` in Part D |
| 1015 | `pub(crate) use crate::routers::shared::metrics_utils::error_type_from_status;` | trivial re-export, leave; remove in Part I |

### A.4 Import-path updates
Every site that today does `use crate::routers::grpc::utils::{<moved fn>};` updates to `use crate::routers::prepare::{<file>::<fn>};` or `use crate::routers::render::finish_reason_mapping::<fn>;`. Touch sites are bounded by:
```bash
grep -rln 'grpc::utils::' src/routers/grpc/ | sort -u    # ~15 files
```

### A.5 Hard constraints
- No new `pub use` re-exports inside `grpc/utils.rs` pointing at the new locations (no backwards-compat shim).
- No function body edits. If a moved function would refuse to compile in its new home, that means the function was actually transport-bound — revert the move and add it to the "stay" list above (then update test A4 expectation).
- `mesh_grpc::*` MUST NOT appear in any file under `prepare/`, `render/`, `worker_stream/`. Hard gate A4.

**Done when**: Test Part A all (auto) checks pass.

---

## Test Part A

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| A1 | auto | Build clean | `cargo build --release` | exit 0 |
| A2 | auto | No test regression | `cargo test --release` | pass count ≥ Part 0 baseline |
| A3 | auto | No new types in `prepare/`, `render/`, `worker_stream/` | `grep -rE '^pub (struct\|enum\|trait) ' src/routers/{prepare,render,worker_stream}/` | empty |
| A4 | auto | No `mesh_grpc::*` leak | `grep -r 'mesh_grpc::' src/routers/{prepare,render,worker_stream}/` | empty |
| A5 | auto | No backwards-compat re-exports in `utils.rs` | `grep -E 'pub use (super\|crate)::routers::(prepare\|render)' src/routers/grpc/utils.rs` | empty |
| A6 | auto | `utils.rs` shrunk to expected size | `wc -l src/routers/grpc/utils.rs` | 450 ≤ N ≤ 700 (was 1214; the "stay" list above totals ~500 ln) |
| A7 | auto | `process_content_format` unit tests still pass at new location | `cargo test --release routers::prepare::chat_template::tests::test_transform_messages` | all 6 tests pass |
| A8 | auto | Old utils call sites are gone | `grep -rE 'grpc::utils::(process_content_format\|generate_tool_constraints\|create_stop_decoder\|parse_finish_reason)' src/routers/grpc/` | empty |

---

## Dev Part B — Single-scenario byte-equality spike

**Maps to design §7 Step 2a.** Goal: validate the central design hypothesis — `GenerationPayload → to_sglang_proto(payload)` can produce byte-identical proto bytes to today's `SglangSchedulerClient::build_generate_request_from_chat()`. **One scenario.** No bulk migration.

### B.1 Create the four spike files (and only these)
- `prepare/generation_payload.rs` — minimal struct: only the fields needed for Scenario A. Per design §3.2:
  ```rust
  pub struct GenerationPayload {
      pub request_id: String,
      pub token_ids: Vec<u32>,
      pub sampling: SamplingParams,
      pub stop: StopConfig,
      pub logprob: LogprobConfig,
      pub tool_constraints: Option<(String, String)>,
      // pd_metadata omitted in spike (Scenario A is single-mode)
  }
  pub struct SamplingParams { pub temperature: f32, pub top_p: f32, pub top_k: i32,
                              pub repetition_penalty: f32, pub max_new_tokens: i32 }
  pub struct StopConfig { pub stop: Option<crate::protocols::common::StringOrArray>,
                          pub stop_token_ids: Option<Vec<u32>>,
                          pub skip_special_tokens: bool, pub no_stop_trim: bool }
  pub struct LogprobConfig { pub return_logprob: bool, pub top_logprobs_num: u32,
                             pub input_logprobs: bool }
  ```
- `grpc/engine/mod.rs` — new subdir; `pub mod payload_to_proto;` only (no `GrpcEngine` struct yet).
- `grpc/engine/payload_to_proto.rs` — implement `pub fn to_sglang_proto(payload: &GenerationPayload, text: String, multimodal: Option<MultimodalInputs>) -> sglang_proto::GenerateRequest`. The `text` and `multimodal` args are temporarily passed in alongside the payload — they migrate INTO `GenerationPayload` in Part C once Q2 is resolved.
- `atom/mesh/tests/grpc_proto_snapshot_spike.rs` — Scenario A fixture and byte-equality test (see B.2).

### B.2 Scenario A fixture (chat + tools + logprobs + non-default sampling)
```rust
// pseudocode of the test body
let req: ChatCompletionRequest = scenario_a_chat_request();   // builder helper in fixture file
let tokenizer = load_test_tokenizer();                         // pinned local fixture, no network
let processed = process_chat_messages(&req, &*tokenizer)?;     // call existing prepare/ helper
let token_ids = tokenizer.encode(&processed.text, false)?.token_ids().to_vec();

// Path A: today's API on the external client (use a stub SglangSchedulerClient — only the builder needs to work)
let upstream_proto: sglang_proto::GenerateRequest = SglangSchedulerClient::build_generate_request_from_chat(
    "req_spike_A".into(), &req, processed.text.clone(), token_ids.clone(),
    processed.multimodal_inputs.clone(), tool_constraints_for_a(),
)?;

// Path B: our new code
let payload = GenerationPayload { /* populate from req */ };
let our_proto: sglang_proto::GenerateRequest = to_sglang_proto(&payload, processed.text, processed.multimodal_inputs);

// Compare
let upstream_bytes = prost::Message::encode_to_vec(&upstream_proto);
let our_bytes      = prost::Message::encode_to_vec(&our_proto);
assert_eq!(upstream_bytes, our_bytes, "byte mismatch — see findings.md");
```

If the spike cannot construct `SglangSchedulerClient` without a real gRPC server, expose the proto-building method via a small helper added to `mesh_grpc` (Outcome iii path) OR inline a copy of the builder logic in `payload_to_proto.rs` (Outcome ii path). Document which.

### B.3 Outcome handling — exactly one of these documented in `findings.md` before Part C
- **(i) byte-equal** → proceed to Part C unchanged.
- **(ii) small deterministic diff** → field-by-field align `to_sglang_proto`; each alignment gets one terse `// match upstream default` line. Re-run until equal. Proceed.
- **(iii) divergence stems from `smg-grpc-client = 1.0.0` internals we cannot replicate without leaking implementation** → STOP. Ask user to pick between:
  - (a) upstream a `to_proto(&payload)` API to `smg-grpc-client` (cleanest; requires version coordination)
  - (b) keep `to_sglang_proto` as a thin shim that internally calls `build_generate_request_from_chat` (preserves single source of truth)
  Update design §3.2 accordingly before Part C.

### B.4 Hard constraints
- Do not add `to_vllm_proto` in Part B.
- Do not migrate any call site away from `build_generate_request_from_chat`.
- Spike file count: at most 4 new files vs Part A.

**Done when**: Test Part B passes and outcome (i/ii/iii) is recorded in `findings.md`.

---

## Test Part B

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| B1 | auto | Build clean | `cargo build --release` | exit 0 |
| B2 | auto | Spike snapshot runs and reaches a documented outcome | `cargo test --release --test grpc_proto_snapshot_spike -- --nocapture` | exit 0 if outcome (i) or (ii)-after-alignment; if outcome (iii), test prints "OUTCOME_III" and exits 1 |
| B3 | auto | Outcome recorded | `grep -E 'Outcome \((i\|ii\|iii)\)' .claude/plans/findings.md` | match found |
| B4 | auto | File-count footprint | `find src/routers/{prepare,grpc/engine} -name '*.rs' -newer /tmp/grpc_refactor/part_a_done` | ≤ 4 files |
| B5 | auto | No production call-site migration yet | `grep -rE 'crate::routers::prepare::generation_payload' src/routers/grpc/{regular,common}/` | empty |
| B6 | manual | `rust-reviewer` on `payload_to_proto.rs` + fixture | dispatched after B1–B5 pass | reviewer confirms one of {(i), (ii)-with-alignment-comments, (iii)-escalated} |

**On B2 = OUTCOME_III**: STOP; do not begin Part C until user resolves Q3 (the upstream-vs-shim decision).

---

## Dev Part C — Full `GenerationPayload` + both proto adapters + introduce `prepare_chat`/`prepare_generate`

**Maps to design §7 Step 2b.** Goal: every gRPC request to a worker is built via `GenerationPayload → to_{sglang,vllm}_proto`; no production caller invokes `build_generate_request_from_*` directly. `prepare_chat()` and `prepare_generate()` exist and return the **real** `(GenerationPayload, ResponseContext)` tuple.

### C.1 Complete `GenerationPayload` (drop the temporary args from Part B)
Add the remaining fields per design §3.2:
- `pub pd_metadata: Option<PdMetadata>` (= bootstrap_host, bootstrap_room, kv_port — used in PD mode only)
- `pub text: String` (was a Part B argument; merged in now)
- **NO `multimodal_inputs` field** (Q2 resolved: dropped). `GenerationPayload` carries no multimodal data; `to_sglang_proto` calls the external builder without that argument. See C.7.
- `pub return_logprob: bool` (subsumed under `LogprobConfig`)
- Any vLLM-specific sampling fields needed by `to_vllm_proto`: `min_p`, `length_penalty`, `seed`, `ignore_eos`, etc. (driven by Scenario D in Test C6).

### C.2 Complete the two proto adapters in `grpc/engine/payload_to_proto.rs`
- `pub fn to_sglang_proto(p: &GenerationPayload) -> sglang_proto::GenerateRequest`
- `pub fn to_vllm_proto(p: &GenerationPayload)   -> vllm_proto::GenerateRequest`
- Pure field copy modulo any (ii) alignment comments from Part B.

### C.3 Introduce `ResponseContext` and `ProtocolRequest`
- New file `prepare/response_context.rs` (per design §3.2 verbatim).
- New helper `prepare/mod.rs::lookup_tokenizer(model_id, registry)` (replaces `utils::resolve_tokenizer`; takes no `ctx`).

### C.4 Implement `prepare_chat` and `prepare_generate` in `prepare/mod.rs`
Signature per design §3.2:
```rust
pub fn prepare_chat(
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &SharedComponents,
) -> Result<(GenerationPayload, ResponseContext), Response>;

pub fn prepare_generate(
    req: Arc<GenerateRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &SharedComponents,
) -> Result<(GenerationPayload, ResponseContext), Response>;
```
Body copies the logic currently in `regular/stages/chat/preparation.rs:40-113` and `regular/stages/generate/preparation.rs` end-to-end. Reads from `request`; writes to a fresh `(GenerationPayload, ResponseContext)` — no shared mutable state.

### C.5 Migrate the 4 `build_generate_request_from_*` call sites
The call sites are (from inventory 0.c):
1. `regular/stages/chat/request_building.rs:69` (Sglang chat)
2. `regular/stages/chat/request_building.rs:89` (vLLM chat)
3. `regular/stages/generate/request_building.rs:??` (Sglang generate)
4. `regular/stages/generate/request_building.rs:??` (vLLM generate)

For Part C, these stages still exist (deleted in Part I). Update them so that instead of calling `build_generate_request_from_chat(...)` they:
1. Construct a `GenerationPayload` from `ctx.state.preparation` (already populated by the old `ChatPreparationStage`)
2. Call `to_{sglang,vllm}_proto(&payload)`
3. Wrap in `ProtoGenerateRequest::Sglang(...)` / `ProtoGenerateRequest::Vllm(...)` as today

This means we have **double-construction temporarily**: old `ChatPreparationStage` still runs; new `GenerationPayload` is built from its output. Acceptable — the path is byte-identical (Test C3–C6) and full removal lands in Part F.

### C.6 Hard constraints
- All 4 byte-snapshot scenarios from design §6.8 must be byte-equal.
- No render-side or engine-dispatch changes in this part.
- No new `mesh_grpc::*` imports under `prepare/`.

### C.7 Multimodal field removal (Q2 resolved — drop)
Apply all of the following in this part:
1. Delete the `multimodal_inputs: Option<MultimodalInputs>` field from `ProcessedMessages` (`grpc/mod.rs:22`).
2. Delete the `use mesh_grpc::sglang_proto::MultimodalInputs;` import from `grpc/mod.rs:3`.
3. In `utils.rs::process_chat_messages` (line 508 / 512), remove the `let multimodal_inputs = None;` line and the corresponding struct field assignment.
4. In `regular/stages/chat/request_building.rs:74-78`, remove the `.multimodal_inputs.clone()` arg from the Sglang `build_generate_request_from_chat(...)` call. **This changes the external call signature from 6 args to 5 args.** Verify against `mesh_grpc = smg-grpc-client = 1.0.0` that the 5-arg overload exists. If only the 6-arg signature is available, pass `None` literal at the call site (still drop the field upstream).
5. Move `ProcessedMessages` (now a 2-field struct: `text`, `stop_sequences`) to `prepare/chat_template.rs`.
6. All proto-snapshot scenarios (C3–C6) use fixtures with **no multimodal**. Add a one-line note to `findings.md`: "Multimodal support removed in Part C; reintroduce as a neutral type if multimodal inference lands."

**Done when**: Test Part C passes + `rust-reviewer` clean.

---

## Test Part C

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| C1 | auto | Build clean | `cargo build --release` | exit 0 |
| C2 | auto | No test regression | `cargo test --release` | pass count ≥ Part A |
| C3 | auto | Scenario A (chat + tools + logprobs + sampling) byte-equal | `cargo test --release --test grpc_proto_snapshot -- scenario_a` | exit 0 |
| C4 | auto | Scenario B (generate + stop array + input_logprobs) byte-equal | `cargo test --release --test grpc_proto_snapshot -- scenario_b` | exit 0 |
| C5 | auto | Scenario C (PD pair + bootstrap + n>1 SGLang) byte-equal | `cargo test --release --test grpc_proto_snapshot -- scenario_c` | exit 0 |
| C6 | auto | Scenario D (vLLM-specific sampling) byte-equal | `cargo test --release --test grpc_proto_snapshot -- scenario_d` | exit 0 |
| C7 | auto | All 4 `build_generate_request_*` direct call sites are gone from `regular/stages/` | `grep -rE 'build_generate_request_from_(chat\|generate)' src/routers/grpc/regular/stages/` | empty (allowed only inside `engine/payload_to_proto.rs` if outcome (iii)) |
| C8 | auto | `mesh_grpc::*` still absent from `prepare/` | `grep -r 'mesh_grpc::' src/routers/prepare/` | empty |
| C9 | auto | `prepare_chat` and `prepare_generate` exist and return the tuple | `grep -E 'fn prepare_(chat\|generate).*Result<\(GenerationPayload, ResponseContext\)' src/routers/prepare/mod.rs` | both match |
| C10 | auto | `resolve_tokenizer` not called by new code | `grep -rE 'resolve_tokenizer' src/routers/prepare/ src/routers/render/ src/routers/grpc/engine/` | empty |
| C11 | manual | `rust-reviewer` | dispatched after C1–C10 pass | no blockers |

---

## Dev Part D — `worker_stream/` types + `engine/` dispatch + PD merge

**Maps to design §7 Step 3.** Goal: transport-neutral boundary types live in `worker_stream/`; `GrpcEngine` + `merge_pd_streams` exist and are unit-tested. **Not wired into the live pipeline yet.**

### D.1 Required reading
Read `docs/2026-05-19-grpc-pd-merge-spec.md` end-to-end before writing any code. Pin the T1–T7 obligations.

### D.2 Create the 3 worker_stream files
- `worker_stream/token_chunk.rs` — `TokenChunk`, `FinishReason`, `MatchedStop`, `TokenLogprobs`, `TokenLogprob`, `Usage`, `WorkerMeta`, `InputLogprobs` (design §3.2 verbatim).
- `worker_stream/worker_stream.rs` — `WorkerStream { inner: WorkerStreamInner }`, with `WorkerStreamInner::Single(tonic::Streaming<T>)` and `WorkerStreamInner::Pd { prefill, decode }`. Impl `Stream<Item = Result<TokenChunk, EngineError>>`. `Drop` propagates by dropping both inner streams (tonic-native H2 cancellation).
- `worker_stream/engine_error.rs` — 7-variant `EngineError` per design §3.2.

### D.3 Build the engine subdir
- `grpc/engine/proto_stream_wrapper.rs` — move `grpc/proto_wrapper.rs` here (rename only, no body edits). 518 ln → ~518 ln.
- `grpc/engine/worker_client_cache.rs` — move `grpc/client.rs` here. 198 ln → ~198 ln.
- `grpc/engine/proto_to_chunk.rs` — NEW. Holds:
  - `fn sglang_complete_to_chunk(complete: sglang_proto::GenerateComplete) -> TokenChunk::Complete`
  - `fn sglang_chunk_to_chunk(chunk: sglang_proto::GenerateChunk) -> TokenChunk::Partial`
  - `fn vllm_complete_to_chunk(...)` and `fn vllm_chunk_to_chunk(...)`
  - Logprob collapse logic moved from `utils.rs` (`convert_proto_to_openai_logprobs`, `convert_generate_output_logprobs`, `convert_generate_input_logprobs`) — adapted to produce neutral `TokenLogprobs` instead of OpenAI-format types.
- `grpc/engine/pd_stream_merge.rs` — NEW. Private `fn merge_pd_streams(prefill: WorkerStream, decode: WorkerStream, input_logprobs: bool) -> WorkerStream`. State machine per PD spec §3. **T1–T7 unit tests live in the same file under `#[cfg(test)] mod tests`.**
- `grpc/engine/mod.rs` — expand to:
  ```rust
  pub mod payload_to_proto;
  pub mod proto_stream_wrapper;
  pub mod proto_to_chunk;
  pub mod worker_client_cache;
  mod pd_stream_merge;     // private — only dispatch uses it
  
  use std::sync::Arc;
  use crate::routers::worker_stream::{WorkerStream, EngineError};
  use crate::routers::prepare::GenerationPayload;
  use crate::core::Worker;
  use proto_stream_wrapper::ClientRegistry;          // or equivalent
  
  pub struct GrpcEngine { client_registry: Arc<ClientRegistry> }
  pub enum PlacementPlan { Single { worker: Arc<dyn Worker> },
                           Pair { prefill: Arc<dyn Worker>, decode: Arc<dyn Worker> } }
  
  impl GrpcEngine {
      pub async fn dispatch(&self, placement: &PlacementPlan, payload: &GenerationPayload)
          -> Result<WorkerStream, EngineError> { /* §3.2 code block */ }
      async fn dispatch_one(...) -> Result<WorkerStream, EngineError> { ... }
  }
  ```

### D.4 Migrate `ResponseContext` integration
`response_context.rs::ResponseContext` was scaffolded in Part C; it is now consumed by the (still-old) `regular/stages/response_processing.rs`. The render-layer migration happens in Part E. So Part D's only `ResponseContext` change is to add the `tokenizer` and `stop_decoder` fields wired into `prepare_*`.

### D.5 Test fixtures for D
- `tests/grpc_pd_merge_tests.rs` — 7 test fns matching T1–T7, each constructing two synthetic in-memory `WorkerStream`s via a tokio `mpsc::channel` shim (so no real tonic). The shim lives at `worker_stream/test_support.rs` under `#[cfg(test)]`.
- `tests/grpc_engine_drop_tests.rs` — 2 tests: drop single-mode stream → confirm inner channel closed; drop PD-mode stream → both inner channels closed.

### D.6 Hard constraints
- `merge_pd_streams` must match PD spec §3 state diagram. Spec ambiguity → fix the spec first.
- `mesh_grpc::*` is confined to `grpc/engine/` after this part: check D12.
- Old `grpc/client.rs` and `grpc/proto_wrapper.rs` are **DELETED** after the move (not left as re-export stubs).
- `Pipeline` is NOT yet calling `GrpcEngine::dispatch`. New code lives alongside old.

**Done when**: Test Part D passes + `rust-reviewer` clean.

---

## Test Part D

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| D1 | auto | Build clean | `cargo build --release` | exit 0 |
| D2 | auto | `worker_stream/` compiles standalone | `cargo test --release --lib routers::worker_stream` | exit 0 |
| D3 | auto | PD merge T1 (input_logprobs=true: prefill drained before decode chunk yielded) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t1` | exit 0 |
| D4 | auto | T2 (input_logprobs=false: decode interleaves) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t2` | exit 0 |
| D5 | auto | T3 (prefill Error → EngineError::Prefill, decode cancelled) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t3` | exit 0 |
| D6 | auto | T4 (decode Error → EngineError::DecodeError) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t4` | exit 0 |
| D7 | auto | T5 (prefill early close → EngineError::PrefillEarlyClose) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t5` | exit 0 |
| D8 | auto | T6 (decode incomplete → EngineError::DecodeIncomplete) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t6` | exit 0 |
| D9 | auto | T7 (Complete merging: input_logprobs from prefill, rest from decode) | `cargo test --release --test grpc_pd_merge_tests pd_merge_t7` | exit 0 |
| D10 | auto | Drop propagates in single mode | `cargo test --release --test grpc_engine_drop_tests worker_stream_drop_single` | exit 0 |
| D11 | auto | Drop propagates in PD mode | `cargo test --release --test grpc_engine_drop_tests worker_stream_drop_pd_both` | exit 0 |
| D12 | auto | `mesh_grpc::*` confined to `grpc/engine/` | `grep -rl 'mesh_grpc::' src/routers/ \| grep -v '/grpc/engine/'` | empty |
| D13 | auto | `worker_stream/` is grpc-free | `grep -rE 'crate::routers::grpc\|mesh_grpc::' src/routers/worker_stream/` | empty |
| D14 | auto | Old `client.rs` and `proto_wrapper.rs` deleted (no re-export stubs) | `! test -f src/routers/grpc/client.rs && ! test -f src/routers/grpc/proto_wrapper.rs` | exit 0 |
| D15 | manual | `rust-reviewer` on `pd_stream_merge.rs` | dispatched after D1–D14 pass | reviewer confirms state machine matches PD spec §3 |

---

## Dev Part E — `render/` layer (rewrite consumer to take `WorkerStream<TokenChunk>`)

**Maps to design §7 Step 4.** Goal: render-layer functions exist and pass synthetic-chunk tests. **Not yet wired into `Pipeline`.** Per V-3, this is a rewrite not a pure split.

### E.1 Create the 4 render files
For each, the signature change from old → new is the key thing:

| File | Old source | Old consumer | New signature |
|---|---|---|---|
| `render/chat_streaming.rs` | `regular/streaming.rs:chat_*` branch (~700 ln of 1326) | `ProtoStream`, `ProtoResponseVariant`, `ProtoGenerateComplete::MatchedStop` | `pub fn process(stream: WorkerStream, ctx: ResponseContext, backend_label: &'static str) -> Response` |
| `render/chat_aggregator.rs` | `regular/processor.rs` chat-flow methods (~250 ln of 465) | `ExecutionResult`, `ProtoGenerateComplete` | `pub async fn process(stream: WorkerStream, ctx: ResponseContext) -> Response` |
| `render/generate_streaming.rs` | `regular/streaming.rs:generate_*` branch (~600 ln) | same | parallel signature |
| `render/generate_aggregator.rs` | `regular/processor.rs` generate-flow methods (~200 ln) | same | parallel signature |

Each new file:
1. Imports neutral types from `worker_stream/` (not `mesh_grpc::*`).
2. Replaces `match complete.matched_stop` against `MatchedStopStr/MatchedTokenId` proto enum with `match completion.matched_stop` against `worker_stream::MatchedStop::{Str, TokenId}`.
3. Replaces `complete.output_ids()` with `chunk.token_ids` (already a `Vec<u32>` in `TokenChunk`).
4. Stop decoder, parser pool, reasoning parser usage is unchanged (they came from `prepare/`).

### E.2 Remove the now-orphaned shared bits
- `grpc/common/response_collection.rs` (91 ln) — DELETE; its job (iterating `ExecutionResult` into `ProtoGenerateComplete`s) is replaced by `WorkerStream` direct iteration in the aggregator.
- `grpc/common/response_formatting.rs` (29 ln) — DELETE; fold the 1-2 small functions into the aggregator files that need them.

### E.3 Construct ResponseProcessor and StreamingProcessor as free functions, not types
The current `ResponseProcessor` and `StreamingProcessor` structs (in `processor.rs` and `streaming.rs`) only carry config (parser factories, backend label). Rather than reproduce those structs, pass the config as function arguments. This matches design §3.2 — no traits, no needless types.

### E.4 Test infrastructure
- `worker_stream/test_support.rs` (`#[cfg(test)]`): helper `pub fn synthetic_stream(chunks: Vec<TokenChunk>) -> WorkerStream` builds an in-memory stream backed by `tokio::sync::mpsc`.
- `tests/grpc_render_tests.rs`: 4 tests (one per render file) feeding a `Vec<TokenChunk>` with one Partial + one Complete and asserting on the produced `Response` (status, content-type, body equal to expected JSON or SSE bytes).

### E.5 Hard constraints
- No `mesh_grpc::*` imports under `render/`.
- No `crate::routers::grpc::*` imports under `render/` (the render layer is grpc-free; it talks only to `worker_stream/` and `prepare/`).
- Old `regular/streaming.rs` and `regular/processor.rs` remain in place — deleted in Part I after Pipeline switches over.

**Done when**: Test Part E passes + `rust-reviewer` clean.

---

## Test Part E

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| E1 | auto | Build clean | `cargo build --release` | exit 0 |
| E2 | auto | No test regression | `cargo test --release` | pass count ≥ Part D |
| E3 | auto | chat_aggregator collapses synthetic stream | `cargo test --release --test grpc_render_tests render_chat_aggregator` | exit 0 |
| E4 | auto | chat_streaming emits SSE sequence | `cargo test --release --test grpc_render_tests render_chat_streaming` | exit 0 |
| E5 | auto | generate_aggregator | `cargo test --release --test grpc_render_tests render_generate_aggregator` | exit 0 |
| E6 | auto | generate_streaming | `cargo test --release --test grpc_render_tests render_generate_streaming` | exit 0 |
| E7 | auto | `render/` is grpc-free | `grep -rE 'crate::routers::grpc\|mesh_grpc::' src/routers/render/` | empty |
| E8 | auto | `render/` consumes `TokenChunk` | `grep -rE 'TokenChunk\|WorkerStream' src/routers/render/` | ≥ 4 distinct files match |
| E9 | auto | `response_collection.rs` and `response_formatting.rs` deleted | `! test -f src/routers/grpc/common/response_collection.rs && ! test -f src/routers/grpc/common/response_formatting.rs` | exit 0 |
| E10 | manual | `rust-reviewer` | dispatched after E1–E9 pass | no blockers |

---

## Dev Part F — New `Pipeline` (4-step body) + switch regular `GrpcRouter` over

**Maps to design §7 Step 5.** Goal: `GrpcRouter` (regular mode) routes every chat/generate request through the new 4-step `Pipeline`. PD mode and `/v1/responses` still use the old path in this part.

### F.1 Write the new `Pipeline`
Replace `grpc/pipeline.rs`. The new module exports ONE type `Pipeline` with 3 fns:
```rust
pub(crate) struct Pipeline {
    planner: Arc<dyn PdPlanner>,
    engine: GrpcEngine,
    components: Arc<SharedComponents>,
    backend_label: &'static str,    // metrics_labels::BACKEND_REGULAR or BACKEND_PD
}

impl Pipeline {
    pub fn new_regular(...) -> Self { ... }    // wires GrpcEngine + DefaultPlanner same way as RequestPipeline::new_regular
    pub fn new_pd(...)      -> Self { ... }

    pub async fn execute_chat(
        &self, req: Arc<ChatCompletionRequest>, headers: Option<HeaderMap>,
        model_id: Option<String>, components: Arc<SharedComponents>,
    ) -> Response { /* see body below */ }

    pub async fn execute_generate(...) -> Response { /* parallel */ }

    pub async fn execute_chat_for_responses(
        &self, req: Arc<ChatCompletionRequest>, headers: Option<HeaderMap>,
        model_id: Option<String>, components: Arc<SharedComponents>,
    ) -> Result<ChatCompletionResponse, Response> { /* parallel, no streaming branch */ }
}
```

`execute_chat` body — 4 explicit steps wrapping the metrics calls that `RequestPipeline::execute_chat` does today:
```rust
let start = Instant::now();
let req_for_metrics = Arc::clone(&req);
let streaming = req.stream;
Metrics::record_router_request(ROUTER_GRPC, self.backend_label, CONNECTION_GRPC,
                                &req_for_metrics.model, ENDPOINT_CHAT, bool_to_static_str(streaming));

// 1. prepare
let (payload, resp_ctx) = match prepare::prepare_chat(req, headers, model_id, &components) {
    Ok(t) => t,
    Err(e) => { record_error_metric(...); return e; }
};

// 2. plan
let placement = match self.planner.plan(/*descriptor from &payload*/).await {
    Ok(p) => p,
    Err(e) => { record_error_metric(...); return placement_err_to_response(e, ...); }
};
let _guards = LoadGuards::new(&placement, resp_ctx.headers.as_ref());

// 3. dispatch
let stream = match self.engine.dispatch(&placement, &payload).await {
    Ok(s) => s,
    Err(e) => { record_error_metric(...); return engine_err_to_response(e); }
};

// 4. render
let response = if resp_ctx.original.is_streaming() {
    render::chat_streaming::process(stream, resp_ctx, self.backend_label)
} else {
    render::chat_aggregator::process(stream, resp_ctx).await
};
record_duration_metric(...);
response
```

### F.2 Switch `GrpcRouter` (regular)
- `grpc/router.rs:38` change `pipeline: RequestPipeline` → `pipeline: Pipeline`.
- `grpc/router.rs:72` change `RequestPipeline::new_regular(...)` → `Pipeline::new_regular(...)`.
- `grpc/router.rs:128-130` `pipeline.execute_chat(...)` continues to work — same signature.
- Same for `execute_generate` and the `ResponsesContext::new(Arc::new(pipeline.clone()), ...)` (Responses still uses `Arc<Pipeline>`).
- Update `ResponsesContext.pipeline: Arc<Pipeline>` accordingly in `common/responses/context.rs:13`.

### F.3 PD router NOT switched yet
`grpc/pd_router.rs` still uses `RequestPipeline::new_pd`. Old `RequestPipeline` and old stages are still alive — they get exercised by PD-mode traffic until Part G.

### F.4 Hard constraints
- `Pipeline::execute_chat` body ≤ 60 lines (gate F4).
- No `Vec<Box<dyn PipelineStage>>` anywhere in `pipeline.rs` (gate F3).
- `RetryExecutor` wrapping in `router.rs` is untouched.
- All metrics labels (`ROUTER_GRPC`, `BACKEND_REGULAR/PD`, `CONNECTION_GRPC`, `ENDPOINT_CHAT/GENERATE`) are recorded with the same arguments as today — no metric semantic drift.

**Done when**: Test Part F (auto) passes + (human) `/mesh-e2e-test` regular matrix recorded passing + `rust-reviewer` clean.

---

## Test Part F

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| F1 | auto | Build clean | `cargo build --release` | exit 0 |
| F2 | auto | No test regression | `cargo test --release` | pass count ≥ Part E |
| F3 | auto | No stage-based plumbing in `pipeline.rs` | `grep -E 'Vec<Box<dyn PipelineStage' src/routers/grpc/pipeline.rs` | empty |
| F4 | auto | `Pipeline::execute_chat` body ≤ 60 lines | `awk '/fn execute_chat\b/,/^    \}/' src/routers/grpc/pipeline.rs \| wc -l` | < 60 |
| F5 | auto | `GrpcRouter.pipeline` is the new `Pipeline` type | `grep -E 'pipeline: Pipeline,' src/routers/grpc/router.rs` | 1 match |
| F6 | auto | `Pipeline::new_regular` and `Pipeline::new_pd` exist | `grep -E 'pub fn new_(regular\|pd)' src/routers/grpc/pipeline.rs` | 2 matches |
| F7 | auto | `ResponsesContext.pipeline: Arc<Pipeline>` | `grep -E 'pipeline: Arc<Pipeline>' src/routers/grpc/common/responses/context.rs` | 1 match |
| F8 | auto | Metrics calls preserved | `grep -c 'Metrics::record_router_' src/routers/grpc/pipeline.rs` | ≥ 6 (3 per chat/generate path) |
| F9 | human | `/mesh-e2e-test` regular matrix (sglang+vllm × chat+generate × streaming+non-streaming) | user runs `/mesh-e2e-test` | all green; record in `findings.md` |
| F10 | manual | `rust-reviewer` on `pipeline.rs` + `router.rs` | dispatched after F1–F8 pass | confirms 4-step structure; retry wrapper untouched |

---

## Dev Part G — Switch PD + `/v1/responses` + lock SSE byte snapshots

**Maps to design §7 Step 6.** Goal: every chat/generate path uses the new `Pipeline`. Responses-API streaming preserves byte-exact SSE versus main branch.

### G.0 Pre-step (run BEFORE any code change in this part)
Record golden SSE bytes from **main branch HEAD** for a fixed prompt set. Use a mock worker fixture (built in Part D under `worker_stream/test_support.rs`) that returns scripted chunks — this makes the bytes deterministic across runs.

```bash
git checkout main -- atom/mesh/src/                          # temporarily switch source to main
cd atom/mesh
cargo test --release --test grpc_sse_snapshot -- --ignored --record-golden    # writes tests/fixtures/sse_golden/*.bin
git checkout HEAD -- atom/mesh/src/                          # restore refactor branch
```
Commit the goldens before any code changes in this part. **If the snapshot helper does not exist yet, build it as the first step of Part G.**

### G.1 Switch PD router
- `grpc/pd_router.rs:31` change type, `:67-78` change constructor call — same shape as Part F's `GrpcRouter` switch.

### G.2 Switch responses-streaming path
`regular/responses/streaming.rs` currently invokes `pipeline.execute_chat(stream=true, ...)` and transforms the resulting SSE into Responses-API SSE events. Since `Pipeline::execute_chat` returns the same `Response` body, the transformation is unchanged — only the inner pipeline call swaps.

`regular/responses/non_streaming.rs` calls `pipeline.execute_chat_for_responses(...)` — already typed; swap type only.

### G.3 Hard constraints
- SSE byte snapshots (G2–G6) MUST be byte-identical. A diff means a render-layer bug — fix the render, do NOT regenerate the golden.
- No relocation of `responses/` subtree in Part G — that is Part H.

**Done when**: Test Part G (auto) passes + (human) `/mesh-e2e-test` PD matrix + `/v1/responses` smoke + `rust-reviewer` clean.

---

## Test Part G

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| G0 | auto | Golden SSE fixtures present | `find atom/mesh/tests/fixtures/sse_golden -name '*.bin' \| wc -l` | ≥ 8 ({sglang,vllm} × {regular,PD} × {chat,generate}) |
| G1 | auto | Build clean | `cargo build --release` | exit 0 |
| G2 | auto | SSE: chat × sglang × regular | `cargo test --release --test grpc_sse_snapshot sse_chat_sglang_regular` | byte-identical |
| G3 | auto | SSE: chat × sglang × PD | `cargo test --release --test grpc_sse_snapshot sse_chat_sglang_pd` | byte-identical |
| G4 | auto | SSE: chat × vllm × regular | `cargo test --release --test grpc_sse_snapshot sse_chat_vllm_regular` | byte-identical |
| G5 | auto | SSE: chat × vllm × PD | `cargo test --release --test grpc_sse_snapshot sse_chat_vllm_pd` | byte-identical |
| G6 | auto | SSE: generate × 4 backends | `cargo test --release --test grpc_sse_snapshot sse_generate_` | all 4 byte-identical |
| G7 | auto | PD router uses new `Pipeline` | `grep -E 'pipeline: Pipeline,' src/routers/grpc/pd_router.rs` | 1 match |
| G8 | auto | Responses uses `execute_chat_for_responses` | `grep -rE 'execute_chat_for_responses' src/routers/grpc/{common,regular}/responses/` | ≥ 1 match |
| G9 | auto | Old `RequestPipeline` no longer used by any router | `grep -rE 'RequestPipeline' src/routers/grpc/{router,pd_router}.rs src/routers/grpc/common/responses/` | empty |
| G10 | human | `/mesh-e2e-test` PD matrix + `/v1/responses` smoke | user runs | all green; record in `findings.md` |
| G11 | manual | `rust-reviewer` | dispatched after G1–G9 pass | no blockers |

**On G2–G6 mismatch**: fix the render layer (not the snapshot). Mismatch is a real behavioral regression.

---

## Dev Part H — Relocate Responses subtree to `routers/openai/responses/`

**Maps to design §7 Step 7, with V-1 rename omitted.** Goal: pure file relocation + 1 merger + 4 renames. Zero business-logic edits.

### H.1 Create the openai namespace
- `atom/mesh/src/routers/openai/mod.rs` (just `pub mod responses;`)
- `atom/mesh/src/routers/openai/responses/mod.rs`
- Register `openai` in `routers/mod.rs`.

### H.2 Move files (per design §4, names corrected for what's actually in each file)

| From | → To | Notes |
|---|---|---|
| `grpc/common/responses/context.rs` (47 ln) | `openai/responses/context.rs` | type field changes `Arc<RequestPipeline>` → `Arc<crate::routers::grpc::Pipeline>` (concrete type, no trait) |
| `grpc/common/responses/streaming.rs` (638 ln) **+** `grpc/regular/responses/streaming.rs` (383 ln) | **MERGED →** `openai/responses/streaming.rs` (~1020 ln) | Verify the merger does not introduce a name collision; if both files define `ResponseStreamEventEmitter` and a free fn of the same name, namespace one inside an `emitter` submodule. |
| `grpc/common/responses/handlers.rs` (71 ln, GET/cancel only) | `openai/responses/retrieve.rs` | rename per design |
| `grpc/common/responses/utils.rs` (64 ln) | `openai/responses/persistence.rs` | rename per design |
| `grpc/regular/responses/handlers.rs` (118 ln, entry dispatch) | `openai/responses/handlers.rs` | rename per design |
| `grpc/regular/responses/non_streaming.rs` (87 ln) | `openai/responses/non_streaming.rs` | move |
| `grpc/regular/responses/common.rs` (195 ln, `load_conversation_history`) | `openai/responses/conversation.rs` | rename per design |
| `grpc/regular/responses/conversions.rs` (425 ln) | `openai/responses/conversions.rs` | move |
| `grpc/common/responses/` directory | DELETE empty | after all moves |
| `grpc/regular/responses/` directory | DELETE empty | after all moves |

### H.3 V-1: Do NOT rename `grpc/router.rs` or `grpc/pd_router.rs`
The design's proposed `router.rs → http_router.rs` / `pd_router.rs → http_router_pd.rs` collides with the pre-existing `routers/http_router.rs` (32 KB) and `routers/http_pd_router.rs` (67 KB). Per §1 Q1 default: keep current names.

### H.4 Update imports
- `grpc/router.rs:7-11` change `use super::{common::responses::..., regular::responses};` → `use crate::routers::openai::responses::{ResponsesContext, handlers as responses, retrieve};`.
- Wherever `cancel_response_impl` / `get_response_impl` are imported, update to the new `openai::responses::retrieve` path.
- `pd_router.rs` does NOT import responses today (PD returns 501 for `/v1/responses`); no change.

### H.5 Hard constraints
- No function-body edits. Only file location, import paths, the one streaming merger, and the `ResponsesContext.pipeline` type field.
- Layering rules (design §6.11 / §6.12 / §6.13) must hold — gates H5/H6/H7.

**Done when**: Test Part H (auto) passes + (human) `/v1/responses` smoke recorded passing + `rust-reviewer` clean.

---

## Test Part H

| # | Tier | Check | Command | Pass criterion |
|---|------|---|---|---|
| H1 | auto | Build clean | `cargo build --release` | exit 0 |
| H2 | auto | No test regression | `cargo test --release` | pass count ≥ Part G |
| H3 | auto | New paths exist | `test -f src/routers/openai/responses/mod.rs && test -f src/routers/openai/responses/streaming.rs && test -f src/routers/openai/responses/conversions.rs` | exit 0 |
| H4 | auto | Old paths gone (responses subdirs only — router.rs/pd_router.rs NOT renamed per V-1) | `! test -d src/routers/grpc/common/responses && ! test -d src/routers/grpc/regular/responses` | exit 0 |
| H5 | auto | §6.11 — `openai/responses/` references `grpc` only via `Pipeline` | `grep -rE 'crate::routers::grpc::[a-z_]+' src/routers/openai/responses/ \| grep -vE '(grpc::pipeline)?::Pipeline\\b'` | empty |
| H6 | auto | §6.12 — `grpc/` does not reference `openai/` | `grep -r 'crate::routers::openai' src/routers/grpc/` | empty |
| H7 | auto | §6.13 — `openai/responses/` has no `mesh_grpc::*` imports | `grep -r 'mesh_grpc::' src/routers/openai/responses/` | empty |
| H8 | auto | The two streaming files merged (not duplicated) | `find src/routers/openai/responses -name 'streaming*.rs' \| wc -l` | exactly 1 |
| H9 | auto | Peer folders still grpc-free | `grep -rE 'crate::routers::grpc' src/routers/{prepare,render,worker_stream}/` | empty |
| H10 | auto | `routers/http_router.rs` and `routers/http_pd_router.rs` (top-level) untouched | `git diff --name-only main -- src/routers/http_router.rs src/routers/http_pd_router.rs` | empty (no V-1 collision) |
| H11 | human | `/v1/responses` streaming + non-streaming smoke | user runs | both succeed; record in `findings.md` |
| H12 | manual | `rust-reviewer` on the relocation diff | dispatched after H1–H10 pass | no blockers; purely structural |

---

## Dev Part I — Delete obsolete code + final acceptance sweep

**Maps to design §7 Step 8.** Goal: drain everything obsoleted by Parts A–H; tree matches design §3.1 layout (modulo V-1).

### I.1 Verify no callers, then delete
For each item below, run `grep -rE '<symbol>' src/` FIRST. Only delete if empty.

- `grpc/context.rs::ProcessingState` and the sibling structs that became dead: `PreparationOutput`, `WorkerSelection`, `ClientSelection`, `DispatchMetadata`, `LoadGuards` (note: `LoadGuards::new` is now called from `Pipeline`, so move it to `pipeline.rs` first then delete from `context.rs`), `ExecutionResult`, `FinalResponse`, `ResponseState`. Verify each is unused before delete.
- Old `RequestPipeline` (may already be gone from F1).
- `grpc/common/stages/` (entire dir, 5 files: `client_acquisition`, `dispatch_metadata`, `helpers`, `request_execution`, `worker_selection`, `mod.rs`).
- `grpc/regular/stages/` (entire dir, 8 files across `chat/`, `generate/`, and top level).
- `grpc/regular/streaming.rs` (1326 ln) and `grpc/regular/processor.rs` (465 ln).
- `grpc/utils.rs` — empty by now if all moves landed. Final check: `wc -l src/routers/grpc/utils.rs` should be 0 or ≤ 5 (re-export only) → delete.
- `PipelineStage` trait + the helper `helpers` module under `common/stages/`.
- Now-empty `grpc/common/` and `grpc/regular/` directories.
- `ProcessedMessages` from `grpc/mod.rs` (already relocated in Part C).

### I.2 Hard constraints
- No `pub use` re-exports for deleted symbols.
- `cargo machete` and `cargo udeps` report no new orphans.
- All design §6.1–§6.15 acceptance criteria pass (Test Part I).
- `findings.md` final entry summarizes line-count delta:
  - `grpc/` before / after (target: 9280 → ~1800)
  - `prepare/`, `render/`, `worker_stream/`, `openai/responses/` new totals.

**Done when**: Test Part I (auto) passes + (human) full `/mesh-e2e-test` matrix recorded passing + final `rust-reviewer` clean.

---

## Test Part I (= global §6 acceptance criteria sweep)

| # | Tier | §6 ref | Check | Command | Pass criterion |
|---|------|---|---|---|---|
| I1 | auto | — | Build clean | `cargo build --release` | exit 0 |
| I2 | auto | — | All tests pass | `cargo test --release` | pass count ≥ baseline + new tests, zero regressions |
| I3 | auto | §6.1 | `mesh_grpc::*` only in `grpc/engine/` | `grep -rl 'mesh_grpc::' src/routers/ \| grep -v '/grpc/engine/'` | empty |
| I4 | auto | §6.2 | `prepare`, `render`, `worker_stream` grpc-free | `grep -rE 'crate::routers::grpc' src/routers/{prepare,render,worker_stream}/` | empty |
| I5 | auto | §6.3 | No god-bag struct (`ProcessingState` gone) | `grep -rE 'struct ProcessingState\b' src/routers/` | empty |
| I6 | auto | §6.4 | No `Vec<Box<dyn PipelineStage>>` | `grep -rE 'Vec<Box<dyn PipelineStage' src/routers/`; also `grep -r 'trait PipelineStage' src/routers/` | both empty |
| I7 | auto | §6.5 | Every Rust file in refactored tree has a one-line header comment | `for f in $(find src/routers/{grpc,prepare,render,worker_stream,openai} -name '*.rs'); do head -1 "$f" \| grep -qE '^//' \|\| echo "MISSING: $f"; done` | no `MISSING:` lines |
| I8 | auto | §6.6 | No umbrella file names | `find src/routers/{grpc,prepare,render,worker_stream,openai} \( -name utils.rs -o -name common.rs -o -name helpers.rs -o -name chunk.rs -o -name payload.rs \)` | empty |
| I9 | auto | §6.7 | Protocol logic unit-testable without `mesh_grpc::` | `cargo test --release routers::prepare routers::render` | exit 0 |
| I10 | auto | §6.8 | Proto snapshots A–D | `cargo test --release --test grpc_proto_snapshot` | all 4 pass |
| I11 | auto | §6.9 | PD merge T1–T7 | `cargo test --release --test grpc_pd_merge_tests` | all 7 pass |
| I12 | auto | §6.10 | `WorkerStream::Drop` propagation (single + PD) | `cargo test --release --test grpc_engine_drop_tests` | both pass |
| I13 | auto | §6.11 | `openai/responses/` references `grpc` only via `Pipeline` | re-run H5 command | empty |
| I14 | auto | §6.12 | `grpc/` does not reference `openai/` | re-run H6 | empty |
| I15 | auto | §6.13 | `openai/responses/` has no `mesh_grpc::*` | re-run H7 | empty |
| I16 | auto | §6.14 | SSE byte snapshots stable | re-run G2–G6 | all byte-identical |
| I17 | auto | — | Dead-code sweep | `cargo machete 2>&1; cargo udeps --release 2>&1 \| grep -E '^(unused\|orphan)'` | no new unused/orphan entries vs Part 0 baseline |
| I18 | auto | — | Old files gone | `! test -f src/routers/grpc/utils.rs && ! test -d src/routers/grpc/common && ! test -d src/routers/grpc/regular && ! grep -E 'struct ProcessingState\b' src/routers/grpc/context.rs 2>/dev/null` | exit 0 |
| I19 | auto | — | Line-count sanity | `find src/routers/grpc -name '*.rs' -exec wc -l {} + \| tail -1` | total ≤ 2500 (design predicts ~1800) |
| I20 | human | §6.15 | Full `/mesh-e2e-test` matrix | user runs | every cell green; record in `findings.md` |
| I21 | manual | — | Final `rust-reviewer` on full diff vs main | dispatched after I1–I19 pass | no blockers |

**On I20**: agent must stop after I19 and request user to run `/mesh-e2e-test`. Refactor is not declared done until I20 passes.

---

## Appendix A — `findings.md` schema (the agent's running log)

```markdown
# gRPC Refactor — Findings Log

## Part 0 — <date>
- baseline_pass_count = N
- inventory: <counts from 0.c>
- Q1 decision: <user choice>
- Q2 decision: <user choice>

## Part B — <date>
- Outcome: (i) | (ii) | (iii)
- Alignment comments added: <count> (if ii)
- Escalated to user: <Y/N> (if iii)

## Part F — <date>
- pass_count = N (delta vs Part E: +<k>)
- /mesh-e2e-test (regular): PENDING | PASS | FAIL (user-reported)

## Part G — <date>
- pass_count = N
- SSE goldens: 8 files committed
- /mesh-e2e-test (PD + responses): PENDING | PASS | FAIL

## Part I — <date>
- Final line counts:
  - grpc/      = 9280 → <N>
  - prepare/   = <N>
  - render/    = <N>
  - worker_stream/ = <N>
  - openai/responses/ = <N>
- /mesh-e2e-test (full): PENDING | PASS | FAIL
- rust-reviewer final: PENDING | CLEAN | BLOCKED
```

## Appendix B — Stop conditions (escalate, do not patch around)

- Part B outcome (iii) → user must pick upstream-vs-shim before Part C.
- Any SSE byte snapshot mismatch in Part G or I that does not collapse to a render-layer bug in one debug pass.
- Any `rust-reviewer` blocker that requires re-shaping a boundary type or trait.
- Structural surprise: file already at target path with non-empty unrelated content; type collision with an unrelated module; etc.
- A Pipeline metric label that was being emitted today but is missing after Part F (verify against `findings.md` Part 0 metric inventory if needed — add to 0.c if useful).
