# Mesh Integration Tests

This directory contains the integration test suite for **atom/mesh**, organized by functional area. All tests run with `cargo test` and do not require a GPU or real inference backend — they use mock workers and in-memory routers.

**Total: ~232 tests across 25 test files.**

## Directory Structure

```
tests/
├── api/                        # HTTP API endpoint tests
├── routing/                    # Routing policy and load balancing tests
├── reliability/                # Fault tolerance and resilience tests
├── security/                   # Authentication and access control tests
├── spec/                       # Protocol contract/spec tests
├── common/                     # Shared test utilities (not tests themselves)
├── test_atomesh/               # Fixture-driven Atomesh test harness
├── fixtures/atomesh_harness/   # Prompt/request and expected-response samples
├── inflight_tracker_test.rs    # Standalone: inflight request tracking
├── load_guard_raii_test.rs     # Standalone: RAII load guard lifecycle
└── metrics_aggregator_test.rs  # Standalone: Prometheus metrics aggregation
```

## Test Infrastructure

The `common/` directory provides shared test utilities used across all test modules:

| File | Purpose |
|------|---------|
| `mock_worker.rs` | Mock HTTP server simulating an inference worker (configurable health, delays, failure rates) |
| `mock_openai_server.rs` | Mock OpenAI-compatible server for end-to-end testing |
| `test_app.rs` | `AppTestContext` — builds a full router app stack for integration testing |
| `test_config.rs` | `TestRouterConfig` / `TestWorkerConfig` — helpers for constructing test configurations |
| `streaming_helpers.rs` | Utilities for testing SSE (Server-Sent Events) streaming responses |

The `test_atomesh/` directory provides a reusable fixture-driven harness:

| File | Purpose |
|------|---------|
| `mock_test_case.rs` | Loads prompt/request, route mode, and expected-response fixtures |
| `virtual_request.rs` | Builds real Atomesh HTTP requests from fixtures |
| `virtual_worker.rs` | Axum virtual worker that replays matched fixture responses |
| `test_harness.rs` | Starts virtual workers, builds Atomesh app context, sends requests, and asserts results |
| `golden_assert.rs` | Shared subset assertions for dynamic response bodies |
| `virtual_grpc_worker.rs` | Unified tonic-backed virtual worker with internal SGLang/vLLM services |

The `fixtures/atomesh_harness/` directory stores the JSON samples used by the
harness. These files are test data, not integration test entrypoints. Each
fixture describes the request payload, expected response, route mode, connection
mode, backend (`sglang` or `vllm` for gRPC), and virtual worker simulation
settings for one reusable harness scenario. gRPC harness fixtures target the
native `/generate` boundary because OpenAI `/v1/completions` is adapted to the
same backend Generate RPC inside the gRPC router.

---

## API Tests (`api/`)

| File | Tests | Description |
|------|-------|-------------|
| `api_endpoints_test.rs` | 35 | Core HTTP endpoint coverage: `/liveness`, `/readiness`, `/health`, `/health_generate`, `/generate`, `/v1/chat/completions`. Tests status codes, response formats, error handling, concurrent requests, and health check behavior with healthy/unhealthy workers. |
| `atomesh_harness_test.rs` | 8 | Atomesh harness smoke tests driven by `fixtures/atomesh_harness/`. Starts local virtual workers, sends real Atomesh API requests, and verifies regular HTTP, streaming, completions, HTTP PD, SGLang/vLLM gRPC regular, and SGLang gRPC PD worker paths. |
| `parser_endpoints_test.rs` | 12 | Tests for `/parse/function_call` and `/parse/reasoning` endpoints. Verifies function call extraction from model output and reasoning/thinking block parsing. |
| `request_formats_test.rs` | 6 | Request format validation for `/generate`, `/v1/chat/completions`, and `/v1/completions`. Tests various payload shapes: text, input_ids, batch requests, sampling params, and special parameters (logprobs, json_schema, ignore_eos). |
| `responses_api_test.rs` | 11 | Responses API (conversations) CRUD operations. Tests creating, listing, retrieving, and deleting conversation sessions via the conversation handlers. |
| `streaming_tests.rs` | 7 | SSE streaming response tests. Verifies streaming output for `/generate` and `/v1/chat/completions`, including chunked transfer and stream termination. |

## Routing Tests (`routing/`)

| File | Tests | Description |
|------|-------|-------------|
| `load_balancing_test.rs` | 8 | Load balancing policy tests: round_robin, random, cache_aware, and other policies. Verifies request distribution across workers. |
| `power_of_two_test.rs` | 3 | Power of Two Choices algorithm tests. Selects the less loaded worker from two random candidates. |
| `cache_aware_backward_compat_test.rs` | 3 | Backward compatibility tests for CacheAwarePolicy with empty model IDs and legacy configurations. |
| `header_forwarding_test.rs` | 6 | Header propagation tests. Verifies that custom headers are correctly forwarded from client through the router to backend workers. |
| `payload_size_test.rs` | 5 | Request payload size limit tests. Verifies behavior with oversized payloads and boundary conditions. |
| `pd_routing_test.rs` | 3 | Prefill/Decode disaggregation routing. Tests routing decisions that split prefill and decode phases to different workers. |
| `pd_topology_test.rs` | 18 | PD topology end-to-end tests: 1P1D, 2P2D, and Regular mode. Verifies complete request flows through the router with mock worker backends. |
| `test_pd_routing.rs` | 23 | PD routing unit-level tests. Covers PD selection policies, worker assignment, context construction, and routing decisions with various topology configurations. |
| `policy_registry_integration.rs` | 3 | PolicyRegistry integration with RouterManager. Tests policy registration, lookup, and lifecycle management. |
| `worker_management_test.rs` | 3 | Dynamic worker management API: listing workers via `GET /workers`, routing with multiple workers, and request handling during worker changes. |

## Reliability Tests (`reliability/`)

| File | Tests | Description |
|------|-------|-------------|
| `circuit_breaker_test.rs` | 4 | Circuit breaker state machine: closed → open → half-open → recovery. Tests failure threshold detection and automatic recovery. |
| `fault_tolerance_test.rs` | 5 | System resilience under worker failures, network issues, and recovery scenarios. Verifies graceful degradation when backends become unavailable. |
| `rate_limiting_test.rs` | 4 | Rate limiting and concurrency control. Tests request throttling, burst handling, and concurrent request limits. |
| `retries_test.rs` | 4 | Retry mechanism: exponential backoff, max retry limits, and retry-on-failure behavior for transient errors. |

## Security Tests (`security/`)

| File | Tests | Description |
|------|-------|-------------|
| `auth_test.rs` | 5 | API key authentication middleware. Tests that requests without a valid API key are rejected, and valid keys are accepted. Covers both missing and invalid key scenarios. |

## Protocol Spec Tests (`spec/`)

Contract tests that verify serialization/deserialization behavior of protocol types used by mesh. These ensure the external `openai-protocol` crate's types behave as mesh expects.

| File | Tests | Description |
|------|-------|-------------|
| `chat_completion.rs` | 16 | `ChatCompletionRequest` normalization via the `Normalizable` trait. Tests deprecated field conversion: `max_tokens` → `max_completion_tokens`, `functions` → `tools`, `function_call` → `tool_choice`. Also covers `stream_options` validation and tool_choice cross-referencing. |
| `chat_message.rs` | 5 | `ChatMessage` tagged enum deserialization. Verifies that the `role` field correctly tags messages into System, User, Assistant, and Tool variants, and that invalid roles are rejected. |
| `responses.rs` | 36 | `ResponsesRequest` validation and construction. Tests field defaults, parameter validation, tool configuration, reasoning settings, and edge cases for the Responses API request type. |

## Standalone Tests (root level)

| File | Tests | Description |
|------|-------|-------------|
| `inflight_tracker_test.rs` | 3 | Inflight request tracker for observability. Tests increment/decrement of active request counts and concurrent tracking accuracy. |
| `load_guard_raii_test.rs` | 6 | `WorkerLoadGuard` RAII pattern with response body attachment. Verifies that load counters properly decrement when response bodies are consumed or dropped (client disconnect simulation), including dual prefill/decode guard scenarios. |
| `metrics_aggregator_test.rs` | 5 | Prometheus metrics aggregation from multiple workers. Tests merging of counter, gauge, and histogram metrics with label injection and deduplication. |

## Running Tests

```bash
# Run all mesh tests
cargo test -p atom-mesh

# Run the Atomesh harness smoke tests with router/worker logs
cargo test -p atom-mesh --test api_tests test_atomesh_harness -- --nocapture

# Run local quick checks used during harness development
cargo test -p atom-mesh --lib
cargo test -p atom-mesh --test spec_test
cargo test -p atom-mesh --test api_tests test_atomesh_harness -- --nocapture

# Run a specific test module
cargo test -p atom-mesh --test api_tests
cargo test -p atom-mesh --test routing_tests
cargo test -p atom-mesh --test reliability_tests
cargo test -p atom-mesh --test security_tests
cargo test -p atom-mesh --test spec_test

# Run a specific test by name
cargo test -p atom-mesh test_list_workers

# Run with output
cargo test -p atom-mesh -- --nocapture
```
