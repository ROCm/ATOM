//! Smoke tests proving the fixture-driven harness reaches regular HTTP routes.

use crate::test_atomesh::{
    ConnectionModeFixture, MockTestCase, TestHarness, TestHarnessResult, VirtualGrpcWorker,
    WorkerKindFixture,
};

const ATOMESH_HARNESS_FIXTURE_DIR: &str = "tests/fixtures/atomesh_harness";

fn load_fixture_case(file_name: &str) -> MockTestCase {
    MockTestCase::from_fixture(format!("{}/{}", ATOMESH_HARNESS_FIXTURE_DIR, file_name)).unwrap()
}

fn log_case_result(case: &MockTestCase, result: &TestHarnessResult) {
    println!(
        concat!(
            "fixture_case={} endpoint={} status={} ",
            "router_mode={} connection_mode={} policy={} ",
            "worker_urls={:?} registered_workers={} healthy_workers={} ",
            "worker_path={:?} stream_events={}"
        ),
        case.name,
        case.endpoint,
        result.status,
        result.router_mode,
        result.connection_mode,
        result.policy,
        result.worker_urls,
        result.registered_workers,
        result.healthy_workers,
        result.worker_path,
        result.stream_events.len()
    );
}

#[tokio::test]
async fn test_atomesh_harness_http_regular_chat() {
    // This case verifies non-streaming regular HTTP chat routing.
    // It passes when Atomesh returns the fixture golden response and the
    // virtual worker receives `/v1/chat/completions`.
    let case = load_fixture_case("http_regular_chat.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/chat/completions");
}

#[tokio::test]
async fn test_atomesh_harness_http_regular_generate() {
    // This case verifies non-streaming regular HTTP /generate routing.
    // It passes when Atomesh returns the fixture golden response and the
    // virtual worker receives `/generate`.
    let case = load_fixture_case("http_regular_generate.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/generate");
}

#[tokio::test]
async fn test_atomesh_harness_http_regular_chat_streaming() {
    // This case verifies regular HTTP chat streaming through Atomesh.
    // It passes when an SSE JSON event matches the fixture and the virtual
    // worker receives `/v1/chat/completions`.
    let case = load_fixture_case("http_regular_chat_streaming.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/chat/completions");
}

#[tokio::test]
async fn test_atomesh_harness_http_regular_completion() {
    // This case ports an existing /v1/completions shape into the harness.
    // It passes when Atomesh returns the fixture golden response and the
    // virtual worker receives `/v1/completions`.
    let case = load_fixture_case("http_regular_completion.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/completions");
}

#[tokio::test]
async fn test_atomesh_harness_http_pd_chat() {
    // This case verifies HTTP PD chat routing through Atomesh.
    // It passes when Atomesh returns the fixture golden response and both
    // prefill/decode workers receive `/v1/chat/completions`.
    let case = load_fixture_case("http_pd_chat.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_count_at_least("/v1/chat/completions", 2);
}

#[tokio::test]
async fn test_atomesh_harness_grpc_regular_generate_boundary() {
    // This case verifies gRPC fixtures are classified before tonic support.
    // It passes when VirtualGrpcWorker owns the fixture and the HTTP harness
    // rejects it with the reserved gRPC transport boundary.
    let case = load_fixture_case("grpc_regular_generate.json");
    assert_eq!(case.route.connection_mode, ConnectionModeFixture::Grpc);

    let mut grpc_worker = VirtualGrpcWorker::new(case.clone()).unwrap();
    assert_eq!(grpc_worker.case_name(), "grpc_regular_generate");
    assert_eq!(grpc_worker.worker_kind(), &WorkerKindFixture::Regular);

    let grpc_error = grpc_worker.start().await.unwrap_err().to_string();
    assert!(grpc_error.contains("tonic transport is not implemented yet"));

    let harness_error = TestHarness::new(case).run().await.unwrap_err().to_string();
    assert!(harness_error.contains("gRPC fixtures are reserved for VirtualGrpcWorker"));
}
