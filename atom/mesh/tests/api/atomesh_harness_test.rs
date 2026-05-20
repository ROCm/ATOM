//! Smoke tests proving the fixture-driven harness reaches regular HTTP routes.

use crate::test_atomesh::{MockTestCase, TestHarness, TestHarnessResult};

const MOCK_TEST_FIXTURE_DIR: &str = "tests/fixtures/mock_tests";

fn load_fixture_case(file_name: &str) -> MockTestCase {
    MockTestCase::from_fixture(format!("{}/{}", MOCK_TEST_FIXTURE_DIR, file_name)).unwrap()
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
async fn test_fixture_driven_chat_regular_http() {
    // This case verifies non-streaming regular HTTP chat routing.
    // It passes when Atomesh returns the fixture golden response and the
    // virtual worker receives `/v1/chat/completions`.
    let case = load_fixture_case("who_are_you_chat.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/chat/completions");
}

#[tokio::test]
async fn test_fixture_driven_generate_regular_http() {
    // This case verifies non-streaming regular HTTP /generate routing.
    // It passes when Atomesh returns the fixture golden response and the
    // virtual worker receives `/generate`.
    let case = load_fixture_case("generate_basic.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/generate");
}

#[tokio::test]
async fn test_fixture_driven_chat_streaming_regular_http() {
    // This case verifies regular HTTP chat streaming through Atomesh.
    // It passes when an SSE JSON event matches the fixture and the virtual
    // worker receives `/v1/chat/completions`.
    let case = load_fixture_case("chat_streaming_text.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/chat/completions");
}

#[tokio::test]
async fn test_fixture_driven_chat_http_pd() {
    // This case verifies HTTP PD chat routing through Atomesh.
    // It passes when Atomesh returns the fixture golden response and both
    // prefill/decode workers receive `/v1/chat/completions`.
    let case = load_fixture_case("pd_prefill_decode.json");
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();
    log_case_result(&case, &result);

    result.assert_response(&case);
    result.assert_worker_path_count_at_least("/v1/chat/completions", 2);
}
