//! Smoke tests proving the fixture-driven harness reaches regular HTTP routes.

use crate::test_atomesh::{MockTestCase, TestHarness};

#[tokio::test]
async fn test_fixture_driven_chat_regular_http() {
    let case =
        MockTestCase::from_fixture("tests/fixtures/mock_tests/who_are_you_chat.json").unwrap();
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();

    result.assert_response(&case);
    result.assert_worker_path_contains("/v1/chat/completions");
}

#[tokio::test]
async fn test_fixture_driven_generate_regular_http() {
    let case = MockTestCase::from_fixture("tests/fixtures/mock_tests/generate_basic.json").unwrap();
    let harness = TestHarness::new(case.clone());

    let result = harness.run().await.unwrap();

    result.assert_response(&case);
    result.assert_worker_path_contains("/generate");
}
