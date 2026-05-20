//! Response assertions for fixture-backed tests.
//!
//! Worker responses often contain dynamic fields such as ids or timestamps.
//! These helpers assert that the actual response contains the stable fixture
//! fields without requiring exact whole-body equality.

use serde_json::Value;

/// Recursively assert that `actual` contains every field in `expected`.
pub fn assert_json_contains(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            for (key, expected_value) in expected {
                let actual_value = actual
                    .get(key)
                    .unwrap_or_else(|| panic!("missing expected key `{}` in {:?}", key, actual));
                assert_json_contains(actual_value, expected_value);
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert!(
                actual.len() >= expected.len(),
                "actual array has fewer items than expected: actual={}, expected={}",
                actual.len(),
                expected.len()
            );
            for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
                assert_json_contains(actual_value, expected_value);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

/// Golden response assertion built from a fixture's expected response.
#[derive(Clone, Debug)]
pub struct GoldenAssert {
    pub expected_status: u16,
    pub expected_body: Value,
}

impl GoldenAssert {
    pub fn assert_response(&self, actual_status: u16, actual_body: &Value) {
        assert_eq!(actual_status, self.expected_status);
        assert_json_contains(actual_body, &self.expected_body);
    }
}

/// Assert that at least one collected SSE event contains the expected fields.
pub fn assert_any_json_contains(actual_events: &[Value], expected: &Value) {
    assert!(
        actual_events
            .iter()
            .any(
                |actual| std::panic::catch_unwind(|| assert_json_contains(actual, expected))
                    .is_ok()
            ),
        "no SSE event matched expected subset: expected={}, actual_events={:?}",
        expected,
        actual_events
    );
}
