//! Response assertions for fixture-backed tests.
//!
//! Worker responses often contain dynamic fields such as ids or timestamps.
//! These helpers assert that the actual response contains the stable fixture
//! fields without requiring exact whole-body equality.

use serde_json::Value;

/// Recursively validate that `actual` contains every field in `expected`.
pub fn json_contains(actual: &Value, expected: &Value) -> Result<(), String> {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            for (key, expected_value) in expected {
                let actual_value = actual
                    .get(key)
                    .ok_or_else(|| format!("missing expected key `{}` in {:?}", key, actual))?;
                json_contains(actual_value, expected_value)?;
            }
            Ok(())
        }
        (Value::Array(actual), Value::Array(expected)) => {
            if actual.len() < expected.len() {
                return Err(format!(
                    "actual array has fewer items than expected: actual={}, expected={}",
                    actual.len(),
                    expected.len()
                ));
            }
            for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
                json_contains(actual_value, expected_value)?;
            }
            Ok(())
        }
        _ if actual == expected => Ok(()),
        _ => Err(format!("expected {}, got {}", expected, actual)),
    }
}

/// Recursively assert that `actual` contains every field in `expected`.
pub fn assert_json_contains(actual: &Value, expected: &Value) {
    json_contains(actual, expected).unwrap();
}

/// Golden response assertion built from a fixture's expected response.
#[derive(Clone, Debug)]
pub struct GoldenAssert {
    pub expected_status: u16,
    pub expected_body: Value,
}

impl GoldenAssert {
    pub fn validate_response(&self, actual_status: u16, actual_body: &Value) -> Result<(), String> {
        if actual_status != self.expected_status {
            return Err(format!(
                "expected status {}, got {} with body {}",
                self.expected_status, actual_status, actual_body
            ));
        }
        json_contains(actual_body, &self.expected_body)
    }

    pub fn assert_response(&self, actual_status: u16, actual_body: &Value) {
        self.validate_response(actual_status, actual_body).unwrap();
    }
}

/// Validate that at least one collected SSE event contains the expected fields.
pub fn any_json_contains(actual_events: &[Value], expected: &Value) -> Result<(), String> {
    if actual_events
        .iter()
        .any(|actual| json_contains(actual, expected).is_ok())
    {
        return Ok(());
    }

    Err(format!(
        "no SSE event matched expected subset: expected={}, actual_events={:?}",
        expected, actual_events
    ))
}

/// Assert that at least one collected SSE event contains the expected fields.
pub fn assert_any_json_contains(actual_events: &[Value], expected: &Value) {
    any_json_contains(actual_events, expected).unwrap();
}
