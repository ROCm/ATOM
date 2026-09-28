//! Stores replayable fixture cases and matches backend requests to them.
//!
//! `VirtualWorker` delegates request matching here so worker handlers stay
//! focused on HTTP behavior. Matching currently uses the request endpoint and
//! the primary prompt text extracted from the request body.

use std::path::Path;

use serde_json::Value;

use super::MockCase;

/// Collection of fixture cases a virtual worker can replay.
#[derive(Clone, Debug)]
pub struct ReplayCaseStore {
    cases: Vec<MockCase>,
}

impl ReplayCaseStore {
    pub fn new(cases: Vec<MockCase>) -> Self {
        Self { cases }
    }

    /// Convenience loader for tests that only need one fixture.
    pub fn from_fixture(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(vec![MockCase::from_fixture(path)?]))
    }

    /// Find the fixture that corresponds to a request forwarded by Atomesh.
    pub fn match_request(&self, endpoint: &str, body: &Value) -> Option<&MockCase> {
        self.cases.iter().find(|case| {
            case.endpoint == endpoint
                && prompt_text(&case.request)
                    .is_none_or(|expected| prompt_text(body).is_some_and(|actual| actual == expected))
        })
    }

    pub fn first(&self) -> Option<&MockCase> {
        self.cases.first()
    }
}

fn prompt_text(value: &Value) -> Option<&str> {
    value
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| value.get("prompt").and_then(Value::as_str))
        .or_else(|| {
            value
                .get("messages")
                .and_then(Value::as_array)
                .and_then(|messages| messages.first())
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
        })
        .or_else(|| value.get("input").and_then(Value::as_str))
}
