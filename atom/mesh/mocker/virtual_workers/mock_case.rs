//! Fixture schema for data-driven Atomesh integration tests.
//!
//! A `MockCase` describes the client request, target routing mode, and the
//! response that a virtual backend worker should return. Keeping this data in
//! JSON fixtures lets the same sample drive regular HTTP, PD, and gRPC
//! scenarios.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKindFixture {
    Regular,
    PrefillDecode,
}

impl Default for WorkerKindFixture {
    fn default() -> Self {
        Self::Regular
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionModeFixture {
    Http,
    Grpc,
}

impl Default for ConnectionModeFixture {
    fn default() -> Self {
        Self::Http
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendFixture {
    Sglang,
    Vllm,
}

impl Default for BackendFixture {
    fn default() -> Self {
        Self::Sglang
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RouteFixture {
    #[serde(default)]
    pub worker_kind: WorkerKindFixture,
    #[serde(default)]
    pub connection_mode: ConnectionModeFixture,
    #[serde(default)]
    pub backend: BackendFixture,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ExpectedResponse {
    pub status: u16,
    #[serde(default)]
    pub body: Value,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SimulationFixture {
    #[serde(default)]
    pub ttft_ms: u64,
    #[serde(default)]
    pub chunk_interval_ms: u64,
}

/// One fixture-backed test sample.
///
/// `request` is the client-facing Atomesh payload. `expected_response` is what
/// the virtual worker returns after Atomesh routes the request to it.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MockCase {
    pub name: String,
    pub model: String,
    pub endpoint: String,
    #[serde(default)]
    pub route: RouteFixture,
    #[serde(default)]
    pub request: Value,
    pub expected_response: ExpectedResponse,
    #[serde(default)]
    pub simulation: SimulationFixture,
}

impl MockCase {
    /// Load a single JSON fixture from disk.
    pub fn from_fixture(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Whether this fixture expects the worker to return an SSE response.
    pub fn is_streaming(&self) -> bool {
        self.request
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}
