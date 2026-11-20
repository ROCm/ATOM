//! Wire types of `/v1/rl/*`, as SMG's RL protocol defines them.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Version of the `/v1/rl` wire contract, reported by `GET /v1/rl/workers`.
pub const RL_PROTOCOL_VERSION: u32 = 1;

/// Where a worker's capability row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RlCapabilitySource {
    /// The built-in per-engine table.
    Static,
    /// At least one `rl.*` worker label overrode the table.
    Label,
}

/// What an engine can do for RL control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RlCapabilities {
    pub source: RlCapabilitySource,
    /// Accepted `mode` values of the engine's pause route.
    pub pause_modes: Vec<String>,
    /// Weight sources the engine can refit from (`disk`, `distributed`, ...).
    pub update_from: Vec<String>,
    pub abort: bool,
    pub flush_cache: bool,
    pub sleep_wake: bool,
    /// Whether the engine reports a weight version after a refit.
    pub reports_weight_version: bool,
}

/// One row of `GET /v1/rl/workers`. DP ranks sharing a `base_url` collapse
/// into one row; `dp_ranks` counts them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlWorkerEntry {
    /// Registry id; the `{id}` of the per-worker routes.
    pub id: String,
    /// Registry URL; carries an `@<rank>` suffix for a DP-aware worker.
    pub url: String,
    pub base_url: String,
    /// `sglang`, `vllm`, `atom`, ...
    pub engine: String,
    pub engine_version: Option<String>,
    pub model_id: String,
    pub worker_type: String,
    pub connection_mode: String,
    /// Base URL of the worker's RL control routes; `null` when it has none.
    pub control_url: Option<String>,
    pub tp_size: Option<u64>,
    pub dp_size: Option<u64>,
    pub pp_size: Option<u64>,
    /// DP ranks collapsed into this row; 1 for a non-DP worker.
    pub dp_ranks: usize,
    pub role: Option<String>,
    /// `ready` or `not_ready`.
    pub health: String,
    /// The `weight_version` worker label, when set.
    pub weight_version: Option<String>,
    pub labels: HashMap<String, String>,
    pub capabilities: RlCapabilities,
}

/// Body of `GET /v1/rl/workers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlWorkersResponse {
    pub protocol_version: u32,
    pub workers: Vec<RlWorkerEntry>,
    pub total: usize,
}

/// One completed engine call, whatever its HTTP status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlCallOutcome {
    pub worker_id: String,
    pub url: String,
    /// The engine's HTTP status.
    pub status: u16,
    pub latency_ms: u64,
    /// The engine's body: parsed JSON when it was JSON, otherwise text.
    pub body: Value,
    /// Set when the body was cut at the size cap.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub body_truncated: bool,
    /// Set when reading the body failed after the status arrived; `body` then
    /// holds the bytes read before the failure, as text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_error: Option<String>,
}

/// One fan-out target that did not succeed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlFailedCall {
    pub worker_id: String,
    pub url: String,
    /// `upstream_error`, `upstream_unreachable`, `upstream_timeout` or
    /// `no_control_endpoint`.
    pub error: String,
    pub message: String,
    /// The engine's HTTP status, for `upstream_error`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// The worker's connection mode, for `no_control_endpoint`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_mode: Option<String>,
}

/// Body of a fan-out: HTTP 200 when `failed` is empty, 207 otherwise.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RlFanoutResponse {
    /// Every completed call by worker id, whatever its status.
    pub results: BTreeMap<String, RlCallOutcome>,
    pub failed: Vec<RlFailedCall>,
    pub total: usize,
    pub succeeded: usize,
}
