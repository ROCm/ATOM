//! Event source description: one `(worker, dp_rank)` event stream.

use std::{collections::HashMap, time::Duration};

use serde::{Deserialize, Serialize};

use super::model::TierTarget;
use crate::core::WorkerType;

/// Overrides for known mislabelled tier names, keyed by the exact name in the event.
pub type TierMap = HashMap<String, TierTarget>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolRole {
    Regular,
    Prefill,
    Decode,
}

impl From<&WorkerType> for PoolRole {
    fn from(worker_type: &WorkerType) -> Self {
        match worker_type {
            WorkerType::Regular => PoolRole::Regular,
            WorkerType::Prefill { .. } => PoolRole::Prefill,
            WorkerType::Decode => PoolRole::Decode,
        }
    }
}

/// Engine-specific decoding rules on top of the shared vLLM wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WireProfile {
    #[default]
    Vllm,
    Sglang,
    Atom,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportSpec {
    Zmq {
        endpoint: String,
        replay_endpoint: Option<String>,
        topic: String,
    },
    /// In-memory channel registered under `name`, for tests and mock workers.
    Mock { name: String },
}

/// When lower-tier hits count toward the engine's `cached_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffloadLoad {
    pub chunk_tokens: u32,
    pub min_load_tokens: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceCaps {
    /// ROUTER replay socket available.
    pub replay: bool,
    pub snapshot: bool,
    /// The engine publishes an un-tiered `AllBlocksCleared` as its first batch.
    pub startup_clear: bool,
    /// Idle heartbeat interval; only sources that declare one can turn `Stale`.
    pub heartbeat: Option<Duration>,
    /// None: only Device hits count toward `cached_tokens`.
    pub offload_load: Option<OffloadLoad>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvEventSource {
    /// Holder identity: the ATOMesh worker URL, `{base}@{rank}` with DP.
    pub holder: String,
    pub model_id: String,
    pub pool: PoolRole,
    pub transport: TransportSpec,
    pub profile: WireProfile,
    /// Block size from discovery. None (vLLM) accepts the size carried by each event.
    pub block_size: Option<u32>,
    pub caps: SourceCaps,
    pub tier_map: TierMap,
    pub incarnation: Option<String>,
}
