//! Settings of one `(worker, dp_rank)` event stream: pool, wire profile, tier overrides and
//! when lower-tier hits load.

use std::collections::HashMap;

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

/// When lower-tier hits count toward the engine's `cached_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OffloadLoad {
    pub chunk_tokens: u32,
    pub min_load_tokens: u32,
}
