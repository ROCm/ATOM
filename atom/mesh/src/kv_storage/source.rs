//! A shared-storage source: a key rule for the writer plus a backend for the store.

use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// `atom-{backend}-{ns8}`, generated from the readers' offload report.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceId(pub Arc<str>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyState {
    Memory,
    SsdOnly,
    Absent,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailReason {
    Timeout,
    Transport,
    BadResponse,
    /// No client for the store, or its writer's keys differ from the router's.
    Unreadable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReason {
    TooShort,
    Multimodal,
    NoReaderInCandidates,
    BreakerOpen,
    Overloaded,
}

/// Never read `Skipped` or `Unavailable` as zero hits: the shared part is unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageLookup {
    /// Leading hit tokens from token 0: `shared` in the store's memory tier, `disk` with
    /// its SSD tier as well.
    Hit {
        shared: u32,
        disk: u32,
    },
    Skipped(GateReason),
    Unavailable(FailReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RankProbe {
    /// Rank 0 only: Mooncake evicts all ranks of a chunk together.
    #[default]
    First,
    All,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SglangKeyLayout(pub String);

/// Computes the writer's object keys from tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum KeyRule {
    /// ATOM's `{namespace}/w{rank}of{world}/{digest}` blake2b chunk chain.
    AtomChunk {
        namespace: String,
        chunk_tokens: u32,
        world: u32,
        #[serde(default)]
        probe: RankProbe,
    },
    SglangPage {
        page_size: u32,
        layout: SglangKeyLayout,
    },
}

/// Workers that can load from the source. Configured as `"discovered"` or a list of
/// worker URLs with `*` wildcards.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReaderRule {
    #[default]
    Discovered,
    Workers(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gate {
    /// Shorter requests are skipped; defaults to the readers' `min_load_tokens`.
    pub min_tokens: u32,
}

/// Sends a batch of keys to the store. Multiple pools are merged inside: memory in any
/// pool wins, then SSD in any pool; keys no pool answered are `Unknown`.
#[async_trait]
pub trait StorageBackend: Send + Sync + fmt::Debug {
    /// One state per key, in key order.
    async fn query(&self, keys: &[String]) -> Result<Vec<KeyState>, FailReason>;
}

#[derive(Debug, Clone)]
pub enum StorageBackendSpec {
    /// Read-only `/batch_query_keys` of each pool's master admin service.
    MooncakeAdmin {
        masters: Vec<String>,
    },
    Mock(Arc<dyn StorageBackend>),
}

#[derive(Debug, Clone)]
pub struct StorageSourceSpec {
    pub id: SourceId,
    pub model_id: String,
    pub backend: StorageBackendSpec,
    pub key_rule: KeyRule,
    pub readers: ReaderRule,
    pub gate: Gate,
    pub deadline: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadBackend {
    MooncakeStore,
    Lmcache,
    #[serde(other)]
    Other,
}

/// ATOM's offload connector, reported with each rank by `/kv_event_sources`
/// (ROCm/ATOM#3277). `lmcache` reports only `chunk_tokens` and `min_load_tokens`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OffloadReport {
    pub backend: OffloadBackend,
    #[serde(default)]
    pub masters: Vec<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    pub chunk_tokens: u32,
    #[serde(default)]
    pub world: Option<u32>,
    pub min_load_tokens: u32,
    /// Hex digest of the key chain over tokens `0..chunk_tokens`, for the key-rule self check.
    #[serde(default)]
    pub chain_check: Option<String>,
}
