//! KV event ingestion: engine events update the router's prefix index.

pub mod discovery;
pub mod model;
pub mod rank_state;
pub mod source;
pub mod transport;
pub mod wire;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;

pub use self::{
    model::{DropReason, EngineHash, Group, GroupKind, KvEvent, Namespace, StoredBlock, Tier},
    rank_state::{Phase, Trust},
    source::{KvEventSource, OffloadLoad, PoolRole, SourceCaps, WireProfile},
};
use crate::{core::Worker, kv_index::PrefixMatchProvider};

/// The `state` label of `mesh_kv_events_sources`; `Down` means connect or discovery failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceState {
    Exact,
    Partial,
    Stale,
    Down,
}

/// One entry of `GET /kv_events/sources`, served as `{"sources": [...]}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceStatus {
    pub holder: String,
    pub model_id: String,
    pub pool: PoolRole,
    pub profile: WireProfile,
    pub endpoint: String,
    pub state: SourceState,
    pub phase: Phase,
    pub expected_seq: u64,
    pub last_batch_ts: Option<f64>,
    pub batches: u64,
    pub events: u64,
    pub rejected: u64,
    pub gaps: u64,
}

/// Driven by the worker registration and removal workflows.
#[async_trait]
pub trait KvEventMonitor: Send + Sync {
    /// Subscribe to every source of the worker; the first batch bootstraps the rank.
    async fn on_worker_added(&self, worker: &Arc<dyn Worker>, sources: Vec<KvEventSource>);
    /// Stop the worker's subscriptions and drop its holders from the index.
    async fn on_worker_removed(&self, worker_url: &str);
    fn provider(&self) -> Arc<dyn PrefixMatchProvider>;
    fn status(&self) -> Vec<SourceStatus>;
}
