//! Shared-storage path: hits in stores shared by many workers (Mooncake Store, UMBP) are
//! looked up by token at routing time instead of being indexed from events, and projected
//! onto the workers that can load from them (readers).

pub mod source;

use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;

pub use self::source::{
    FailReason, Gate, GateReason, KeyRule, KeyState, RankProbe, ReaderRule, SourceId,
    StorageBackend, StorageBackendSpec, StorageHitSource, StorageLookup, StorageSourceSpec,
};
use crate::{
    core::Worker,
    kv_events::discovery::OffloadReport,
    kv_index::{LoadPath, PrefixRequest},
};

/// The storage sources a worker reads, and where it loads from on an HBM miss.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReaderInfo {
    pub sources: Vec<SourceId>,
    pub load_path: Option<LoadPath>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Outcome of checking the reported `chain_check` against the router's key rule; a
/// failed source answers every lookup `Unreadable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChainCheck {
    Passed,
    Failed,
    NotReported,
}

/// One entry of `GET /kv_storage/sources`, served as `{"sources": [...]}`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StorageSourceStatus {
    pub id: String,
    pub model_id: String,
    /// `mooncake_admin`, `umbp_master` or `mock`.
    pub backend: &'static str,
    pub endpoints: Vec<String>,
    pub key_rule: KeyRule,
    pub chain_check: ChainCheck,
    pub breaker: BreakerState,
    pub tainted_pools: u32,
    pub last_lookup_ms: Option<f64>,
    pub readers: Vec<String>,
}

#[async_trait]
pub trait StorageHub: Send + Sync {
    /// Adds the worker as a reader of the source its `offload` report describes (or of
    /// configured sources whose readers match it), creating the source on first use.
    async fn on_worker_added(
        &self,
        worker_url: &str,
        model_id: &str,
        offload: Option<&OffloadReport>,
    );
    /// A source left without readers is removed.
    async fn on_worker_removed(&self, worker_url: &str);
    /// One lookup per source with a reader among `candidates`, made once before selection.
    async fn lookup_all(
        &self,
        req: &PrefixRequest<'_>,
        candidates: &[Arc<dyn Worker>],
    ) -> Vec<(SourceId, StorageLookup)>;
    fn reader(&self, worker_url: &str) -> ReaderInfo;
    fn status(&self) -> Vec<StorageSourceStatus>;
}
