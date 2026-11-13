//! Per-worker cache hits for policies: event index, predicted source and shared storage.

use std::sync::Arc;

use crate::{
    core::Worker,
    kv_events::Namespace,
    kv_storage::{SourceId, StorageLookup},
};

pub struct PrefixRequest<'a> {
    pub model_id: &'a str,
    pub tokens: &'a [u32],
    pub ns: Namespace,
    /// Set by the planner for inputs the token ids do not identify (images, audio): the
    /// request scores no hit and records no prediction, but stays a candidate.
    pub multimodal: bool,
    storage: Vec<(SourceId, StorageLookup)>,
}

impl<'a> PrefixRequest<'a> {
    pub fn new(model_id: &'a str, tokens: &'a [u32], ns: Namespace) -> Self {
        Self {
            model_id,
            tokens,
            ns,
            multimodal: false,
            storage: Vec::new(),
        }
    }

    /// One result per storage source, filled by the planner before selection and shared
    /// by the prefill and decode selections.
    pub fn storage(&self) -> &[(SourceId, StorageLookup)] {
        &self.storage
    }

    pub fn set_storage(&mut self, storage: Vec<(SourceId, StorageLookup)>) {
        self.storage = storage;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadPath {
    /// The engine's own lower tiers (e.g. LMCache).
    Local,
    Storage(SourceId),
}

/// Lower-tier hit along the worker's load path, used to estimate `cached_tokens`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerHit {
    /// No lower tier, or the source declares no `offload_load`.
    None,
    /// The storage source on the load path timed out, failed, was skipped, is open or
    /// cannot be read.
    Unknown(SourceId),
    Hit {
        tokens: u32,
        from: LoadPath,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitSource {
    Events,
    Predicted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheHit {
    /// Cumulative leading hit tokens allowing Device, Host, Shared and Disk in turn.
    pub tokens: [u32; 4],
    pub lower: LowerHit,
    pub source: HitSource,
    /// Whether the Device part comes from an `Exact` event source.
    pub exact: bool,
}

pub trait PrefixMatchProvider: Send + Sync {
    /// Same order as `workers`. None means no evidence: the worker stays a candidate but
    /// scores no hit. Does no I/O.
    fn match_workers(
        &self,
        req: &PrefixRequest<'_>,
        workers: &[Arc<dyn Worker>],
    ) -> Vec<Option<CacheHit>>;

    /// Called with the selected worker; records the predicted source.
    fn on_routed(&self, _worker: &dyn Worker, _req: &PrefixRequest<'_>) {}
}
