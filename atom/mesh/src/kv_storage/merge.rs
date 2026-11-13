//! Shared-storage hits folded into the index's cache hits.

use std::sync::Arc;

use super::{GateReason, ReaderInfo, SourceId, StorageHub, StorageLookup};
use crate::{
    core::Worker,
    kv_index::{CacheHit, HitSource, LoadPath, LowerHit, PrefixMatchProvider, PrefixRequest},
};

/// Adds the lookups the planner stored in the request to `inner`'s hits.
pub struct WithStorage<P> {
    inner: P,
    hub: Arc<dyn StorageHub>,
}

impl<P> WithStorage<P> {
    pub fn new(inner: P, hub: Arc<dyn StorageHub>) -> Self {
        Self { inner, hub }
    }
}

impl<P: PrefixMatchProvider> PrefixMatchProvider for WithStorage<P> {
    fn match_workers(
        &self,
        req: &PrefixRequest<'_>,
        workers: &[Arc<dyn Worker>],
    ) -> Vec<Option<CacheHit>> {
        let hits = self.inner.match_workers(req, workers);
        if req.storage().is_empty() {
            return hits;
        }
        hits.into_iter()
            .zip(workers)
            .map(|(hit, worker)| merge(hit, &self.hub.reader(worker.url()), req.storage()))
            .collect()
    }

    fn on_routed(&self, worker: &dyn Worker, req: &PrefixRequest<'_>) {
        self.inner.on_routed(worker, req);
    }
}

/// A reader's `tokens` become `[D, H, max(H, shared), max(K, disk, Shared)]`, the larger
/// over the sources it reads. `lower` follows its load path, unknown unless that source
/// answered. A reader whose load path failed keeps a hit even without evidence, so the
/// failure is not taken for a miss.
pub fn merge(
    hit: Option<CacheHit>,
    reader: &ReaderInfo,
    storage: &[(SourceId, StorageLookup)],
) -> Option<CacheHit> {
    if reader.sources.is_empty() {
        return hit;
    }
    let lookup = |id: &SourceId| {
        storage
            .iter()
            .find(|(source, _)| source == id)
            .map(|(_, lookup)| *lookup)
    };
    let evidence = hit.is_some();
    let mut hit = hit.unwrap_or(CacheHit {
        tokens: [0; 4],
        lower: LowerHit::None,
        source: HitSource::Storage,
        exact: false,
    });
    let mut stored = false;
    for id in &reader.sources {
        if let Some(StorageLookup::Hit { shared, disk }) = lookup(id) {
            let [_, host, with_shared, with_disk] = &mut hit.tokens;
            *with_shared = (*with_shared).max(shared).max(*host);
            *with_disk = (*with_disk).max(disk).max(*with_shared);
            stored |= disk > 0;
        }
    }
    let mut failed = false;
    if let Some(LoadPath::Storage(id)) = &reader.load_path {
        let lookup = lookup(id);
        failed = matches!(
            lookup,
            Some(
                StorageLookup::Unavailable(_)
                    | StorageLookup::Skipped(GateReason::BreakerOpen | GateReason::Overloaded)
            )
        );
        hit.lower = match lookup {
            Some(StorageLookup::Hit { disk, .. }) => LowerHit::Hit {
                tokens: disk,
                from: LoadPath::Storage(id.clone()),
            },
            _ => LowerHit::Unknown(id.clone()),
        };
    }
    (evidence || stored || failed).then_some(hit)
}
