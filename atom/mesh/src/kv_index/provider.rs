//! Per-worker cache hits for policies: event index, predicted source and shared storage.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::Mutex;

use super::{hash, HolderId, IndexOp, KvIndex, MatchQuery, PrefixHash, PrefixIndex, MAX_HOLDERS};
use crate::{
    core::Worker,
    kv_events::{Namespace, PoolRole, Tier, Trust},
    kv_storage::{SourceId, StorageLookup},
};

#[derive(Debug)]
pub struct PrefixRequest<'a> {
    pub model_id: &'a str,
    pub tokens: &'a [u32],
    pub ns: Namespace,
    /// Set by the planner for inputs the token ids do not identify (images, audio): the
    /// request scores no hit and records no prediction, but stays a candidate.
    pub multimodal: bool,
    hashes: Mutex<Vec<(u32, Arc<[PrefixHash]>)>>,
    storage: Vec<(SourceId, StorageLookup)>,
}

impl<'a> PrefixRequest<'a> {
    pub fn new(model_id: &'a str, tokens: &'a [u32], ns: Namespace) -> Self {
        Self {
            model_id,
            tokens,
            ns,
            multimodal: false,
            hashes: Mutex::new(Vec::new()),
            storage: Vec::new(),
        }
    }

    /// Prefix hashes of the full blocks, computed once per block size and shared by the
    /// prefill and decode selections.
    pub fn hashes(&self, block_size: u32) -> Arc<[PrefixHash]> {
        let mut cache = self.hashes.lock();
        if let Some((_, hashes)) = cache.iter().find(|(size, _)| *size == block_size) {
            return hashes.clone();
        }
        let hashes: Arc<[PrefixHash]> =
            hash::prefix_hashes(self.tokens, block_size as usize, &self.ns).into();
        cache.push((block_size, hashes.clone()));
        hashes
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionKey {
    pub model_id: Arc<str>,
    pub pool: PoolRole,
    pub block_size: u32,
}

/// One partition. A worker's event holder and its predictions take separate lanes of the
/// same index, so one walk answers both and predictions never mix into event results.
#[derive(Default)]
pub struct Partition {
    pub index: PrefixIndex,
    lanes: Mutex<u64>,
}

impl Partition {
    fn take_lane(&self) -> Option<HolderId> {
        let mut used = self.lanes.lock();
        let free = used.trailing_ones();
        (free < MAX_HOLDERS).then(|| {
            *used |= 1 << free;
            HolderId(free)
        })
    }

    /// Gives back a lane that holds no blocks.
    fn release_lane(&self, holder: HolderId) {
        *self.lanes.lock() &= !(1 << holder.0);
    }

    fn free_lane(&self, holder: HolderId) {
        self.index.remove_holder(holder);
        self.release_lane(holder);
    }
}

#[derive(Clone)]
struct Holder {
    key: PartitionKey,
    partition: Arc<Partition>,
    id: HolderId,
    trust: Trust,
}

/// The routing decisions of one worker, each held for the TTL; routing the same prefix
/// again extends it. Queries read only the lane and `until`, so they never wait for a
/// routing.
struct Prediction {
    holder: Holder,
    /// End of the last routing's TTL, in nanoseconds since the provider's epoch.
    until: AtomicU64,
    state: Mutex<Predicted>,
}

struct Predicted {
    /// Deadline of each predicted block.
    expiry: HashMap<PrefixHash, Instant>,
    next_sweep: Instant,
    /// The lane went back to its partition, so the prediction takes no more blocks.
    released: bool,
}

impl Predicted {
    fn expire(&mut self, now: Instant, ops: &mut Vec<IndexOp>) {
        self.expiry.retain(|&hash, deadline| {
            let live = *deadline > now;
            if !live {
                ops.push(IndexOp::Remove {
                    hash,
                    tier: Tier::Device,
                });
            }
            live
        });
    }
}

/// Walks one partition for every worker that holds lanes in it.
struct Walk {
    partition: Arc<Partition>,
    block_size: u32,
    /// Worker, lane, and whether the lane holds predictions.
    lanes: Vec<(usize, HolderId, bool)>,
}

/// What the index knows about one worker: event holders and their trust, and predictions.
#[derive(Clone, Default)]
struct Evidence {
    events: Option<[u32; 3]>,
    exact: bool,
    predicted: [u32; 3],
}

fn merge(into: &mut [u32; 3], hit: [u32; 3]) {
    for (into, hit) in into.iter_mut().zip(hit) {
        *into = (*into).max(hit);
    }
}

impl Evidence {
    /// Exact sources answer alone; partial ones are a lower bound that predictions extend.
    /// A Device depth raised by predictions is reported as predicted.
    fn into_hit(self) -> Option<CacheHit> {
        let (tiers, source) = match self.events {
            Some(events) if self.exact => (events, HitSource::Events),
            Some(mut events) => {
                let source = if self.predicted[0] > events[0] {
                    HitSource::Predicted
                } else {
                    HitSource::Events
                };
                merge(&mut events, self.predicted);
                (events, source)
            }
            None if self.predicted[2] > 0 => (self.predicted, HitSource::Predicted),
            None => return None,
        };
        let [device, host, disk] = tiers;
        Some(CacheHit {
            tokens: [device, host, host, disk],
            lower: LowerHit::None,
            source,
            exact: self.exact,
        })
    }
}

/// Cache hits from the event index and from the predicted source.
pub struct IndexProvider {
    ttl: Duration,
    block_size: u32,
    epoch: Instant,
    partitions: DashMap<PartitionKey, Arc<Partition>>,
    holders: DashMap<String, Vec<Holder>>,
    predictions: DashMap<String, Arc<Prediction>>,
    next_reclaim: Mutex<Instant>,
}

impl IndexProvider {
    /// Predictions last `ttl` and use `block_size` for workers without an event holder.
    pub fn new(ttl: Duration, block_size: u32) -> Self {
        let epoch = Instant::now();
        Self {
            ttl,
            block_size,
            epoch,
            partitions: DashMap::new(),
            holders: DashMap::new(),
            predictions: DashMap::new(),
            next_reclaim: Mutex::new(epoch),
        }
    }

    fn nanos(&self, at: Instant) -> u64 {
        at.saturating_duration_since(self.epoch).as_nanos() as u64
    }

    fn partition(&self, key: &PartitionKey) -> Arc<Partition> {
        self.partitions.entry(key.clone()).or_default().clone()
    }

    /// The worker's event holder in a partition, created as `Partial` on first use. None
    /// once the partition's lanes are taken.
    pub fn holder(
        &self,
        worker_url: &str,
        key: &PartitionKey,
    ) -> Option<(HolderId, Arc<Partition>)> {
        let mut holders = self.holders.entry(worker_url.to_string()).or_default();
        if let Some(holder) = holders.iter().find(|holder| holder.key == *key) {
            return Some((holder.id, holder.partition.clone()));
        }
        let partition = self.partition(key);
        let id = partition.take_lane()?;
        holders.push(Holder {
            key: key.clone(),
            partition: partition.clone(),
            id,
            trust: Trust::Partial,
        });
        Some((id, partition))
    }

    pub fn set_trust(&self, worker_url: &str, key: &PartitionKey, trust: Trust) {
        if let Some(mut holders) = self.holders.get_mut(worker_url) {
            for holder in holders.iter_mut().filter(|holder| holder.key == *key) {
                holder.trust = trust;
            }
        }
    }

    /// Drops the worker's event holders and predictions from the index.
    pub fn remove_worker(&self, worker_url: &str) {
        for holder in self
            .holders
            .remove(worker_url)
            .map(|(_, h)| h)
            .unwrap_or_default()
        {
            holder.partition.free_lane(holder.id);
        }
        if let Some((_, prediction)) = self.predictions.remove(worker_url) {
            prediction.state.lock().released = true;
            prediction.holder.partition.free_lane(prediction.holder.id);
        }
    }

    /// The worker's prediction. A new one goes to the partition of the worker's first
    /// event holder, or of its model and pool at the configured block size. None when the
    /// request has no full block there or the partition has no lane left.
    fn prediction(&self, worker: &dyn Worker, req: &PrefixRequest<'_>) -> Option<Arc<Prediction>> {
        if let Some(prediction) = self.predictions.get(worker.url()) {
            return Some(prediction.clone());
        }
        let key = self
            .holders
            .get(worker.url())
            .and_then(|holders| holders.first().map(|holder| holder.key.clone()))
            .unwrap_or_else(|| PartitionKey {
                model_id: worker.model_id().into(),
                pool: PoolRole::from(worker.worker_type()),
                block_size: self.block_size,
            });
        if req.hashes(key.block_size).is_empty() {
            return None;
        }
        match self.predictions.entry(worker.url().to_string()) {
            dashmap::Entry::Occupied(entry) => Some(entry.get().clone()),
            dashmap::Entry::Vacant(entry) => {
                let partition = self.partition(&key);
                let id = partition.take_lane()?;
                let prediction = Arc::new(Prediction {
                    holder: Holder {
                        key,
                        partition,
                        id,
                        trust: Trust::Partial,
                    },
                    until: AtomicU64::new(0),
                    state: Mutex::new(Predicted {
                        expiry: HashMap::new(),
                        next_sweep: Instant::now(),
                        released: false,
                    }),
                });
                entry.insert(prediction.clone());
                Some(prediction)
            }
        }
    }

    /// Gives back the lanes of predictions whose last TTL has ended, so an idle or removed
    /// worker keeps none. Runs at most once per eighth of the TTL.
    fn reclaim(&self, now: Instant) {
        {
            let mut next = self.next_reclaim.lock();
            if now < *next {
                return;
            }
            *next = now + self.ttl / 8;
        }
        let now = self.nanos(now);
        let idle: Vec<(String, Arc<Prediction>)> = self
            .predictions
            .iter()
            .filter(|entry| entry.until.load(Ordering::Relaxed) <= now)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        for (url, prediction) in idle {
            let mut state = prediction.state.lock();
            if state.released || prediction.until.load(Ordering::Relaxed) > now {
                continue;
            }
            let ops: Vec<IndexOp> = state
                .expiry
                .drain()
                .map(|(hash, _)| IndexOp::Remove {
                    hash,
                    tier: Tier::Device,
                })
                .collect();
            prediction
                .holder
                .partition
                .index
                .apply(prediction.holder.id, &ops);
            state.released = true;
            drop(state);
            self.predictions
                .remove_if(&url, |_, current| Arc::ptr_eq(current, &prediction));
            prediction
                .holder
                .partition
                .release_lane(prediction.holder.id);
        }
    }

    fn walk<'w>(walks: &'w mut Vec<Walk>, holder: &Holder) -> &'w mut Walk {
        let at = walks
            .iter()
            .position(|walk| Arc::ptr_eq(&walk.partition, &holder.partition))
            .unwrap_or_else(|| {
                walks.push(Walk {
                    partition: holder.partition.clone(),
                    block_size: holder.key.block_size,
                    lanes: Vec::new(),
                });
                walks.len() - 1
            });
        &mut walks[at]
    }
}

impl PrefixMatchProvider for IndexProvider {
    fn match_workers(
        &self,
        req: &PrefixRequest<'_>,
        workers: &[Arc<dyn Worker>],
    ) -> Vec<Option<CacheHit>> {
        let now = self.nanos(Instant::now());
        let mut evidence = vec![Evidence::default(); workers.len()];
        let mut walks = Vec::new();
        for (i, worker) in workers.iter().enumerate() {
            if let Some(holders) = self.holders.get(worker.url()).filter(|h| !h.is_empty()) {
                evidence[i].events = Some([0; 3]);
                evidence[i].exact = holders.iter().all(|holder| holder.trust == Trust::Exact);
                for holder in holders.iter() {
                    Self::walk(&mut walks, holder)
                        .lanes
                        .push((i, holder.id, false));
                }
            }
            // Exact sources answer alone, so their predictions are not walked.
            if evidence[i].exact {
                continue;
            }
            if let Some(prediction) = self
                .predictions
                .get(worker.url())
                .filter(|prediction| prediction.until.load(Ordering::Relaxed) > now)
            {
                Self::walk(&mut walks, &prediction.holder).lanes.push((
                    i,
                    prediction.holder.id,
                    true,
                ));
            }
        }
        for walk in walks {
            let prefix = req.hashes(walk.block_size);
            let holders: Vec<HolderId> = walk.lanes.iter().map(|(_, id, _)| *id).collect();
            for found in walk.partition.index.find_matches(&MatchQuery {
                prefix: &prefix,
                holders: &holders,
            }) {
                let tokens = found.blocks.map(|blocks| blocks * walk.block_size);
                for (i, _, predicted) in walk.lanes.iter().filter(|(_, id, _)| *id == found.holder)
                {
                    let evidence = &mut evidence[*i];
                    if *predicted {
                        merge(&mut evidence.predicted, tokens);
                    } else if let Some(events) = &mut evidence.events {
                        merge(events, tokens);
                    }
                }
            }
        }
        evidence.into_iter().map(Evidence::into_hit).collect()
    }

    /// Exact workers get predictions too, used once their trust drops. Expired blocks
    /// leave the index at most an eighth of the TTL late.
    fn on_routed(&self, worker: &dyn Worker, req: &PrefixRequest<'_>) {
        self.reclaim(Instant::now());
        // A prediction released meanwhile is replaced once.
        for _ in 0..2 {
            let Some(prediction) = self.prediction(worker, req) else {
                return;
            };
            let mut state = prediction.state.lock();
            if state.released {
                continue;
            }
            let now = Instant::now();
            let mut ops = Vec::new();
            if now >= state.next_sweep {
                state.next_sweep = now + self.ttl / 8;
                state.expire(now, &mut ops);
            }
            let deadline = now + self.ttl;
            for &hash in req.hashes(prediction.holder.key.block_size).iter() {
                if state.expiry.insert(hash, deadline).is_none() {
                    ops.push(IndexOp::Store {
                        hash,
                        tier: Tier::Device,
                    });
                }
            }
            prediction
                .holder
                .partition
                .index
                .apply(prediction.holder.id, &ops);
            prediction
                .until
                .store(self.nanos(deadline), Ordering::Relaxed);
            return;
        }
    }
}
