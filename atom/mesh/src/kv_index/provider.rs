//! Per-worker cache hits for policies: event index, predicted source and shared storage.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
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
    hashes: Mutex<Vec<(u32, Arc<[PrefixHash]>)>>,
    storage: Vec<(SourceId, StorageLookup)>,
}

impl<'a> PrefixRequest<'a> {
    pub fn new(model_id: &'a str, tokens: &'a [u32], ns: Namespace) -> Self {
        Self {
            model_id,
            tokens,
            ns,
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
    /// The storage source on the load path timed out, failed, was skipped or is open.
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
    /// Only shared storage holds the prefix.
    Storage,
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

    fn free_lane(&self, holder: HolderId) {
        self.index.remove_holder(holder);
        *self.lanes.lock() &= !(1 << holder.0);
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
/// again extends it.
struct Prediction {
    holder: Holder,
    expiry: HashMap<PrefixHash, Instant>,
    queue: VecDeque<(Instant, Arc<[PrefixHash]>)>,
}

impl Prediction {
    fn expire(&mut self, now: Instant) {
        let mut ops = Vec::new();
        while let Some((deadline, hashes)) = self.queue.front() {
            if *deadline > now {
                break;
            }
            for hash in hashes.iter() {
                if self
                    .expiry
                    .get(hash)
                    .is_some_and(|deadline| *deadline <= now)
                {
                    self.expiry.remove(hash);
                    ops.push(IndexOp::Remove {
                        hash: *hash,
                        tier: Tier::Device,
                    });
                }
            }
            self.queue.pop_front();
        }
        self.holder.partition.index.apply(self.holder.id, &ops);
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
    fn into_hit(self) -> Option<CacheHit> {
        let (tiers, source) = match self.events {
            Some(events) if self.exact => (events, HitSource::Events),
            Some(mut events) => {
                merge(&mut events, self.predicted);
                (events, HitSource::Events)
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
    partitions: DashMap<PartitionKey, Arc<Partition>>,
    holders: DashMap<String, Vec<Holder>>,
    predictions: DashMap<String, Prediction>,
}

impl IndexProvider {
    /// Predictions last `ttl` and use `block_size` for workers without an event holder.
    pub fn new(ttl: Duration, block_size: u32) -> Self {
        Self {
            ttl,
            block_size,
            partitions: DashMap::new(),
            holders: DashMap::new(),
            predictions: DashMap::new(),
        }
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
            prediction.holder.partition.free_lane(prediction.holder.id);
        }
    }

    fn exact(&self, worker_url: &str) -> bool {
        self.holders.get(worker_url).is_some_and(|holders| {
            !holders.is_empty() && holders.iter().all(|holder| holder.trust == Trust::Exact)
        })
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
        let now = Instant::now();
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
            if let Some(mut prediction) = self.predictions.get_mut(worker.url()) {
                prediction.expire(now);
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

    fn on_routed(&self, worker: &dyn Worker, req: &PrefixRequest<'_>) {
        if self.exact(worker.url()) {
            return;
        }
        let now = Instant::now();
        let mut prediction = match self.predictions.entry(worker.url().to_string()) {
            dashmap::Entry::Occupied(entry) => entry.into_ref(),
            dashmap::Entry::Vacant(entry) => {
                let key = self
                    .holders
                    .get(worker.url())
                    .and_then(|holders| holders.first().map(|holder| holder.key.clone()))
                    .unwrap_or_else(|| PartitionKey {
                        model_id: worker.model_id().into(),
                        pool: PoolRole::from(worker.worker_type()),
                        block_size: self.block_size,
                    });
                let partition = self.partition(&key);
                let Some(id) = partition.take_lane() else {
                    return;
                };
                entry.insert(Prediction {
                    holder: Holder {
                        key,
                        partition,
                        id,
                        trust: Trust::Partial,
                    },
                    expiry: HashMap::new(),
                    queue: VecDeque::new(),
                })
            }
        };
        prediction.expire(now);
        let hashes = req.hashes(prediction.holder.key.block_size);
        let deadline = now + self.ttl;
        let ops: Vec<IndexOp> = hashes
            .iter()
            .filter(|hash| prediction.expiry.insert(**hash, deadline).is_none())
            .map(|&hash| IndexOp::Store {
                hash,
                tier: Tier::Device,
            })
            .collect();
        prediction
            .holder
            .partition
            .index
            .apply(prediction.holder.id, &ops);
        prediction.queue.push_back((deadline, hashes));
    }
}
