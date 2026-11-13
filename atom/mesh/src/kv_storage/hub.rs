//! The hub: one source per shared store, looked up once per request before selection,
//! under a gate, a deadline, a breaker and a cap on lookups in flight.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use futures::future::join_all;
use parking_lot::{Mutex, RwLock};
use tokio::time::Instant;
use tracing::warn;

use super::{
    keys::{self, AtomChunk},
    BreakerState, ChainCheck, FailReason, Gate, GateReason, RankProbe, ReaderInfo, ReaderRule,
    SourceId, StorageBackend, StorageBackendSpec, StorageHub, StorageLookup, StorageSourceSpec,
    StorageSourceStatus,
};
use crate::{
    config::KvStorageConfig,
    core::Worker,
    kv_events::discovery::{OffloadBackend, OffloadReport},
    kv_index::{LoadPath, PrefixRequest},
};

/// Builds the client of a source's backend; None leaves the source unreadable.
pub type Connect =
    Arc<dyn Fn(&StorageBackendSpec) -> Option<Arc<dyn StorageBackend>> + Send + Sync>;

pub struct Hub {
    config: KvStorageConfig,
    connect: Connect,
    sources: RwLock<Vec<Arc<Source>>>,
}

struct Source {
    spec: StorageSourceSpec,
    rule: AtomChunk,
    /// None when the router has no client for the store or computes other keys than its
    /// writer: lookups answer `Unreadable`, so the readers' lower hits stay unknown.
    backend: Option<Arc<dyn StorageBackend>>,
    chain_check: ChainCheck,
    readers: Mutex<BTreeSet<String>>,
    breaker: Mutex<Breaker>,
    inflight: AtomicU32,
    /// Microseconds the last query took; `u64::MAX` before the first.
    last_lookup_us: AtomicU64,
}

/// Releases its lookup's in-flight slot.
struct Slot<'a>(&'a AtomicU32);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Source {
    fn enter(&self, max: u32) -> Option<Slot<'_>> {
        let slot = Slot(&self.inflight);
        (self.inflight.fetch_add(1, Ordering::Relaxed) < max).then_some(slot)
    }
}

#[derive(Debug, Default)]
struct Breaker {
    failures: u32,
    open_until: Option<Instant>,
    probing: bool,
}

impl Breaker {
    /// Once open, lets one probe through per `open` period.
    fn admit(&mut self, now: Instant, open: Duration) -> bool {
        match self.open_until {
            Some(until) if now < until => false,
            Some(_) => {
                self.open_until = Some(now + open);
                self.probing = true;
                true
            }
            None => true,
        }
    }

    fn record(&mut self, ok: bool, now: Instant, threshold: u32, open: Duration) {
        if ok {
            *self = Self::default();
            return;
        }
        self.failures += 1;
        self.probing = false;
        if self.failures >= threshold {
            self.open_until = Some(now + open);
        }
    }

    fn state(&self, now: Instant) -> BreakerState {
        match self.open_until {
            None => BreakerState::Closed,
            Some(until) if self.probing || now >= until => BreakerState::HalfOpen,
            Some(_) => BreakerState::Open,
        }
    }
}

/// A master's admin service: its RPC host on the admin port.
fn admin_url(master: &str, port: u16) -> String {
    let host = master.rsplit_once(':').map_or(master, |(host, _)| host);
    format!("http://{host}:{port}")
}

impl Hub {
    pub fn new(config: KvStorageConfig, connect: Connect) -> Self {
        Self {
            config,
            connect,
            sources: RwLock::new(Vec::new()),
        }
    }

    fn open(&self) -> Duration {
        Duration::from_secs(self.config.breaker.open_secs)
    }

    async fn lookup(
        &self,
        source: &Source,
        req: &PrefixRequest<'_>,
        candidates: &[Arc<dyn Worker>],
    ) -> StorageLookup {
        if req.multimodal {
            return StorageLookup::Skipped(GateReason::Multimodal);
        }
        let min_tokens = source.spec.gate.min_tokens.max(source.rule.chunk_tokens);
        if req.tokens.len() < min_tokens as usize {
            return StorageLookup::Skipped(GateReason::TooShort);
        }
        {
            let readers = source.readers.lock();
            if !candidates
                .iter()
                .any(|worker| readers.contains(worker.url()))
            {
                return StorageLookup::Skipped(GateReason::NoReaderInCandidates);
            }
        }
        let Some(backend) = &source.backend else {
            return StorageLookup::Unavailable(FailReason::Unreadable);
        };
        let Some(_slot) = source.enter(self.config.max_inflight) else {
            return StorageLookup::Skipped(GateReason::Overloaded);
        };
        if !source.breaker.lock().admit(Instant::now(), self.open()) {
            return StorageLookup::Skipped(GateReason::BreakerOpen);
        }
        let keys = source.rule.keys(req.tokens);
        let sent = Instant::now();
        let lookup = match tokio::time::timeout(source.spec.deadline, backend.query(&keys)).await {
            Err(_) => StorageLookup::Unavailable(FailReason::Timeout),
            Ok(Err(reason)) => StorageLookup::Unavailable(reason),
            Ok(Ok(states)) if states.len() != keys.len() => {
                StorageLookup::Unavailable(FailReason::BadResponse)
            }
            Ok(Ok(states)) => source.rule.fold(&states),
        };
        let now = Instant::now();
        let took = now.duration_since(sent).as_micros() as u64;
        source.last_lookup_us.store(took, Ordering::Relaxed);
        let ok = matches!(lookup, StorageLookup::Hit { .. });
        source
            .breaker
            .lock()
            .record(ok, now, self.config.breaker.failures, self.open());
        lookup
    }
}

#[async_trait]
impl StorageHub for Hub {
    async fn on_worker_added(
        &self,
        worker_url: &str,
        model_id: &str,
        offload: Option<&OffloadReport>,
    ) {
        // A worker registered again keeps only its new role.
        self.on_worker_removed(worker_url).await;
        let Some(report) = offload else {
            return;
        };
        let (kind, backend, probe) = match report.backend {
            OffloadBackend::MooncakeStore => {
                let port = self.config.mooncake_admin_port;
                let masters = report.masters.iter().map(|m| admin_url(m, port)).collect();
                let backend = StorageBackendSpec::MooncakeAdmin { masters };
                ("mooncake", backend, self.config.probe)
            }
            // ATOM does not know the UMBP master; UMBP probes every rank.
            OffloadBackend::UmbpStore => {
                let addr = self.config.umbp_master.clone().unwrap_or_default();
                (
                    "umbp",
                    StorageBackendSpec::UmbpMaster { addr },
                    RankProbe::All,
                )
            }
            _ => return,
        };
        let (Some(namespace), Some(world)) = (&report.namespace, report.world) else {
            return;
        };
        if report.chunk_tokens == 0 || world == 0 {
            return;
        }
        let chain_check = match &report.chain_check {
            None => ChainCheck::NotReported,
            Some(reported) if *reported == keys::chain_check(report.chunk_tokens) => {
                ChainCheck::Passed
            }
            Some(reported) => {
                warn!(
                    worker_url,
                    %namespace,
                    %reported,
                    "chain_check differs from AtomChunk; the worker's lower hits stay unknown"
                );
                ChainCheck::Failed
            }
        };
        let mut sources = self.sources.write();
        let same = |source: &&Arc<Source>| {
            let backends = match (&source.spec.backend, &backend) {
                (
                    StorageBackendSpec::MooncakeAdmin { masters: known },
                    StorageBackendSpec::MooncakeAdmin { masters },
                ) => known == masters,
                (
                    StorageBackendSpec::UmbpMaster { addr: known },
                    StorageBackendSpec::UmbpMaster { addr },
                ) => known == addr,
                _ => false,
            };
            backends
                && source.spec.model_id == model_id
                && source.rule.namespace == *namespace
                && (source.chain_check == ChainCheck::Failed) == (chain_check == ChainCheck::Failed)
        };
        if let Some(source) = sources.iter().find(same) {
            source.readers.lock().insert(worker_url.to_string());
            return;
        }
        let fingerprint = namespace.rsplit('-').next().unwrap_or(namespace);
        let fingerprint: String = fingerprint.chars().take(8).collect();
        let base = format!("atom-{kind}-{fingerprint}");
        let mut id = base.clone();
        for n in 2.. {
            if !sources.iter().any(|source| *source.spec.id.0 == *id) {
                break;
            }
            id = format!("{base}-{n}");
        }
        let rule = AtomChunk {
            namespace: namespace.clone(),
            chunk_tokens: report.chunk_tokens,
            world,
            probe,
        };
        let spec = StorageSourceSpec {
            id: SourceId(id.into()),
            model_id: model_id.to_string(),
            backend,
            key_rule: rule.rule(),
            readers: ReaderRule::Discovered,
            gate: Gate {
                min_tokens: report.min_load_tokens,
            },
            deadline: Duration::from_millis(self.config.deadline_ms),
        };
        let no_master =
            matches!(&spec.backend, StorageBackendSpec::UmbpMaster { addr } if addr.is_empty());
        let readable = chain_check != ChainCheck::Failed && !no_master;
        let backend = readable.then(|| (self.connect)(&spec.backend)).flatten();
        sources.push(Arc::new(Source {
            spec,
            rule,
            backend,
            chain_check,
            readers: Mutex::new(BTreeSet::from([worker_url.to_string()])),
            breaker: Mutex::new(Breaker::default()),
            inflight: AtomicU32::new(0),
            last_lookup_us: AtomicU64::new(u64::MAX),
        }));
    }

    async fn on_worker_removed(&self, worker_url: &str) {
        self.sources.write().retain(|source| {
            let mut readers = source.readers.lock();
            readers.remove(worker_url);
            !readers.is_empty()
        });
    }

    async fn lookup_all(
        &self,
        req: &PrefixRequest<'_>,
        candidates: &[Arc<dyn Worker>],
    ) -> Vec<(SourceId, StorageLookup)> {
        let sources: Vec<Arc<Source>> = self
            .sources
            .read()
            .iter()
            .filter(|source| source.spec.model_id == req.model_id)
            .cloned()
            .collect();
        join_all(sources.iter().map(|source| async move {
            let lookup = self.lookup(source, req, candidates).await;
            (source.spec.id.clone(), lookup)
        }))
        .await
    }

    fn reader(&self, worker_url: &str) -> ReaderInfo {
        let sources: Vec<SourceId> = self
            .sources
            .read()
            .iter()
            .filter(|source| source.readers.lock().contains(worker_url))
            .map(|source| source.spec.id.clone())
            .collect();
        ReaderInfo {
            load_path: sources.first().cloned().map(LoadPath::Storage),
            sources,
        }
    }

    fn status(&self) -> Vec<StorageSourceStatus> {
        let now = Instant::now();
        let sources = self.sources.read();
        sources
            .iter()
            .map(|source| {
                let (backend, endpoints) = match &source.spec.backend {
                    StorageBackendSpec::MooncakeAdmin { masters } => {
                        ("mooncake_admin", masters.clone())
                    }
                    StorageBackendSpec::UmbpMaster { addr } => (
                        "umbp_master",
                        Vec::from_iter((!addr.is_empty()).then(|| addr.clone())),
                    ),
                    StorageBackendSpec::Mock(_) => ("mock", Vec::new()),
                };
                let took = source.last_lookup_us.load(Ordering::Relaxed);
                StorageSourceStatus {
                    id: source.spec.id.0.to_string(),
                    model_id: source.spec.model_id.clone(),
                    backend,
                    endpoints,
                    key_rule: source.spec.key_rule.clone(),
                    chain_check: source.chain_check,
                    breaker: source.breaker.lock().state(now),
                    tainted_pools: source.backend.as_ref().map_or(0, |b| b.tainted_pools()),
                    last_lookup_ms: (took != u64::MAX).then(|| took as f64 / 1000.0),
                    readers: source.readers.lock().iter().cloned().collect(),
                }
            })
            .collect()
    }
}
