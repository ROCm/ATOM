//! The Mooncake Stores the workers offload to, one source per store a model's
//! workers report, read by every worker that reports it.

use std::{collections::BTreeSet, time::Duration};

use futures::future::join_all;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::{
    keys,
    mooncake::{self, KeyState},
};

const MOONCAKE_STORE: &str = "mooncake_store";
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// The `offload` a worker reports at `/kv_event_sources`, the same for every rank.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct OffloadReport {
    /// Only `mooncake_store` makes a source.
    pub backend: String,
    /// The RPC address of each pool's master.
    pub masters: Vec<String>,
    /// The key rule: `{namespace}/w{rank}of{world}/{digest}`, a digest per chunk.
    pub namespace: String,
    pub chunk_tokens: u32,
    pub world: u32,
    /// The fewest tokens the engine loads rather than recomputes.
    pub min_load_tokens: u32,
    /// The digest of tokens `0..chunk_tokens`, by the rule the engine writes with.
    pub chain_check: String,
}

impl OffloadReport {
    /// The engine's report sizes what the router hashes per chunk, the keys it builds
    /// per rank and the masters it asks, so it must stay within reason.
    fn in_bounds(&self) -> bool {
        (1..=65_536).contains(&self.chunk_tokens)
            && (1..=1024).contains(&self.world)
            && self.namespace.len() <= 128
            && self.masters.len() <= 64
    }
}

/// A store as `/kv_storage/sources` shows it.
#[derive(Debug, Clone, Serialize)]
pub struct StorageSource {
    pub id: String,
    pub model_id: String,
    #[serde(flatten)]
    pub report: OffloadReport,
    /// Whether the router computes the reported `chain_check`, and so the engine's keys.
    pub chain_check_passed: bool,
    pub readers: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StorageSourceStatus {
    #[serde(flatten)]
    pub source: StorageSource,
    pub pools: Vec<PoolStatus>,
}

/// A pool's master, asked as `/kv_storage/sources` is read.
#[derive(Debug, Clone, Serialize)]
pub struct PoolStatus {
    pub master: String,
    pub admin: String,
    /// Whether the pool holds ATOM's taint key, so that the engine loads nothing from it.
    pub tainted: Option<bool>,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct StorageSources {
    /// Each master's admin port.
    admin_port: u16,
    sources: Mutex<Vec<StorageSource>>,
}

impl StorageSources {
    pub fn new(admin_port: u16) -> Self {
        Self {
            admin_port,
            sources: Mutex::new(Vec::new()),
        }
    }

    /// Makes `worker` a reader of the store `offload` reports, and of no other.
    pub fn add(&self, worker: &str, model_id: &str, offload: &OffloadReport) {
        let mut sources = self.sources.lock();
        remove_reader(&mut sources, worker);
        if offload.backend != MOONCAKE_STORE {
            return;
        }
        if !offload.in_bounds() {
            warn!(
                worker,
                chunk_tokens = offload.chunk_tokens,
                world = offload.world,
                masters = offload.masters.len(),
                "The engine's offload report is out of bounds"
            );
            return;
        }
        let mut report = offload.clone();
        report.masters.sort();
        report.masters.dedup();
        let same = |source: &&mut StorageSource| {
            (source.model_id == model_id)
                && (source.report.namespace == report.namespace)
                && (source.report.masters == report.masters)
        };
        if let Some(source) = sources.iter_mut().find(same) {
            source.readers.insert(worker.into());
            return;
        }
        let chain_check_passed = keys::chain_check(report.chunk_tokens) == report.chain_check;
        let id = source_id(&sources, &report.namespace);
        if chain_check_passed {
            info!(%id, worker, masters = ?report.masters, "Reading a Mooncake Store");
        } else {
            warn!(%id, worker, chain_check = %report.chain_check,
                "The engine keys its chunks otherwise than the router computes");
        }
        sources.push(StorageSource {
            id,
            model_id: model_id.into(),
            report,
            chain_check_passed,
            readers: BTreeSet::from([worker.into()]),
        });
    }

    pub fn remove(&self, worker: &str) {
        remove_reader(&mut self.sources.lock(), worker);
    }

    /// Every source, its pools' masters asked meanwhile.
    pub async fn status(&self, client: &reqwest::Client) -> Vec<StorageSourceStatus> {
        let sources = self.sources.lock().clone();
        join_all(sources.into_iter().map(|source| async move {
            let pools = join_all(source.report.masters.iter().map(|m| self.probe(client, m))).await;
            StorageSourceStatus { source, pools }
        }))
        .await
    }

    async fn probe(&self, client: &reqwest::Client, master: &str) -> PoolStatus {
        let admin = mooncake::admin_url(master, self.admin_port);
        let taint = [keys::TAINT_KEY.to_string()];
        let (tainted, error) = match mooncake::query(client, &admin, &taint, PROBE_TIMEOUT).await {
            Ok(states) => match states[..] {
                [KeyState::Absent] => (Some(false), None),
                [KeyState::Memory | KeyState::SsdOnly] => (Some(true), None),
                _ => (None, Some("no answer for the taint key".into())),
            },
            Err(error) => (None, Some(error)),
        };
        PoolStatus {
            master: master.into(),
            admin,
            tainted,
            error,
        }
    }
}

/// `worker` reads no source, and a source left without readers is gone.
fn remove_reader(sources: &mut Vec<StorageSource>, worker: &str) {
    sources.retain_mut(|source| !source.readers.remove(worker) || !source.readers.is_empty());
}

/// `atom-mooncake-` and the namespace's first 8 hex digits, numbered past the first.
fn source_id(sources: &[StorageSource], namespace: &str) -> String {
    let digits = namespace.strip_prefix("atomkv1-").unwrap_or(namespace);
    let base = format!("atom-mooncake-{}", digits.get(..8).unwrap_or(digits));
    let taken = |id: &String| sources.iter().any(|source| &source.id == id);
    std::iter::once(base.clone())
        .chain((2..).map(|n| format!("{base}-{n}")))
        .find(|id| !taken(id))
        .unwrap_or(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(namespace: &str, masters: &[&str]) -> OffloadReport {
        OffloadReport {
            backend: MOONCAKE_STORE.into(),
            masters: masters.iter().map(|m| m.to_string()).collect(),
            namespace: namespace.into(),
            chunk_tokens: 64,
            world: 8,
            min_load_tokens: 8192,
            chain_check: keys::chain_check(64),
        }
    }

    fn listed(sources: &StorageSources) -> Vec<(String, bool, Vec<String>)> {
        let sources = sources.sources.lock();
        let list = |s: &StorageSource| {
            (
                s.id.clone(),
                s.chain_check_passed,
                s.readers.iter().cloned().collect(),
            )
        };
        sources.iter().map(list).collect()
    }

    #[test]
    fn a_store_is_read_by_the_workers_that_report_it() {
        let sources = StorageSources::new(9003);
        let ns = "atomkv1-0123456789abcdef01234567";
        sources.add(
            "http://a:8000@0",
            "m",
            &store(ns, &["10.0.0.1:50051", "10.0.0.3:50051"]),
        );
        sources.add(
            "http://a:8000@1",
            "m",
            &store(ns, &["10.0.0.3:50051", "10.0.0.1:50051", "10.0.0.1:50051"]),
        );
        // Another model's workers, the same store's other pools, another rule.
        sources.add(
            "http://b:8000",
            "n",
            &store(ns, &["10.0.0.1:50051", "10.0.0.3:50051"]),
        );
        sources.add("http://c:8000", "m", &store(ns, &["10.0.0.1:50051"]));
        let mut altered = store("atomkv1-fedcba98", &["10.0.0.1:50051"]);
        altered.chain_check = keys::chain_check(16);
        sources.add("http://d:8000", "m", &altered);
        let mut lmcache = store(ns, &[]);
        lmcache.backend = "lmcache".into();
        sources.add("http://e:8000", "m", &lmcache);
        let id = |n: &str| n.to_string();
        let readers = |workers: &[&str]| workers.iter().map(|w| w.to_string()).collect::<Vec<_>>();
        assert_eq!(
            listed(&sources),
            [
                (
                    id("atom-mooncake-01234567"),
                    true,
                    readers(&["http://a:8000@0", "http://a:8000@1"])
                ),
                (
                    id("atom-mooncake-01234567-2"),
                    true,
                    readers(&["http://b:8000"])
                ),
                (
                    id("atom-mooncake-01234567-3"),
                    true,
                    readers(&["http://c:8000"])
                ),
                (
                    id("atom-mooncake-fedcba98"),
                    false,
                    readers(&["http://d:8000"])
                ),
            ]
        );
        assert_eq!(
            sources.sources.lock()[0].report.masters,
            ["10.0.0.1:50051", "10.0.0.3:50051"]
        );

        // Registered again, a worker reads only what it reports now; ids stay.
        sources.add("http://a:8000@1", "m", &store(ns, &["10.0.0.1:50051"]));
        sources.add("http://d:8000", "m", &lmcache);
        for worker in ["http://b:8000", "http://c:8000"] {
            sources.remove(worker);
        }
        assert_eq!(
            listed(&sources),
            [
                (
                    id("atom-mooncake-01234567"),
                    true,
                    readers(&["http://a:8000@0"])
                ),
                (
                    id("atom-mooncake-01234567-3"),
                    true,
                    readers(&["http://a:8000@1"])
                ),
            ]
        );
        sources.remove("http://a:8000@0");
        sources.remove("http://a:8000@1");
        assert!(listed(&sources).is_empty());
    }

    #[test]
    fn a_report_out_of_bounds_is_not_read() {
        let sources = StorageSources::new(9003);
        let good = store("atomkv1-0123456789abcdef01234567", &["10.0.0.1:50051"]);
        let masters = (0..65).map(|port| format!("10.0.0.1:{port}")).collect();
        for bad in [
            OffloadReport {
                chunk_tokens: u32::MAX,
                ..good.clone()
            },
            OffloadReport {
                world: 0,
                ..good.clone()
            },
            OffloadReport {
                namespace: "a".repeat(129),
                ..good.clone()
            },
            OffloadReport {
                masters,
                ..good.clone()
            },
        ] {
            sources.add("http://a:8000", "m", &good);
            sources.add("http://a:8000", "m", &bad);
            assert!(listed(&sources).is_empty(), "{bad:?}");
        }
    }
}
