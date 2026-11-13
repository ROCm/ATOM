//! Event source discovery, and the engine-side formats it reads.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::source::{KvEventSource, PoolRole};

/// What a discovery method knows about a base worker before its ranks are registered.
#[derive(Debug, Clone, Copy)]
pub struct WorkerProbe<'a> {
    pub base_url: &'a str,
    pub model_id: &'a str,
    pub pool: PoolRole,
    pub dp_size: u32,
    pub api_key: Option<&'a str>,
    pub client: &'a reqwest::Client,
}

#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("request to {url} failed: {reason}")]
    Request { url: String, reason: String },
    #[error("bad response from {url}: {reason}")]
    BadResponse { url: String, reason: String },
}

/// Methods are tried in the configured order; the first non-empty result wins.
#[async_trait]
pub trait SourceDiscovery: Send + Sync {
    fn name(&self) -> &'static str;
    /// Sources for every rank of the base worker. `Ok(vec![])` means the method does not
    /// apply and the next one is tried.
    async fn discover(
        &self,
        worker: &WorkerProbe<'_>,
    ) -> Result<Vec<KvEventSource>, DiscoveryError>;
}

/// Body of `GET /kv_event_sources`, keyed by DP rank. ATOM also reports the same map as
/// `kv_event_sources` in `/server_info`.
pub type EngineEventSources = BTreeMap<u32, EngineEventSource>;

/// vLLM's `KVEventsConfig`, plus the fields ATOM adds (PR-2). Unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineEventSource {
    #[serde(default)]
    pub enable_kv_cache_events: bool,
    #[serde(default)]
    pub publisher: Option<String>,
    /// May bind a wildcard host; discovery substitutes the worker's host.
    pub endpoint: String,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub hwm: Option<u64>,
    #[serde(default)]
    pub buffer_steps: Option<u64>,
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub schema_version: Option<u32>,
    #[serde(default)]
    pub incarnation: Option<String>,
    #[serde(default)]
    pub block_size: Option<u32>,
    #[serde(default)]
    pub hash_algo: Option<String>,
    /// A report that does not parse or is out of bounds turns off storage reads for the
    /// rank, not its events.
    #[serde(default, deserialize_with = "bounded_offload")]
    pub offload: Option<OffloadReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadBackend {
    MooncakeStore,
    UmbpStore,
    Lmcache,
    #[serde(other)]
    Other,
}

/// ATOM's offload connector, reported with each rank (PR-2). `masters` is empty for
/// `umbp_store`; `lmcache` reports only `chunk_tokens` and `min_load_tokens`.
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

impl OffloadReport {
    /// The router builds keys per chunk and rank from these, at registration and on every
    /// request.
    fn in_bounds(&self) -> bool {
        (1..=65_536).contains(&self.chunk_tokens)
            && self.world.is_none_or(|world| (1..=1024).contains(&world))
            && self.namespace.as_ref().is_none_or(|ns| ns.len() <= 128)
            && self.masters.len() <= 64
    }
}

fn bounded_offload<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<OffloadReport>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Lenient {
        Report(OffloadReport),
        Other(serde::de::IgnoredAny),
    }
    Ok(match Option::<Lenient>::deserialize(deserializer)? {
        Some(Lenient::Report(report)) if report.in_bounds() => Some(report),
        Some(_) => {
            tracing::warn!("offload report malformed or out of bounds; storage reads are off");
            None
        }
        None => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_atom_pr2_sources() {
        let body = r#"{"0": {"enable_kv_cache_events": true, "publisher": "zmq",
            "endpoint": "tcp://10.0.0.5:5557", "replay_endpoint": "tcp://10.0.0.5:5657",
            "topic": "", "hwm": 100000, "buffer_steps": 10000,
            "encoding": "msgpack", "schema_version": 1,
            "incarnation": "3f0c9a6e5b2d4c1e8f7a6b5c4d3e2f10",
            "block_size": 64, "hash_algo": "xxh64-chain-i64le",
            "offload": {"backend": "mooncake_store", "masters": ["10.0.0.9:26051"],
                        "namespace": "atomkv1-5f0e3c9a2b7d41e6a8c3f1d2",
                        "chunk_tokens": 256, "world": 8, "min_load_tokens": 8192,
                        "chain_check": "8a1f"}},
            "1": {"endpoint": "tcp://10.0.0.5:5558",
            "offload": {"backend": "lmcache", "chunk_tokens": 256, "min_load_tokens": 8192}}}"#;
        let sources: EngineEventSources = serde_json::from_str(body).unwrap();
        let rank0 = &sources[&0];
        assert_eq!(rank0.block_size, Some(64));
        let offload = rank0.offload.as_ref().unwrap();
        assert_eq!(offload.backend, OffloadBackend::MooncakeStore);
        assert_eq!(offload.world, Some(8));
        assert_eq!(offload.chain_check.as_deref(), Some("8a1f"));
        let lmcache = sources[&1].offload.as_ref().unwrap();
        assert_eq!(lmcache.backend, OffloadBackend::Lmcache);
        assert!(lmcache.masters.is_empty() && lmcache.namespace.is_none());
    }

    #[test]
    fn parses_vllm_sources_and_unknown_backend() {
        let body = r#"{"0": {"enable_kv_cache_events": true, "publisher": "zmq",
            "endpoint": "tcp://*:5557", "replay_endpoint": null, "buffer_steps": 10000,
            "hwm": 100000, "max_queue_size": 100000, "topic": ""},
            "1": {"endpoint": "tcp://*:5558",
            "offload": {"backend": "future_store", "chunk_tokens": 1, "min_load_tokens": 1}}}"#;
        let sources: EngineEventSources = serde_json::from_str(body).unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[&0].block_size, None);
        assert_eq!(sources[&0].incarnation, None);
        assert_eq!(
            sources[&1].offload.as_ref().unwrap().backend,
            OffloadBackend::Other
        );
    }

    #[test]
    fn bad_offload_reports_only_turn_storage_off() {
        let body = r#"{"0": {"endpoint": "tcp://a:1", "offload": {"backend": "mooncake_store",
                "namespace": "x", "world": 1, "chunk_tokens": 4294967295, "min_load_tokens": 0}},
            "1": {"endpoint": "tcp://a:2", "offload": {"backend": "nixl"}},
            "2": {"endpoint": "tcp://a:3", "offload": {"backend": "mooncake_store",
                "world": 0, "chunk_tokens": 256, "min_load_tokens": 0}},
            "3": {"endpoint": "tcp://a:4", "offload": {"backend": "lmcache",
                "chunk_tokens": 256, "min_load_tokens": 0}}}"#;
        let sources: EngineEventSources = serde_json::from_str(body).unwrap();
        let offload: Vec<bool> = sources.values().map(|s| s.offload.is_some()).collect();
        assert_eq!(offload, [false, false, false, true]);
    }
}
