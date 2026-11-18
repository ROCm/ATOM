//! Where a worker's DP ranks publish KV events: ATOM serves them at
//! `GET /kv_event_sources`, keyed by the global DP rank each publisher binds for.

use std::{collections::BTreeMap, time::Duration};

use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct KvEventSource {
    /// The ZMQ PUB endpoint; a wildcard host stands for the worker's own.
    pub endpoint: String,
    #[serde(default)]
    pub topic: String,
    /// Tokens per hashed block.
    pub block_size: u32,
    /// The ZMQ ROUTER endpoint that replays recent batches, if the engine keeps them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_endpoint: Option<String>,
}

impl KvEventSource {
    /// The source as a router reaches it for the worker at `worker_url`.
    fn resolve(mut self, worker_url: &str) -> Self {
        let host = Url::parse(worker_url)
            .ok()
            .and_then(|url| url.host_str().map(str::to_string));
        if let Some(host) = host {
            self.endpoint = reach(self.endpoint, &host);
            self.replay_endpoint = self.replay_endpoint.map(|e| reach(e, &host));
        }
        self
    }
}

/// `endpoint` with a wildcard host replaced by `host`.
fn reach(endpoint: String, host: &str) -> String {
    let bound = endpoint.strip_prefix("tcp://");
    match bound.and_then(|b| b.rsplit_once(':')) {
        Some(("*" | "0.0.0.0" | "[::]", port)) => format!("tcp://{host}:{port}"),
        _ => endpoint,
    }
}

/// Every rank's source as the worker at `base_url` reports them.
pub async fn fetch(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<BTreeMap<u32, KvEventSource>, String> {
    let url = format!("{}/kv_event_sources", base_url.trim_end_matches('/'));
    let mut request = client.get(&url).timeout(Duration::from_secs(5));
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("{url}: {e}"))?;
    let sources = response.json().await.map_err(|e| format!("{url}: {e}"))?;
    parse(sources, base_url).map_err(|e| format!("{url}: {e}"))
}

fn parse(
    sources: BTreeMap<String, KvEventSource>,
    base_url: &str,
) -> Result<BTreeMap<u32, KvEventSource>, String> {
    let rank = |(rank, source): (String, KvEventSource)| {
        let rank = rank
            .parse()
            .map_err(|_| format!("rank {rank:?} is not a number"))?;
        if source.block_size == 0 {
            return Err(format!("rank {rank} reports blocks of 0 tokens"));
        }
        Ok((rank, source.resolve(base_url)))
    };
    sources.into_iter().map(rank).collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_wildcard_host_is_the_workers() {
        for (endpoint, worker, want) in [
            (
                "tcp://*:5557",
                "http://10.1.2.3:8000",
                "tcp://10.1.2.3:5557",
            ),
            (
                "tcp://0.0.0.0:5558",
                "http://node-7:8000",
                "tcp://node-7:5558",
            ),
            (
                "tcp://[::]:5557",
                "http://[fd00::1]:8000",
                "tcp://[fd00::1]:5557",
            ),
            (
                "tcp://127.0.0.1:5557",
                "http://10.1.2.3:8000",
                "tcp://127.0.0.1:5557",
            ),
            ("ipc:///tmp/kv_1", "http://10.1.2.3:8000", "ipc:///tmp/kv_1"),
        ] {
            let source = KvEventSource {
                endpoint: endpoint.into(),
                replay_endpoint: Some(endpoint.into()),
                ..Default::default()
            };
            let source = source.resolve(worker);
            assert_eq!(source.endpoint, want, "{endpoint}");
            assert_eq!(source.replay_endpoint.as_deref(), Some(want), "{endpoint}");
        }
    }

    #[test]
    fn ranks_are_numbers_and_blocks_hold_tokens() {
        let sources = |body| parse(serde_json::from_value(body).unwrap(), "http://h:8000");
        let two = sources(json!({
            "2": {"endpoint": "tcp://*:5559", "topic": "kv", "block_size": 16, "replay_endpoint": "tcp://*:5602"},
            "3": {"endpoint": "tcp://*:5560", "block_size": 16},
        }))
        .unwrap();
        assert_eq!(two.keys().copied().collect::<Vec<_>>(), [2, 3]);
        assert_eq!(two[&2].endpoint, "tcp://h:5559");
        assert_eq!((two[&2].topic.as_str(), two[&3].topic.as_str()), ("kv", ""));
        let replay = |rank| two[&rank].replay_endpoint.as_deref();
        assert_eq!((replay(2), replay(3)), (Some("tcp://h:5602"), None));
        assert!(sources(json!({})).unwrap().is_empty());
        let one = |rank: &str, block_size| json!({rank: {"endpoint": "tcp://*:1", "block_size": block_size}});
        assert!(sources(one("rank0", 16)).is_err());
        assert!(sources(one("0", 0)).is_err());
    }
}
