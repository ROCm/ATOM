//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use futures::{
    future,
    stream::{self, StreamExt},
};
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    config::BackendType,
    core::{ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
        backend: BackendType,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let mut groups: HashMap<String, Vec<Arc<dyn Worker>>> = HashMap::new();
        let mut non_http = Vec::new();
        for worker in workers {
            if matches!(worker.connection_mode(), ConnectionMode::Http) {
                groups
                    .entry(worker.base_url().to_string())
                    .or_default()
                    .push(worker);
            } else {
                non_http.push(worker);
            }
        }

        let mut loads = Vec::with_capacity(total_workers);
        for worker in non_http {
            loads.push(worker_load_info(&worker, -1));
        }

        let group_loads = future::join_all(groups.into_iter().map(|(base, group)| {
            let client = client.clone();
            async move {
                let fetched = fetch_base_loads(&client, &base, &group, backend).await;
                group
                    .into_iter()
                    .map(|worker| {
                        let load = fetched.get(worker.url()).copied().unwrap_or(-1);
                        worker_load_info(&worker, load)
                    })
                    .collect::<Vec<_>>()
            }
        }))
        .await;
        for group in group_loads {
            loads.extend(group);
        }

        let successful = loads.iter().filter(|info| info.load >= 0).count();
        let failed = loads.iter().filter(|info| info.load < 0).count();
        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }
}

fn worker_load_info(worker: &Arc<dyn Worker>, load: isize) -> WorkerLoadInfo {
    let worker_type = match worker.worker_type() {
        WorkerType::Regular => None,
        WorkerType::Prefill { .. } => Some("prefill".to_string()),
        WorkerType::Decode => Some("decode".to_string()),
    };
    WorkerLoadInfo {
        worker: worker.url().to_string(),
        worker_type,
        load,
    }
}

/// One request per instance. A failed fetch leaves every worker in the group
/// out of the map, which [`WorkerManager::get_all_worker_loads`] reports as `-1`.
async fn fetch_base_loads(
    client: &reqwest::Client,
    base: &str,
    group: &[Arc<dyn Worker>],
    backend: BackendType,
) -> HashMap<String, isize> {
    let api_key = group.first().and_then(|worker| worker.api_key().clone());
    if backend == BackendType::Atom {
        let body = match get_text(client, &format!("{base}/metrics"), api_key.as_deref()).await {
            Some(body) => body,
            None => return HashMap::new(),
        };
        let by_rank = parse_dp_inflight(&body);
        // The engine charges dp_inflight only when it routes among its own DP ranks, so
        // it stays 0 on a single-engine replica: a worker without a rank reads the
        // scheduler queues instead.
        let queued = parse_queued_requests(&body);
        let mut loads = HashMap::new();
        for worker in group {
            let load = match worker.dp_rank() {
                Some(rank) => by_rank.get(&rank).copied(),
                None => queued,
            };
            if let Some(load) = load {
                loads.insert(worker.url().to_string(), load);
            }
        }
        return loads;
    }

    let body = match get_text(client, &format!("{base}/get_load"), api_key.as_deref()).await {
        Some(body) => body,
        None => return HashMap::new(),
    };
    let Ok(json) = serde_json::from_str::<Value>(&body) else {
        return HashMap::new();
    };
    let Some(total) = sum_num_tokens(&json) else {
        return HashMap::new();
    };
    let mut loads = HashMap::new();
    for worker in group {
        // An instance total is not a per-rank load. DP ranks stay absent so
        // power_of_two falls back to its local request count.
        if worker.dp_rank().is_none() {
            loads.insert(worker.url().to_string(), total);
        }
    }
    loads
}

async fn get_text(client: &reqwest::Client, url: &str, api_key: Option<&str>) -> Option<String> {
    let mut req = client.get(url).timeout(REQUEST_TIMEOUT);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let response = req.send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.text().await.ok()
}

fn sum_num_tokens(json: &Value) -> Option<isize> {
    let entries = json.as_array()?;
    Some(
        entries
            .iter()
            .filter_map(|entry| entry.get("num_tokens").and_then(|value| value.as_i64()))
            .sum::<i64>() as isize,
    )
}

/// `atom:dp_inflight_requests{rank="k"}` values. Lines without a rank label are skipped.
pub(crate) fn parse_dp_inflight(body: &str) -> HashMap<usize, isize> {
    samples(body, "atom:dp_inflight_requests")
        .filter_map(|(rest, value)| {
            let rank = label_value(rest, "rank")?.parse().ok()?;
            Some((rank, value as isize))
        })
        .collect()
}

/// Requests the instance's scheduler holds across its DP ranks, running or waiting.
pub(crate) fn parse_queued_requests(body: &str) -> Option<isize> {
    let gauge = |name| samples(body, name).next().map(|(_, value)| value);
    let running = gauge("atom:requests_running")?;
    let waiting = gauge("atom:requests_waiting").unwrap_or(0.0);
    Some((running + waiting) as isize)
}

/// The label block and finite value of each `name` sample in a Prometheus text body.
fn samples<'a>(body: &'a str, name: &'a str) -> impl Iterator<Item = (&'a str, f64)> + 'a {
    body.lines().filter_map(move |line| {
        let rest = line.trim().strip_prefix(name)?;
        // `atom:dp_inflight_requests_total` and similar names are different metrics.
        if rest.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ':') {
            return None;
        }
        let value = match rest.trim_start().strip_prefix('{') {
            Some(labelled) => labelled.split_once('}')?.1,
            None => rest,
        };
        let value: f64 = value.split_whitespace().next()?.parse().ok()?;
        value.is_finite().then_some((rest, value))
    })
}

fn label_value<'a>(metric_rest: &'a str, name: &str) -> Option<&'a str> {
    let labels = metric_rest
        .trim_start()
        .strip_prefix('{')?
        .split('}')
        .next()?;
    let needle = format!("{name}=\"");
    let start = labels.find(&needle)? + needle.len();
    let value = labels[start..].split('"').next()?;
    Some(value)
}

/// Loads `power_of_two` may compare. Failures stay out so a cached `-1` cannot
/// look like a real token count.
pub(crate) fn cacheable_loads(loads: &[WorkerLoadInfo]) -> HashMap<String, isize> {
    loads
        .iter()
        .filter(|info| info.load >= 0)
        .map(|info| (info.worker.clone(), info.load))
        .collect()
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    backend: BackendType,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        backend: BackendType,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            backend,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let backend = self.backend;
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(
                worker_registry,
                policy_registry,
                client,
                backend,
                interval,
                tx,
            )
            .await;
        });

        *handle_guard = Some(handle);
    }

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        backend: BackendType,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let result =
                WorkerManager::get_all_worker_loads(&worker_registry, &client, backend).await;
            let loads = cacheable_loads(&result.loads);

            // An empty map is published too: loads from an earlier tick are stale, and
            // power_of_two falls back to its local request count without them.
            if loads.is_empty() {
                warn!("No loads fetched from workers");
            } else {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
            }
            for policy in &power_of_two_policies {
                policy.update_loads(&loads);
            }
            let _ = tx.send(loads);
        }
    }

    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder};
    use crate::protocols::worker_spec::WorkerLoadInfo;

    #[test]
    fn parse_dp_inflight_maps_ranks_and_skips_the_instance_gauge() {
        let body = r#"
# HELP atom:dp_inflight_requests Current in-flight requests charged to each data-parallel rank.
# TYPE atom:dp_inflight_requests gauge
atom:dp_inflight_requests{rank="0"} 4
atom:dp_inflight_requests{rank="1"} 9
atom:kv_cache_usage_ratio 0.5
atom:dp_inflight_requests 13
"#;
        let loads = parse_dp_inflight(body);
        assert_eq!(loads.get(&0), Some(&4));
        assert_eq!(loads.get(&1), Some(&9));
        assert_eq!(loads.len(), 2);
    }

    #[test]
    fn parse_queued_requests_needs_the_running_gauge() {
        let body = "atom:requests_running 2.0\n\
                    atom:requests_waiting_total 7\n\
                    atom:requests_waiting 3.0\n";
        assert_eq!(parse_queued_requests(body), Some(5));
        assert_eq!(parse_queued_requests("atom:requests_waiting 3\n"), None);
    }

    /// Serves `body` at `/metrics` and returns the server's base URL.
    async fn metrics_server(body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app =
            axum::Router::new().route("/metrics", axum::routing::get(move || async move { body }));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    #[tokio::test]
    async fn atom_loads_read_each_rank_of_an_instance_and_the_queues_of_one_engine() {
        let dp = metrics_server(
            "atom:dp_inflight_requests{rank=\"0\"} 4\n\
             atom:dp_inflight_requests{rank=\"1\"} 9\n\
             atom:requests_running 11\n",
        )
        .await;
        // A single-engine replica never charges dp_inflight.
        let single = metrics_server(
            "atom:dp_inflight_requests{rank=\"0\"} 0\n\
             atom:requests_running 2.0\n\
             atom:requests_waiting 3.0\n",
        )
        .await;
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(DPAwareWorkerBuilder::new(&dp, 0, 2).build()),
            Arc::new(DPAwareWorkerBuilder::new(&dp, 1, 2).build()),
            Arc::new(BasicWorkerBuilder::new(&single).build()),
            Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:1").build()),
        ];
        let registry = WorkerRegistry::new();
        for worker in &workers {
            registry.register(Arc::clone(worker));
        }

        let result = WorkerManager::get_all_worker_loads(
            &registry,
            &reqwest::Client::new(),
            BackendType::Atom,
        )
        .await;
        let loads: HashMap<&str, isize> = result
            .loads
            .iter()
            .map(|info| (info.worker.as_str(), info.load))
            .collect();
        let load = |index: usize| loads[workers[index].url()];
        assert_eq!((load(0), load(1), load(2), load(3)), (4, 9, 5, -1));
        assert_eq!((result.successful, result.failed), (3, 1));
    }

    #[test]
    fn cacheable_loads_drops_failures() {
        let loads = vec![
            WorkerLoadInfo {
                worker: "http://ok".to_string(),
                worker_type: None,
                load: 3,
            },
            WorkerLoadInfo {
                worker: "http://down".to_string(),
                worker_type: None,
                load: -1,
            },
        ];
        let cached = cacheable_loads(&loads);
        assert_eq!(cached.get("http://ok"), Some(&3));
        assert!(!cached.contains_key("http://down"));
    }
}
