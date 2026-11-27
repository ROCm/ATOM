/*
    Cache-Aware Load Balancing Router

    This router combines two strategies to optimize both cache utilization and request distribution:

    1. Cache-Aware Routing (Approximate Tree)
    2. Load Balancing (Shortest Queue with Balance Thresholds)

    The router dynamically switches between these strategies based on load conditions:
    - Uses load balancing when the system is imbalanced
    - Uses cache-aware routing when the system is balanced

    A system is considered imbalanced if both conditions are met:
    1. (max - min) > abs_threshold
    2. max > rel_threshold * min

    Strategy Details:

    1. Cache-Aware Routing (Approximate Tree)
    -------------------------------------------
    This strategy maintains an approximate radix tree for each worker based on request history,
    eliminating the need for direct cache state queries. The tree stores raw text characters
    instead of token IDs to avoid tokenization overhead.

    Process:
    a. For each request, find the healthy candidate with the highest prefix match
    b. If match rate > cache_threshold:
    Route to the worker with highest match (likely has relevant data cached)
    c. If match rate ≤ cache_threshold:
    Route to the least-loaded candidate
    d. Background maintenance:
    Periodically evict least recently used leaf nodes to prevent memory overflow

    2. Load Balancing (Shortest Queue)
    -------------------------------------------
    This strategy tracks pending request counts per worker and routes new requests
    to the least busy worker when the system is detected to be imbalanced.

    Configuration Parameters:
    ------------------------
    1. cache_threshold: (float, 0.0 to 1.0)
    Minimum prefix match ratio to use highest-match routing.
    At or below this threshold, routes to the least-loaded candidate.

    2. balance_abs_threshold: (integer)
    Absolute difference threshold for load imbalance detection.
    System is potentially imbalanced if (max_load - min_load) > abs_threshold

    3. balance_rel_threshold: (float)
    Relative ratio threshold for load imbalance detection.
    System is potentially imbalanced if max_load > min_load * rel_threshold
    Used in conjunction with abs_threshold to determine final imbalance state.

    4. eviction_interval_secs: (integer)
    Interval between LRU eviction cycles for the approximate trees.

    5. max_tree_size: (integer)
    Maximum characters each worker may hold in a tree. Above it, that worker's
    LRU leaves are evicted during the next eviction cycle.

    With --kv-events, a request the router can tokenize goes to the worker whose
    device cache, as its KV events show, holds the largest share of its tokens,
    if that share is above cache_threshold. A worker's share also counts the
    tokens its engine will copy in from the device pools of its share group
    (ATOM's umbp_pool), at --kv-share-weight each, and with --kv-storage-weight
    above 0, those it will load from the Mooncake Store it reads, at that weight
    each. The tree places all other requests:
    those at or below the threshold, those with media, those under imbalanced
    load, and those whose candidates have no indexed events.
*/

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use rand::seq::IteratorRandom;
use rand::Rng;
use tracing::debug;

use super::{
    get_healthy_worker_indices, normalize_model_key,
    tree::{Candidates, Tree},
    utils::PeriodicTask,
    CacheAwareConfig, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo, NO_HEALTHY_WORKERS,
};
use crate::{
    core::{Worker, WorkerType},
    kv_index::PrefixMatchProvider,
};

/// Prefill and decode workers hold entirely different cache state for the same
/// model: P owns the prompt prefix, D owns the generated suffix. Sharing one
/// radix tree between them makes a P hit look like a D hit and routes on
/// fiction, so each pool gets its own tree. Mirrors SGLang Model Gateway.
fn pool_tag(worker_type: &WorkerType) -> &'static str {
    match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    }
}

fn make_tree_key(pool: &str, model: &str) -> String {
    format!("{}::{}", pool, model)
}

fn tree_key_for_worker(worker: &dyn Worker) -> String {
    make_tree_key(
        pool_tag(worker.worker_type()),
        normalize_model_key(worker.model_id()),
    )
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Tree keys already reported as missing, so the request-path warning
    /// fires once per key instead of once per request.
    missing_tree_warned: Arc<DashSet<String>>,
    kv_events: OnceLock<Arc<dyn PrefixMatchProvider>>,
    _eviction_task: Option<PeriodicTask>,
}

/// The URLs of the healthy candidates, which a tree match may credit.
struct HealthyUrls<'a> {
    workers: &'a [Arc<dyn Worker>],
    healthy_indices: &'a [usize],
}

impl Candidates for HealthyUrls<'_> {
    fn count(&self) -> usize {
        self.healthy_indices.len()
    }

    fn name(&self, index: usize) -> &str {
        self.workers[self.healthy_indices[index]].url()
    }
}

/// The least loaded of `healthy_indices`, ties broken at random so that equally loaded
/// workers do not all collapse onto the lowest index. Loads are read once: `load()` is a
/// live atomic, so reading it again could leave no worker at the minimum just computed.
fn least_loaded(workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> Option<usize> {
    let loads: Vec<(usize, usize)> = healthy_indices
        .iter()
        .map(|&idx| (idx, workers[idx].load()))
        .collect();
    let min_load = loads.iter().map(|&(_, load)| load).min();
    loads
        .iter()
        .copied()
        .filter(|&(_, load)| Some(load) == min_load)
        .map(|(idx, _)| idx)
        .choose(&mut rand::rng())
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    for tree_ref in trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "Cache eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                },
            ))
        } else {
            None
        };

        Self {
            config,
            trees,
            missing_tree_warned: Arc::new(DashSet::new()),
            kv_events: OnceLock::new(),
            _eviction_task: eviction_task,
        }
    }

    /// Places requests the router can tokenize by the KV event index from now on.
    pub(crate) fn set_kv_events(&self, provider: Arc<dyn PrefixMatchProvider>) {
        let _ = self.kv_events.set(provider);
    }

    /// What the KV tiers below a worker's device cache count for in this policy's config.
    #[cfg(test)]
    pub(crate) fn tier_weights(&self) -> super::TierWeights {
        self.config.tier_weights
    }

    /// Initialize the tree with worker URLs (used only during initial setup)
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Group workers by model
        let mut model_workers: std::collections::HashMap<String, Vec<&Arc<dyn Worker>>> =
            std::collections::HashMap::new();
        for worker in workers {
            let tree_key = tree_key_for_worker(worker.as_ref());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize tree for each model
        for (tree_key, model_workers) in model_workers {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()));
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }

    /// Add a single worker to the tree (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        let tree = self
            .trees
            .entry(tree_key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        // This URL-only API has no WorkerType, so it seeds the regular pool.
        // It must still use the namespaced key: a raw model_id would build a
        // tree that select_worker never looks up, silently disabling affinity.
        let tree_key = make_tree_key(
            pool_tag(&WorkerType::Regular),
            normalize_model_key(model_id),
        );
        let tree = self
            .trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(Tree::new()));
        tree.insert("", url);
    }

    /// Remove a worker from the tree
    pub fn remove_worker(&self, worker: &dyn Worker) {
        let tree_key = tree_key_for_worker(worker);
        if let Some(tree) = self.trees.get(&tree_key) {
            tree.remove_tenant(worker.url());
        }
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    pub fn remove_worker_by_url(&self, url: &str) {
        // Remove from all trees since we don't know which model it belongs to
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        for tree_ref in self.trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Cache eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: &Option<&str>,
        healthy_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        // Every firing of this branch bypasses prefix affinity, so the rate is
        // the signal for whether the thresholds fit the deployment's load scale.
        debug!(
            "cache_aware imbalance branch: max_load={} min_load={}",
            max_load, min_load
        );
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        // Use shortest queue when imbalanced.
        let min_load_idx = least_loaded(workers, healthy_indices)?;

        // Even in imbalanced mode, update the tree to maintain cache state
        if let Some(text) = request_text {
            // Get the tree reference without locking the entire HashMap
            // DashMap only locks the specific shard containing this key
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                let worker_url = workers[min_load_idx].url();
                // Now we can work with the tree without holding the HashMap lock
                tree.insert(text, worker_url);
            } else {
                debug!(
                    "Warning: No tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();

        Some(min_load_idx)
    }

    /// The tree's affinity branch, with the match rate of the KV event index: the share of
    /// the request's tokens a worker's device cache holds, each token its engine will then
    /// copy in from its share group counted at `--kv-share-weight`, and each it will load
    /// from its store at `--kv-storage-weight`; the highest first, then the lower load.
    /// `kv_shared` where the worker counts copies. None leaves the request to the tree: it
    /// has no tokens or has media, or no candidate's match rate is above `cache_threshold`.
    async fn select_by_kv_events(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<(Option<usize>, RouteDecision)> {
        let tokens = info
            .tokens
            .filter(|tokens| !tokens.is_empty() && !info.carries_media)?;
        let provider = self.kv_events.get()?;
        let hits = provider.match_workers(tokens, workers);
        let shared = provider.shared_tokens(tokens, workers, &hits);
        let held = shared.as_deref().unwrap_or(&hits);
        let (share_weight, store_weight) = (provider.share_weight(), provider.store_weight());
        let stored = if store_weight > 0.0 {
            Some(provider.cached_tokens(tokens, workers, held).await)
        } else {
            None
        };
        let cached = |idx: usize| stored.as_ref().map_or(held[idx], |stored| stored[idx]);
        let copied = |idx: usize| {
            held[idx]
                .zip(hits[idx])
                .map_or(0, |(s, d)| s.saturating_sub(d))
        };
        let score = |idx: usize| {
            let device = hits[idx]?;
            let copied = copied(idx);
            let loaded = cached(idx).map_or(0, |cached| cached.saturating_sub(device + copied));
            Some(
                f64::from(device)
                    + share_weight * f64::from(copied)
                    + store_weight * f64::from(loaded),
            )
        };
        let (idx, score) = healthy_indices
            .iter()
            .filter_map(|&idx| Some((idx, score(idx)?)))
            .max_by(|&(a, x), &(b, y)| {
                let lower_load = || workers[b].load().cmp(&workers[a].load());
                x.total_cmp(&y).then_with(lower_load)
            })?;
        let match_rate = score / tokens.len() as f64;
        // Below it, the tree, which learns every placement, may know where the prompt went.
        if match_rate <= f64::from(self.config.cache_threshold) {
            return None;
        }
        // The tree keeps learning, for requests without tokens.
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
        if let (Some(text), Some(tree)) = (info.request_text, tree) {
            tree.insert(text, workers[idx].url());
        }
        self.routed(info, workers[idx].as_ref());
        workers[idx].increment_processed();
        let estimate = cached(idx).map(|cached| f64::from(cached) / tokens.len() as f64);
        let branch = if copied(idx) > 0 {
            "kv_shared"
        } else {
            "kv_affinity"
        };
        let decision = RouteDecision {
            match_ratio: Some(match_rate),
            estimated_hit_ratio: estimate,
            ..RouteDecision::branch(branch)
        };
        Some((Some(idx), decision))
    }

    /// Where the tree learns a request's text, the KV event index predicts its tokens, but
    /// not those of a request with media, whose blocks an engine keys by the media (ATOM
    /// every block, by the media and the text up to the last, DeepSeek-V4.1 also by the
    /// prompt's length, which the routing text's front digest does not see; vLLM and
    /// SGLang the blocks from the media on).
    fn routed(&self, info: &SelectWorkerInfo<'_>, worker: &dyn Worker) {
        let tokens = info
            .tokens
            .filter(|tokens| !tokens.is_empty() && !info.carries_media);
        if let (Some(provider), Some(tokens)) = (self.kv_events.get(), tokens) {
            provider.on_routed(tokens, worker);
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwarePolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let request_text = info.request_text;
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        }

        // The router pre-filters candidates by model and, in PD mode, by role,
        // so the first healthy worker identifies both the model and the pool.
        // Keying the tree by pool as well keeps prefill and decode caches
        // separate -- see tree_key_for_worker.
        let model_id = tree_key_for_worker(workers[healthy_indices[0]].as_ref());

        // Get current load statistics - compute min/max in single pass without allocation
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(min, max), w| {
            let load = w.load();
            (min.min(load), max.max(load))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Check if load is imbalanced
        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            let selected = self.select_worker_min_load(
                workers,
                &request_text,
                &healthy_indices,
                &model_id,
                max_load,
                min_load,
            );
            if let Some(idx) = selected {
                self.routed(info, workers[idx].as_ref());
            }
            return (selected, RouteDecision::branch("imbalanced"));
        }

        let by_kv_events = self.select_by_kv_events(workers, info, &healthy_indices, &model_id);
        if let Some(selected) = by_kv_events.await {
            return selected;
        }

        // Use cache-aware routing when balanced
        let text = request_text.unwrap_or("");

        // Get the tree reference without locking the entire HashMap
        // DashMap only locks the specific shard containing this key
        let tree = self.trees.get(&model_id).map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Match among the healthy candidates only. Other tenants may be alive outside
            // them (another DP rank under idx2idx, or an ext-proc subset), so they stay in
            // the tree; removed workers leave it through `remove_worker_by_url`.
            let candidates = HealthyUrls {
                workers,
                healthy_indices: &healthy_indices,
            };
            let result = tree.prefix_match_in(text, &candidates);
            let match_rate = if result.input_char_count == 0 {
                0.0
            } else {
                result.matched_char_count as f32 / result.input_char_count as f32
            };
            let holder = result.candidate.map(|index| healthy_indices[index]);
            let tenant_idx = holder.filter(|_| match_rate > self.config.cache_threshold);
            let idx = tenant_idx
                .or_else(|| least_loaded(workers, &healthy_indices))
                .unwrap_or(healthy_indices[0]);

            // Update the tree with this request (use worker URL directly, no allocation)
            tree.insert(text, workers[idx].url());
            self.routed(info, workers[idx].as_ref());

            // Increment processed counter
            workers[idx].increment_processed();

            let decision = RouteDecision {
                match_ratio: Some(match_rate as f64),
                estimated_hit_ratio: (holder == Some(idx)).then_some(match_rate as f64),
                ..RouteDecision::branch(if tenant_idx.is_some() {
                    "affinity"
                } else {
                    "relocate"
                })
            };
            (Some(idx), decision)
        } else {
            // No tree for this key: cache affinity is dead and every request is
            // placed at random, which is indistinguishable from "cache_aware is
            // enabled but useless". warn (not debug) so it is visible at the
            // default log level instead of silently degrading throughput.
            // Warn once per key: this fires on the request path, and a missing
            // tree affects every request, so an unthrottled warn would flood
            // the log with one line per request.
            if self.missing_tree_warned.insert(model_id.clone()) {
                tracing::warn!(
                    "cache_aware: no tree for key '{}', falling back to random \
                     placement — pool was not seeded \
                     (init_pd_cache_aware_policies missed, or a race during \
                     worker registration)",
                    model_id
                );
            }
            // Return a random healthy worker
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            (
                Some(healthy_indices[random_idx]),
                RouteDecision::branch("no_tree"),
            )
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    fn wants_tokens(&self) -> bool {
        self.kv_events.get().is_some()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn seeded_prefill_pool_routes_a_repeated_prefix_to_one_worker() {
        // End-to-end guard for the namespaced key: init_workers and
        // select_worker must agree on `pool::model`. If they diverge the tree
        // lookup misses, selection silently falls back to random placement,
        // and affinity is lost -- which key-string comparisons alone cannot
        // catch.
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://p{i}:8000"))
                        .worker_type(WorkerType::Prefill {
                            bootstrap_port: None,
                        })
                        .model_id("m")
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        policy.init_workers(&workers);

        let text = "a shared conversation prefix that should pin one worker";
        let info = SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        };
        let first = policy.select_worker(&workers, &info).await.unwrap();
        for _ in 0..8 {
            assert_eq!(
                policy.select_worker(&workers, &info).await,
                Some(first),
                "a repeated prefix must keep landing on its cached worker"
            );
        }
    }

    #[test]
    fn prefill_and_decode_get_separate_cache_pools() {
        // The same model served by a P and a D worker must not share one radix
        // tree: P caches the prompt prefix while D caches the generated suffix,
        // so a shared tree would report a P hit for text only D has seen.
        use crate::core::{BasicWorkerBuilder, WorkerType};
        let p = BasicWorkerBuilder::new("http://p:8000")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: None,
            })
            .model_id("m")
            .build();
        let d = BasicWorkerBuilder::new("http://d:8000")
            .worker_type(WorkerType::Decode)
            .model_id("m")
            .build();
        let (kp, kd) = (tree_key_for_worker(&p), tree_key_for_worker(&d));
        assert_ne!(kp, kd, "prefill and decode must key into different trees");
        assert!(kp.starts_with("prefill::"), "got {kp}");
        assert!(kd.starts_with("decode::"), "got {kd}");
    }

    #[tokio::test]
    async fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    #[tokio::test]
    async fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
            ..Default::default()
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        let info = SelectWorkerInfo {
            request_text: Some("test"),
            ..Default::default()
        };
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, &info).await.unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    #[tokio::test]
    async fn each_exit_reports_its_branch_and_match_ratio() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            max_tree_size: 10000,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);

        let select = |text: &'static str| {
            let (policy, workers) = (&policy, &workers);
            async move {
                let info = SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                };
                let (selected, decision) = policy.select_with_decision(workers, &info).await;
                let ratios = (decision.match_ratio, decision.estimated_hit_ratio);
                (selected.unwrap(), decision.branch, ratios)
            }
        };
        let ratio = |matched: f32, input: f32| Some((matched / input) as f64);

        let (first, branch, (match_ratio, _)) = select("hello world").await;
        assert_eq!((branch, match_ratio), ("relocate", Some(0.0)));
        assert_eq!(
            select("hello world").await,
            (first, "affinity", (Some(1.0), Some(1.0)))
        );
        // A relocation estimates a hit only on the worker that holds the match.
        workers[first].increment_load();
        assert_eq!(
            select("help me").await,
            (1 - first, "relocate", (ratio(3.0, 7.0), None))
        );
        workers[1 - first].increment_load();
        workers[1 - first].increment_load();
        assert_eq!(
            select("hello, friend").await,
            (first, "relocate", (ratio(5.0, 13.0), ratio(5.0, 13.0)))
        );
        // An unhealthy tenant's match does not count; the healthy worker is credited its own.
        workers[first].set_healthy(false);
        assert_eq!(
            select("hello world").await,
            (1 - first, "relocate", (ratio(3.0, 11.0), ratio(3.0, 11.0)))
        );
        workers[first].set_healthy(true);
        for _ in 0..20 {
            workers[first].increment_load();
        }
        assert_eq!(
            select("hello world").await,
            (1 - first, "imbalanced", (None, None))
        );
    }

    /// Hits of the KV event index, by worker URL.
    #[derive(Debug)]
    struct Hits(Vec<(&'static str, u32)>);

    impl PrefixMatchProvider for Hits {
        fn match_workers(&self, _tokens: &[u32], workers: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            let hit = |worker: &Arc<dyn Worker>| {
                let found = self.0.iter().find(|(url, _)| *url == worker.url());
                found.map(|&(_, hit)| hit)
            };
            workers.iter().map(hit).collect()
        }
    }

    async fn decide(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (usize, &'static str, Option<f64>) {
        let (selected, decision) = policy.select_with_decision(workers, info).await;
        (selected.unwrap(), decision.branch, decision.match_ratio)
    }

    /// The workers routed to, each then holding the whole prompt.
    #[derive(Debug, Default)]
    struct Routed(parking_lot::Mutex<Vec<String>>);

    impl PrefixMatchProvider for Routed {
        fn match_workers(&self, tokens: &[u32], workers: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            let routed = self.0.lock();
            let held = |worker: &Arc<dyn Worker>| routed.iter().any(|url| url == worker.url());
            let held = |worker| held(worker).then_some(tokens.len() as u32);
            workers.iter().map(held).collect()
        }

        fn on_routed(&self, _tokens: &[u32], worker: &dyn Worker) {
            self.0.lock().push(worker.url().into());
        }
    }

    #[tokio::test]
    async fn every_placement_the_tree_learns_with_tokens_is_predicted() {
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            ..Default::default()
        });
        policy.init_workers(&workers);
        let routed = Arc::new(Routed::default());
        policy.set_kv_events(routed.clone());
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            request_text: Some("hello world"),
            tokens: Some(&tokens),
            ..Default::default()
        };
        let mut placed: Vec<usize> = Vec::new();
        for branch in ["relocate", "kv_affinity", "imbalanced"] {
            if branch == "imbalanced" {
                for _ in 0..20 {
                    workers[placed[1]].increment_load();
                }
            }
            let (idx, taken, _) = decide(&policy, &workers, &info).await;
            assert_eq!(taken, branch);
            placed.push(idx);
        }
        let text = SelectWorkerInfo {
            tokens: None,
            ..info.clone()
        };
        decide(&policy, &workers, &text).await;
        let media = SelectWorkerInfo {
            carries_media: true,
            ..info.clone()
        };
        decide(&policy, &workers, &media).await;
        let urls: Vec<_> = placed.iter().map(|&idx| workers[idx].url()).collect();
        assert_eq!(*routed.0.lock(), urls);
    }

    #[tokio::test]
    async fn kv_events_place_requests_with_tokens_and_leave_the_rest_to_the_tree() {
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000", "http://w3:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        let policy = |hits: Option<Vec<(&'static str, u32)>>| {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                cache_threshold: 0.5,
                eviction_interval_secs: 0,
                ..Default::default()
            });
            policy.init_workers(&workers);
            if let Some(hits) = hits {
                policy.set_kv_events(Arc::new(Hits(hits)));
            }
            policy
        };
        let tokens: Vec<u32> = (0..100).collect();
        let text = SelectWorkerInfo {
            request_text: Some("hello world"),
            ..Default::default()
        };
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..text.clone()
        };

        let tree = policy(None);
        assert!(!tree.wants_tokens());
        assert_eq!(decide(&tree, &workers, &info).await.1, "relocate");
        let events = policy(Some(vec![("http://w1:8000", 40), ("http://w2:8000", 80)]));
        assert!(events.wants_tokens());
        assert_eq!(
            decide(&events, &workers, &info).await,
            (1, "kv_affinity", Some(0.8))
        );
        // The tree learned that placement, for the request without tokens.
        assert_eq!(
            decide(&events, &workers, &text).await,
            (1, "affinity", Some(1.0))
        );
        assert_eq!(
            decide(&policy(Some(vec![])), &workers, &info).await.1,
            "relocate"
        );
        let media = SelectWorkerInfo {
            carries_media: true,
            ..info.clone()
        };
        let held = || policy(Some(vec![("http://w1:8000", 90)]));
        assert_eq!(decide(&held(), &workers, &media).await.1, "relocate");

        workers[1].increment_load();
        let tied = policy(Some(vec![("http://w1:8000", 60), ("http://w2:8000", 60)]));
        assert_eq!(
            decide(&tied, &workers, &info).await,
            (0, "kv_affinity", Some(0.6))
        );
        workers[0].increment_load();
        // At most cache_threshold of the prompt cached, the tree places it, then knows it.
        let cold = policy(Some(vec![("http://w1:8000", 50)]));
        assert_eq!(
            decide(&cold, &workers, &info).await,
            (2, "relocate", Some(0.0))
        );
        assert_eq!(
            decide(&cold, &workers, &info).await,
            (2, "affinity", Some(1.0))
        );
        for _ in 0..40 {
            workers[0].increment_load();
        }
        assert_eq!(
            decide(&held(), &workers, &info).await,
            (2, "imbalanced", None)
        );
    }

    /// Each worker's device hit and what its engine reports cached once it loads from a
    /// store (None when the store does not answer), and the weight of those loads.
    #[derive(Debug)]
    struct Stored(Vec<(u32, Option<u32>)>, f64);

    #[async_trait]
    impl PrefixMatchProvider for Stored {
        fn match_workers(&self, _tokens: &[u32], _: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            self.0.iter().map(|&(device, _)| Some(device)).collect()
        }

        async fn cached_tokens(
            &self,
            _: &[u32],
            _: &[Arc<dyn Worker>],
            device: &[Option<u32>],
        ) -> Vec<Option<u32>> {
            assert!(self.1 > 0.0, "a store asked at weight 0");
            assert_eq!(device, self.match_workers(&[], &[]));
            self.0.iter().map(|&(_, cached)| cached).collect()
        }

        fn store_weight(&self) -> f64 {
            self.1
        }
    }

    #[tokio::test]
    async fn the_kv_match_rate_weighs_what_the_engine_loads_from_a_store() {
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // w1's device cache holds 60 of the 100 tokens; w2's none, but its store 90.
        let stored = |weight| Stored(vec![(60, Some(60)), (0, Some(90))], weight);
        for (stored, want) in [
            // At weight 0 the store is not asked: the device caches alone.
            (stored(0.0), Some((0, 0.6, Some(0.6)))),
            // A loaded token weighing 1 is worth one on the device, and 0.5 half of it.
            (stored(1.0), Some((1, 0.9, Some(0.9)))),
            (stored(0.5), Some((0, 0.6, Some(0.6)))),
            // A store that does not answer adds nothing, and leaves no estimate.
            (
                Stored(vec![(60, None), (0, None)], 1.0),
                Some((0, 0.6, None)),
            ),
            // At most cache_threshold of the prompt, the tree places it.
            (Stored(vec![(30, Some(30)), (0, Some(50))], 1.0), None),
        ] {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                eviction_interval_secs: 0,
                ..Default::default()
            });
            policy.init_workers(&workers);
            policy.set_kv_events(Arc::new(stored));
            let (selected, decision) = policy.select_with_decision(&workers, &info).await;
            let got = (decision.branch == "kv_affinity").then(|| {
                let match_rate = decision.match_ratio.unwrap();
                (selected.unwrap(), match_rate, decision.estimated_hit_ratio)
            });
            assert_eq!(got, want);
        }
    }

    /// Each worker's device hit and what its device cache will hold once its engine copies
    /// in from its share group, and the weight of those copies.
    #[derive(Debug)]
    struct Shared(Vec<(u32, u32)>, f64);

    #[async_trait]
    impl PrefixMatchProvider for Shared {
        fn match_workers(&self, _tokens: &[u32], _: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            self.0.iter().map(|&(device, _)| Some(device)).collect()
        }

        fn shared_tokens(
            &self,
            _: &[u32],
            _: &[Arc<dyn Worker>],
            device: &[Option<u32>],
        ) -> Option<Vec<Option<u32>>> {
            assert_eq!(device, self.match_workers(&[], &[]));
            Some(self.0.iter().map(|&(_, shared)| Some(shared)).collect())
        }

        fn share_weight(&self) -> f64 {
            self.1
        }
    }

    #[tokio::test]
    async fn the_kv_match_rate_weighs_what_the_engine_copies_from_its_share_group() {
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // w1's device cache holds 60 of the 100 tokens; w2's 20, and its share group 80.
        let shared = |weight| Shared(vec![(60, 60), (20, 80)], weight);
        for (shared, want) in [
            // w2's 20 + 0.9 * 60 beats w1's 60, and its engine reports the copies cached.
            (shared(0.9), Some((1, "kv_shared", 0.74, Some(0.8)))),
            (shared(0.5), Some((0, "kv_affinity", 0.6, Some(0.6)))),
            // At most cache_threshold of the prompt, the tree places it.
            (Shared(vec![(10, 10), (0, 40)], 0.9), None),
        ] {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                eviction_interval_secs: 0,
                ..Default::default()
            });
            policy.init_workers(&workers);
            policy.set_kv_events(Arc::new(shared));
            let (selected, decision) = policy.select_with_decision(&workers, &info).await;
            let got = decision.branch.starts_with("kv_").then(|| {
                let match_rate = decision.match_ratio.unwrap();
                let estimate = decision.estimated_hit_ratio;
                (selected.unwrap(), decision.branch, match_rate, estimate)
            });
            assert_eq!(got, want);
        }
    }

    #[tokio::test]
    async fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await;
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test2"),
                    ..Default::default()
                },
            )
            .await;

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_healthy(false);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[tokio::test]
    async fn test_cache_aware_without_mesh() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        policy.init_workers(&workers);

        // Should work without mesh
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(idx, 0);
    }
}
