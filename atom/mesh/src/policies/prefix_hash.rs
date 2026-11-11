//! Prefix Hash routing policy for KV cache-aware load balancing
//!
//! A lightweight alternative to the full radix tree cache_aware policy.
//! Routes requests based on a hash of their prefix tokens to maximize
//! KV cache hits across workers.
//!
//! ## Algorithm
//!
//! 1. Extract first N tokens from the request (configurable prefix length)
//! 2. Hash the token sequence using xxhash for fast, stable hashing
//! 3. Use consistent hash ring to find the target worker
//! 4. If worker is overloaded (load > avg * load_factor), find least loaded
//! 5. Return least loaded worker that passes load check, or initial if all overloaded
//!
//! ## Complexity
//!
//! - Hash computation: O(prefix_length)
//! - Ring lookup: O(log n) binary search
//! - Load balance fallback: O(n) scan for least loaded
//!
//! ## Comparison with cache_aware
//!
//! | Aspect          | prefix_hash       | cache_aware (radix) |
//! |-----------------|-------------------|---------------------|
//! | Lookup          | O(log n)          | O(prefix_len)       |
//! | Memory          | O(workers × vn)   | O(total_tokens)     |
//! | Update          | O(1)              | O(prefix_len)       |
//! | Precision       | Prefix grouping   | Exact matching      |
//!
//! prefix_hash trades optimal cache utilization for predictable O(log n) performance.

use std::sync::Arc;

use super::{LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, observability::metrics::MeshMetrics};

/// Text bytes taken per configured prefix token when hashing text instead.
const TEXT_BYTES_PER_TOKEN: usize = 4;
/// Keeps text-prefix keys apart from token-prefix keys on the ring.
const TEXT_PREFIX_SEED: u64 = 0x7465_7874;

/// Configuration for the PrefixHash load balancing policy
#[derive(Debug, Clone)]
pub struct PrefixHashConfig {
    /// Number of prefix tokens to use for hashing.
    /// Longer prefixes = more precise routing but less grouping.
    /// Shorter prefixes = more requests grouped together.
    /// Default: 256 tokens (~1 paragraph of text)
    pub prefix_token_count: usize,

    /// Load factor threshold for walking the ring.
    /// If a worker's load > (total_load / num_workers) * load_factor,
    /// walk clockwise to the next worker.
    /// Default: 1.25 (125% of average load)
    pub load_factor: f64,
}

impl Default for PrefixHashConfig {
    fn default() -> Self {
        Self {
            prefix_token_count: 256,
            load_factor: 1.25,
        }
    }
}

/// Execution branch for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    NoHealthyWorkers,
    NoTokens,
    RingHit,
    LoadBalanceWalk,
    FallbackLeastLoad,
}

impl Branch {
    #[inline]
    const fn as_str(&self) -> &'static str {
        match self {
            Self::NoHealthyWorkers => "no_healthy_workers",
            Self::NoTokens => "no_tokens",
            Self::RingHit => "ring_hit",
            Self::LoadBalanceWalk => "load_balance_walk",
            Self::FallbackLeastLoad => "fallback_least_load",
        }
    }
}

/// Prefix Hash load balancing policy
///
/// Routes requests based on prefix token hash for KV cache locality.
/// Uses consistent hashing with bounded load balancing.
#[derive(Debug)]
pub struct PrefixHashPolicy {
    config: PrefixHashConfig,
}

impl PrefixHashPolicy {
    /// Create a new PrefixHashPolicy with the given configuration
    pub fn new(config: PrefixHashConfig) -> Self {
        Self { config }
    }

    /// Create a new PrefixHashPolicy with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PrefixHashConfig::default())
    }

    /// Compute hash of prefix tokens using xxhash
    #[inline]
    fn compute_prefix_hash(&self, tokens: &[u32]) -> u64 {
        let prefix_len = tokens.len().min(self.config.prefix_token_count);
        let prefix = &tokens[..prefix_len];

        let bytes: &[u8] = bytemuck::cast_slice(prefix);
        xxhash_rust::xxh3::xxh3_64(bytes)
    }

    /// Hash of the text prefix, for requests the ingress could not tokenize.
    fn compute_text_prefix_hash(&self, text: &str) -> u64 {
        let bytes = text.as_bytes();
        let end = bytes
            .len()
            .min(self.config.prefix_token_count * TEXT_BYTES_PER_TOKEN);
        xxhash_rust::xxh3::xxh3_64_with_seed(&bytes[..end], TEXT_PREFIX_SEED)
    }

    /// Check if a worker's load is acceptable
    #[inline]
    fn load_ok(&self, worker_load: usize, total_load: usize, num_workers: usize) -> bool {
        if total_load == 0 || num_workers == 0 {
            return true;
        }

        // Average load per worker (with +1 to simulate incoming request)
        let avg_load = (total_load + 1) as f64 / num_workers as f64;
        let threshold = avg_load * self.config.load_factor;

        (worker_load as f64) <= threshold
    }

    /// Find worker using consistent hash ring with load balancing
    fn find_worker_with_load_balance(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        prefix_hash: u64,
    ) -> (Option<usize>, Branch) {
        // Build healthy worker URL to index map
        let healthy_workers: Vec<(usize, &Arc<dyn Worker>)> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| w.is_healthy())
            .collect();

        if healthy_workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        // Calculate total load for load balancing
        let total_load: usize = healthy_workers.iter().map(|(_, w)| w.load()).sum();
        let num_workers = healthy_workers.len();

        // Use pre-computed ring if available
        if let Some(ref ring) = info.hash_ring {
            // Convert prefix hash to a ring key string for lookup
            let key = format!("{:016x}", prefix_hash);

            // Build URL to (index, worker) map for healthy workers
            let healthy_url_map: std::collections::HashMap<&str, (usize, &Arc<dyn Worker>)> =
                healthy_workers
                    .iter()
                    .map(|(idx, w)| (w.url(), (*idx, *w)))
                    .collect();

            // Find initial worker from ring
            if let Some(initial_url) =
                ring.find_healthy_url(&key, |url| healthy_url_map.contains_key(url))
            {
                if let Some(&(idx, worker)) = healthy_url_map.get(initial_url) {
                    let worker_load = worker.load();

                    // Check if initial worker has acceptable load
                    if self.load_ok(worker_load, total_load, num_workers) {
                        return (Some(idx), Branch::RingHit);
                    }

                    // Initial worker overloaded, find least loaded healthy worker
                    // This is a simpler approach than walking the ring
                    let least_loaded = healthy_workers
                        .iter()
                        .filter(|(_, w)| self.load_ok(w.load(), total_load, num_workers))
                        .min_by_key(|(_, w)| w.load());

                    if let Some(&(idx, _)) = least_loaded {
                        return (Some(idx), Branch::LoadBalanceWalk);
                    }

                    // All workers overloaded, use initial worker anyway
                    return (Some(idx), Branch::LoadBalanceWalk);
                }
            }
        }

        // Fallback: no ring or ring lookup failed, use least loaded worker
        let least_loaded = healthy_workers
            .iter()
            .min_by_key(|(_, w)| w.load())
            .map(|(idx, _)| *idx);

        (least_loaded, Branch::FallbackLeastLoad)
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> (Option<usize>, Branch) {
        if workers.is_empty() {
            return (None, Branch::NoHealthyWorkers);
        }

        let prefix_hash = match info.tokens {
            Some(tokens) if !tokens.is_empty() => self.compute_prefix_hash(tokens),
            // No tokens: keep prefix locality on the text and still place the request.
            _ => {
                let text_hash = self.compute_text_prefix_hash(info.request_text.unwrap_or(""));
                let (selected, _) = self.find_worker_with_load_balance(workers, info, text_hash);
                return (selected, Branch::NoTokens);
            }
        };

        // Find worker using ring with load balancing
        self.find_worker_with_load_balance(workers, info, prefix_hash)
    }
}

#[async_trait::async_trait]
impl LoadBalancingPolicy for PrefixHashPolicy {
    fn needs_tokens(&self) -> bool {
        true
    }
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let (result, branch) = self.select_worker_impl(workers, info);
        MeshMetrics::record_worker_prefix_hash_policy_branch(branch.as_str());
        result
    }

    fn name(&self) -> &'static str {
        "prefix_hash"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, HashRing, WorkerType};

    fn create_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(*url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_prefix_hash_consistent_routing() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Same tokens should always route to same worker
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (first_result, _) = policy.select_worker_impl(&workers, &info);
        let first_idx = first_result.unwrap();

        // Verify consistency
        for _ in 0..10 {
            let (result, _) = policy.select_worker_impl(&workers, &info);
            assert_eq!(result, Some(first_idx));
        }
    }

    #[test]
    fn test_different_prefixes_distribute() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        let mut distribution = std::collections::HashMap::new();

        // Different token sequences should distribute across workers
        for i in 0..100 {
            let tokens: Vec<u32> = vec![i, i + 1, i + 2, i + 3];
            let info = SelectWorkerInfo {
                tokens: Some(&tokens),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };

            let (result, _) = policy.select_worker_impl(&workers, &info);
            *distribution.entry(result.unwrap()).or_insert(0) += 1;
        }

        assert!(
            distribution.len() > 1,
            "Should distribute across workers, got {:?}",
            distribution
        );
    }

    #[test]
    fn test_shared_prefix_routes_same() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 5, // Only look at first 5 tokens
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        // Two sequences with same first 5 tokens should route to same worker
        let tokens1: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let tokens2: Vec<u32> = vec![1, 2, 3, 4, 5, 999, 888, 777];

        let info1 = SelectWorkerInfo {
            tokens: Some(&tokens1),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };
        let info2 = SelectWorkerInfo {
            tokens: Some(&tokens2),
            hash_ring: Some(ring.clone()),
            ..Default::default()
        };

        let (result1, _) = policy.select_worker_impl(&workers, &info1);
        let (result2, _) = policy.select_worker_impl(&workers, &info2);

        assert_eq!(result1, result2, "Same prefix should route to same worker");
    }

    #[test]
    fn test_no_tokens_still_places_the_request() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        let ring = Arc::new(HashRing::new(&workers));

        for tokens in [Some(&[][..]), None] {
            let info = SelectWorkerInfo {
                tokens,
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            assert_eq!(
                policy.select_worker_impl(&workers, &info),
                (Some(0), Branch::NoTokens)
            );
        }
    }

    #[test]
    fn test_text_prefix_keeps_locality_without_tokens() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            prefix_token_count: 4,
            ..Default::default()
        });
        let workers = create_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let ring = Arc::new(HashRing::new(&workers));
        let select = |text: &'static str| {
            let info = SelectWorkerInfo {
                request_text: Some(text),
                hash_ring: Some(ring.clone()),
                ..Default::default()
            };
            policy.select_worker_impl(&workers, &info)
        };

        // Only the first 16 bytes key the ring.
        let (first, branch) = select("shared system prompt, turn one");
        assert_eq!(branch, Branch::NoTokens);
        assert_eq!(select("shared system prompt, turn two").0, first);

        let spread: std::collections::HashSet<_> = (0..64)
            .map(|i| select(format!("{i:016}").leak()).0)
            .collect();
        assert!(spread.len() > 1, "text prefixes should spread: {spread:?}");
    }

    #[test]
    fn test_no_healthy_workers() {
        let policy = PrefixHashPolicy::with_defaults();
        let workers = create_workers(&["http://w1:8000"]);
        workers[0].set_healthy(false);

        let ring = Arc::new(HashRing::new(&workers));
        let tokens: Vec<u32> = vec![1, 2, 3];
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            hash_ring: Some(ring),
            ..Default::default()
        };

        let (result, branch) = policy.select_worker_impl(&workers, &info);
        assert_eq!(result, None);
        assert_eq!(branch, Branch::NoHealthyWorkers);
    }

    #[test]
    fn test_load_ok_calculation() {
        let policy = PrefixHashPolicy::new(PrefixHashConfig {
            load_factor: 1.25,
            ..Default::default()
        });

        // Total load 100, 4 workers -> avg 25, threshold 31.25
        assert!(policy.load_ok(30, 100, 4)); // 30 <= 31.25
        assert!(!policy.load_ok(35, 100, 4)); // 35 > 31.25

        // Edge cases
        assert!(policy.load_ok(0, 0, 4)); // No load = OK
        assert!(policy.load_ok(100, 0, 0)); // No workers = OK (shouldn't happen)
    }

    #[test]
    fn test_policy_name() {
        let policy = PrefixHashPolicy::with_defaults();
        assert_eq!(policy.name(), "prefix_hash");
    }
}
