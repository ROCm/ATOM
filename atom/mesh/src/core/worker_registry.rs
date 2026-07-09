//! Worker Registry for multi-router support
//!
//! Provides a centralized registry for workers organized as a layered pool:
//! `model_id -> framework -> {regular, pd-prefill, pd-decode}`.
//!
//! # Performance Optimizations
//! Each pool bucket is an immutable Arc slice updated copy-on-write, so reads
//! are lock-free. This is critical for high-concurrency scenarios where many
//! requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed hash ring per model for O(log n) consistent hashing.
//! The ring is rebuilt only when workers are added/removed, not per-request.
//! Uses virtual nodes (150 per worker) for even distribution and blake3 for stable hashing.

use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use crate::{
    core::{
        circuit_breaker::CircuitState,
        worker::{Framework, HealthChecker, PoolRole, WorkerType},
        ConnectionMode, Worker,
    },
    observability::metrics::MeshMetrics,
};

/// Number of virtual nodes per physical worker for even distribution.
/// 150 is a common choice that provides good balance between memory and distribution.
const VIRTUAL_NODES_PER_WORKER: usize = 150;

/// Consistent hash ring for O(log n) worker selection.
///
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index). This provides:
/// - Even key distribution across workers
/// - Minimal key redistribution when workers are added/removed (~1/N keys move)
/// - O(log n) lookup via binary search
///
/// Uses blake3 for stable, fast hashing that's consistent across Rust versions.
#[derive(Debug, Clone)]
pub struct HashRing {
    /// Sorted list of (ring_position, worker_url)
    /// Multiple entries per worker (virtual nodes) for even distribution.
    /// Uses Arc<str> to share URL across all virtual nodes (150 refs vs 150 copies).
    entries: Arc<[(u64, Arc<str>)]>,
}

impl HashRing {
    /// Build a hash ring from a list of workers.
    /// Creates VIRTUAL_NODES_PER_WORKER entries per worker for even distribution.
    pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
        let mut entries: Vec<(u64, Arc<str>)> =
            Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

        for worker in workers {
            // Create Arc<str> once per worker, share across all virtual nodes
            let url: Arc<str> = Arc::from(worker.url());
            let url_bytes = url.as_bytes();

            // Create multiple virtual nodes per worker
            for vnode in 0..VIRTUAL_NODES_PER_WORKER {
                let mut hasher = blake3::Hasher::new();
                hasher.update(url_bytes);
                hasher.update(b"#");
                hasher.update(&(vnode as u64).to_le_bytes());
                let hash = hasher.finalize();
                let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
                entries.push((pos, Arc::clone(&url)));
            }
        }

        // Sort by ring position for binary search
        entries.sort_unstable_by_key(|(pos, _)| *pos);

        Self {
            entries: Arc::from(entries.into_boxed_slice()),
        }
    }

    /// Hash a string to a ring position using blake3 (stable across versions).
    #[inline]
    fn hash_position(s: &str) -> u64 {
        let hash = blake3::hash(s.as_bytes());
        // Take first 8 bytes as u64
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    /// Find worker URL for a key using consistent hashing.
    /// Returns the first healthy worker URL at or after the key's position (clockwise).
    ///
    /// - `key`: The routing key to hash
    /// - `is_healthy`: Function to check if a worker URL is healthy
    pub fn find_healthy_url<F>(&self, key: &str, is_healthy: F) -> Option<&str>
    where
        F: Fn(&str) -> bool,
    {
        if self.entries.is_empty() {
            return None;
        }

        let key_pos = Self::hash_position(key);

        // Binary search to find first entry at or after key_pos
        let start = self.entries.partition_point(|(pos, _)| *pos < key_pos);

        // Walk clockwise from start, wrapping around
        // Track visited URLs to avoid checking same worker multiple times (virtual nodes)
        let mut checked_urls =
            std::collections::HashSet::with_capacity(self.worker_count().min(16));

        for i in 0..self.entries.len() {
            let (_, url) = &self.entries[(start + i) % self.entries.len()];
            let url_str: &str = url;

            // Skip if we already checked this worker (from another virtual node)
            if !checked_urls.insert(url_str) {
                continue;
            }

            if is_healthy(url_str) {
                return Some(url_str);
            }
        }

        None
    }

    /// Check if the ring is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the number of entries in the ring (including virtual nodes)
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get the number of unique workers in the ring
    pub fn worker_count(&self) -> usize {
        self.entries.len() / VIRTUAL_NODES_PER_WORKER.max(1)
    }
}

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Empty immutable worker slice (shared, avoids per-bucket allocation).
fn empty_worker_slice() -> Arc<[Arc<dyn Worker>]> {
    Arc::from(Vec::new().into_boxed_slice())
}

/// The three role buckets for a single (model, framework) pair.
///
/// `regular` holds standard (non-disaggregated) workers; `prefill` and `decode`
/// hold the P and D nodes of the PD pool, kept separate so each can be selected
/// independently. Each bucket is an immutable Arc slice updated copy-on-write,
/// preserving the lock-free read path.
#[derive(Debug, Clone)]
pub struct FrameworkBuckets {
    regular: Arc<[Arc<dyn Worker>]>,
    prefill: Arc<[Arc<dyn Worker>]>,
    decode: Arc<[Arc<dyn Worker>]>,
}

impl Default for FrameworkBuckets {
    fn default() -> Self {
        Self {
            regular: empty_worker_slice(),
            prefill: empty_worker_slice(),
            decode: empty_worker_slice(),
        }
    }
}

impl FrameworkBuckets {
    /// Immutable view of the bucket for a given role.
    pub fn bucket(&self, role: PoolRole) -> Arc<[Arc<dyn Worker>]> {
        match role {
            PoolRole::Regular => Arc::clone(&self.regular),
            PoolRole::PrefillPD => Arc::clone(&self.prefill),
            PoolRole::DecodePD => Arc::clone(&self.decode),
        }
    }

    /// All workers across the three buckets.
    pub fn all(&self) -> Vec<Arc<dyn Worker>> {
        let mut out =
            Vec::with_capacity(self.regular.len() + self.prefill.len() + self.decode.len());
        out.extend(self.regular.iter().cloned());
        out.extend(self.prefill.iter().cloned());
        out.extend(self.decode.iter().cloned());
        out
    }

    fn is_empty(&self) -> bool {
        self.regular.is_empty() && self.prefill.is_empty() && self.decode.is_empty()
    }

    fn slot_mut(&mut self, role: PoolRole) -> &mut Arc<[Arc<dyn Worker>]> {
        match role {
            PoolRole::Regular => &mut self.regular,
            PoolRole::PrefillPD => &mut self.prefill,
            PoolRole::DecodePD => &mut self.decode,
        }
    }

    /// Return a new buckets snapshot with `worker` added to `role`'s bucket
    /// (replacing any existing entry with the same URL). Copy-on-write.
    fn with_added(&self, role: PoolRole, worker: Arc<dyn Worker>) -> Self {
        let mut next = self.clone();
        let slot = next.slot_mut(role);
        let mut v: Vec<Arc<dyn Worker>> = slot
            .iter()
            .filter(|w| w.url() != worker.url())
            .cloned()
            .collect();
        v.push(worker);
        *slot = Arc::from(v.into_boxed_slice());
        next
    }

    /// Return a new buckets snapshot with the worker at `url` removed from
    /// `role`'s bucket. Copy-on-write.
    fn with_removed(&self, role: PoolRole, url: &str) -> Self {
        let mut next = self.clone();
        let slot = next.slot_mut(role);
        let v: Vec<Arc<dyn Worker>> = slot.iter().filter(|w| w.url() != url).cloned().collect();
        *slot = Arc::from(v.into_boxed_slice());
        next
    }
}

/// All workers for one model, organized by framework then role.
///
/// Layer hierarchy: `model_id -> framework -> {regular, pd-prefill, pd-decode}`.
#[derive(Debug, Default)]
pub struct ModelPool {
    /// framework -> role buckets (copy-on-write snapshots for lock-free reads)
    buckets: DashMap<Framework, FrameworkBuckets>,
}

impl ModelPool {
    /// Buckets for a specific framework, if any workers are registered under it.
    pub fn framework_buckets(&self, framework: &Framework) -> Option<FrameworkBuckets> {
        self.buckets.get(framework).map(|b| b.clone())
    }

    /// Frameworks that currently have at least one worker in this model pool.
    pub fn frameworks(&self) -> Vec<Framework> {
        self.buckets
            .iter()
            .filter(|e| !e.value().is_empty())
            .map(|e| *e.key())
            .collect()
    }

    /// All workers for this model across every framework and role.
    pub fn all(&self) -> Vec<Arc<dyn Worker>> {
        self.buckets.iter().flat_map(|e| e.value().all()).collect()
    }

    fn is_empty(&self) -> bool {
        self.buckets.iter().all(|e| e.value().is_empty())
    }
}

/// Worker registry with a layered pool structure.
///
/// Workers are organized as `model_id -> framework -> {regular, pd-prefill,
/// pd-decode}` in `pools`. The other maps are secondary indexes over the same
/// workers for lookup by ID / URL / connection mode and for consistent hashing.
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Layered pools: model_id -> framework -> role buckets.
    /// Replaces the former flat model_index + type_workers indexes.
    pools: Arc<DashMap<String, Arc<ModelPool>>>,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(DashMap::new()),
            pools: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
        }
    }

    /// Rebuild the hash ring for a model based on the current workers in its pool
    fn rebuild_hash_ring(&self, model_id: &str) {
        let workers = self.get_by_model(model_id);
        if workers.is_empty() {
            // No workers for this model, remove the ring
            self.hash_rings.remove(model_id);
        } else {
            let ring = HashRing::new(&workers);
            self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
        }
    }

    /// Get the hash ring for a model (O(1) lookup)
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    /// Add a worker into its `model -> framework -> role` bucket (copy-on-write).
    fn pool_add(&self, worker: &Arc<dyn Worker>) {
        let model_id = worker.model_id().to_string();
        let framework = *worker.framework();
        let role = worker.worker_type().pool_role();

        let pool = self
            .pools
            .entry(model_id)
            .or_insert_with(|| Arc::new(ModelPool::default()))
            .clone();

        let mut bucket = pool.buckets.entry(framework).or_default();
        *bucket = bucket.with_added(role, worker.clone());
    }

    /// Remove a worker (identified by url) from its `model -> framework -> role`
    /// bucket (copy-on-write). Cleans up empty pools.
    fn pool_remove(&self, model_id: &str, framework: Framework, role: PoolRole, url: &str) {
        if let Some(pool) = self.pools.get(model_id).map(|p| p.clone()) {
            if let Some(mut bucket) = pool.buckets.get_mut(&framework) {
                *bucket = bucket.with_removed(role, url);
            }
            pool.buckets.retain(|_, b| !b.is_empty());
            if pool.is_empty() {
                self.pools.remove(model_id);
            }
        }
    }

    /// Register a new worker
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let worker_id = if let Some(existing_id) = self.url_to_id.get(worker.url()) {
            // Worker with this URL already exists, update it
            existing_id.clone()
        } else {
            WorkerId::new()
        };

        // If a worker with this URL already exists, drop its old pool entry first
        // (its model/framework/role may have changed on update).
        if let Some(old) = self.workers.get(&worker_id).map(|w| w.clone()) {
            let old_model = old.model_id().to_string();
            self.pool_remove(
                &old_model,
                *old.framework(),
                old.worker_type().pool_role(),
                old.url(),
            );
            if let Some(mut conn_workers) = self.connection_workers.get_mut(old.connection_mode()) {
                conn_workers.retain(|id| id != &worker_id);
            }
            if old_model != worker.model_id() {
                self.rebuild_hash_ring(&old_model);
            }
        }

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Insert into the layered pool (model -> framework -> role), copy-on-write.
        let model_id = worker.model_id().to_string();
        self.pool_add(&worker);

        // Rebuild hash ring for this model
        self.rebuild_hash_ring(&model_id);

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(worker.connection_mode().clone())
            .or_default()
            .push(worker_id.clone());

        worker_id
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    /// Uses atomic entry API to avoid race conditions between check and insert.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Best-effort lookup of the URL for a given worker ID.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker)) = self.workers.remove(worker_id) {
            // Remove from URL mapping
            self.url_to_id.remove(worker.url());

            // Remove from the layered pool (model -> framework -> role), copy-on-write.
            let model_id = worker.model_id().to_string();
            self.pool_remove(
                &model_id,
                *worker.framework(),
                worker.worker_type().pool_role(),
                worker.url(),
            );

            // Rebuild hash ring for this model
            self.rebuild_hash_ring(&model_id);

            // Remove from connection mode index
            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            worker.set_healthy(false);
            MeshMetrics::remove_worker_metrics(worker.url());

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        if let Some((_, worker_id)) = self.url_to_id.remove(url) {
            self.remove(&worker_id)
        } else {
            None
        }
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.url_to_id.get(url).and_then(|id| self.get(&id))
    }

    /// Get the layered pool for a model, if it exists.
    pub fn get_model_pool(&self, model_id: &str) -> Option<Arc<ModelPool>> {
        self.pools.get(model_id).map(|p| p.clone())
    }

    /// Frameworks that currently have workers registered for a model.
    pub fn get_frameworks(&self, model_id: &str) -> Vec<Framework> {
        self.pools
            .get(model_id)
            .map(|p| p.frameworks())
            .unwrap_or_default()
    }

    /// Direct access to a single pool bucket: (model, framework, role).
    /// Returns an immutable Arc slice — lock-free, just a refcount bump.
    pub fn get_pool(
        &self,
        model_id: &str,
        framework: &Framework,
        role: PoolRole,
    ) -> Arc<[Arc<dyn Worker>]> {
        self.pools
            .get(model_id)
            .and_then(|p| p.framework_buckets(framework))
            .map(|b| b.bucket(role))
            .unwrap_or_else(empty_worker_slice)
    }

    /// Get all workers for a model (across every framework and role).
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        match self.pools.get(model_id) {
            Some(pool) => Arc::from(pool.all().into_boxed_slice()),
            None => empty_worker_slice(),
        }
    }

    /// Get all workers matching a worker type (across all models/frameworks).
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        let role = worker_type.pool_role();
        self.pools
            .iter()
            .flat_map(|pool| {
                pool.buckets
                    .iter()
                    .flat_map(|b| b.value().bucket(role).iter().cloned().collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .filter(|w| w.worker_type() == worker_type)
            .collect()
    }

    /// Collect every worker in a given pool role across all models/frameworks.
    fn collect_role(&self, role: PoolRole) -> Vec<Arc<dyn Worker>> {
        self.pools
            .iter()
            .flat_map(|pool| {
                pool.buckets
                    .iter()
                    .flat_map(|b| b.value().bucket(role).iter().cloned().collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.collect_role(PoolRole::PrefillPD)
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.collect_role(PoolRole::DecodePD)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        self.connection_workers
            .get(connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get the number of workers in the registry
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().clone(),
                )
            })
            .collect()
    }

    /// Get all model IDs with workers (lock-free)
    pub fn get_models(&self) -> Vec<String> {
        self.pools
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode (Http, Grpc)
    /// - framework: Filter by framework (Sglang, Vllm, Atom, Anonymous)
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        framework: Option<Framework>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the narrowest collection the pool layout allows.
        // Best case: (model, framework, role) hits a single bucket directly.
        let workers: Vec<Arc<dyn Worker>> = match (model_id, &framework, &worker_type) {
            (Some(model), Some(fw), Some(wt)) => self.get_pool(model, fw, wt.pool_role()).to_vec(),
            (Some(model), _, _) => self.get_by_model(model).to_vec(),
            _ => self.get_all(),
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if *w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified (using matches for flexible gRPC matching)
                if let Some(ref conn) = connection_mode {
                    if !w.connection_mode().matches(conn) {
                        return false;
                    }
                }

                // Check framework if specified
                if let Some(ref fw) = framework {
                    if w.framework() != fw {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics (lock-free)
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .pools
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc { .. } => grpc_count += 1,
            }

            match worker.circuit_breaker().state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Get counts of regular and PD workers efficiently (O(1))
    /// This avoids the overhead of get_all() which allocates memory and iterates all workers
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Sum the regular buckets across every model/framework pool.
        let regular_count: usize = self
            .pools
            .iter()
            .map(|pool| {
                pool.buckets
                    .iter()
                    .map(|b| b.value().bucket(PoolRole::Regular).len())
                    .sum::<usize>()
            })
            .sum();

        // Get total workers count efficiently from DashMap
        let total_workers = self.workers.len();

        // PD workers are any workers that are not Regular
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub(crate) fn start_health_checker(&self, check_interval_secs: u64) -> HealthChecker {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks in parallel for better performance
                // This is especially important when there are many workers
                let health_futures: Vec<_> = workers
                    .iter()
                    .filter(|worker| !worker.metadata().health_config.disable_health_check)
                    .map(|worker| {
                        let worker = worker.clone();
                        async move {
                            let _ = worker.check_health_async().await;
                        }
                    })
                    .collect();
                futures::future::join_all(health_futures).await;
            }
        });

        HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{
        circuit_breaker::CircuitBreakerConfig, BasicWorkerBuilder, UNKNOWN_MODEL_ID,
    };

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker));

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    /// Helper to create a worker Arc with given url, type, and model_id
    fn make_worker(url: &str, wtype: WorkerType, model_id: &str) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), model_id.to_string());
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(wtype)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        )
    }

    #[test]
    fn test_hash_ring_basic() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", WorkerType::Regular, "m"),
            make_worker("http://w2:8000", WorkerType::Regular, "m"),
            make_worker("http://w3:8000", WorkerType::Regular, "m"),
        ];
        let ring = HashRing::new(&workers);

        assert!(!ring.is_empty());
        assert_eq!(ring.worker_count(), 3);
        // 3 workers * 150 virtual nodes = 450 entries
        assert_eq!(ring.len(), 3 * VIRTUAL_NODES_PER_WORKER);
    }

    #[test]
    fn test_hash_ring_empty() {
        let ring = HashRing::new(&[]);
        assert!(ring.is_empty());
        assert_eq!(ring.worker_count(), 0);
        assert_eq!(ring.find_healthy_url("key", |_| true), None);
    }

    #[test]
    fn test_hash_ring_find_healthy() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", WorkerType::Regular, "m"),
            make_worker("http://w2:8000", WorkerType::Regular, "m"),
        ];
        let ring = HashRing::new(&workers);

        // All healthy -> should find something
        let result = ring.find_healthy_url("test-key", |_| true);
        assert!(result.is_some());

        // None healthy -> should return None
        let result = ring.find_healthy_url("test-key", |_| false);
        assert!(result.is_none());

        // Only w2 healthy -> should always return w2
        let result = ring.find_healthy_url("test-key", |url| url == "http://w2:8000");
        assert_eq!(result, Some("http://w2:8000"));
    }

    #[test]
    fn test_hash_ring_deterministic() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", WorkerType::Regular, "m"),
            make_worker("http://w2:8000", WorkerType::Regular, "m"),
            make_worker("http://w3:8000", WorkerType::Regular, "m"),
        ];
        let ring = HashRing::new(&workers);

        // Same key should always map to same worker
        let result1 = ring.find_healthy_url("consistent-key", |_| true);
        let result2 = ring.find_healthy_url("consistent-key", |_| true);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_hash_ring_distribution() {
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", WorkerType::Regular, "m"),
            make_worker("http://w2:8000", WorkerType::Regular, "m"),
            make_worker("http://w3:8000", WorkerType::Regular, "m"),
        ];
        let ring = HashRing::new(&workers);

        // Different keys should distribute across workers
        let mut counts = HashMap::new();
        for i in 0..300 {
            let key = format!("key-{}", i);
            if let Some(url) = ring.find_healthy_url(&key, |_| true) {
                *counts.entry(url.to_string()).or_insert(0) += 1;
            }
        }
        // All three workers should receive some keys
        assert_eq!(counts.len(), 3);
        assert!(counts.values().all(|&c| c > 20)); // Reasonable distribution
    }

    #[test]
    fn test_registry_pd_workers() {
        let registry = WorkerRegistry::new();

        let prefill = make_worker(
            "http://p1:8000",
            WorkerType::Prefill {
                bootstrap_port: Some(9000),
            },
            "llama",
        );
        let decode = make_worker("http://d1:8000", WorkerType::Decode, "llama");
        let regular = make_worker("http://r1:8000", WorkerType::Regular, "llama");

        registry.register(prefill);
        registry.register(decode);
        registry.register(regular);

        let prefill_workers = registry.get_prefill_workers();
        assert_eq!(prefill_workers.len(), 1);
        assert_eq!(prefill_workers[0].url(), "http://p1:8000");

        let decode_workers = registry.get_decode_workers();
        assert_eq!(decode_workers.len(), 1);
        assert_eq!(decode_workers[0].url(), "http://d1:8000");

        let stats = registry.stats();
        assert_eq!(stats.prefill_workers, 1);
        assert_eq!(stats.decode_workers, 1);
        assert_eq!(stats.regular_workers, 1);
        assert_eq!(stats.total_workers, 3);
    }

    #[test]
    fn test_registry_get_workers_filtered() {
        let registry = WorkerRegistry::new();

        let w1 = make_worker(
            "http://p1:8000",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            "llama",
        );
        let w2 = make_worker("http://d1:8000", WorkerType::Decode, "llama");
        let w3 = make_worker("http://r1:8000", WorkerType::Regular, "gpt-4");

        registry.register(w1);
        registry.register(w2);
        registry.register(w3);

        // Filter by model
        let llama = registry.get_workers_filtered(Some("llama"), None, None, None, false);
        assert_eq!(llama.len(), 2);

        // Filter by type
        let prefill = registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            None,
            None,
            false,
        );
        assert_eq!(prefill.len(), 1);

        // Filter healthy only
        registry
            .get_by_url("http://p1:8000")
            .unwrap()
            .set_healthy(false);
        let healthy = registry.get_workers_filtered(None, None, None, None, true);
        assert_eq!(healthy.len(), 2); // p1 is unhealthy

        // Filter by connection mode
        let http_workers =
            registry.get_workers_filtered(None, None, Some(ConnectionMode::Http), None, false);
        assert_eq!(http_workers.len(), 3);
    }

    #[test]
    fn test_registry_worker_distribution() {
        let registry = WorkerRegistry::new();

        registry.register(make_worker("http://r1:8000", WorkerType::Regular, "m"));
        registry.register(make_worker("http://r2:8000", WorkerType::Regular, "m"));
        registry.register(make_worker(
            "http://p1:8000",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            "m",
        ));

        let (regular, pd) = registry.get_worker_distribution();
        assert_eq!(regular, 2);
        assert_eq!(pd, 1);
    }

    #[test]
    fn test_registry_reserve_id_for_url() {
        let registry = WorkerRegistry::new();

        let id1 = registry.reserve_id_for_url("http://w1:8000");
        let id2 = registry.reserve_id_for_url("http://w1:8000");

        // Same URL should get same ID
        assert_eq!(id1.as_str(), id2.as_str());

        // Different URL should get different ID
        let id3 = registry.reserve_id_for_url("http://w2:8000");
        assert_ne!(id1.as_str(), id3.as_str());
    }

    #[test]
    fn test_registry_get_url_by_id() {
        let registry = WorkerRegistry::new();

        let worker = make_worker("http://w1:8000", WorkerType::Regular, "m");
        let id = registry.register(worker);

        assert_eq!(
            registry.get_url_by_id(&id),
            Some("http://w1:8000".to_string())
        );

        // Unknown ID
        let unknown = WorkerId::from_string("nonexistent".to_string());
        assert_eq!(registry.get_url_by_id(&unknown), None);
    }

    #[test]
    fn test_registry_remove_by_url() {
        let registry = WorkerRegistry::new();

        let worker = make_worker("http://w1:8000", WorkerType::Regular, "m");
        registry.register(worker);

        assert_eq!(registry.len(), 1);
        let removed = registry.remove_by_url("http://w1:8000");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 0);

        // Remove non-existent
        let removed = registry.remove_by_url("http://nonexistent:8000");
        assert!(removed.is_none());
    }

    #[test]
    fn test_registry_stats_health_tracking() {
        let registry = WorkerRegistry::new();

        let w1 = make_worker("http://w1:8000", WorkerType::Regular, "m");
        let w2 = make_worker("http://w2:8000", WorkerType::Regular, "m");
        registry.register(w1);
        registry.register(w2);

        let stats = registry.stats();
        assert_eq!(stats.healthy_workers, 2);
        assert_eq!(stats.unhealthy_workers, 0);

        // Mark one unhealthy
        registry
            .get_by_url("http://w1:8000")
            .unwrap()
            .set_healthy(false);
        let stats = registry.stats();
        assert_eq!(stats.healthy_workers, 1);
        assert_eq!(stats.unhealthy_workers, 1);
    }

    #[test]
    fn test_registry_hash_ring_rebuilt_on_change() {
        let registry = WorkerRegistry::new();

        let w1 = make_worker("http://w1:8000", WorkerType::Regular, "llama");
        registry.register(w1);

        let ring1 = registry.get_hash_ring("llama");
        assert!(ring1.is_some());
        assert_eq!(ring1.unwrap().worker_count(), 1);

        let w2 = make_worker("http://w2:8000", WorkerType::Regular, "llama");
        registry.register(w2);

        let ring2 = registry.get_hash_ring("llama");
        assert_eq!(ring2.unwrap().worker_count(), 2);

        // Remove a worker -> ring should shrink
        registry.remove_by_url("http://w1:8000");
        let ring3 = registry.get_hash_ring("llama");
        assert_eq!(ring3.unwrap().worker_count(), 1);
    }

    #[test]
    fn test_registry_get_all_urls() {
        let registry = WorkerRegistry::new();

        registry.register(make_worker("http://w1:8000", WorkerType::Regular, "m"));
        registry.register(make_worker("http://w2:8000", WorkerType::Regular, "m"));

        let urls = registry.get_all_urls();
        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://w1:8000".to_string()));
        assert!(urls.contains(&"http://w2:8000".to_string()));
    }

    #[test]
    fn test_registry_get_models() {
        let registry = WorkerRegistry::new();

        registry.register(make_worker("http://w1:8000", WorkerType::Regular, "llama"));
        registry.register(make_worker("http://w2:8000", WorkerType::Regular, "gpt-4"));
        registry.register(make_worker("http://w3:8000", WorkerType::Regular, "llama"));

        let models = registry.get_models();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"llama".to_string()));
        assert!(models.contains(&"gpt-4".to_string()));
    }

    #[test]
    fn test_registry_empty() {
        let registry = WorkerRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 0);
        assert_eq!(stats.total_models, 0);

        assert!(registry.get_all().is_empty());
        assert!(registry.get_all_urls().is_empty());
        assert!(registry.get_models().is_empty());
        assert!(registry.get_prefill_workers().is_empty());
        assert!(registry.get_decode_workers().is_empty());
    }

    #[test]
    fn test_registry_reregister_same_url() {
        let registry = WorkerRegistry::new();

        let w1 = make_worker("http://w1:8000", WorkerType::Regular, "m");
        let id1 = registry.register(w1);

        // Re-register same URL -> should reuse ID
        let w1_again = make_worker("http://w1:8000", WorkerType::Regular, "m");
        let id2 = registry.register(w1_again);

        assert_eq!(id1.as_str(), id2.as_str());
        assert_eq!(registry.len(), 1);
    }

    fn make_worker_fw(
        url: &str,
        wtype: WorkerType,
        model_id: &str,
        framework: Framework,
    ) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), model_id.to_string());
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(wtype)
                .framework(framework)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        )
    }

    #[test]
    fn test_layered_pool_direct_bucket_access() {
        let registry = WorkerRegistry::new();
        // Same model, different frameworks and roles.
        registry.register(make_worker_fw(
            "http://r1:8000",
            WorkerType::Regular,
            "m",
            Framework::Vllm,
        ));
        registry.register(make_worker_fw(
            "http://p1:8000",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            "m",
            Framework::Vllm,
        ));
        registry.register(make_worker_fw(
            "http://d1:8000",
            WorkerType::Decode,
            "m",
            Framework::Vllm,
        ));
        registry.register(make_worker_fw(
            "http://r2:8000",
            WorkerType::Regular,
            "m",
            Framework::Sglang,
        ));

        // Direct bucket access hits exactly one (model, framework, role).
        assert_eq!(
            registry
                .get_pool("m", &Framework::Vllm, PoolRole::Regular)
                .len(),
            1
        );
        assert_eq!(
            registry
                .get_pool("m", &Framework::Vllm, PoolRole::PrefillPD)
                .len(),
            1
        );
        assert_eq!(
            registry
                .get_pool("m", &Framework::Vllm, PoolRole::DecodePD)
                .len(),
            1
        );
        assert_eq!(
            registry
                .get_pool("m", &Framework::Sglang, PoolRole::Regular)
                .len(),
            1
        );
        assert_eq!(
            registry
                .get_pool("m", &Framework::Sglang, PoolRole::PrefillPD)
                .len(),
            0
        );

        // Frameworks present for the model.
        let mut fws = registry.get_frameworks("m");
        fws.sort_by_key(|f| f.to_string());
        assert_eq!(fws, vec![Framework::Sglang, Framework::Vllm]);

        // Aggregates across frameworks/roles.
        assert_eq!(registry.get_by_model("m").len(), 4);
        assert_eq!(registry.get_prefill_workers().len(), 1);
        assert_eq!(registry.get_decode_workers().len(), 1);
    }

    #[test]
    fn test_layered_pool_defaults_unknown_and_anonymous() {
        let registry = WorkerRegistry::new();
        // No model_id label and default (Anonymous) framework.
        let w: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8000")
                .worker_type(WorkerType::Regular)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        registry.register(w);

        assert_eq!(
            registry
                .get_pool(UNKNOWN_MODEL_ID, &Framework::Anonymous, PoolRole::Regular)
                .len(),
            1
        );
        assert_eq!(registry.get_models(), vec![UNKNOWN_MODEL_ID.to_string()]);
    }

    #[test]
    fn test_layered_pool_remove_shrinks_bucket() {
        let registry = WorkerRegistry::new();
        registry.register(make_worker_fw(
            "http://a:8000",
            WorkerType::Regular,
            "m",
            Framework::Atom,
        ));
        registry.register(make_worker_fw(
            "http://b:8000",
            WorkerType::Regular,
            "m",
            Framework::Atom,
        ));
        assert_eq!(
            registry
                .get_pool("m", &Framework::Atom, PoolRole::Regular)
                .len(),
            2
        );

        registry.remove_by_url("http://a:8000");
        assert_eq!(
            registry
                .get_pool("m", &Framework::Atom, PoolRole::Regular)
                .len(),
            1
        );

        registry.remove_by_url("http://b:8000");
        // Pool emptied -> model gone.
        assert_eq!(
            registry
                .get_pool("m", &Framework::Atom, PoolRole::Regular)
                .len(),
            0
        );
        assert!(registry.get_models().is_empty());
    }
}
