use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;
use async_trait::async_trait;
use std::sync::Arc;

/// Exact P/D scoring is performed jointly by the HTTP ATOM relay. A source that
/// is unavailable, uncalibrated or lacks exact tokens uses this load fallback.
#[derive(Debug, Default)]
pub struct KvCacheAwarePolicy;

#[async_trait]
impl LoadBalancingPolicy for KvCacheAwarePolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|i| workers[*i].load())
    }
    fn name(&self) -> &'static str {
        "kv_cache_aware"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
