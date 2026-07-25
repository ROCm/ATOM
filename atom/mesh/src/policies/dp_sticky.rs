//! Data-parallel worker routing with session affinity.
//!
//! Requests carrying `X-Session-ID` are pinned to the first healthy worker
//! selected for that session. New sessions and requests without a session ID
//! use minimum-load balancing. A stale mapping is removed and
//! reassigned when its worker is no longer healthy.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, routers::comm::header_utils::extract_sticky_routing_key};

#[derive(Debug)]
pub struct DpStickyPolicy {
    /// Session ID -> worker URL. URLs remain valid when a worker slice is reordered.
    assignments: DashMap<String, String>,
}

impl DpStickyPolicy {
    pub fn new() -> Self {
        Self {
            assignments: DashMap::new(),
        }
    }

    fn healthy_worker_for_url(workers: &[Arc<dyn Worker>], url: &str) -> Option<usize> {
        workers
            .iter()
            .enumerate()
            .find(|(_, worker)| {
                worker.url() == url && worker.is_healthy() && worker.circuit_breaker().can_execute()
            })
            .map(|(index, _)| index)
    }

    /// Select the healthy worker with the smallest current request load.
    fn select_low_load_worker(workers: &[Arc<dyn Worker>]) -> Option<usize> {
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| workers[index].load())
    }

    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let session_id = extract_sticky_routing_key(info.headers);
        let Some(session_id) = session_id else {
            return Self::select_low_load_worker(workers);
        };

        if let Some(assignment) = self.assignments.get(session_id) {
            if let Some(index) = Self::healthy_worker_for_url(workers, assignment.value()) {
                return Some(index);
            }
        }

        // The assigned worker was removed or became unhealthy. The next
        // selection establishes a replacement affinity.
        self.assignments.remove(session_id);
        let selected_index = Self::select_low_load_worker(workers)?;
        let selected_url = workers[selected_index].url().to_string();

        match self.assignments.entry(session_id.to_string()) {
            Entry::Occupied(mut assignment) => {
                if let Some(index) = Self::healthy_worker_for_url(workers, assignment.get()) {
                    Some(index)
                } else {
                    assignment.insert(selected_url);
                    Some(selected_index)
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(selected_url);
                Some(selected_index)
            }
        }
    }
}

impl Default for DpStickyPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for DpStickyPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select_worker_impl(workers, info)
    }

    fn name(&self) -> &'static str {
        "dp_sticky"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn workers() -> Vec<Arc<dyn Worker>> {
        ["http://worker-1:8000", "http://worker-2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn headers(session_id: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-session-id", session_id.parse().unwrap());
        headers
    }

    #[tokio::test]
    async fn session_id_is_sticky_while_worker_is_healthy() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let selected = policy.select_worker(&workers, &info).await.unwrap();
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, &info).await, Some(selected));
        }
    }

    #[tokio::test]
    async fn unhealthy_assignment_is_replaced() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let initial = policy.select_worker(&workers, &info).await.unwrap();
        workers[initial].set_healthy(false);

        let replacement = policy.select_worker(&workers, &info).await.unwrap();
        assert_ne!(replacement, initial);
        assert_eq!(
            policy.select_worker(&workers, &info).await,
            Some(replacement)
        );
    }

    #[tokio::test]
    async fn missing_session_id_uses_load_balancing_fallback() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        workers[0].increment_load();
        workers[0].increment_load();

        let selected = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await;
        assert_eq!(selected, Some(1));
    }
}
