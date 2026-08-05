//! Data-parallel worker routing with session affinity.
//!
//! Requests carrying `X-Session-ID` are pinned to the first healthy worker
//! selected for that session. New sessions and requests without a session ID
//! use minimum-load balancing. A stale mapping is replaced when its worker is
//! no longer healthy or the session has been idle for too long.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, routers::comm::header_utils::extract_sticky_routing_key};

const SESSION_REASSIGNMENT_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Debug)]
struct StickyAssignment {
    worker: WorkerIdentity,
    last_access: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkerIdentity {
    url: String,
    dp_rank: Option<usize>,
}

#[derive(Debug)]
pub struct DpStickyPolicy {
    /// Session ID -> worker identity. The DP rank distinguishes logical workers
    /// that share a common endpoint URL.
    assignments: DashMap<String, StickyAssignment>,
}

impl DpStickyPolicy {
    pub fn new() -> Self {
        Self {
            assignments: DashMap::new(),
        }
    }

    fn worker_identity(worker: &Arc<dyn Worker>) -> WorkerIdentity {
        WorkerIdentity {
            url: worker.url().to_string(),
            dp_rank: worker.dp_rank(),
        }
    }

    fn healthy_worker_for_identity(
        workers: &[Arc<dyn Worker>],
        identity: &WorkerIdentity,
    ) -> Option<usize> {
        workers
            .iter()
            .enumerate()
            .find(|(_, worker)| {
                worker.url() == identity.url
                    && worker.dp_rank() == identity.dp_rank
                    && worker.is_healthy()
                    && worker.circuit_breaker().can_execute()
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
        let now = Instant::now();

        if let Some(mut assignment) = self.assignments.get_mut(session_id) {
            let is_expired = now.saturating_duration_since(assignment.last_access)
                > SESSION_REASSIGNMENT_IDLE_TIMEOUT;

            if !is_expired {
                if let Some(index) = Self::healthy_worker_for_identity(workers, &assignment.worker)
                {
                    assignment.last_access = now;
                    return Some(index);
                }
            }
        }

        let selected_index = Self::select_low_load_worker(workers)?;
        let selected_worker = Self::worker_identity(&workers[selected_index]);

        match self.assignments.entry(session_id.to_string()) {
            Entry::Occupied(mut entry) => {
                let existing_index = {
                    let assignment = entry.get_mut();
                    let is_expired = now.saturating_duration_since(assignment.last_access)
                        > SESSION_REASSIGNMENT_IDLE_TIMEOUT;

                    if !is_expired {
                        if let Some(index) =
                            Self::healthy_worker_for_identity(workers, &assignment.worker)
                        {
                            assignment.last_access = now;
                            Some(index)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some(index) = existing_index {
                    Some(index)
                } else {
                    entry.insert(StickyAssignment {
                        worker: selected_worker,
                        last_access: now,
                    });
                    Some(selected_index)
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(StickyAssignment {
                    worker: selected_worker,
                    last_access: now,
                });
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
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

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

    fn dp_workers() -> Vec<Arc<dyn Worker>> {
        (0..2)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://worker:8000", rank, 2).build())
                    as Arc<dyn Worker>
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
    async fn session_id_stays_on_the_same_dp_rank() {
        let policy = DpStickyPolicy::new();
        let workers = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        workers[0].increment_load();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
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
    async fn expired_session_assignment_is_rebalanced_to_lowest_load_worker() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        workers[0].increment_load();
        workers[0].increment_load();
        policy.assignments.get_mut("session-1").unwrap().last_access =
            Instant::now() - SESSION_REASSIGNMENT_IDLE_TIMEOUT - Duration::from_secs(1);

        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
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
