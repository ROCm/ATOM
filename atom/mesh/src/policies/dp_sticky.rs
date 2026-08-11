//! Data-parallel worker routing with session affinity.
//!
//! Requests carrying `X-Session-ID` are pinned to the first healthy worker
//! selected for that session. A new session goes to the worker holding the
//! fewest live sessions (load breaks ties); requests without a session ID use
//! minimum-load balancing. A stale mapping is replaced when its worker is no
//! longer healthy or the session has been idle for too long.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::{core::Worker, routers::comm::header_utils::extract_sticky_routing_key};

const SESSION_REASSIGNMENT_IDLE_TIMEOUT: Duration = Duration::from_secs(120 * 60);

#[derive(Debug)]
struct StickyAssignment {
    worker: WorkerIdentity,
    last_access: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkerIdentity {
    url: String,
    dp_rank: Option<usize>,
}

#[derive(Debug)]
pub struct DpStickyPolicy {
    /// Session ID -> worker identity. The DP rank distinguishes logical workers
    /// that share a common endpoint URL.
    assignments: DashMap<String, StickyAssignment>,
    /// Serializes count -> choose -> insert for new sessions so a burst of them
    /// cannot all read the same pre-burst assignment counts and pile onto one
    /// worker. Only new/expired sessions contend; established ones take the
    /// lock-free path above.
    assign_lock: Mutex<()>,
}

impl DpStickyPolicy {
    pub fn new() -> Self {
        Self {
            assignments: DashMap::new(),
            assign_lock: Mutex::new(()),
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

    /// Select the healthy worker holding the fewest live session assignments,
    /// tie-broken by current load.
    ///
    /// Load alone cannot spread new sessions. On the streaming PD path the
    /// router creates a request's `WorkerLoadGuard` only once the upstream
    /// response headers arrive, so every request of a simultaneous burst reads
    /// `load() == 0` and `min_by_key` hands them all the same worker — which
    /// this policy then pins for the rest of each session. Assignment counts
    /// update the instant a session is pinned, so the N-th new session of a
    /// burst sees the N-1 pins made before it.
    ///
    /// Caller must hold `assign_lock`.
    fn select_new_session_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        now: Instant,
    ) -> Option<usize> {
        let mut live_sessions: HashMap<WorkerIdentity, usize> = HashMap::new();
        for entry in self.assignments.iter() {
            if now.saturating_duration_since(entry.last_access) <= SESSION_REASSIGNMENT_IDLE_TIMEOUT
            {
                *live_sessions.entry(entry.worker.clone()).or_insert(0) += 1;
            }
        }

        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| {
                let identity = Self::worker_identity(&workers[index]);
                (
                    live_sessions.get(&identity).copied().unwrap_or(0),
                    workers[index].load(),
                )
            })
    }

    /// Return the worker index this session is already pinned to, refreshing
    /// its access time. `None` when the session is new, idle-expired, or its
    /// worker is no longer usable.
    fn resolve_existing(
        &self,
        workers: &[Arc<dyn Worker>],
        session_id: &str,
        now: Instant,
    ) -> Option<usize> {
        let mut assignment = self.assignments.get_mut(session_id)?;
        if now.saturating_duration_since(assignment.last_access) > SESSION_REASSIGNMENT_IDLE_TIMEOUT
        {
            return None;
        }
        let index = Self::healthy_worker_for_identity(workers, &assignment.worker)?;
        assignment.last_access = now;
        Some(index)
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

        if let Some(index) = self.resolve_existing(workers, session_id, now) {
            return Some(index);
        }

        // New, expired, or orphaned session: assign under the lock so concurrent
        // newcomers observe each other's pins instead of all reading the same
        // counts and landing on one worker.
        let _guard = self.assign_lock.lock().unwrap_or_else(|e| e.into_inner());

        // Another thread may have assigned this session while we waited.
        if let Some(index) = self.resolve_existing(workers, session_id, now) {
            return Some(index);
        }

        let selected_index = self.select_new_session_worker(workers, now)?;
        let selected_worker = Self::worker_identity(&workers[selected_index]);
        self.assignments.insert(
            session_id.to_string(),
            StickyAssignment {
                worker: selected_worker,
                last_access: now,
            },
        );
        Some(selected_index)
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
    async fn new_sessions_are_spread_by_assignment_count() {
        let policy = DpStickyPolicy::new();
        let workers = dp_workers();

        for (session_id, expected_worker) in [
            ("session-1", 0),
            ("session-2", 1),
            ("session-3", 0),
            ("session-4", 1),
        ] {
            let headers = headers(session_id);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            assert_eq!(
                policy.select_worker(&workers, &info).await,
                Some(expected_worker)
            );
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
