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

use super::{
    decision::{self, RouteDecision, NO_HEALTHY_WORKERS},
    get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::{
    core::Worker,
    observability::metrics::metrics_labels::{
        AFFINITY_EXPIRED, AFFINITY_HIT, AFFINITY_NEW, AFFINITY_REBIND,
    },
    routers::comm::header_utils::extract_sticky_routing_key,
};

const SESSION_REASSIGNMENT_IDLE_TIMEOUT: Duration = Duration::from_secs(120 * 60);
const NO_SESSION: &str = "no_session";

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
    /// Serializes count -> choose -> insert for new sessions so a burst cannot
    /// pin every session to the same DP rank. Established sessions use the
    /// lock-free assignment lookup above.
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

    /// Select the healthy worker holding the fewest non-expired assignments,
    /// tie-broken by current request load. Caller must hold `assign_lock`.
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

    /// Returns the selection with the session outcome: hit, new, rebind,
    /// expired, or no_session for requests without a session ID.
    fn select_worker_impl(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, &'static str) {
        let session_id = extract_sticky_routing_key(info.headers);
        let Some(session_id) = session_id else {
            return (Self::select_low_load_worker(workers), NO_SESSION);
        };
        let now = Instant::now();

        if let Some(mut assignment) = self.assignments.get_mut(session_id) {
            let is_expired = now.saturating_duration_since(assignment.last_access)
                > SESSION_REASSIGNMENT_IDLE_TIMEOUT;
            if !is_expired {
                if let Some(index) = Self::healthy_worker_for_identity(workers, &assignment.worker)
                {
                    assignment.last_access = now;
                    return (Some(index), AFFINITY_HIT);
                }
            }
        }

        // New, expired, or orphaned sessions must assign atomically: another
        // request may have pinned this session while this one waited for the lock.
        let _assign_guard = self
            .assign_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let outcome = match self.assignments.get_mut(session_id) {
            None => AFFINITY_NEW,
            Some(mut assignment) => {
                let is_expired = now.saturating_duration_since(assignment.last_access)
                    > SESSION_REASSIGNMENT_IDLE_TIMEOUT;
                if is_expired {
                    AFFINITY_EXPIRED
                } else if let Some(index) =
                    Self::healthy_worker_for_identity(workers, &assignment.worker)
                {
                    assignment.last_access = now;
                    return (Some(index), AFFINITY_HIT);
                } else {
                    AFFINITY_REBIND
                }
            }
        };

        let Some(selected_index) = self.select_new_session_worker(workers, now) else {
            return (None, outcome);
        };
        self.assignments.insert(
            session_id.to_string(),
            StickyAssignment {
                worker: Self::worker_identity(&workers[selected_index]),
                last_access: now,
            },
        );
        (Some(selected_index), outcome)
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
        let (selected, outcome) = self.select_worker_impl(workers, info);
        let decision = match selected {
            None => RouteDecision::branch(NO_HEALTHY_WORKERS),
            Some(index) if outcome == NO_SESSION => RouteDecision {
                cost: Some(workers[index].load() as f64),
                ..RouteDecision::branch(NO_SESSION)
            },
            Some(_) => RouteDecision {
                affinity: Some(outcome),
                ..RouteDecision::branch(outcome)
            },
        };
        decision::report(self, workers, selected, decision);
        selected
    }

    fn name(&self) -> &'static str {
        "dp_sticky"
    }

    fn decision_branches(&self) -> &'static [&'static str] {
        &[
            AFFINITY_HIT,
            AFFINITY_NEW,
            AFFINITY_REBIND,
            AFFINITY_EXPIRED,
            NO_SESSION,
            NO_HEALTHY_WORKERS,
        ]
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
    async fn session_outcomes_are_counted_by_result() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let first = policy.select_worker(&workers, &info).await.unwrap();
        policy.select_worker(&workers, &info).await.unwrap();
        workers[first].set_healthy(false);
        policy.select_worker(&workers, &info).await.unwrap();
        policy.assignments.get_mut("session-1").unwrap().last_access =
            Instant::now() - SESSION_REASSIGNMENT_IDLE_TIMEOUT - Duration::from_secs(1);
        policy.select_worker(&workers, &info).await.unwrap();
        policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await
            .unwrap();

        let rendered = handle.render();
        for result in ["new", "hit", "rebind", "expired"] {
            let line =
                format!(r#"mesh_session_affinity_total{{pool="regular",result="{result}"}} 1"#);
            assert!(rendered.contains(&line), "missing {line} in\n{rendered}");
        }
        assert!(
            rendered.contains(
                r#"mesh_route_branch_total{pool="regular",policy="dp_sticky",branch="no_session"} 1"#
            ),
            "{rendered}"
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
