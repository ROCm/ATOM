//! Data-parallel worker routing with session affinity.
//!
//! Requests carrying `X-Session-ID` are pinned to the first healthy worker
//! selected for that session. A new session goes to the worker holding the
//! fewest live sessions (load breaks ties); requests without a session ID use
//! minimum-load balancing. A stale mapping is replaced when its worker is no
//! longer healthy, the session has been idle for too long, or a request of the
//! session failed on it with a server error or got no response.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo,
    NO_HEALTHY_WORKERS,
};
use crate::{
    core::Worker,
    observability::metrics::{
        metrics_labels::{
            AFFINITY_EXPIRED, AFFINITY_HIT, AFFINITY_NEW, AFFINITY_REBIND,
            WORKER_CHANGE_RANK_MAPPING,
        },
        MeshMetrics,
    },
    routers::comm::header_utils::extract_sticky_routing_key,
};

const SESSION_REASSIGNMENT_IDLE_TIMEOUT: Duration = Duration::from_secs(120 * 60);
const NO_SESSION: &str = "no_session";

#[derive(Debug)]
struct StickyAssignment {
    worker: WorkerIdentity,
    last_access: Instant,
    /// Where the session's last request was sent, after DP rank mapping.
    dispatched: Option<WorkerIdentity>,
    /// Why the session was last reassigned, until a request is sent.
    reassigned: Option<&'static str>,
    /// A request of the session failed on the worker it was sent to, so the
    /// next request, a retry included, is placed again.
    failed: bool,
    /// Workers the session's requests failed on since its last hit. Placing
    /// it again leaves them out while another worker is healthy.
    avoid: Vec<WorkerIdentity>,
}

impl StickyAssignment {
    /// Whether a request sent to `worker` belongs to this binding: `worker` is
    /// the bound worker, or, under DP rank mapping, the one the session's last
    /// request was sent to while no reassignment is pending.
    fn owns(&self, worker: &WorkerIdentity) -> bool {
        self.worker == *worker
            || (self.reassigned.is_none() && self.dispatched.as_ref() == Some(worker))
    }
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
    /// tie-broken by current request load, other than the workers in `avoid`
    /// unless every healthy worker is in it. Caller must hold `assign_lock`.
    fn select_new_session_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        now: Instant,
        avoid: &[WorkerIdentity],
    ) -> Option<usize> {
        let mut live_sessions: HashMap<WorkerIdentity, usize> = HashMap::new();
        for entry in self.assignments.iter() {
            if now.saturating_duration_since(entry.last_access) <= SESSION_REASSIGNMENT_IDLE_TIMEOUT
            {
                *live_sessions.entry(entry.worker.clone()).or_insert(0) += 1;
            }
        }

        let mut candidates = get_healthy_worker_indices(workers);
        if !avoid.is_empty() {
            let others: Vec<usize> = candidates
                .iter()
                .copied()
                .filter(|&index| !avoid.contains(&Self::worker_identity(&workers[index])))
                .collect();
            if !others.is_empty() {
                candidates = others;
            }
        }
        candidates.into_iter().min_by_key(|&index| {
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
            if !is_expired && !assignment.failed {
                if let Some(index) = Self::healthy_worker_for_identity(workers, &assignment.worker)
                {
                    assignment.last_access = now;
                    assignment.avoid.clear();
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
        let (outcome, dispatched, avoid) = match self.assignments.get_mut(session_id) {
            None => (AFFINITY_NEW, None, Vec::new()),
            Some(mut assignment) => {
                let is_expired = now.saturating_duration_since(assignment.last_access)
                    > SESSION_REASSIGNMENT_IDLE_TIMEOUT;
                if is_expired {
                    (
                        AFFINITY_EXPIRED,
                        assignment.dispatched.clone(),
                        std::mem::take(&mut assignment.avoid),
                    )
                } else if let Some(index) =
                    Self::healthy_worker_for_identity(workers, &assignment.worker)
                        .filter(|_| !assignment.failed)
                {
                    assignment.last_access = now;
                    assignment.avoid.clear();
                    return (Some(index), AFFINITY_HIT);
                } else {
                    (
                        AFFINITY_REBIND,
                        assignment.dispatched.clone(),
                        std::mem::take(&mut assignment.avoid),
                    )
                }
            }
        };

        let Some(selected_index) = self.select_new_session_worker(workers, now, &avoid) else {
            return (None, outcome);
        };
        self.assignments.insert(
            session_id.to_string(),
            StickyAssignment {
                worker: Self::worker_identity(&workers[selected_index]),
                last_access: now,
                dispatched,
                reassigned: (outcome != AFFINITY_NEW).then_some(outcome),
                failed: false,
                avoid,
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
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let (selected, outcome) = self.select_worker_impl(workers, info);
        let branch = if selected.is_some() {
            outcome
        } else {
            NO_HEALTHY_WORKERS
        };
        (selected, RouteDecision::branch(branch))
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        let Some(session_id) = extract_sticky_routing_key(headers) else {
            return;
        };
        let Some(mut assignment) = self.assignments.get_mut(session_id) else {
            return;
        };
        let identity = Self::worker_identity(worker);
        // Without a reassignment, only rank mapping can move a pinned session.
        let reason = assignment
            .reassigned
            .take()
            .unwrap_or(WORKER_CHANGE_RANK_MAPPING);
        if assignment
            .dispatched
            .as_ref()
            .is_some_and(|previous| *previous != identity)
        {
            MeshMetrics::record_session_worker_change(pool, reason);
        }
        assignment.dispatched = Some(identity);
    }

    /// A failed request releases its session's binding: the next request, a
    /// retry included, is placed again as `rebind`, away from every worker the
    /// session failed on since its last hit while another one is healthy. A
    /// late failure on a worker the session has since left keeps the new
    /// binding.
    fn on_dispatch_failed(&self, headers: Option<&http::HeaderMap>, worker: &Arc<dyn Worker>) {
        let Some(session_id) = extract_sticky_routing_key(headers) else {
            return;
        };
        let identity = Self::worker_identity(worker);
        if let Some(mut assignment) = self.assignments.get_mut(session_id) {
            if assignment.owns(&identity) {
                assignment.failed = true;
                if !assignment.avoid.contains(&identity) {
                    assignment.avoid.push(identity);
                }
            }
        }
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
    async fn session_outcomes_are_reported_as_branches() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let session = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let no_session = SelectWorkerInfo::default();

        let (first, decision) = policy.select_with_decision(&workers, &session).await;
        let mut branches = vec![decision.branch];
        branches.push(branch(&policy, &workers, &session).await);
        workers[first.unwrap()].set_healthy(false);
        branches.push(branch(&policy, &workers, &session).await);
        expire(&policy, "session-1");
        branches.push(branch(&policy, &workers, &session).await);
        branches.push(branch(&policy, &workers, &no_session).await);
        workers.iter().for_each(|worker| worker.set_healthy(false));
        branches.push(branch(&policy, &workers, &session).await);

        assert_eq!(
            branches,
            [
                "new",
                "hit",
                "rebind",
                "expired",
                "no_session",
                NO_HEALTHY_WORKERS
            ]
        );
    }

    async fn branch(
        policy: &DpStickyPolicy,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> &'static str {
        policy.select_with_decision(workers, info).await.1.branch
    }

    /// Selects a worker for `session` and dispatches the request to it.
    async fn route(
        policy: &DpStickyPolicy,
        workers: &[Arc<dyn Worker>],
        pool: &'static str,
        session: &str,
    ) -> usize {
        let headers = headers(session);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let selected = policy.select_worker(workers, &info).await.unwrap();
        policy.on_dispatch(pool, Some(&headers), &workers[selected]);
        selected
    }

    fn expire(policy: &DpStickyPolicy, session: &str) {
        policy.assignments.get_mut(session).unwrap().last_access =
            Instant::now() - SESSION_REASSIGNMENT_IDLE_TIMEOUT - Duration::from_secs(1);
    }

    fn worker_changes(rendered: &str) -> Vec<&str> {
        let mut changes: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("mesh_session_worker_change_total{"))
            .collect();
        changes.sort_unstable();
        changes
    }

    #[tokio::test]
    async fn worker_changes_count_only_a_different_dispatched_worker() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let workers = workers();

        // New, then hit: no change.
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        // Worker 0 is unavailable: rebind to worker 1.
        workers[0].set_healthy(false);
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 1);
        workers[0].set_healthy(true);
        // Expired: reassigned to worker 0, a different worker.
        expire(&policy, "session-1");
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        // Expired again: reassigned to worker 0, the same worker.
        expire(&policy, "session-1");
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);

        // Ranks behind one URL are different workers.
        let ranks = dp_workers();
        assert_eq!(route(&policy, &ranks, "decode", "session-2").await, 0);
        ranks[0].set_healthy(false);
        assert_eq!(route(&policy, &ranks, "decode", "session-2").await, 1);
        assert_eq!(ranks[0].base_url(), ranks[1].base_url());

        assert_eq!(
            worker_changes(&handle.render()),
            [
                r#"mesh_session_worker_change_total{pool="decode",reason="rebind"} 1"#,
                r#"mesh_session_worker_change_total{pool="regular",reason="expired"} 1"#,
                r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#,
            ]
        );
    }

    #[tokio::test]
    async fn worker_changes_follow_the_dispatched_rank() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let ranks = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // The pin stays on rank 0 while rank mapping dispatches to rank 1, 1, then 0.
        for dispatched in [1, 1, 0] {
            assert_eq!(policy.select_worker(&ranks, &info).await, Some(0));
            policy.on_dispatch("prefill", Some(&headers), &ranks[dispatched]);
        }

        assert_eq!(
            worker_changes(&handle.render()),
            [r#"mesh_session_worker_change_total{pool="prefill",reason="rank_mapping"} 1"#]
        );
    }

    #[tokio::test]
    async fn dispatch_failure_places_the_retry_again() {
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

        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        // The retry avoids worker 0, where it failed.
        let (retry, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((retry, decision.branch), (Some(1), "rebind"));
        policy.on_dispatch("regular", Some(&headers), &workers[1]);
        assert_eq!(branch(&policy, &workers, &info).await, "hit");

        assert_eq!(
            worker_changes(&handle.render()),
            [r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#]
        );
    }

    #[tokio::test]
    async fn a_retry_avoids_the_worker_it_failed_on() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        assert_eq!(route(&policy, &workers, "regular", "s1").await, 0);
        assert_eq!(route(&policy, &workers, "regular", "s2").await, 1);
        assert_eq!(route(&policy, &workers, "regular", "s3").await, 0);
        let s2 = headers("s2");
        let info = SelectWorkerInfo {
            headers: Some(&s2),
            ..Default::default()
        };

        // Worker 1 holds fewer sessions, but s2 failed there.
        policy.on_dispatch_failed(Some(&s2), &workers[1]);
        let (retry, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((retry, decision.branch), (Some(0), "rebind"));

        // With no other healthy worker, the retry stays where it failed.
        let only = &workers[1..];
        assert_eq!(route(&policy, only, "regular", "s4").await, 0);
        let s4 = headers("s4");
        policy.on_dispatch_failed(Some(&s4), &only[0]);
        let info = SelectWorkerInfo {
            headers: Some(&s4),
            ..Default::default()
        };
        let (retry, decision) = policy.select_with_decision(only, &info).await;
        assert_eq!((retry, decision.branch), (Some(0), "rebind"));
    }

    #[tokio::test]
    async fn retries_avoid_every_worker_the_session_failed_on_until_a_hit() {
        let policy = DpStickyPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = ["http://a:8000", "http://b:8000", "http://c:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        // Worker c holds two other sessions, so a and b are lighter.
        for other in ["other-1", "other-2"] {
            route(&policy, &workers[2..], "regular", other).await;
        }
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
        let s = headers("s");

        // a fails, then b: the next retry goes to c, not back to a.
        policy.on_dispatch_failed(Some(&s), &workers[0]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 1);
        policy.on_dispatch_failed(Some(&s), &workers[1]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 2);

        // A hit on c clears the set: a failure on c then avoids c alone.
        assert_eq!(route(&policy, &workers, "regular", "s").await, 2);
        policy.on_dispatch_failed(Some(&s), &workers[2]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
    }

    #[tokio::test]
    async fn a_late_failure_on_the_previous_worker_keeps_the_new_binding() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        workers[0].set_healthy(false);
        assert_eq!(branch(&policy, &workers, &info).await, "rebind");
        // A request still running on worker 0 fails, before and after the
        // rebound session's next request is sent to worker 1.
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        policy.on_dispatch("regular", Some(&headers), &workers[1]);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        workers[0].set_healthy(true);
        let (selected, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "hit"));
    }

    #[tokio::test]
    async fn a_failure_on_the_mapped_rank_places_the_session_again() {
        let policy = DpStickyPolicy::new();
        let ranks = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // The pin is rank 0; rank mapping sends the request to rank 1.
        assert_eq!(policy.select_worker(&ranks, &info).await, Some(0));
        policy.on_dispatch("prefill", Some(&headers), &ranks[1]);
        policy.on_dispatch_failed(Some(&headers), &ranks[1]);
        assert_eq!(branch(&policy, &ranks, &info).await, "rebind");
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
