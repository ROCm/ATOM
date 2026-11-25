//! Data-parallel worker routing with session affinity.
//!
//! Requests carrying `X-Session-ID` are pinned to the first healthy worker
//! selected for that session. A new session goes to the worker holding the
//! fewest live sessions (load breaks ties); requests without a session ID use
//! minimum-load balancing. A stale mapping is replaced when its worker is no
//! longer healthy or a candidate (another DP rank under idx2idx), or the session
//! has been idle for too long.
//!
//! With a rebalance threshold, affinity is bounded by load: a session moves to
//! the least-loaded worker when its own runs more than `1 + threshold` times
//! that load and at least two more requests, and new sessions go to the
//! least-loaded worker. Lock-step DP decode runs every rank at the pace of the
//! busiest one, so a crowded rank costs all of them.

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
            AFFINITY_EXPIRED, AFFINITY_HIT, AFFINITY_NEW, AFFINITY_REBALANCE, AFFINITY_REBIND,
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
    /// See the module docs; `None` keeps strict affinity.
    rebalance_threshold: Option<f32>,
}

impl DpStickyPolicy {
    pub fn new() -> Self {
        Self::with_rebalance_threshold(None)
    }

    pub fn with_rebalance_threshold(rebalance_threshold: Option<f32>) -> Self {
        Self {
            assignments: DashMap::new(),
            assign_lock: Mutex::new(()),
            rebalance_threshold,
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

    /// The least-loaded healthy worker when the pinned worker's load exceeds
    /// `1 + threshold` times it and by at least two: a lead of one would only
    /// move the imbalance once this request lands.
    fn rebalance_target(&self, workers: &[Arc<dyn Worker>], pinned: usize) -> Option<usize> {
        let threshold = f64::from(self.rebalance_threshold?);
        let target = Self::select_low_load_worker(workers)?;
        let (pinned_load, min_load) = (workers[pinned].load(), workers[target].load());
        (pinned_load > min_load + 1 && pinned_load as f64 > min_load as f64 * (1.0 + threshold))
            .then_some(target)
    }

    /// Route a request of a session whose pinned worker is still usable: the pin,
    /// or with a rebalance threshold the least-loaded worker, which becomes the pin.
    fn route_pinned(
        &self,
        workers: &[Arc<dyn Worker>],
        assignment: &mut StickyAssignment,
        pinned: usize,
        now: Instant,
    ) -> (Option<usize>, &'static str) {
        assignment.last_access = now;
        match self.rebalance_target(workers, pinned) {
            Some(target) => {
                assignment.worker = Self::worker_identity(&workers[target]);
                assignment.reassigned = Some(AFFINITY_REBALANCE);
                (Some(target), AFFINITY_REBALANCE)
            }
            None => (Some(pinned), AFFINITY_HIT),
        }
    }

    /// Select the healthy worker holding the fewest non-expired assignments,
    /// tie-broken by current request load; with a rebalance threshold, the
    /// least-loaded worker, tie-broken by assignments. Caller must hold
    /// `assign_lock`.
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

        let by_load = self.rebalance_threshold.is_some();
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| {
                let identity = Self::worker_identity(&workers[index]);
                let sessions = live_sessions.get(&identity).copied().unwrap_or(0);
                let load = workers[index].load();
                if by_load {
                    (load, sessions)
                } else {
                    (sessions, load)
                }
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
                    return self.route_pinned(workers, &mut assignment, index, now);
                }
            }
        }

        // New, expired, or orphaned sessions must assign atomically: another
        // request may have pinned this session while this one waited for the lock.
        let _assign_guard = self
            .assign_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let (outcome, dispatched) = match self.assignments.get_mut(session_id) {
            None => (AFFINITY_NEW, None),
            Some(mut assignment) => {
                let is_expired = now.saturating_duration_since(assignment.last_access)
                    > SESSION_REASSIGNMENT_IDLE_TIMEOUT;
                if is_expired {
                    (AFFINITY_EXPIRED, assignment.dispatched.clone())
                } else if let Some(index) =
                    Self::healthy_worker_for_identity(workers, &assignment.worker)
                {
                    return self.route_pinned(workers, &mut assignment, index, now);
                } else {
                    (AFFINITY_REBIND, assignment.dispatched.clone())
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
                dispatched,
                reassigned: (outcome != AFFINITY_NEW).then_some(outcome),
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
        // Without a reassignment, rank mapping moved the pinned session, or another in-flight
        // request of the session took the reassignment.
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

    fn dp4_workers() -> Vec<Arc<dyn Worker>> {
        (0..4)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://decode:8000", rank, 4).build())
                    as Arc<dyn Worker>
            })
            .collect()
    }

    fn set_loads(workers: &[Arc<dyn Worker>], loads: &[usize]) {
        for (worker, &load) in workers.iter().zip(loads) {
            worker.reset_load();
            (0..load).for_each(|_| worker.increment_load());
        }
    }

    #[tokio::test]
    async fn strict_affinity_ignores_load_without_a_threshold() {
        let policy = DpStickyPolicy::new();
        let ranks = dp4_workers();
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);
        set_loads(&ranks, &[110, 93, 46, 110]);
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);
    }

    #[tokio::test]
    async fn rebalance_moves_a_session_off_an_overloaded_rank_and_repins_it() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::with_rebalance_threshold(Some(0.2));
        let ranks = dp4_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);
        // Rank 0 runs 110 > 46 * 1.2: the session moves to rank 2 and stays there.
        set_loads(&ranks, &[110, 93, 46, 110]);
        let (selected, decision) = policy.select_with_decision(&ranks, &info).await;
        assert_eq!((selected, decision.branch), (Some(2), AFFINITY_REBALANCE));
        policy.on_dispatch("decode", Some(&headers), &ranks[2]);
        set_loads(&ranks, &[110, 93, 47, 110]);
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 2);

        assert_eq!(
            worker_changes(&handle.render()),
            [r#"mesh_session_worker_change_total{pool="decode",reason="rebalance"} 1"#]
        );
    }

    #[tokio::test]
    async fn rebalance_keeps_affinity_within_the_bound() {
        let policy = DpStickyPolicy::with_rebalance_threshold(Some(0.2));
        let ranks = dp4_workers();
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);

        // 60 <= 50 * 1.2: within the relative bound.
        set_loads(&ranks, &[60, 55, 50, 58]);
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);
        // A lead of one would only move the imbalance: 2 > 1 * 1.2 but stays.
        set_loads(&ranks, &[2, 1, 1, 1]);
        assert_eq!(route(&policy, &ranks, "decode", "session-1").await, 0);
        // Threshold 0 still needs a lead of two.
        let strict = DpStickyPolicy::with_rebalance_threshold(Some(0.0));
        set_loads(&ranks, &[0, 0, 0, 0]);
        assert_eq!(route(&strict, &ranks, "decode", "session-2").await, 0);
        set_loads(&ranks, &[3, 2, 2, 2]);
        assert_eq!(route(&strict, &ranks, "decode", "session-2").await, 0);
        set_loads(&ranks, &[3, 2, 1, 2]);
        assert_eq!(route(&strict, &ranks, "decode", "session-2").await, 2);
    }

    #[tokio::test]
    async fn rebalance_places_new_sessions_by_load() {
        let ranks = dp4_workers();
        let strict = DpStickyPolicy::new();
        let bounded = DpStickyPolicy::with_rebalance_threshold(Some(0.2));
        // Ranks 1-3 hold one live session each, rank 0 none.
        for (session, rank) in [("a", 1), ("b", 2), ("c", 3)] {
            for policy in [&strict, &bounded] {
                policy.assignments.insert(
                    session.to_string(),
                    StickyAssignment {
                        worker: DpStickyPolicy::worker_identity(&ranks[rank]),
                        last_access: Instant::now(),
                        dispatched: None,
                        reassigned: None,
                    },
                );
            }
        }
        // Strict placement counts sessions; bounded placement counts load.
        set_loads(&ranks, &[5, 1, 9, 9]);
        assert_eq!(route(&strict, &ranks, "decode", "new").await, 0);
        assert_eq!(route(&bounded, &ranks, "decode", "new").await, 1);
    }
}
