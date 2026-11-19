//! Session table for `dp_sticky`.
//!
//! The first request for a session goes to the healthy worker with the fewest
//! live sessions, then the lowest load. Later requests stay on that worker
//! until the idle TTL expires, the worker is unusable, or a request of the
//! session fails on it.

use std::{
    collections::HashMap,
    sync::Arc,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WorkerIdentity {
    url: String,
    dp_rank: Option<usize>,
}

impl WorkerIdentity {
    fn from_worker(worker: &Arc<dyn Worker>) -> Self {
        Self {
            url: worker.url().to_string(),
            dp_rank: worker.dp_rank(),
        }
    }
}

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
}

#[derive(Default)]
pub(crate) struct SessionTable {
    assignments: DashMap<String, StickyAssignment>,
}

impl SessionTable {
    /// Bind `key` to `worker`. A rebound session keeps where its last request
    /// was sent, so the next dispatch can tell whether the worker changed.
    fn assign(
        &self,
        key: &str,
        worker: WorkerIdentity,
        now: Instant,
        reassigned: Option<&'static str>,
    ) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            entry.worker = worker;
            entry.last_access = now;
            entry.reassigned = reassigned;
            entry.failed = false;
            return;
        }
        self.assignments.insert(
            key.to_string(),
            StickyAssignment {
                worker,
                last_access: now,
                dispatched: None,
                reassigned,
                failed: false,
            },
        );
    }

    fn mark_failed(&self, key: &str) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            entry.failed = true;
        }
    }

    fn touch(&self, key: &str, now: Instant) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            entry.last_access = now;
        }
    }

    /// Record where a request of `key` was sent. Returns why the session moved
    /// when this worker differs from the one its previous request was sent to.
    fn record_dispatch(&self, key: &str, worker: WorkerIdentity) -> Option<&'static str> {
        let mut entry = self.assignments.get_mut(key)?;
        // Without a reassignment, only rank mapping can move a pinned session.
        let reason = entry
            .reassigned
            .take()
            .unwrap_or(WORKER_CHANGE_RANK_MAPPING);
        let moved = entry
            .dispatched
            .as_ref()
            .is_some_and(|previous| *previous != worker);
        entry.dispatched = Some(worker);
        moved.then_some(reason)
    }

    fn live_counts(&self, now: Instant, ttl: Duration) -> HashMap<WorkerIdentity, usize> {
        let mut live = HashMap::new();
        for entry in self.assignments.iter() {
            if now.saturating_duration_since(entry.last_access) <= ttl {
                *live.entry(entry.worker.clone()).or_insert(0) += 1;
            }
        }
        live
    }
}

pub struct SessionAffinityPolicy {
    ttl: Duration,
    table: SessionTable,
    /// Serializes count -> choose -> insert for a session without a usable
    /// binding, so a burst of new sessions does not land on one worker.
    assign_lock: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for SessionAffinityPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionAffinityPolicy")
            .field("ttl", &self.ttl)
            .finish_non_exhaustive()
    }
}

impl SessionAffinityPolicy {
    pub(crate) fn dp_sticky() -> Self {
        Self {
            ttl: SESSION_REASSIGNMENT_IDLE_TIMEOUT,
            table: SessionTable::default(),
            assign_lock: tokio::sync::Mutex::new(()),
        }
    }

    #[cfg(test)]
    pub(crate) fn set_last_access(&self, key: &str, when: Instant) {
        if let Some(mut entry) = self.table.assignments.get_mut(key) {
            entry.last_access = when;
        }
    }

    /// The table key for a request.
    fn binding_key<'a>(&self, headers: Option<&'a http::HeaderMap>) -> Option<&'a str> {
        extract_sticky_routing_key(headers)
    }

    fn is_expired(&self, assignment: &StickyAssignment, now: Instant) -> bool {
        now.saturating_duration_since(assignment.last_access) > self.ttl
    }

    fn fresh_index(&self, key: &str, workers: &[Arc<dyn Worker>], now: Instant) -> Option<usize> {
        let assignment = self.table.assignments.get(key)?;
        if assignment.failed || self.is_expired(&assignment, now) {
            return None;
        }
        healthy_index(workers, &assignment.worker)
    }

    fn miss_result(&self, key: &str, now: Instant) -> &'static str {
        match self.table.assignments.get(key) {
            None => AFFINITY_NEW,
            Some(assignment) if self.is_expired(&assignment, now) => AFFINITY_EXPIRED,
            Some(_) => AFFINITY_REBIND,
        }
    }

    /// The healthy worker with the fewest non-expired sessions, then the lowest load.
    fn select_fewest(&self, workers: &[Arc<dyn Worker>], now: Instant) -> Option<usize> {
        let live = self.table.live_counts(now, self.ttl);
        get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| {
                let identity = WorkerIdentity::from_worker(&workers[index]);
                (
                    live.get(&identity).copied().unwrap_or(0),
                    workers[index].load(),
                )
            })
    }

    /// The bound worker at `index` serves the request.
    fn hit(&self, key: &str, index: usize, now: Instant) -> (Option<usize>, RouteDecision) {
        self.table.touch(key, now);
        (Some(index), RouteDecision::branch(AFFINITY_HIT))
    }

    fn record_placement(
        &self,
        key: &str,
        worker_index: usize,
        workers: &[Arc<dyn Worker>],
        now: Instant,
        result: &'static str,
    ) {
        let reassigned = (result != AFFINITY_NEW).then_some(result);
        self.table.assign(
            key,
            WorkerIdentity::from_worker(&workers[worker_index]),
            now,
            reassigned,
        );
    }
}

fn healthy_index(workers: &[Arc<dyn Worker>], identity: &WorkerIdentity) -> Option<usize> {
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

#[async_trait]
impl LoadBalancingPolicy for SessionAffinityPolicy {
    /// The branch is the affinity result (`hit`, `new`, `rebind`, `expired`),
    /// or `no_session` for a request without a session key, which goes to the
    /// least-loaded worker.
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let Some(key) = self.binding_key(info.headers) else {
            let selected = get_healthy_worker_indices(workers)
                .into_iter()
                .min_by_key(|&index| workers[index].load());
            let branch = if selected.is_some() {
                NO_SESSION
            } else {
                NO_HEALTHY_WORKERS
            };
            return (selected, RouteDecision::branch(branch));
        };
        let now = Instant::now();
        if let Some(index) = self.fresh_index(key, workers, now) {
            return self.hit(key, index, now);
        }

        let _assign_guard = self.assign_lock.lock().await;
        if let Some(index) = self.fresh_index(key, workers, now) {
            return self.hit(key, index, now);
        }
        let Some(selected) = self.select_fewest(workers, now) else {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        };
        let result = self.miss_result(key, now);
        self.record_placement(key, selected, workers, now, result);
        (Some(selected), RouteDecision::branch(result))
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        let Some(key) = self.binding_key(headers) else {
            return;
        };
        if let Some(reason) = self
            .table
            .record_dispatch(key, WorkerIdentity::from_worker(worker))
        {
            MeshMetrics::record_session_worker_change(pool, reason);
        }
    }

    /// A failed request releases its session's binding: the next request, a
    /// retry included, is placed again and counts as `rebind`.
    fn on_dispatch_failed(
        &self,
        _pool: &'static str,
        headers: Option<&http::HeaderMap>,
        _worker: &Arc<dyn Worker>,
    ) {
        if let Some(key) = self.binding_key(headers) {
            self.table.mark_failed(key);
        }
    }

    fn name(&self) -> &'static str {
        "dp_sticky"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
