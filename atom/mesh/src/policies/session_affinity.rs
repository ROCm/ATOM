//! Session table around a load-balancing policy.
//!
//! The inner policy places a session's first request; for `dp_sticky` that is
//! [`LeastSessionsPolicy`], which reads the live session counts the table
//! passes in. Later requests stay on that worker until the idle TTL expires,
//! the worker is unusable or not a candidate (another DP rank under idx2idx), or
//! a request of the session fails. A failure counts
//! only for a request placed on the session's current binding, on the worker it
//! was sent to after DP rank mapping. Under ATOM PD idx2idx rank mapping, a
//! failed prefill also releases the decode binding, although the decode was
//! never sent.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;

use super::{dp_sticky::LeastSessionsPolicy, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
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
    /// Identifies this placement of the session. A request carries it from
    /// selection to its dispatch and failure hooks, so a request placed before
    /// the session moved cannot act on the new binding.
    binding: u64,
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

#[derive(Default)]
pub(crate) struct SessionTable {
    assignments: DashMap<String, StickyAssignment>,
    next_binding: AtomicU64,
}

impl SessionTable {
    /// Bind `key` to `worker` and return the new binding. A rebound session
    /// keeps where its last request was sent, so the next dispatch can tell
    /// whether the worker changed.
    fn assign(
        &self,
        key: &str,
        worker: WorkerIdentity,
        now: Instant,
        reassigned: Option<&'static str>,
    ) -> u64 {
        let binding = self.next_binding.fetch_add(1, Ordering::Relaxed);
        if let Some(mut entry) = self.assignments.get_mut(key) {
            entry.worker = worker;
            entry.binding = binding;
            entry.last_access = now;
            entry.reassigned = reassigned;
            entry.failed = false;
            return binding;
        }
        self.assignments.insert(
            key.to_string(),
            StickyAssignment {
                worker,
                binding,
                last_access: now,
                dispatched: None,
                reassigned,
                failed: false,
                avoid: Vec::new(),
            },
        );
        binding
    }

    /// Release `key`'s binding after a request placed on `binding` failed on
    /// `worker`. A request placed before the session moved keeps the new
    /// binding, whenever it fails.
    fn mark_failed(&self, key: &str, worker: &WorkerIdentity, binding: Option<u64>) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            if binding == Some(entry.binding) {
                entry.failed = true;
                if !entry.avoid.contains(worker) {
                    entry.avoid.push(worker.clone());
                }
            }
        }
    }

    /// Record where a request of `key` placed on `binding` was sent. Returns
    /// why the session moved when this worker differs from the one its previous
    /// request was sent to.
    fn record_dispatch(
        &self,
        key: &str,
        worker: WorkerIdentity,
        binding: Option<u64>,
    ) -> Option<&'static str> {
        let mut entry = self.assignments.get_mut(key)?;
        // A request placed before the session moved does not move it back.
        if binding != Some(entry.binding) {
            return None;
        }
        // Without a reassignment, only rank mapping can move a pinned session: a
        // request placed on an earlier binding returned above.
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
    /// Places a session without a usable binding, and every request without a
    /// session key.
    inner: Arc<dyn LoadBalancingPolicy>,
    /// Serializes count -> choose -> insert for a session without a usable
    /// binding, so a burst of new sessions does not land on one worker and the
    /// inner policy's pick is the one that is bound.
    assign_lock: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for SessionAffinityPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionAffinityPolicy")
            .field("ttl", &self.ttl)
            .field("inner", &self.inner.name())
            .finish_non_exhaustive()
    }
}

impl SessionAffinityPolicy {
    /// `dp_sticky`: sessions keyed on `x-session-id`, placed by fewest live sessions.
    pub(crate) fn dp_sticky() -> Self {
        Self::new(Arc::new(LeastSessionsPolicy))
    }

    fn new(inner: Arc<dyn LoadBalancingPolicy>) -> Self {
        Self {
            ttl: SESSION_REASSIGNMENT_IDLE_TIMEOUT,
            table: SessionTable::default(),
            inner,
            assign_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// The session's current binding.
    #[cfg(test)]
    pub(crate) fn binding(&self, key: &str) -> Option<u64> {
        self.table.assignments.get(key).map(|entry| entry.binding)
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

    /// The bound worker serves the request when the session is not idle for
    /// too long, no request failed since it was placed, and the worker is
    /// usable. A hit forgets the workers the session failed on. One entry
    /// lock covers the check and the binding read, so a placement in between
    /// cannot hand this request the new binding for the old worker.
    fn hit(
        &self,
        key: &str,
        workers: &[Arc<dyn Worker>],
        now: Instant,
    ) -> Option<(Option<usize>, RouteDecision)> {
        let mut assignment = self.table.assignments.get_mut(key)?;
        if assignment.failed || self.is_expired(&assignment, now) {
            return None;
        }
        let index = healthy_index(workers, &assignment.worker)?;
        assignment.last_access = now;
        assignment.avoid.clear();
        let decision = RouteDecision {
            binding: Some(assignment.binding),
            ..RouteDecision::branch(AFFINITY_HIT)
        };
        Some((Some(index), decision))
    }

    fn miss_result(&self, key: &str, now: Instant) -> &'static str {
        match self.table.assignments.get(key) {
            None => AFFINITY_NEW,
            Some(assignment) if self.is_expired(&assignment, now) => AFFINITY_EXPIRED,
            Some(_) => AFFINITY_REBIND,
        }
    }

    /// Live sessions bound to each of `workers`, in their order.
    fn session_counts(&self, workers: &[Arc<dyn Worker>], now: Instant) -> Vec<usize> {
        let live = self.table.live_counts(now, self.ttl);
        workers
            .iter()
            .map(|worker| {
                live.get(&WorkerIdentity::from_worker(worker))
                    .copied()
                    .unwrap_or(0)
            })
            .collect()
    }

    fn record_placement(
        &self,
        key: &str,
        worker_index: usize,
        workers: &[Arc<dyn Worker>],
        now: Instant,
        result: &'static str,
    ) -> u64 {
        let reassigned = (result != AFFINITY_NEW).then_some(result);
        self.table.assign(
            key,
            WorkerIdentity::from_worker(&workers[worker_index]),
            now,
            reassigned,
        )
    }

    /// Places `key` with the inner policy, away from the workers it failed on
    /// since its last hit while another one is healthy. Caller holds `assign_lock`.
    async fn place(
        &self,
        key: &str,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        now: Instant,
    ) -> (Option<usize>, RouteDecision) {
        let avoid = self
            .table
            .assignments
            .get(key)
            .map(|assignment| assignment.avoid.clone())
            .unwrap_or_default();
        let others = excluding(workers, &avoid);
        let candidates = others.as_deref().unwrap_or(workers);
        let counts = self.session_counts(candidates, now);
        let placing = SelectWorkerInfo {
            session_counts: Some(&counts),
            ..info.clone()
        };
        let (selected, decision) = self.inner.select_with_decision(candidates, &placing).await;
        let selected = selected.and_then(|index| match &others {
            Some(others) => workers
                .iter()
                .position(|worker| Arc::ptr_eq(worker, &others[index])),
            None => Some(index),
        });
        let Some(selected) = selected else {
            return (None, decision);
        };
        let result = self.miss_result(key, now);
        let binding = self.record_placement(key, selected, workers, now, result);
        let decision = if self.inner.reports_session_result() {
            RouteDecision::branch(result)
        } else {
            decision
        };
        let decision = RouteDecision {
            binding: Some(binding),
            ..decision
        };
        (Some(selected), decision)
    }
}

/// `workers` without those in `avoid`, if one of the others is healthy.
fn excluding(
    workers: &[Arc<dyn Worker>],
    avoid: &[WorkerIdentity],
) -> Option<Vec<Arc<dyn Worker>>> {
    if avoid.is_empty() {
        return None;
    }
    let others: Vec<_> = workers
        .iter()
        .filter(|worker| !avoid.contains(&WorkerIdentity::from_worker(worker)))
        .cloned()
        .collect();
    others
        .iter()
        .any(|worker| worker.is_healthy() && worker.circuit_breaker().can_execute())
        .then_some(others)
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
    /// A hit reports `hit`. A placement reports the inner policy's decision, or
    /// the session result (`new`, `rebind`, `expired`) for a policy that asks
    /// for it, like `dp_sticky`. A request without a session key is the inner
    /// policy's alone.
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let Some(key) = self.binding_key(info.headers) else {
            return self.inner.select_with_decision(workers, info).await;
        };
        let now = Instant::now();
        if let Some(hit) = self.hit(key, workers, now) {
            return hit;
        }

        let _assign_guard = self.assign_lock.lock().await;
        if let Some(hit) = self.hit(key, workers, now) {
            return hit;
        }
        self.place(key, workers, info, now).await
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        self.inner.on_request_complete(worker_url, success);
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
        binding: Option<u64>,
    ) {
        self.inner.on_dispatch(pool, headers, worker, binding);
        let Some(key) = self.binding_key(headers) else {
            return;
        };
        if let Some(reason) =
            self.table
                .record_dispatch(key, WorkerIdentity::from_worker(worker), binding)
        {
            MeshMetrics::record_session_worker_change(pool, reason);
        }
    }

    /// A failed request placed on the session's current binding releases it:
    /// the next request, a retry included, is placed again (`rebind`, or
    /// `expired` after the idle timeout), away from every worker the session
    /// failed on since its last hit while another one is healthy. The worker
    /// avoided is the one the request was sent to, after DP rank mapping; under
    /// ATOM PD idx2idx, a failed prefill also releases the decode binding,
    /// although the decode was never sent. A request placed before the session
    /// moved keeps the new binding, whenever it fails.
    fn on_dispatch_failed(
        &self,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
        binding: Option<u64>,
    ) {
        self.inner.on_dispatch_failed(headers, worker, binding);
        if let Some(key) = self.binding_key(headers) {
            self.table
                .mark_failed(key, &WorkerIdentity::from_worker(worker), binding);
        }
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn needs_request_text(&self) -> bool {
        self.inner.needs_request_text()
    }

    fn needs_tokens(&self) -> bool {
        self.inner.needs_tokens()
    }

    fn wants_tokens(&self) -> bool {
        self.inner.wants_tokens()
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        self.inner.update_loads(loads);
    }

    fn reset(&self) {
        self.inner.reset();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
