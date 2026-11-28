//! Session table around a load-balancing policy.
//!
//! The inner policy places a session's first request; for `dp_sticky` that is
//! [`LeastSessionsPolicy`], which reads the live session counts the table
//! passes in. Later requests stay on that worker until the idle TTL expires,
//! the worker is unusable or not a candidate (another DP rank under idx2idx), or
//! a request of the session fails on its bound worker. Under ATOM PD idx2idx
//! rank mapping, a failed prefill also releases the decode binding, although
//! the decode was never sent.
//!
//! `hard` always returns the bound worker. `soft` does too, unless the inner
//! policy opts in via [`LoadBalancingPolicy::overrides_session_affinity`].

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::{mapref::entry::Entry, DashMap};

use super::{dp_sticky::LeastSessionsPolicy, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
use crate::{
    config::{SessionAffinityConfig, SessionAffinityMode, SessionAffinityPool, SessionParentMode},
    core::Worker,
    observability::metrics::{
        metrics_labels::{
            AFFINITY_EXPIRED, AFFINITY_HIT, AFFINITY_NEW, AFFINITY_REBIND,
            WORKER_CHANGE_RANK_MAPPING,
        },
        MeshMetrics,
    },
    routers::comm::header_utils::{
        extract_first_header, extract_header_value, PARENT_SESSION_HEADER,
    },
};

const SWEEP_INTERVAL: Duration = Duration::from_secs(30);
/// Prefixes a parent group key. The header values a key is read from are
/// visible ASCII, so no session id can start with this control character.
const PARENT_KEY_PREFIX: &str = "\u{1}parent:";

/// Which session table produced a decision. Prefill and decode are separate
/// counters so a healthy decode table cannot hide prefill misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionAffinityScope {
    Regular,
    Prefill,
    Decode,
}

impl SessionAffinityScope {
    pub(crate) fn pool(self) -> SessionAffinityPool {
        match self {
            Self::Regular => SessionAffinityPool::Regular,
            Self::Prefill => SessionAffinityPool::Prefill,
            Self::Decode => SessionAffinityPool::Decode,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

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
    /// A request of the session failed on its bound worker, so the next
    /// request, a retry included, is placed again.
    failed: bool,
    /// Workers the session's requests failed on since its last hit. Placing
    /// it again leaves them out while another worker is healthy.
    avoid: Vec<WorkerIdentity>,
}

#[derive(Default)]
pub(crate) struct SessionTable {
    assignments: DashMap<String, StickyAssignment>,
    /// Bindings each worker holds, kept with `assignments` so placing a session
    /// does not scan the table. An idle binding counts until a sweep drops it.
    counts: std::sync::Mutex<HashMap<WorkerIdentity, usize>>,
}

impl SessionTable {
    /// Bind `key` to `worker`. A rebound session
    /// keeps where its last request was sent, so the next dispatch can tell
    /// whether the worker changed.
    fn assign(
        &self,
        key: &str,
        worker: WorkerIdentity,
        now: Instant,
        reassigned: Option<&'static str>,
    ) {
        // The entry's shard lock is held while the count changes, so a sweep
        // cannot drop the binding in between.
        match self.assignments.entry(key.to_string()) {
            Entry::Occupied(mut entry) => {
                let entry = entry.get_mut();
                if entry.worker != worker {
                    let mut counts = self.counts();
                    uncount(&mut counts, &entry.worker);
                    *counts.entry(worker.clone()).or_insert(0) += 1;
                }
                entry.worker = worker;
                entry.last_access = now;
                entry.reassigned = reassigned;
                entry.failed = false;
            }
            Entry::Vacant(entry) => {
                *self.counts().entry(worker.clone()).or_insert(0) += 1;
                entry.insert(StickyAssignment {
                    worker,
                    last_access: now,
                    dispatched: None,
                    reassigned,
                    failed: false,
                    avoid: Vec::new(),
                });
            }
        }
    }

    fn counts(&self) -> std::sync::MutexGuard<'_, HashMap<WorkerIdentity, usize>> {
        self.counts
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Bindings each of `workers` holds, in their order, without the binding
    /// of `replacing`, the session being placed again.
    fn counts_for(&self, workers: &[Arc<dyn Worker>], replacing: Option<&str>) -> Vec<usize> {
        let own = replacing
            .and_then(|key| self.assignments.get(key))
            .map(|assignment| assignment.worker.clone());
        let counts = self.counts();
        workers
            .iter()
            .map(|worker| {
                let identity = WorkerIdentity::from_worker(worker);
                let count = counts.get(&identity).copied().unwrap_or(0);
                count.saturating_sub(usize::from(own.as_ref() == Some(&identity)))
            })
            .collect()
    }

    /// Drop the bindings `remove` selects. Counts change after the shards are
    /// released, so the count lock is never taken before a shard lock.
    fn remove_where(&self, mut remove: impl FnMut(&String, &StickyAssignment) -> bool) {
        let mut removed = Vec::new();
        self.assignments.retain(|key, assignment| {
            let drop = remove(key, assignment);
            if drop {
                removed.push(assignment.worker.clone());
            }
            !drop
        });
        if removed.is_empty() {
            return;
        }
        let mut counts = self.counts();
        for worker in &removed {
            uncount(&mut counts, worker);
        }
    }

    /// Drop the least recently used bindings, other than `keep`, until at most
    /// `target` remain. Returns how many were dropped.
    fn evict_least_recent(&self, target: usize, keep: &str) -> usize {
        let excess = self.assignments.len().saturating_sub(target);
        if excess == 0 {
            return 0;
        }
        let mut ages: Vec<Instant> = self
            .assignments
            .iter()
            .filter(|entry| entry.key() != keep)
            .map(|entry| entry.last_access)
            .collect();
        if ages.is_empty() {
            return 0;
        }
        let nth = excess.min(ages.len()) - 1;
        let cutoff = *ages.select_nth_unstable(nth).1;
        let mut evicted = 0;
        self.remove_where(|key, assignment| {
            let evict = evicted < excess && assignment.last_access <= cutoff && key != keep;
            evicted += usize::from(evict);
            evict
        });
        evicted
    }

    fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Release `key`'s binding after a request failed on `worker`, if it is the
    /// bound worker. A late failure on a worker the session has left keeps the
    /// new binding.
    fn mark_failed(&self, key: &str, worker: &WorkerIdentity) {
        if let Some(mut entry) = self.assignments.get_mut(key) {
            if entry.worker == *worker {
                entry.failed = true;
                if !entry.avoid.contains(worker) {
                    entry.avoid.push(worker.clone());
                }
            }
        }
    }

    /// Record where a request of `key` was sent. Returns
    /// why the session moved when this worker differs from the one its previous
    /// request was sent to.
    fn record_dispatch(&self, key: &str, worker: WorkerIdentity) -> Option<&'static str> {
        let mut entry = self.assignments.get_mut(key)?;
        // Without a reassignment, rank mapping moved the pinned session, or another
        // in-flight request of the session took the reassignment.
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

    /// Drop idle bindings. `keep` is left in place so the request being routed
    /// can still tell a new session from an expired one.
    fn retain_live(&self, now: Instant, ttl: Duration, keep: Option<&str>) {
        self.remove_where(|key, assignment| {
            keep != Some(key.as_str())
                && now.saturating_duration_since(assignment.last_access) > ttl
        });
    }
}

fn uncount(counts: &mut HashMap<WorkerIdentity, usize>, worker: &WorkerIdentity) {
    if let Some(count) = counts.get_mut(worker) {
        *count -= 1;
        if *count == 0 {
            counts.remove(worker);
        }
    }
}

pub struct SessionAffinityPolicy {
    config: SessionAffinityConfig,
    ttl: Duration,
    table: SessionTable,
    /// Places a session without a usable binding, and every request without a
    /// session key.
    inner: Arc<dyn LoadBalancingPolicy>,
    scope: SessionAffinityScope,
    /// Serializes count -> choose -> insert for a session without a usable
    /// binding, so a burst of new sessions does not land on one worker and the
    /// inner policy's pick is the one that is bound.
    assign_lock: tokio::sync::Mutex<()>,
    last_sweep: std::sync::Mutex<Instant>,
}

impl std::fmt::Debug for SessionAffinityPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionAffinityPolicy")
            .field("config", &self.config)
            .field("inner", &self.inner.name())
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl SessionAffinityPolicy {
    #[cfg(test)]
    pub(crate) fn around(
        config: SessionAffinityConfig,
        inner: Arc<dyn LoadBalancingPolicy>,
    ) -> Self {
        Self::around_scoped(config, inner, SessionAffinityScope::Regular)
    }

    pub(crate) fn around_scoped(
        config: SessionAffinityConfig,
        inner: Arc<dyn LoadBalancingPolicy>,
        scope: SessionAffinityScope,
    ) -> Self {
        Self::new(config, inner, scope)
    }

    /// `dp_sticky` with the default settings.
    pub(crate) fn dp_sticky() -> Self {
        Self::dp_sticky_scoped(
            SessionAffinityConfig::default(),
            SessionAffinityScope::Regular,
        )
    }

    /// `dp_sticky`: the table around [`LeastSessionsPolicy`].
    pub(crate) fn dp_sticky_scoped(
        config: SessionAffinityConfig,
        scope: SessionAffinityScope,
    ) -> Self {
        Self::new(config, Arc::new(LeastSessionsPolicy), scope)
    }

    fn new(
        config: SessionAffinityConfig,
        inner: Arc<dyn LoadBalancingPolicy>,
        scope: SessionAffinityScope,
    ) -> Self {
        if config.mode == SessionAffinityMode::Soft && !inner.overrides_session_affinity() {
            tracing::warn!(
                policy = inner.name(),
                scope = scope.as_str(),
                "session-affinity mode is soft, but this policy keeps the bound worker; only cache_aware moves sessions"
            );
        }
        Self {
            ttl: Duration::from_secs(config.ttl_secs),
            config,
            table: SessionTable::default(),
            inner,
            scope,
            assign_lock: tokio::sync::Mutex::new(()),
            last_sweep: std::sync::Mutex::new(
                Instant::now()
                    .checked_sub(SWEEP_INTERVAL)
                    .unwrap_or_else(Instant::now),
            ),
        }
    }

    #[cfg(test)]
    pub(crate) fn scope_label(&self) -> &'static str {
        self.scope.as_str()
    }

    #[cfg(test)]
    pub(crate) fn set_last_access(&self, key: &str, when: Instant) {
        if let Some(mut entry) = self.table.assignments.get_mut(key) {
            entry.last_access = when;
        }
    }

    /// The table key for a request: borrowed from its headers, except a parent
    /// group key, which is built.
    fn binding_key<'a>(&self, headers: Option<&'a http::HeaderMap>) -> Option<Cow<'a, str>> {
        let session = extract_first_header(headers, &self.config.session_id_headers);
        match self.config.parent_mode {
            SessionParentMode::Group => extract_header_value(headers, PARENT_SESSION_HEADER)
                .map(|parent_id| Cow::Owned(format!("{PARENT_KEY_PREFIX}{parent_id}")))
                .or(session.map(Cow::Borrowed)),
            SessionParentMode::Independent => session.map(Cow::Borrowed),
        }
    }

    fn is_expired(&self, assignment: &StickyAssignment, now: Instant) -> bool {
        now.saturating_duration_since(assignment.last_access) > self.ttl
    }

    /// The bound worker serves the request when the session is not idle for
    /// too long, no request failed since it was placed, and the worker is
    /// usable. A hit forgets the workers the session failed on. Returns the
    /// worker's index.
    fn hit(&self, key: &str, workers: &[Arc<dyn Worker>], now: Instant) -> Option<usize> {
        let mut assignment = self.table.assignments.get_mut(key)?;
        if assignment.failed || self.is_expired(&assignment, now) {
            return None;
        }
        let index = healthy_index(workers, &assignment.worker)?;
        assignment.last_access = now;
        assignment.avoid.clear();
        Some(index)
    }

    /// Counts a hit and routes the request to its bound worker.
    fn served(&self, index: usize) -> (Option<usize>, RouteDecision) {
        MeshMetrics::record_session_affinity(AFFINITY_HIT, self.scope.as_str());
        (Some(index), RouteDecision::branch(AFFINITY_HIT))
    }

    fn miss_result(&self, key: &str, now: Instant) -> &'static str {
        match self.table.assignments.get(key) {
            None => AFFINITY_NEW,
            Some(assignment) if self.is_expired(&assignment, now) => AFFINITY_EXPIRED,
            Some(_) => AFFINITY_REBIND,
        }
    }

    fn maybe_sweep(&self, now: Instant, keep: Option<&str>) {
        let interval = SWEEP_INTERVAL.min(self.ttl);
        let mut last = self
            .last_sweep
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if now.saturating_duration_since(*last) < interval {
            return;
        }
        *last = now;
        drop(last);
        self.table.retain_live(now, self.ttl, keep);
        MeshMetrics::set_session_affinity_entries(self.scope.as_str(), self.table.len());
    }

    /// Keeps the table within `max_entries`: idle bindings go first, then the
    /// least recently used tenth, so a full table does not evict on every new
    /// session. Caller holds `assign_lock`.
    fn enforce_capacity(&self, now: Instant, keep: &str) {
        let max = self.config.max_entries;
        if self.table.len() <= max {
            return;
        }
        self.table.retain_live(now, self.ttl, Some(keep));
        let evicted = self.table.evict_least_recent(max - max / 10, keep);
        if evicted > 0 {
            MeshMetrics::record_session_affinity_evictions(self.scope.as_str(), evicted);
        }
        MeshMetrics::set_session_affinity_entries(self.scope.as_str(), self.table.len());
    }

    fn allows_soft_move(&self) -> bool {
        self.config.mode == SessionAffinityMode::Soft && self.inner.overrides_session_affinity()
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
        if result == AFFINITY_NEW {
            self.enforce_capacity(now, key);
        }
        MeshMetrics::record_session_affinity(result, self.scope.as_str());
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
        let counts = self.table.counts_for(candidates, Some(key));
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
        self.record_placement(key, selected, workers, now, result);
        let decision = if self.inner.reports_session_result() {
            RouteDecision::branch(result)
        } else {
            decision
        };
        (Some(selected), decision)
    }

    /// Soft mode: the inner policy sees the bound worker of `hit` as
    /// `preferred_worker` and may pick another, which rebinds the session.
    async fn commit_soft_move(
        &self,
        key: &str,
        hit: usize,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        now: Instant,
    ) -> (Option<usize>, RouteDecision) {
        let mut delegated = info.clone();
        delegated.preferred_worker = Some(hit);
        let (chosen, decision) = self.inner.select_with_decision(workers, &delegated).await;
        let Some(chosen) = chosen else {
            return (None, decision);
        };
        if chosen == hit {
            return self.served(hit);
        }
        let _assign_guard = self.assign_lock.lock().await;
        self.record_placement(key, chosen, workers, now, AFFINITY_REBIND);
        (Some(chosen), decision)
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

/// Downcast through the session-affinity wrapper so cache-aware tree seeding
/// still sees the inner policy.
pub(crate) fn as_cache_aware(policy: &dyn LoadBalancingPolicy) -> Option<&super::CacheAwarePolicy> {
    if let Some(wrapper) = policy.as_any().downcast_ref::<SessionAffinityPolicy>() {
        return wrapper.inner.as_any().downcast_ref();
    }
    policy.as_any().downcast_ref()
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
        self.maybe_sweep(now, Some(&*key));
        if let Some(hit) = self.hit(&key, workers, now) {
            if self.allows_soft_move() {
                return self.commit_soft_move(&key, hit, workers, info, now).await;
            }
            self.inner.on_session_hit(workers, info, hit);
            return self.served(hit);
        }

        let _assign_guard = self.assign_lock.lock().await;
        if let Some(hit) = self.hit(&key, workers, now) {
            return self.served(hit);
        }
        self.place(&key, workers, info, now).await
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        self.inner.on_request_complete(worker_url, success);
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        self.inner.on_dispatch(pool, headers, worker);
        let Some(key) = self.binding_key(headers) else {
            return;
        };
        if let Some(reason) = self
            .table
            .record_dispatch(&key, WorkerIdentity::from_worker(worker))
        {
            MeshMetrics::record_session_worker_change(pool, reason);
        }
    }

    /// A failure on the session's bound worker releases the binding: the next
    /// request, a retry included, is placed again (`rebind`, or `expired`
    /// after the idle timeout), away from every worker the session failed on
    /// since its last hit while another one is healthy. Under ATOM PD idx2idx,
    /// a failed prefill also releases the decode binding, although the decode
    /// was never sent. A late failure on a worker the session has left keeps
    /// the new binding; one from an older placement on the worker the session
    /// is bound to again releases it, at the cost of one extra rebind.
    fn on_dispatch_failed(&self, headers: Option<&http::HeaderMap>, worker: &Arc<dyn Worker>) {
        self.inner.on_dispatch_failed(headers, worker);
        if let Some(key) = self.binding_key(headers) {
            self.table
                .mark_failed(&key, &WorkerIdentity::from_worker(worker));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::dp_sticky::NO_SESSION;
    use crate::{
        config::{SessionAffinityMode, SessionParentMode},
        core::{BasicWorkerBuilder, WorkerType},
    };

    fn dp_sticky(config: SessionAffinityConfig) -> SessionAffinityPolicy {
        SessionAffinityPolicy::dp_sticky_scoped(config, SessionAffinityScope::Regular)
    }

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

    fn header_map(pairs: &[(&str, &str)]) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                http::HeaderValue::from_str(value).unwrap(),
            );
        }
        headers
    }

    fn info(headers: &http::HeaderMap) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            headers: Some(headers),
            ..Default::default()
        }
    }

    /// The selected worker and branch for a request with `headers`.
    async fn route(
        policy: &SessionAffinityPolicy,
        workers: &[Arc<dyn Worker>],
        headers: &http::HeaderMap,
    ) -> (Option<usize>, &'static str) {
        let (selected, decision) = policy.select_with_decision(workers, &info(headers)).await;
        (selected, decision.branch)
    }

    fn bound(policy: &SessionAffinityPolicy, key: &str) -> bool {
        policy.table.assignments.contains_key(key)
    }

    #[tokio::test]
    async fn default_headers_use_the_first_non_empty_key() {
        let policy = dp_sticky(SessionAffinityConfig::default());
        let workers = workers();
        // agentx-harness 1.0.13 sends x-session-affinity by default, and no x-session-id.
        let affinity = header_map(&[("x-session-affinity", "a"), ("x-correlation-id", "a")]);
        assert_eq!(
            route(&policy, &workers, &affinity).await,
            (Some(0), AFFINITY_NEW)
        );
        assert_eq!(
            route(&policy, &workers, &affinity).await,
            (Some(0), AFFINITY_HIT)
        );

        let session_id = header_map(&[("x-session-id", "b"), ("x-session-affinity", "a")]);
        assert_eq!(
            route(&policy, &workers, &session_id).await,
            (Some(1), AFFINITY_NEW)
        );
        let empty_session_id = header_map(&[("x-session-id", ""), ("x-session-affinity", "a")]);
        assert_eq!(
            route(&policy, &workers, &empty_session_id).await,
            (Some(0), AFFINITY_HIT)
        );
    }

    #[tokio::test]
    async fn correlation_id_alone_does_not_enter_the_table() {
        let policy = dp_sticky(SessionAffinityConfig::default());
        let headers = header_map(&[("x-correlation-id", "req-1")]);
        assert_eq!(
            route(&policy, &workers(), &headers).await,
            (Some(0), NO_SESSION)
        );
        assert_eq!(policy.table.len(), 0);
    }

    #[tokio::test]
    async fn sweep_drops_idle_bindings_and_a_returning_session_is_new() {
        let config = SessionAffinityConfig {
            ttl_secs: 30,
            ..SessionAffinityConfig::default()
        };
        let policy = dp_sticky(config);
        let workers = workers();
        let session_a = header_map(&[("x-session-id", "session-a")]);
        let session_b = header_map(&[("x-session-id", "session-b")]);
        assert_eq!(route(&policy, &workers, &session_a).await.1, AFFINITY_NEW);
        assert_eq!(route(&policy, &workers, &session_b).await.1, AFFINITY_NEW);
        let idle = Instant::now() - Duration::from_secs(31);
        policy.set_last_access("session-a", idle);
        policy.set_last_access("session-b", idle);
        *policy.last_sweep.lock().unwrap() = idle;

        // The sweep keeps session-a's binding for the request being routed.
        assert_eq!(
            route(&policy, &workers, &session_a).await.1,
            AFFINITY_EXPIRED
        );
        assert!(bound(&policy, "session-a"));
        assert!(!bound(&policy, "session-b"));
        assert_eq!(route(&policy, &workers, &session_b).await.1, AFFINITY_NEW);
    }

    #[tokio::test]
    async fn an_older_failure_after_a_sweep_places_the_session_again() {
        let policy = dp_sticky(SessionAffinityConfig {
            ttl_secs: 30,
            ..SessionAffinityConfig::default()
        });
        let workers = workers();
        let workers = &workers[..1];
        let session_a = header_map(&[("x-session-id", "session-a")]);
        policy.select_worker(workers, &info(&session_a)).await;
        let idle = Instant::now() - Duration::from_secs(31);
        policy.set_last_access("session-a", idle);
        *policy.last_sweep.lock().unwrap() = idle;
        let session_b = header_map(&[("x-session-id", "session-b")]);
        route(&policy, workers, &session_b).await;
        assert!(!bound(&policy, "session-a"));

        // session-a returns to the same worker, then its request from before
        // the sweep fails there: the worker check cannot tell the placements
        // apart, so the session is placed once more.
        assert_eq!(
            route(&policy, workers, &session_a).await,
            (Some(0), AFFINITY_NEW)
        );
        policy.on_dispatch_failed(Some(&session_a), &workers[0]);
        assert_eq!(
            route(&policy, workers, &session_a).await,
            (Some(0), AFFINITY_REBIND)
        );
    }

    #[tokio::test]
    async fn group_mode_shares_one_binding_across_a_parents_children() {
        for (parent_mode, second_child) in [
            (SessionParentMode::Group, Some(1)),
            (SessionParentMode::Independent, Some(0)),
        ] {
            let policy = dp_sticky(SessionAffinityConfig {
                parent_mode,
                ..SessionAffinityConfig::default()
            });
            let workers = workers();
            let child = |id| {
                header_map(&[
                    ("x-dynamo-session-id", id),
                    ("x-dynamo-parent-session-id", "root"),
                ])
            };
            // The parent session keeps its own binding in both modes.
            let parent = header_map(&[("x-dynamo-session-id", "root")]);
            assert_eq!(route(&policy, &workers, &parent).await.0, Some(0));
            assert_eq!(route(&policy, &workers, &child("child-a")).await.0, Some(1));
            assert_eq!(
                route(&policy, &workers, &child("child-b")).await.0,
                second_child,
                "{parent_mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_session_id_cannot_join_a_parent_group() {
        let grouped = SessionAffinityConfig {
            parent_mode: SessionParentMode::Group,
            ..SessionAffinityConfig::default()
        };
        let policy = dp_sticky(grouped);
        let workers = workers();
        let child = header_map(&[
            ("x-dynamo-session-id", "child-a"),
            ("x-dynamo-parent-session-id", "root"),
        ]);
        assert_eq!(
            route(&policy, &workers, &child).await,
            (Some(0), AFFINITY_NEW)
        );

        // The group's key cannot be spelled as a session id.
        let lookalike = header_map(&[("x-session-id", "parent:root")]);
        assert_eq!(
            route(&policy, &workers, &lookalike).await,
            (Some(1), AFFINITY_NEW)
        );
    }

    #[test]
    fn session_counts_follow_binds_rebinds_sweeps_and_evictions() {
        let table = SessionTable::default();
        let workers = workers();
        let identity = |index: usize| WorkerIdentity::from_worker(&workers[index]);
        let now = Instant::now();
        table.assign("a", identity(0), now, None);
        table.assign("b", identity(0), now, None);
        table.assign("c", identity(1), now, None);
        assert_eq!(table.counts_for(&workers, None), [2, 1]);

        table.assign("b", identity(1), now, Some(AFFINITY_REBIND));
        assert_eq!(table.counts_for(&workers, None), [1, 2]);

        let idle = now - Duration::from_secs(60);
        table.assignments.get_mut("a").unwrap().last_access = idle;
        table.retain_live(now, Duration::from_secs(30), None);
        assert_eq!(table.counts_for(&workers, None), [0, 2]);

        table.assignments.get_mut("c").unwrap().last_access = idle;
        assert_eq!(table.evict_least_recent(1, "b"), 1);
        assert_eq!(table.counts_for(&workers, None), [0, 1]);
        // A session placed again does not count its own binding.
        assert_eq!(table.counts_for(&workers, Some("b")), [0, 0]);
    }

    #[tokio::test]
    async fn a_full_table_drops_its_least_recently_used_bindings() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let config = SessionAffinityConfig {
            max_entries: 2,
            ..SessionAffinityConfig::default()
        };
        let policy = dp_sticky(config);
        let workers = workers();
        for session in ["session-a", "session-b"] {
            route(&policy, &workers, &header_map(&[("x-session-id", session)])).await;
        }
        policy.set_last_access("session-a", Instant::now() - Duration::from_secs(10));

        route(
            &policy,
            &workers,
            &header_map(&[("x-session-id", "session-c")]),
        )
        .await;

        assert!(!bound(&policy, "session-a"));
        assert!(bound(&policy, "session-b"));
        assert!(bound(&policy, "session-c"));
        let rendered = handle.render();
        assert!(
            rendered.contains(r#"mesh_session_affinity_evicted_total{scope="regular"} 1"#),
            "{rendered}"
        );
        assert!(
            rendered.contains(r#"mesh_session_affinity_entries{scope="regular"} 2"#),
            "{rendered}"
        );
    }

    /// The samples of `mesh_session_affinity_total` in `rendered`, sorted.
    fn affinity_samples(rendered: &str) -> Vec<&str> {
        let mut samples: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("mesh_session_affinity_total{"))
            .collect();
        samples.sort_unstable();
        samples
    }

    #[derive(Debug)]
    struct MoveOffPreferred;

    #[async_trait]
    impl LoadBalancingPolicy for MoveOffPreferred {
        async fn select_with_decision(
            &self,
            workers: &[Arc<dyn Worker>],
            info: &SelectWorkerInfo<'_>,
        ) -> (Option<usize>, RouteDecision) {
            let selected = match info.preferred_worker {
                Some(preferred) if workers.len() > 1 => usize::from(preferred == 0),
                Some(preferred) => preferred,
                None => 0,
            };
            (Some(selected), RouteDecision::branch("move_off_preferred"))
        }

        fn overrides_session_affinity(&self) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            "move_off_preferred"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn hard_mode_keeps_binding_when_inner_would_move() {
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(MoveOffPreferred),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        assert_eq!(route(&policy, &workers, &headers).await.0, Some(0));
        assert_eq!(
            route(&policy, &workers, &headers).await,
            (Some(0), AFFINITY_HIT)
        );
    }

    #[tokio::test]
    async fn a_wrapped_policy_places_a_retry_away_from_the_worker_it_failed_on() {
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(MoveOffPreferred),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        assert_eq!(route(&policy, &workers, &headers).await.0, Some(0));
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        // The inner policy picks its first candidate, which is now worker 1.
        assert_eq!(route(&policy, &workers, &headers).await.0, Some(1));
    }

    #[tokio::test]
    async fn wrapped_policy_keeps_its_branch_on_placement_and_counts_worker_changes() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(crate::policies::RandomPolicy::new()),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);

        workers[1].set_healthy(false);
        assert_eq!(
            route(&policy, &workers, &headers).await,
            (Some(0), "random")
        );
        policy.on_dispatch("regular", Some(&headers), &workers[0]);

        workers[1].set_healthy(true);
        assert_eq!(
            route(&policy, &workers, &headers).await,
            (Some(0), AFFINITY_HIT)
        );
        policy.on_dispatch("regular", Some(&headers), &workers[0]);

        workers[0].set_healthy(false);
        assert_eq!(
            route(&policy, &workers, &headers).await,
            (Some(1), "random")
        );
        policy.on_dispatch("regular", Some(&headers), &workers[1]);

        assert_eq!(
            affinity_samples(&handle.render()),
            [
                r#"mesh_session_affinity_total{result="hit",scope="regular"} 1"#,
                r#"mesh_session_affinity_total{result="new",scope="regular"} 1"#,
                r#"mesh_session_affinity_total{result="rebind",scope="regular"} 1"#,
            ]
        );
        let changes: Vec<_> = handle
            .render()
            .lines()
            .filter(|line| line.starts_with("mesh_session_worker_change_total{"))
            .map(str::to_string)
            .collect();
        assert_eq!(
            changes,
            [r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#]
        );
    }

    #[tokio::test]
    async fn soft_mode_lets_cache_aware_move_an_overloaded_session() {
        for mode in [SessionAffinityMode::Hard, SessionAffinityMode::Soft] {
            let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
            let handle = recorder.handle();
            let _recorder = metrics::set_default_local_recorder(&recorder);
            let cache_aware = Arc::new(crate::policies::CacheAwarePolicy::with_config(
                crate::policies::CacheAwareConfig {
                    cache_threshold: 0.5,
                    balance_abs_threshold: 2,
                    balance_rel_threshold: 1.5,
                    eviction_interval_secs: 0,
                    max_tree_size: 1000,
                },
            ));
            let workers = workers();
            cache_aware.init_workers(&workers);
            let config = SessionAffinityConfig {
                mode,
                ..SessionAffinityConfig::default()
            };
            let policy = SessionAffinityPolicy::around(config, cache_aware.clone());
            let headers = header_map(&[("x-session-id", "session-1")]);
            let turn = |text: &'static str| SelectWorkerInfo {
                request_text: Some(text),
                headers: Some(&headers),
                ..Default::default()
            };

            // Worker 1 is busier, so the first turn goes to worker 0.
            workers[1].increment_load();
            assert_eq!(policy.select_worker(&workers, &turn("a")).await, Some(0));

            // Balanced: both modes keep the session.
            let (kept, decision) = policy
                .select_with_decision(&workers, &turn("abcdefghij"))
                .await;
            assert_eq!((kept, decision.branch), (Some(0), AFFINITY_HIT), "{mode:?}");
            // The kept turn reached the tree in both modes: a request without a
            // session key matches its whole prompt on worker 0.
            let probe = SelectWorkerInfo {
                request_text: Some("abcdefghij"),
                ..Default::default()
            };
            let (owner, decision) = cache_aware.select_with_decision(&workers, &probe).await;
            assert_eq!((owner, decision.branch), (Some(0), "affinity"), "{mode:?}");

            // Worker 0 is over both thresholds against worker 1.
            for _ in 0..5 {
                workers[0].increment_load();
            }
            let (selected, decision) = policy
                .select_with_decision(&workers, &turn("abcdefghijk"))
                .await;
            let expected = match mode {
                SessionAffinityMode::Hard => (Some(0), AFFINITY_HIT),
                SessionAffinityMode::Soft => (Some(1), "imbalanced"),
            };
            assert_eq!((selected, decision.branch), expected);
            let rebind = r#"mesh_session_affinity_total{result="rebind",scope="regular"} 1"#;
            assert_eq!(
                handle.render().contains(rebind),
                mode == SessionAffinityMode::Soft,
                "{mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn soft_mode_commits_inner_rebind() {
        let config = SessionAffinityConfig {
            mode: SessionAffinityMode::Soft,
            ..SessionAffinityConfig::default()
        };
        let policy = SessionAffinityPolicy::around(config, Arc::new(MoveOffPreferred));
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        // The move was committed: worker 1 is now the preferred worker.
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
    }

    #[tokio::test]
    async fn wrapped_cache_aware_keeps_its_name_and_downcast() {
        let policy = crate::policies::PolicyFactory::create_in(
            &crate::config::PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 30,
                max_tree_size: 1000,
            },
            &SessionAffinityConfig {
                pools: vec![SessionAffinityPool::Regular],
                ..SessionAffinityConfig::default()
            },
            SessionAffinityScope::Regular,
        );
        assert_eq!(policy.name(), "cache_aware");
        assert!(as_cache_aware(policy.as_ref()).is_some());
    }

    fn soft_around(inner: Arc<dyn LoadBalancingPolicy>) -> SessionAffinityPolicy {
        let config = SessionAffinityConfig {
            mode: SessionAffinityMode::Soft,
            ..SessionAffinityConfig::default()
        };
        SessionAffinityPolicy::around(config, inner)
    }

    fn released(policy: &SessionAffinityPolicy, key: &str) -> bool {
        policy
            .table
            .assignments
            .get(key)
            .is_some_and(|assignment| assignment.failed)
    }

    #[tokio::test]
    async fn a_wrapped_policy_keeps_the_new_binding_when_an_earlier_request_fails() {
        let policy = SessionAffinityPolicy::around(
            SessionAffinityConfig::default(),
            Arc::new(MoveOffPreferred),
        );
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);

        // A PD relay request reserves decode worker 0; its decode is sent
        // after its prefill.
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        // Meanwhile another request of the session hits worker 0 and fails,
        // and the next one is placed on worker 1 and sent there.
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        policy.on_dispatch("decode", Some(&headers), &workers[1]);
        // The first request's decode is now sent to worker 0 and fails.
        policy.on_dispatch("decode", Some(&headers), &workers[0]);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);

        assert_eq!(
            route(&policy, &workers, &headers).await,
            (Some(1), AFFINITY_HIT)
        );
    }

    #[tokio::test]
    async fn a_soft_move_keeps_its_binding_when_a_request_left_behind_fails() {
        let policy = soft_around(Arc::new(MoveOffPreferred));
        let workers = workers();
        let headers = header_map(&[("x-session-id", "session-1")]);
        let info = info(&headers);

        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        // The inner policy moves the session off worker 0.
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        policy.on_dispatch("regular", Some(&headers), &workers[1]);
        // The request placed before the move is sent to worker 0 and fails.
        policy.on_dispatch("regular", Some(&headers), &workers[0]);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        assert!(!released(&policy, "session-1"));
        // The moved request's failure releases the session.
        policy.on_dispatch_failed(Some(&headers), &workers[1]);
        assert!(released(&policy, "session-1"));
    }
}
