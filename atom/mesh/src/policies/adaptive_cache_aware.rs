//! Experimental prefill routing by incremental cache/work cost.
//!
//! The first version uses uncached Unicode-character work, not seconds or token
//! counts. Outstanding work is a queue proxy (chunk progress is not available).
//! It assumes comparable P workers and one router. Completion usage calibrates
//! cache prediction error; it must not be used to infer service speed from TTFT.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;
use tracing::debug;

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LoadBalancingPolicy, PrefillFeedback, PrefillReservation, PrefillSelection, SelectWorkerInfo,
};
use crate::{
    core::{Worker, WorkerType},
    routers::comm::header_utils::extract_sticky_routing_key,
};

const EMA_ALPHA: f64 = 1.0 / 32.0;
const SESSION_TTL: Duration = Duration::from_secs(7200);
const MAX_SESSIONS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Identity {
    url: String,
    rank: Option<usize>,
}

impl Identity {
    fn of(worker: &dyn Worker) -> Self {
        Self {
            url: worker.url().into(),
            rank: worker.dp_rank(),
        }
    }

    fn tenant(&self) -> String {
        // Length-prefix the URL so endpoint text cannot collide with a rank.
        format!("{}:{}:{:?}", self.url.len(), self.url, self.rank)
    }
}

#[derive(Debug)]
struct Assignment {
    worker: Identity,
    generation: u64,
    confirmed: Instant,
}

#[derive(Debug)]
struct Pending {
    text: Arc<str>,
    input_chars: usize,
    work: f64,
    session: Option<[u8; 32]>,
    dispatched: bool,
}

#[derive(Debug, Default)]
struct WorkerWork {
    pending: HashMap<u64, Pending>,
    cache_error: f64,
    observations: u64,
}

impl WorkerWork {
    fn queued_work(&self) -> f64 {
        self.pending.values().map(|p| p.work).sum()
    }
}

#[derive(Debug, Default)]
struct State {
    next_id: u64,
    workers: HashMap<Identity, WorkerWork>,
    sessions: HashMap<[u8; 32], Assignment>,
    mean_work: Option<f64>,
}

#[derive(Debug, Default)]
struct Pool {
    tree: Tree,
    state: Mutex<State>,
}

#[derive(Debug)]
pub struct AdaptiveCacheAwarePolicy {
    pools: Arc<DashMap<String, Arc<Pool>>>,
    _eviction_task: Option<PeriodicTask>,
}

#[derive(Debug)]
struct WorkReservation {
    pool: Arc<Pool>,
    worker: Identity,
    id: u64,
}

impl PrefillReservation for WorkReservation {
    fn mark_dispatched(&mut self) {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pending) = state
            .workers
            .get_mut(&self.worker)
            .and_then(|w| w.pending.get_mut(&self.id))
        {
            pending.dispatched = true;
        }
    }

    fn complete(&mut self, feedback: PrefillFeedback) {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(worker) = state.workers.get_mut(&self.worker) else {
            return;
        };
        let Some(pending) = worker.pending.remove(&self.id) else {
            return;
        };
        // Invalid/missing usage never becomes a perfect-hit training sample.
        let actual_work = feedback
            .input_tokens
            .zip(feedback.cached_tokens)
            .filter(|&(input, cached)| input > 0 && cached <= input)
            .map(|(input, cached)| {
                (pending.input_chars as f64 * (input - cached) as f64 / input as f64).max(1.0)
            });
        if let Some(actual) = actual_work {
            let error = (actual - pending.work).abs();
            worker.cache_error = if worker.observations == 0 {
                error
            } else {
                worker.cache_error + EMA_ALPHA * (error - worker.cache_error)
            };
            worker.observations += 1;
            state.mean_work = Some(
                state
                    .mean_work
                    .map_or(actual, |mean| mean + EMA_ALPHA * (actual - mean)),
            );
        }
        // Only a confirmed P response populates the prefix tree. Cancellation,
        // failed dispatch, and an abandoned plan only drop the reservation.
        if !pending.text.is_empty() {
            self.pool.tree.insert(&pending.text, &self.worker.tenant());
        }
        if let Some(session) = pending.session {
            let now = Instant::now();
            if state.sessions.len() >= MAX_SESSIONS {
                state
                    .sessions
                    .retain(|_, entry| now.duration_since(entry.confirmed) <= SESSION_TTL);
            }
            let can_insert =
                state.sessions.contains_key(&session) || state.sessions.len() < MAX_SESSIONS;
            let newer = state
                .sessions
                .get(&session)
                .is_none_or(|old| self.id > old.generation);
            if can_insert && newer {
                state.sessions.insert(
                    session,
                    Assignment {
                        worker: self.worker.clone(),
                        generation: self.id,
                        confirmed: now,
                    },
                );
            }
        }
        debug!(worker = %self.worker.url, dp_rank = ?self.worker.rank, reservation_id = self.id,
            predicted_work = pending.work, actual_work = ?actual_work,
            "adaptive_cache_aware prefill completed");
    }
}

impl Drop for WorkReservation {
    fn drop(&mut self) {
        let mut state = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(worker) = state.workers.get_mut(&self.worker) {
            worker.pending.remove(&self.id);
        }
    }
}

#[derive(Debug)]
struct Candidate {
    index: usize,
    identity: Identity,
    cached: usize,
    work: f64,
    queue: f64,
    error: f64,
    untracked_requests: usize,
}

impl AdaptiveCacheAwarePolicy {
    pub fn new(eviction_interval_secs: u64, max_tree_size: usize) -> Self {
        let pools = Arc::new(DashMap::<String, Arc<Pool>>::new());
        let eviction_task = (eviction_interval_secs > 0).then(|| {
            let pools = pools.clone();
            PeriodicTask::spawn(
                eviction_interval_secs,
                "Adaptive cache eviction",
                move || {
                    for pool in pools.iter() {
                        pool.tree.evict_tenant_by_size(max_tree_size);
                        let now = Instant::now();
                        pool.state
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .sessions
                            .retain(|_, a| now.duration_since(a.confirmed) <= SESSION_TTL);
                    }
                },
            )
        });
        Self {
            pools,
            _eviction_task: eviction_task,
        }
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for pool in self.pools.iter() {
            let mut state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            state.workers.retain(|identity, _| {
                if identity.url == url {
                    pool.tree.remove_tenant(&identity.tenant());
                    false
                } else {
                    true
                }
            });
            state.sessions.retain(|_, a| a.worker.url != url);
        }
    }

    fn select(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
        reserve: bool,
    ) -> Option<PrefillSelection> {
        let healthy = get_healthy_worker_indices(workers);
        let first = workers.get(*healthy.first()?)?;
        // P work and D generation have different cost/lifetime semantics.
        if healthy.iter().any(|&i| {
            !matches!(workers[i].worker_type(), WorkerType::Prefill { .. })
                || normalize_model_key(workers[i].model_id())
                    != normalize_model_key(first.model_id())
        }) {
            return None;
        }
        let key = normalize_model_key(first.model_id()).to_string();
        let pool = self.pools.entry(key).or_default().clone();
        let text = info.request_text.unwrap_or("");
        let input_chars = text.chars().count();
        let session = extract_sticky_routing_key(info.headers)
            .map(|s| *blake3::hash(s.as_bytes()).as_bytes());
        // Serialize snapshot -> cost comparison -> reserve. A concurrent caller
        // sees this choice before any dispatch or backend load update occurs.
        let mut state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        let identities: Vec<_> = healthy
            .iter()
            .map(|&i| Identity::of(workers[i].as_ref()))
            .collect();
        let tenants: Vec<_> = identities.iter().map(Identity::tenant).collect();
        let tenant_refs: Vec<_> = tenants.iter().map(String::as_str).collect();
        let matches = pool
            .tree
            .prefix_match_counts_for_tenants(text, &tenant_refs);
        let mean_work = state.mean_work.unwrap_or(input_chars.max(1) as f64);
        let candidates: Vec<_> = healthy
            .iter()
            .zip(identities)
            .zip(matches)
            .map(|((&index, identity), cached)| {
                let tracked = state.workers.entry(identity.clone()).or_default();
                let dispatched = tracked.pending.values().filter(|p| p.dispatched).count();
                // worker.load already includes dispatched reservations. Count only
                // the remainder as unknown work; do not add the same request twice.
                let untracked_requests = workers[index].load().saturating_sub(dispatched);
                Candidate {
                    index,
                    identity,
                    cached,
                    work: input_chars.saturating_sub(cached).max(1) as f64,
                    queue: tracked.queued_work() + untracked_requests as f64 * mean_work,
                    error: tracked.cache_error,
                    untracked_requests,
                }
            })
            .collect();
        let now = Instant::now();
        let previous = session
            .and_then(|s| state.sessions.get(&s))
            .filter(|a| now.duration_since(a.confirmed) <= SESSION_TTL)
            .and_then(|a| candidates.iter().position(|c| c.identity == a.worker));
        let most_cached = candidates.iter().map(|c| c.cached).max().unwrap_or(0);
        let affinity = previous.or_else(|| {
            (most_cached > 0).then(|| {
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.cached == most_cached)
                    .min_by(|(_, a), (_, b)| (a.queue + a.work).total_cmp(&(b.queue + b.work)))
                    .unwrap()
                    .0
            })
        });
        let chosen = if let Some(a) = affinity {
            let source = &candidates[a];
            let mut best = a;
            let mut best_gain = 0.0;
            for (i, target) in candidates.iter().enumerate() {
                if i == a {
                    continue;
                }
                // Same work units on both sides: under equal service capacity
                // the unknown common rate cancels. No absolute/relative load
                // threshold, low-match bypass, or migration quota is involved.
                let wait_saving = source.queue - target.queue;
                let extra_work = target.work - source.work;
                let error_margin = source.error + target.error;
                let gain = wait_saving - extra_work - error_margin;
                debug!(source = %source.identity.url, target = %target.identity.url,
                    source_dp_rank = ?source.identity.rank, target_dp_rank = ?target.identity.rank,
                    source_cached_chars = source.cached, target_cached_chars = target.cached,
                    wait_saving_work = wait_saving, extra_work, error_margin_work = error_margin,
                    net_gain_work = gain, "adaptive_cache_aware candidate");
                if gain > best_gain {
                    best = i;
                    best_gain = gain;
                }
            }
            best
        } else {
            candidates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.queue + a.work + a.error).total_cmp(&(b.queue + b.work + b.error))
                })
                .unwrap()
                .0
        };
        let selected = &candidates[chosen];
        debug!(worker = %selected.identity.url, dp_rank = ?selected.identity.rank,
            input_chars, cached_chars = selected.cached, queue_work = selected.queue,
            added_work = selected.work, cache_error_work = selected.error,
            untracked_requests = selected.untracked_requests,
            changed_affinity = affinity.is_some_and(|a| a != chosen), reserved = reserve,
            "adaptive_cache_aware selected");
        let reservation = if reserve {
            state.next_id = state.next_id.checked_add(1)?;
            let id = state.next_id;
            state.workers.get_mut(&selected.identity)?.pending.insert(
                id,
                Pending {
                    text: Arc::from(text),
                    input_chars,
                    work: selected.work,
                    session,
                    dispatched: false,
                },
            );
            workers[selected.index].increment_processed();
            Some(Box::new(WorkReservation {
                pool: pool.clone(),
                worker: selected.identity.clone(),
                id,
            }) as Box<dyn PrefillReservation>)
        } else {
            None
        };
        Some(PrefillSelection {
            index: selected.index,
            reservation,
        })
    }
}

#[async_trait]
impl LoadBalancingPolicy for AdaptiveCacheAwarePolicy {
    // Index-only callers can inspect a decision, but cannot reserve its lifetime.
    // The PD planner always uses select_prefill_worker below.
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select(workers, info, false).map(|s| s.index)
    }

    async fn select_prefill_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<PrefillSelection> {
        self.select(workers, info, true)
    }

    fn name(&self) -> &'static str {
        "adaptive_cache_aware"
    }
    fn needs_request_text(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests;
