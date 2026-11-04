//! Request-count/context-budget placement, independent from P and sessions.
use std::{collections::HashMap, sync::Arc};

use parking_lot::Mutex;

use crate::config::types::AtomDecodeBalancePolicy;
use crate::core::Worker;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Key(String, Option<usize>);

#[derive(Clone, Copy, Debug, Default)]
struct Load {
    requests: u64,
    tokens: u64,
}

fn publish_load(key: &Key, load: Load) {
    let rank = key
        .1
        .map(|r| r.to_string())
        .unwrap_or_else(|| "none".into());
    metrics::gauge!("atomesh_pd_decode_reserved_requests", "worker"=>key.0.clone(), "dp_rank"=>rank.clone())
        .set(load.requests as f64);
    metrics::gauge!("atomesh_pd_decode_reserved_context_tokens", "worker"=>key.0.clone(), "dp_rank"=>rank)
        .set(load.tokens as f64);
}

#[derive(Default)]
struct State {
    loads: HashMap<Key, Load>,
    cursor: usize,
}

#[derive(Default)]
pub(super) struct DecodeBalancer(Mutex<State>);

pub(super) struct DecodeReservation {
    balancer: Arc<DecodeBalancer>,
    key: Key,
    tokens: u64,
}

impl DecodeBalancer {
    pub(super) fn new(_policy: AtomDecodeBalancePolicy) -> Self {
        Self(Mutex::new(State::default()))
    }

    #[cfg(test)]
    pub(super) fn outstanding(&self) -> u64 {
        self.0.lock().loads.values().map(|l| l.requests).sum()
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        candidates: &[Arc<dyn Worker>],
        prompt_tokens: u64,
        output_budget: u64,
    ) -> Option<(Arc<dyn Worker>, DecodeReservation)> {
        let mut state = self.0.lock();
        let n = candidates.len();
        if n == 0 {
            return None;
        }
        let offset = state.cursor % n;
        state.cursor = state.cursor.wrapping_add(1);
        let tokens = prompt_tokens.checked_add(output_budget)?;
        let index = (0..n)
            .map(|i| (offset + i) % n)
            .filter(|&i| candidates[i].is_available())
            .filter(|&i| {
                let w = &candidates[i];
                let load = state
                    .loads
                    .get(&Key(w.url().into(), w.dp_rank()))
                    .copied()
                    .unwrap_or_default();
                load.tokens.checked_add(tokens).is_some() && load.requests < u64::MAX
            })
            .min_by_key(|&i| {
                let w = &candidates[i];
                let load = state
                    .loads
                    .get(&Key(w.url().into(), w.dp_rank()))
                    .copied()
                    .unwrap_or_default();
                (load.requests, load.tokens)
            })?;
        let worker = candidates[index].clone();
        let key = Key(worker.url().into(), worker.dp_rank());
        let load = state.loads.entry(key.clone()).or_default();
        load.requests += 1;
        load.tokens += tokens;
        publish_load(&key, *load);
        Some((
            worker,
            DecodeReservation {
                balancer: self.clone(),
                key,
                tokens,
            },
        ))
    }
}

impl Drop for DecodeReservation {
    fn drop(&mut self) {
        let mut state = self.balancer.0.lock();
        let load = state
            .loads
            .get_mut(&self.key)
            .expect("reservation exists until drop");
        load.requests -= 1;
        load.tokens -= self.tokens;
        publish_load(&self.key, *load);
        if load.requests == 0 {
            state.loads.remove(&self.key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

    fn workers() -> Vec<Arc<dyn Worker>> {
        (0..2)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://d{i}:8000"))
                        .worker_type(WorkerType::Decode)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn request_count_baseline_does_not_pack_short_requests_on_heavy_context() {
        let b = Arc::new(DecodeBalancer::default());
        let ws = workers();
        let (_, a) = b.reserve(&ws[..1], 700_000, 0).unwrap();
        let (_, c) = b.reserve(&ws[1..], 1_000, 0).unwrap();
        let (chosen, d) = b.reserve(&ws, 100, 0).unwrap();
        assert_eq!(chosen.url(), ws[1].url());
        drop((a, c, d));
        assert!(b.0.lock().loads.is_empty());
    }

    #[test]
    fn request_count_always_precedes_context_budget() {
        let b = Arc::new(DecodeBalancer::default());
        let ws = workers();
        let (_, a) = b.reserve(&ws[..1], 700_000, 0).unwrap();
        let (_, c) = b.reserve(&ws[1..], 100, 0).unwrap();
        let (_, d) = b.reserve(&ws[1..], 100, 0).unwrap();
        for prompt in [10, 100_000] {
            let (chosen, guard) = b.reserve(&ws, prompt, 20).unwrap();
            assert_eq!(chosen.url(), ws[0].url());
            drop(guard);
        }
        drop((a, c, d));
    }

    #[test]
    fn exact_ties_rotate_after_requests_finish() {
        let b = Arc::new(DecodeBalancer::default());
        let ws = workers();
        let (first, a) = b.reserve(&ws, 100, 0).unwrap();
        drop(a);
        let (second, c) = b.reserve(&ws, 100, 0).unwrap();
        assert_ne!(first.url(), second.url());
        drop(c);
    }

    #[test]
    fn invalid_token_budget_does_not_leave_a_reservation() {
        let b = Arc::new(DecodeBalancer::default());
        assert!(b.reserve(&workers(), u64::MAX, 1).is_none());
        assert!(b.0.lock().loads.is_empty());
    }

    #[test]
    fn unhealthy_workers_never_receive_reservations() {
        let b = Arc::new(DecodeBalancer::default());
        let ws = workers();
        ws[0].set_healthy(false);
        assert_eq!(b.reserve(&ws, 10, 0).unwrap().0.url(), ws[1].url());
        ws[1].set_healthy(false);
        assert!(b.reserve(&ws, 10, 0).is_none());
        assert!(b.0.lock().loads.is_empty());
    }

    #[test]
    fn burst_selection_and_reservation_are_atomic() {
        let b = Arc::new(DecodeBalancer::default());
        let ws = workers();
        let handles: Vec<_> = (0..64)
            .map(|_| {
                let b = b.clone();
                let ws = ws.clone();
                std::thread::spawn(move || b.reserve(&ws, 100, 20).unwrap().1)
            })
            .collect();
        let guards: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert!(b
            .0
            .lock()
            .loads
            .values()
            .all(|l| l.requests == 32 && l.tokens == 32 * 120));
        drop(guards);
        assert!(b.0.lock().loads.is_empty());
    }

    #[test]
    fn ranks_on_one_endpoint_have_independent_loads() {
        let b = Arc::new(DecodeBalancer::default());
        let ws: Vec<Arc<dyn Worker>> = (0..2)
            .map(|rank| {
                Arc::new(
                    DPAwareWorkerBuilder::new_with_type(
                        "http://d:8000",
                        rank,
                        2,
                        WorkerType::Decode,
                    )
                    .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let (first, a) = b.reserve(&ws, 100, 0).unwrap();
        let (second, c) = b.reserve(&ws, 100, 0).unwrap();
        assert_ne!(first.dp_rank(), second.dp_rank());
        drop((a, c));
        assert!(b.0.lock().loads.is_empty());
    }
}
