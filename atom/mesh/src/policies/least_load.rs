//! `least_load`: place a request on the worker (`url@rank`) with the lowest
//! outstanding decode cost.
//!
//! The cost is the KV already reserved on that worker plus one request-equivalent,
//! the same 512-token weight the engine's `least_tokens` balancer uses. The
//! reservation is taken inside the selection, before the lock is released, so a
//! request still in prefill or waiting on decode is visible to the next choice.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
use crate::core::{LoadReservation, Worker};

/// Token weight of one in-flight request. Matches `ATOM_DP_LB_REQ_EQUIV` and
/// [`super::AgenticCost::request_token_equivalent`].
pub const REQUEST_TOKEN_EQUIVALENT: u64 = 512;

#[derive(Debug)]
pub struct LeastLoadPolicy {
    request_token_equivalent: u64,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    next_id: u64,
    holds: HashMap<u64, LoadReservation>,
}

impl LeastLoadPolicy {
    pub fn new(request_token_equivalent: u64) -> Self {
        Self {
            request_token_equivalent,
            state: Mutex::new(State {
                next_id: 1,
                holds: HashMap::new(),
            }),
        }
    }

    fn prompt_tokens(info: &SelectWorkerInfo<'_>) -> Option<usize> {
        info.tokens.map(<[u32]>::len)
    }
}

impl Default for LeastLoadPolicy {
    fn default() -> Self {
        Self::new(REQUEST_TOKEN_EQUIVALENT)
    }
}

#[async_trait]
impl LoadBalancingPolicy for LeastLoadPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return (None, RouteDecision::branch(super::NO_HEALTHY_WORKERS));
        }

        let prompt = Self::prompt_tokens(info);
        let compare_tokens = prompt.is_some()
            && healthy
                .iter()
                .all(|&idx| workers[idx].decode_untokenized() == 0);

        let mut best: Option<(u128, usize, usize)> = None;
        for idx in healthy {
            let worker = &workers[idx];
            let load = worker.load();
            let cost = if compare_tokens {
                worker.decode_kv_tokens() as u128
                    + u128::from(self.request_token_equivalent) * load as u128
            } else {
                load as u128
            };
            let rank = (cost, load, idx);
            if best.is_none_or(|current| rank < current) {
                best = Some(rank);
            }
        }
        let Some((_, _, idx)) = best else {
            return (None, RouteDecision::branch(super::NO_HEALTHY_WORKERS));
        };

        let tokens = if compare_tokens { prompt } else { None };
        let hold = LoadReservation::charge(workers[idx].clone(), tokens);
        let id = state.next_id;
        state.next_id = state.next_id.wrapping_add(1);
        state.holds.insert(id, hold);
        let mut decision = RouteDecision::branch("least_load");
        decision.reservation_id = Some(id);
        (Some(idx), decision)
    }

    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let (idx, decision) = self.select_with_decision(workers, info).await;
        // Callers of this entry point never adopt the hold. Drop it here so the
        // charge does not stay in the policy until the policy itself is dropped.
        if let Some(id) = decision.reservation_id {
            drop(self.take_load_reservation(id));
        }
        idx
    }

    fn take_load_reservation(&self, id: u64) -> Option<LoadReservation> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .holds
            .remove(&id)
    }

    fn name(&self) -> &'static str {
        "least_load"
    }

    fn wants_tokens(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .build(),
        )
    }

    /// Leave `kv` tokens and `requests` in-flight. One charge carries the tokens;
    /// the rest are empty requests. `untokenized` adds requests with no token count.
    fn seed(worker: &Arc<dyn Worker>, kv: usize, requests: usize, untokenized: usize) {
        if requests > 0 {
            std::mem::forget(LoadReservation::charge(worker.clone(), Some(kv)));
            for _ in 1..requests {
                std::mem::forget(LoadReservation::charge(worker.clone(), Some(0)));
            }
        }
        for _ in 0..untokenized {
            std::mem::forget(LoadReservation::charge(worker.clone(), None));
        }
    }

    fn info_with(tokens: &[u32]) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            tokens: Some(tokens),
            ..SelectWorkerInfo::default()
        }
    }

    #[tokio::test]
    async fn lower_kv_wins() {
        let policy = LeastLoadPolicy::default();
        let workers = vec![worker("http://d0"), worker("http://d1")];
        seed(&workers[0], 2_000, 1, 0);
        seed(&workers[1], 100, 1, 0);
        let prompt = [0u32; 8];
        let (idx, decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(idx, Some(1));
        let hold = policy
            .take_load_reservation(decision.reservation_id.unwrap())
            .unwrap();
        assert_eq!(workers[1].load(), 2);
        assert_eq!(workers[1].decode_kv_tokens(), 108);
        drop(hold);
        assert_eq!(workers[1].load(), 1);
        assert_eq!(workers[1].decode_kv_tokens(), 100);
    }

    #[tokio::test]
    async fn request_weight_breaks_a_token_tie() {
        let policy = LeastLoadPolicy::new(512);
        let workers = vec![worker("http://d0"), worker("http://d1")];
        seed(&workers[0], 100, 2, 0);
        seed(&workers[1], 100, 1, 0);
        let prompt = [0u32; 4];
        let (idx, decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(idx, Some(1));
        drop(policy.take_load_reservation(decision.reservation_id.unwrap()));
    }

    #[tokio::test]
    async fn equal_cost_keeps_the_lower_index() {
        let policy = LeastLoadPolicy::default();
        let workers = vec![worker("http://d0"), worker("http://d1")];
        seed(&workers[0], 10, 1, 0);
        seed(&workers[1], 10, 1, 0);
        let prompt = [0u32; 1];
        let (idx, decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(idx, Some(0));
        drop(policy.take_load_reservation(decision.reservation_id.unwrap()));
    }

    #[tokio::test]
    async fn an_untokenized_worker_makes_the_choice_use_inflight() {
        let policy = LeastLoadPolicy::default();
        // Token scores would pick d0 (512 * 5 vs 100_000 + 512). Inflight picks d1.
        let workers = vec![worker("http://d0"), worker("http://d1")];
        seed(&workers[0], 0, 0, 5);
        seed(&workers[1], 100_000, 1, 0);
        let prompt = [0u32; 4];
        let (idx, decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(idx, Some(1));
        let hold = policy
            .take_load_reservation(decision.reservation_id.unwrap())
            .unwrap();
        // The new request is also untokenized, because the pool already was.
        assert_eq!(workers[1].decode_untokenized(), 1);
        drop(hold);
        assert_eq!(workers[1].decode_untokenized(), 0);
    }

    #[tokio::test]
    async fn a_later_selection_sees_the_reservation() {
        let policy = LeastLoadPolicy::default();
        let workers = vec![worker("http://d0"), worker("http://d1")];
        let prompt = [0u32; 32];
        let (first, first_decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        let (second, second_decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(first, Some(0));
        assert_eq!(second, Some(1));
        drop(policy.take_load_reservation(first_decision.reservation_id.unwrap()));
        drop(policy.take_load_reservation(second_decision.reservation_id.unwrap()));
        assert_eq!(workers[0].load(), 0);
        assert_eq!(workers[1].load(), 0);
    }

    #[tokio::test]
    async fn missing_tokens_reserves_a_request_only() {
        let policy = LeastLoadPolicy::default();
        let workers = vec![worker("http://d0")];
        let (idx, decision) = policy
            .select_with_decision(&workers, &SelectWorkerInfo::default())
            .await;
        assert_eq!(idx, Some(0));
        assert_eq!(workers[0].decode_kv_tokens(), 0);
        assert_eq!(workers[0].decode_untokenized(), 1);
        drop(policy.take_load_reservation(decision.reservation_id.unwrap()));
        assert_eq!(workers[0].load(), 0);
        assert_eq!(workers[0].decode_untokenized(), 0);
    }
}
