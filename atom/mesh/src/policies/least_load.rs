//! `least_load`: place a request on the worker (`url@rank`) with the lowest
//! outstanding decode load.
//!
//! Two orders ([`LeastLoadOrder`]):
//! - `tokens` (default): the prompt tokens reserved on that worker plus one
//!   request-equivalent per in-flight request, the same 512-token weight the engine's
//!   `least_tokens` balancer uses. With prompts far longer than the weight this evens
//!   the workers' tokens and leaves their request counts apart.
//! - `requests`: in-flight requests first, reserved prompt tokens among workers with
//!   equal counts, as the engine's `least_requests` does. The DP ranks of one engine
//!   step together, so the rank with the most rows sets every rank's decode step.
//!
//! Generated tokens are not counted: ATOM reports `usage` only on a stream's last
//! frame. The reservation is taken inside the selection, before the lock is released,
//! so a request still in prefill or waiting on decode is visible to the next choice.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
use crate::{
    config::LeastLoadOrder,
    core::{LoadReservation, Worker},
};

/// Token weight of one in-flight request. Matches `ATOM_DP_LB_REQ_EQUIV` and
/// [`super::AgenticCost::request_token_equivalent`].
pub const REQUEST_TOKEN_EQUIVALENT: u64 = 512;

#[derive(Debug)]
pub struct LeastLoadPolicy {
    request_token_equivalent: u64,
    order: LeastLoadOrder,
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    next_id: u64,
    holds: HashMap<u64, LoadReservation>,
}

impl LeastLoadPolicy {
    pub fn new(request_token_equivalent: u64) -> Self {
        Self::with_order(request_token_equivalent, LeastLoadOrder::Tokens)
    }

    pub fn with_order(request_token_equivalent: u64, order: LeastLoadOrder) -> Self {
        Self {
            request_token_equivalent,
            order,
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

        // Compared element by element: the first decides, the next ones break its ties.
        let mut best: Option<(u128, u128, usize)> = None;
        for idx in healthy {
            let worker = &workers[idx];
            let load = worker.load() as u128;
            let kv_tokens = if compare_tokens {
                worker.decode_kv_tokens() as u128
            } else {
                0
            };
            let rank = match self.order {
                LeastLoadOrder::Tokens if compare_tokens => (
                    kv_tokens + u128::from(self.request_token_equivalent) * load,
                    load,
                    idx,
                ),
                LeastLoadOrder::Tokens | LeastLoadOrder::Requests => (load, kv_tokens, idx),
            };
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
    async fn the_requests_order_puts_the_count_before_the_tokens() {
        // d0 holds one long prompt, d1 three short ones. `tokens` takes d1
        // (1_000 + 3 * 512 against 400_000 + 512), `requests` takes d0.
        let prompt = [0u32; 4];
        for (order, expected) in [(LeastLoadOrder::Tokens, 1), (LeastLoadOrder::Requests, 0)] {
            let policy = LeastLoadPolicy::with_order(512, order);
            let workers = vec![worker("http://d0"), worker("http://d1")];
            seed(&workers[0], 400_000, 1, 0);
            seed(&workers[1], 1_000, 3, 0);
            let (idx, decision) = policy
                .select_with_decision(&workers, &info_with(&prompt))
                .await;
            assert_eq!(idx, Some(expected), "{order}");
            let hold = policy
                .take_load_reservation(decision.reservation_id.unwrap())
                .unwrap();
            // The reservation carries the prompt under either order.
            assert_eq!(workers[expected].decode_untokenized(), 0);
            drop(hold);
        }
    }

    #[tokio::test]
    async fn the_requests_order_breaks_a_count_tie_by_tokens() {
        let policy = LeastLoadPolicy::with_order(512, LeastLoadOrder::Requests);
        let workers = vec![
            worker("http://d0"),
            worker("http://d1"),
            worker("http://d2"),
        ];
        seed(&workers[0], 300_000, 2, 0);
        seed(&workers[1], 90_000, 2, 0);
        seed(&workers[2], 10, 3, 0);
        let prompt = [0u32; 4];
        let (idx, decision) = policy
            .select_with_decision(&workers, &info_with(&prompt))
            .await;
        assert_eq!(idx, Some(1));
        drop(policy.take_load_reservation(decision.reservation_id.unwrap()));
    }

    #[tokio::test]
    async fn the_requests_order_keeps_counts_within_one() {
        // Prompts from 1k to 400k tokens, the mix that left `tokens` ten requests
        // apart: with every request still in flight the counts differ by at most one.
        let policy = LeastLoadPolicy::with_order(512, LeastLoadOrder::Requests);
        let workers: Vec<_> = (0..8).map(|i| worker(&format!("http://d{i}"))).collect();
        let lengths = [1_000usize, 400_000, 20_000, 150_000, 3_000, 250_000, 80_000];
        let mut holds = Vec::new();
        for i in 0..100 {
            let prompt = vec![0u32; lengths[i % lengths.len()]];
            let (_, decision) = policy
                .select_with_decision(&workers, &info_with(&prompt))
                .await;
            holds.push(
                policy
                    .take_load_reservation(decision.reservation_id.unwrap())
                    .unwrap(),
            );
            let loads: Vec<_> = workers.iter().map(|w| w.load()).collect();
            let spread = loads.iter().max().unwrap() - loads.iter().min().unwrap();
            assert!(spread <= 1, "after {} requests: {loads:?}", i + 1);
        }
        drop(holds);
        assert!(workers.iter().all(|w| w.load() == 0));
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
