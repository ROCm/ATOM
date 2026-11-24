//! Data-parallel load as ATOM's CoreManager keeps it, and the least-load
//! policies that place by it (`least_requests`, `least_tokens`).
//!
//! A request is charged one request and its prompt tokens on the rank it is
//! placed on. The tokens are released when its first output arrives (its
//! prefill is done), the request when it ends. A policy here selects and
//! charges under one lock, so a burst spreads over the ranks instead of all
//! landing on the one that was lightest when it began; the dispatcher then
//! owns the charge as a `DpCharge`, which releases it whatever way the
//! request ends.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;
use parking_lot::Mutex;

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo,
    NO_HEALTHY_WORKERS,
};
use crate::core::Worker;

/// In-flight requests and queued prefill tokens of one DP rank.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RankLoad {
    pub requests: u64,
    pub tokens: u64,
}

/// Every DP rank's load, shared by the policy that charges it and the
/// dispatcher that releases it.
#[derive(Debug)]
pub struct DpLoad {
    ranks: Mutex<Vec<RankLoad>>,
}

impl DpLoad {
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ranks: Mutex::new(vec![RankLoad::default(); ranks]),
        })
    }

    pub fn loads(&self) -> Vec<RankLoad> {
        self.ranks.lock().clone()
    }

    /// Charge a request placed on `rank` by something that did not charge it
    /// (an explicit rank, a policy that keeps no load).
    pub fn charge(self: &Arc<Self>, rank: usize, tokens: u64) -> DpCharge {
        if let Some(load) = self.ranks.lock().get_mut(rank) {
            load.requests += 1;
            load.tokens += tokens;
        }
        self.adopt(rank, tokens)
    }

    /// Take over the charge a policy here made when it placed the request.
    pub fn adopt(self: &Arc<Self>, rank: usize, tokens: u64) -> DpCharge {
        DpCharge {
            load: Arc::clone(self),
            rank,
            tokens,
            prefill_released: false,
        }
    }

    fn release(&self, rank: usize, requests: u64, tokens: u64) {
        if let Some(load) = self.ranks.lock().get_mut(rank) {
            load.requests = load.requests.saturating_sub(requests);
            load.tokens = load.tokens.saturating_sub(tokens);
        }
    }
}

/// One request's charge on its rank: its tokens until `prefill_done`, the
/// request until dropped.
#[derive(Debug)]
pub struct DpCharge {
    load: Arc<DpLoad>,
    rank: usize,
    tokens: u64,
    prefill_released: bool,
}

impl DpCharge {
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The request's first output arrived: its prompt is no longer queued.
    pub fn prefill_done(&mut self) {
        if !self.prefill_released {
            self.prefill_released = true;
            self.load.release(self.rank, 0, self.tokens);
        }
    }
}

impl Drop for DpCharge {
    fn drop(&mut self) {
        let tokens = if self.prefill_released {
            0
        } else {
            self.tokens
        };
        self.load.release(self.rank, 1, tokens);
    }
}

/// What a least-load policy compares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeastLoad {
    /// Fewest in-flight requests, then fewest queued tokens: the request
    /// count keeps lockstep DP ranks in phase.
    Requests,
    /// Lowest `tokens + req_equiv * requests`: prefill and decode pressure.
    Tokens { req_equiv: u64 },
}

/// CoreManager's `least_requests` / `least_tokens` over mesh's workers, one
/// per DP rank (`Worker::dp_rank`). Fully tied ranks take turns.
#[derive(Debug)]
pub struct LeastLoadPolicy {
    load: Arc<DpLoad>,
    mode: LeastLoad,
    cursor: AtomicUsize,
}

impl LeastLoadPolicy {
    pub fn new(load: Arc<DpLoad>, mode: LeastLoad) -> Self {
        Self {
            load,
            mode,
            cursor: AtomicUsize::new(0),
        }
    }

    fn score(&self, load: RankLoad) -> (u64, u64) {
        match self.mode {
            LeastLoad::Requests => (load.requests, load.tokens),
            LeastLoad::Tokens { req_equiv } => (load.tokens + req_equiv * load.requests, 0),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for LeastLoadPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        }
        let rank_of = |idx: usize| workers[idx].dp_rank().unwrap_or(idx);
        let tokens = info.tokens.map_or(0, |t| t.len() as u64);
        let start = self.cursor.fetch_add(1, Ordering::Relaxed);
        let mut ranks = self.load.ranks.lock();
        let load_of = |idx: usize| ranks.get(rank_of(idx)).copied().unwrap_or_default();
        let chosen = (0..healthy.len())
            .map(|i| healthy[(start + i) % healthy.len()])
            .min_by_key(|&idx| self.score(load_of(idx)))
            .expect("healthy is not empty");
        if let Some(load) = ranks.get_mut(rank_of(chosen)) {
            load.requests += 1;
            load.tokens += tokens;
        }
        (Some(chosen), RouteDecision::branch(self.name()))
    }

    fn name(&self) -> &'static str {
        match self.mode {
            LeastLoad::Requests => "least_requests",
            LeastLoad::Tokens { .. } => "least_tokens",
        }
    }

    fn needs_tokens(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DPAwareWorkerBuilder;

    fn workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|r| -> Arc<dyn Worker> {
                Arc::new(DPAwareWorkerBuilder::new("atom://local", r, n).build())
            })
            .collect()
    }

    async fn place(policy: &LeastLoadPolicy, workers: &[Arc<dyn Worker>], tokens: usize) -> usize {
        let prompt = vec![1u32; tokens];
        let info = SelectWorkerInfo {
            tokens: Some(&prompt),
            ..Default::default()
        };
        policy.select_worker(workers, &info).await.unwrap()
    }

    #[tokio::test]
    async fn a_burst_spreads_because_each_placement_is_charged_at_once() {
        let load = DpLoad::new(4);
        let policy = LeastLoadPolicy::new(Arc::clone(&load), LeastLoad::Requests);
        let workers = workers(4);
        let mut placed = Vec::new();
        for _ in 0..8 {
            placed.push(place(&policy, &workers, 10).await);
        }
        placed.sort();
        assert_eq!(placed, [0, 0, 1, 1, 2, 2, 3, 3]);
        assert!(load
            .loads()
            .iter()
            .all(|l| l.requests == 2 && l.tokens == 20));
    }

    #[tokio::test]
    async fn least_requests_breaks_a_request_tie_on_queued_tokens() {
        let load = DpLoad::new(2);
        let _a = load.charge(0, 100);
        let _b = load.charge(1, 10);
        let policy = LeastLoadPolicy::new(Arc::clone(&load), LeastLoad::Requests);
        assert_eq!(place(&policy, &workers(2), 1).await, 1);
    }

    #[tokio::test]
    async fn least_tokens_weighs_requests_by_req_equiv() {
        let load = DpLoad::new(2);
        // Rank 0: 1 request, 600 tokens = 1112; rank 1: 3 requests, 0 tokens = 1536.
        let _a = load.charge(0, 600);
        let _b = [load.charge(1, 0), load.charge(1, 0), load.charge(1, 0)];
        let policy = LeastLoadPolicy::new(Arc::clone(&load), LeastLoad::Tokens { req_equiv: 512 });
        assert_eq!(place(&policy, &workers(2), 1).await, 0);
    }

    #[tokio::test]
    async fn an_unhealthy_rank_is_passed_over() {
        let load = DpLoad::new(2);
        let policy = LeastLoadPolicy::new(Arc::clone(&load), LeastLoad::Requests);
        let workers = workers(2);
        workers[0].set_healthy(false);
        for _ in 0..3 {
            assert_eq!(place(&policy, &workers, 1).await, 1);
        }
        workers[1].set_healthy(false);
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, None);
    }

    #[test]
    fn a_charge_releases_its_tokens_at_prefill_and_its_request_when_dropped() {
        let load = DpLoad::new(1);
        let mut charge = load.charge(0, 30);
        assert_eq!(
            load.loads()[0],
            RankLoad {
                requests: 1,
                tokens: 30
            }
        );
        charge.prefill_done();
        charge.prefill_done();
        assert_eq!(
            load.loads()[0],
            RankLoad {
                requests: 1,
                tokens: 0
            }
        );
        drop(charge);
        assert_eq!(load.loads()[0], RankLoad::default());
        // Ended before any output: both go at once.
        drop(load.charge(0, 30));
        assert_eq!(load.loads()[0], RankLoad::default());
    }
}
