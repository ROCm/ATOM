//! `agentic`: the worker (`url@rank`) with the lowest cost wins, ties at random:
//! alpha * (inflight_prefill + uncached_tokens) + beta * inflight_requests - gamma * hit_blocks.

use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::IndexedRandom;
use tracing::debug;

use super::{
    block_index::{chain_keys, BlockIndex},
    get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;

/// Text bytes standing in for one token when a request has no tokens.
const TEXT_BYTES_PER_TOKEN: usize = 4;
/// Keeps text-keyed blocks apart from token-keyed blocks.
const TOKEN_SEED: u64 = 0;
const TEXT_SEED: u64 = 0x7465_7874;

#[derive(Debug, Clone, PartialEq)]
pub struct AgenticConfig {
    /// Tokens per block; match the engine's KV block size.
    pub block_size: usize,
    /// Weight of queued plus this request's uncached prefill tokens.
    pub alpha: f64,
    /// Weight of in-flight requests, in tokens per request.
    pub beta: f64,
    /// Extra credit per hit block, on top of the prefill it saves.
    pub gamma: f64,
    /// Most (block, worker) entries the hit index keeps.
    pub max_indexed_blocks: usize,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            alpha: 1.0,
            beta: 512.0,
            gamma: 0.0,
            max_indexed_blocks: 1 << 21,
        }
    }
}

#[derive(Debug)]
pub struct AgenticPolicy {
    config: AgenticConfig,
    index: BlockIndex,
}

/// One candidate's cost and the terms the decision log reports.
#[derive(Debug, Clone, Copy)]
struct Score {
    cost: f64,
    hit_blocks: usize,
    uncached_tokens: usize,
}

impl AgenticPolicy {
    pub fn new(config: AgenticConfig) -> Self {
        let index = BlockIndex::new(config.max_indexed_blocks);
        Self { config, index }
    }

    /// Block keys of the request and its prompt length in tokens.
    fn request_blocks(&self, info: &SelectWorkerInfo<'_>) -> (Vec<u64>, usize) {
        let block = self.config.block_size;
        match info.tokens {
            Some(tokens) if !tokens.is_empty() => {
                (chain_keys(tokens, block, TOKEN_SEED), tokens.len())
            }
            _ => {
                let text = info.request_text.unwrap_or("").as_bytes();
                let keys = chain_keys(text, block * TEXT_BYTES_PER_TOKEN, TEXT_SEED);
                (keys, text.len().div_ceil(TEXT_BYTES_PER_TOKEN))
            }
        }
    }

    fn score(&self, worker: &dyn Worker, hit_blocks: usize, prompt_tokens: usize) -> Score {
        let hit_tokens = (hit_blocks * self.config.block_size).min(prompt_tokens);
        let uncached_tokens = prompt_tokens - hit_tokens;
        let queued = (worker.prefill_tokens() + uncached_tokens) as f64;
        let cost = self.config.alpha * queued + self.config.beta * worker.load() as f64
            - self.config.gamma * hit_blocks as f64;
        Score {
            cost,
            hit_blocks,
            uncached_tokens,
        }
    }

    fn select(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo<'_>) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let (keys, prompt_tokens) = self.request_blocks(info);
        let urls: Vec<&str> = healthy.iter().map(|&i| workers[i].url()).collect();
        let hits = self.index.matched_blocks(&keys, &urls);
        let scores: Vec<Score> = healthy
            .iter()
            .zip(&hits)
            .map(|(&i, &hit_blocks)| self.score(workers[i].as_ref(), hit_blocks, prompt_tokens))
            .collect();
        let best = scores.iter().map(|s| s.cost).fold(f64::INFINITY, f64::min);
        let tied: Vec<usize> = (0..healthy.len())
            .filter(|&position| scores[position].cost == best)
            .collect();
        let position = *tied.choose(&mut rand::rng())?;
        let (selected, score) = (healthy[position], scores[position]);

        self.index.insert(workers[selected].url(), &keys);
        if let Some(estimate) = info.prefill_estimate {
            estimate.set(score.uncached_tokens);
        }
        debug!(
            worker = workers[selected].url(),
            cost = score.cost,
            hit_blocks = score.hit_blocks,
            uncached_tokens = score.uncached_tokens,
            prompt_tokens,
            by_tokens = info.tokens.is_some_and(|t| !t.is_empty()),
            "agentic placement"
        );
        Some(selected)
    }
}

impl Default for AgenticPolicy {
    fn default() -> Self {
        Self::new(AgenticConfig::default())
    }
}

#[async_trait]
impl LoadBalancingPolicy for AgenticPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        self.select(workers, info)
    }

    fn name(&self) -> &'static str {
        "agentic"
    }

    fn needs_request_text(&self) -> bool {
        true
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
    use crate::{
        core::{worker::PrefillTokenGuard, DPAwareWorkerBuilder},
        policies::PrefillEstimate,
    };

    fn ranks(count: usize) -> Vec<Arc<dyn Worker>> {
        (0..count)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://p:8000", rank, count).build())
                    as Arc<dyn Worker>
            })
            .collect()
    }

    fn policy(beta: f64, gamma: f64) -> AgenticPolicy {
        AgenticPolicy::new(AgenticConfig {
            block_size: 4,
            beta,
            gamma,
            ..Default::default()
        })
    }

    fn place(
        policy: &AgenticPolicy,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
    ) -> (usize, Option<usize>) {
        let estimate = PrefillEstimate::default();
        let info = SelectWorkerInfo {
            tokens: Some(tokens),
            prefill_estimate: Some(&estimate),
            ..Default::default()
        };
        (policy.select(workers, &info).unwrap(), estimate.get())
    }

    fn session(id: u32, len: u32) -> Vec<u32> {
        (0..len).map(|i| id * 100_000 + i).collect()
    }

    #[test]
    fn a_later_turn_follows_its_prefix_and_is_charged_only_the_new_blocks() {
        let policy = policy(0.0, 0.0);
        let workers = ranks(4);
        let (owner, charged) = place(&policy, &workers, &session(1, 64));
        assert_eq!(charged, Some(64));
        let _held = PrefillTokenGuard::new(workers[owner].clone(), 40);
        // 40 queued + 8 new tokens beats 72 uncached tokens anywhere else.
        assert_eq!(place(&policy, &workers, &session(1, 72)), (owner, Some(8)));
    }

    #[test]
    fn a_hit_moves_once_the_owner_queue_outweighs_recomputing_it() {
        let policy = policy(0.0, 0.0);
        let workers = ranks(2);
        let (owner, _) = place(&policy, &workers, &session(1, 64));
        let _held = PrefillTokenGuard::new(workers[owner].clone(), 100);
        // 100 queued + 8 new > 72 uncached on the idle rank.
        assert_eq!(
            place(&policy, &workers, &session(1, 72)),
            (1 - owner, Some(72))
        );
    }

    #[test]
    fn new_sessions_spread_by_queued_prefill_and_requests() {
        let policy = policy(512.0, 0.0);
        let workers = ranks(4);
        let mut held = Vec::new();
        let mut placed = std::collections::HashSet::new();
        for id in 0..4 {
            let (rank, charged) = place(&policy, &workers, &session(id, 32));
            held.push(PrefillTokenGuard::new(
                workers[rank].clone(),
                charged.unwrap(),
            ));
            workers[rank].increment_load();
            placed.insert(rank);
        }
        assert_eq!(placed.len(), 4, "each new session takes the idlest rank");
    }

    #[test]
    fn gamma_credits_hits_beyond_the_tokens_they_save() {
        let workers = ranks(2);
        for (gamma, stays) in [(0.0, false), (10.0, true)] {
            let policy = policy(0.0, gamma);
            let (owner, _) = place(&policy, &workers, &session(1, 64));
            // 16 hit blocks: 160 credit outweighs 70 queued tokens only with gamma.
            let _held = PrefillTokenGuard::new(workers[owner].clone(), 70);
            let (rank, _) = place(&policy, &workers, &session(1, 64));
            assert_eq!(rank == owner, stays, "gamma={gamma}");
        }
    }

    #[test]
    fn requests_without_tokens_match_on_text() {
        let policy = policy(0.0, 0.0);
        let workers = ranks(4);
        let text = "s".repeat(256);
        let select = |text: &str| {
            let estimate = PrefillEstimate::default();
            let info = SelectWorkerInfo {
                request_text: Some(text),
                prefill_estimate: Some(&estimate),
                ..Default::default()
            };
            (policy.select(&workers, &info).unwrap(), estimate.get())
        };
        let (owner, charged) = select(&text);
        assert_eq!(charged, Some(64));
        let _held = PrefillTokenGuard::new(workers[owner].clone(), 30);
        // 16-byte blocks: 256 shared bytes are 16 hit blocks of 4 tokens.
        assert_eq!(
            select(&format!("{text}{}", "t".repeat(32))),
            (owner, Some(8))
        );
    }

    #[test]
    fn unhealthy_workers_are_skipped() {
        let policy = policy(512.0, 0.0);
        let workers = ranks(2);
        workers[0].set_healthy(false);
        assert_eq!(place(&policy, &workers, &session(1, 8)).0, 1);
        workers[1].set_healthy(false);
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select(&workers, &info), None);
    }
}
