//! `agentic`: places each request on the worker (`url@rank`) with the lowest
//! prefill cost, weighing the prefix the worker's GPU KV cache holds against the
//! prefill queued on it.
//!
//! [`AgenticCost`] scores snapshots and owns no state. [`AgenticPolicy`] feeds it
//! each worker's cached prefix (from the KV event index, or the text tree when the
//! request cannot match the index) and in-flight prefill tokens and requests, and
//! reports the selected worker's uncached tokens as the prefill it is charged.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::DashMap;
use rand::{seq::IteratorRandom, Rng};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LoadBalancingPolicy, RouteDecision, SelectWorkerInfo, NO_HEALTHY_WORKERS,
};
use crate::{core::Worker, kv_index::PrefixMatchProvider};

/// Text bytes standing in for one token when a request's tokens are unknown.
const TEXT_BYTES_PER_TOKEN: usize = 4;

/// The agentic cost's weight:
///
/// ```text
/// cost = inflight_prefill_tokens + uncached_tokens
///      + request_token_equivalent * inflight_requests
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgenticCost {
    /// Cost of one in-flight request, in tokens. The engine's `least_tokens`
    /// DP balancer weighs requests by `ATOM_DP_LB_REQ_EQUIV`; keep the two
    /// equal so the router and the engine rank a rank's load the same way.
    /// Default: 512.
    pub request_token_equivalent: u64,
}

impl Default for AgenticCost {
    fn default() -> Self {
        Self {
            request_token_equivalent: 512,
        }
    }
}

/// One eligible worker's inputs for this request, at `url@rank` granularity.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// Index in the caller's worker list, preserved even after health filtering.
    pub worker_index: usize,
    /// Leading prompt tokens the worker's GPU KV cache is estimated to hold.
    /// `None` means unknown and receives no cache credit; `Some(0)` is a known
    /// miss. Both stay eligible.
    pub cached_tokens: Option<usize>,
    /// Tokens assigned to this worker whose prefill has not completed.
    pub inflight_prefill_tokens: usize,
    /// Requests assigned to this worker that have not completed.
    pub inflight_requests: usize,
}

/// A candidate's cost and the prefill charge to report on placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub worker_index: usize,
    pub uncached_tokens: usize,
    /// Integer token-equivalent cost. Widen before arithmetic so large counters
    /// neither wrap nor lose the one-token differences used to rank workers.
    pub cost: u128,
}

impl AgenticCost {
    /// Score one candidate.
    pub fn score(&self, prompt_tokens: usize, candidate: &Candidate) -> Selection {
        let uncached_tokens = prompt_tokens.saturating_sub(candidate.cached_tokens.unwrap_or(0));
        Selection {
            worker_index: candidate.worker_index,
            uncached_tokens,
            cost: candidate.inflight_prefill_tokens as u128
                + uncached_tokens as u128
                + u128::from(self.request_token_equivalent) * candidate.inflight_requests as u128,
        }
    }

    /// Select the lowest-cost eligible worker, breaking ties uniformly at random.
    ///
    /// An empty candidate list returns `None`. The caller remains responsible for
    /// worker health, session affinity, rank mapping and charging the final placement.
    pub fn select_worker(
        &self,
        prompt_tokens: usize,
        candidates: &[Candidate],
    ) -> Option<Selection> {
        self.select_with_rng(prompt_tokens, candidates, &mut rand::rng())
    }

    fn select_with_rng(
        &self,
        prompt_tokens: usize,
        candidates: &[Candidate],
        rng: &mut impl Rng,
    ) -> Option<Selection> {
        let scored = candidates
            .iter()
            .map(|candidate| self.score(prompt_tokens, candidate));
        let best = scored.clone().map(|selection| selection.cost).min()?;
        scored
            .filter(|selection| selection.cost == best)
            .choose(rng)
    }
}

/// Configuration for the `agentic` policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgenticConfig {
    pub cost: AgenticCost,
    /// Seconds between text tree evictions; 0 never evicts.
    pub eviction_interval_secs: u64,
    /// Size each text tree is evicted down to, as `cache_aware`'s `max_tree_size`.
    pub max_tree_size: usize,
}

impl Default for AgenticConfig {
    fn default() -> Self {
        Self {
            cost: AgenticCost::default(),
            eviction_interval_secs: 120,
            max_tree_size: 1 << 26,
        }
    }
}

/// Where a selection's cache hits came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hits {
    Index,
    Text,
    Unmatched,
}

/// The `agentic` routing policy.
///
/// A request the KV event index can match takes its hits from the index. Others
/// (without tokens, with media, or with no candidate's events indexed) take them
/// from a text tree of the requests this policy placed, which only credits the
/// worker holding the longest text match.
#[derive(Debug)]
pub struct AgenticPolicy {
    cost: AgenticCost,
    kv_events: OnceLock<Arc<dyn PrefixMatchProvider>>,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    _eviction_task: Option<PeriodicTask>,
}

impl AgenticPolicy {
    pub fn new(config: AgenticConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let eviction_task = (config.eviction_interval_secs > 0).then(|| {
            let trees = Arc::clone(&trees);
            PeriodicTask::spawn(
                config.eviction_interval_secs,
                "AgenticEviction",
                move || {
                    for tree in trees.iter() {
                        tree.evict_tenant_by_size(config.max_tree_size);
                    }
                },
            )
        });
        Self {
            cost: config.cost,
            kv_events: OnceLock::new(),
            trees,
            _eviction_task: eviction_task,
        }
    }

    /// Credits cache hits from the KV event index from now on.
    pub(crate) fn set_kv_events(&self, provider: Arc<dyn PrefixMatchProvider>) {
        let _ = self.kv_events.set(provider);
    }

    fn tree(&self, worker: &dyn Worker) -> Arc<Tree> {
        let key = normalize_model_key(worker.model_id());
        if let Some(tree) = self.trees.get(key) {
            return tree.clone();
        }
        self.trees
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .clone()
    }
}

impl Default for AgenticPolicy {
    fn default() -> Self {
        Self::new(AgenticConfig::default())
    }
}

/// Tokens estimated for the first `bytes` bytes of a request's text.
fn estimated_tokens(bytes: usize) -> usize {
    bytes.div_ceil(TEXT_BYTES_PER_TOKEN)
}

/// Each worker's hit in `tree`: the longest text match's worker holds its tokens,
/// estimated from its bytes, and every other worker none.
fn text_hits(tree: &Tree, text: &str, workers: &[Arc<dyn Worker>]) -> Vec<Option<usize>> {
    let matched = tree.prefix_match_with_counts(text);
    let bytes = text
        .char_indices()
        .nth(matched.matched_char_count)
        .map_or(text.len(), |(byte, _)| byte);
    workers
        .iter()
        .map(|worker| {
            let holds = bytes > 0 && worker.url() == matched.tenant.as_ref();
            Some(if holds { estimated_tokens(bytes) } else { 0 })
        })
        .collect()
}

#[async_trait]
impl LoadBalancingPolicy for AgenticPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        }
        let tokens = info.tokens.filter(|tokens| !tokens.is_empty());
        let text = info.request_text.filter(|text| !text.is_empty());
        let prompt_tokens = match (tokens, text) {
            (Some(tokens), _) => tokens.len(),
            (None, Some(text)) => estimated_tokens(text.len()),
            (None, None) => 0,
        };
        // The index cannot match media.
        let matchable = tokens.filter(|_| !info.carries_media);
        let provider = self.kv_events.get();
        let index_hits = matchable
            .zip(provider)
            .map(|(tokens, provider)| provider.match_workers(tokens, workers))
            .filter(|hits| healthy.iter().any(|&idx| hits[idx].is_some()));
        let tree = text.map(|_| self.tree(workers[healthy[0]].as_ref()));
        let (source, hits) = match (index_hits, text.zip(tree.as_deref())) {
            (Some(hits), _) => (
                Hits::Index,
                hits.into_iter()
                    .map(|hit| hit.map(|tokens| tokens as usize))
                    .collect(),
            ),
            (None, Some((text, tree))) => (Hits::Text, text_hits(tree, text, workers)),
            (None, None) => (Hits::Unmatched, vec![None; workers.len()]),
        };

        let candidates: Vec<Candidate> = healthy
            .iter()
            .map(|&idx| Candidate {
                worker_index: idx,
                cached_tokens: hits[idx].map(|cached| cached.min(prompt_tokens)),
                inflight_prefill_tokens: workers[idx].prefill_tokens(),
                inflight_requests: workers[idx].load(),
            })
            .collect();
        let Some(selection) = self.cost.select_worker(prompt_tokens, &candidates) else {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        };
        let selected = workers[selection.worker_index].as_ref();
        if let (Some(tokens), Some(provider)) = (matchable, provider) {
            provider.on_routed(tokens, selected);
        }
        if let (Some(text), Some(tree)) = (text, &tree) {
            tree.insert(text, selected.url());
        }
        selected.increment_processed();

        let longest = candidates
            .iter()
            .filter_map(|candidate| candidate.cached_tokens)
            .max()
            .unwrap_or(0);
        let reused = prompt_tokens - selection.uncached_tokens;
        let ratio =
            |tokens: usize| (prompt_tokens > 0).then(|| tokens as f64 / prompt_tokens as f64);
        let selected_known = hits[selection.worker_index].is_some();
        let (branch, estimated_hit_ratio) = match source {
            Hits::Index if longest == 0 => ("miss", ratio(0).filter(|_| selected_known)),
            Hits::Index if reused == longest => ("hit", ratio(reused)),
            Hits::Index => ("relocate", ratio(reused).filter(|_| selected_known)),
            Hits::Text if longest == 0 => ("text_miss", None),
            Hits::Text if reused == longest => ("text_hit", ratio(reused)),
            Hits::Text => ("text_relocate", None),
            Hits::Unmatched => ("unmatched", None),
        };
        let decision = RouteDecision {
            branch,
            match_ratio: (source != Hits::Unmatched)
                .then(|| ratio(longest))
                .flatten(),
            estimated_hit_ratio,
            prefill_tokens: Some(selection.uncached_tokens),
        };
        (Some(selection.worker_index), decision)
    }

    fn name(&self) -> &'static str {
        "agentic"
    }

    fn needs_request_text(&self) -> bool {
        true
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
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn candidate(index: usize, cached: Option<usize>, queued: usize, requests: usize) -> Candidate {
        Candidate {
            worker_index: index,
            cached_tokens: cached,
            inflight_prefill_tokens: queued,
            inflight_requests: requests,
        }
    }

    fn score(prompt_tokens: usize, candidate: &Candidate) -> Selection {
        AgenticCost::default().score(prompt_tokens, candidate)
    }

    fn select_worker(prompt_tokens: usize, candidates: &[Candidate]) -> Option<Selection> {
        AgenticCost::default().select_worker(prompt_tokens, candidates)
    }

    #[test]
    fn cache_reuse_wins_until_waiting_cost_exceeds_recomputation() {
        let mut workers = [
            candidate(3, Some(96_000), 10_000, 1),
            candidate(7, Some(0), 0, 1),
        ];
        assert_eq!(
            select_worker(100_000, &workers),
            Some(Selection {
                worker_index: 3,
                uncached_tokens: 4_000,
                cost: 14_512,
            })
        );
        workers[0].inflight_prefill_tokens = 120_000;
        assert_eq!(select_worker(100_000, &workers).unwrap().worker_index, 7);
    }

    #[test]
    fn request_pressure_can_outweigh_a_cache_hit() {
        let workers = [candidate(0, Some(512), 0, 2), candidate(1, Some(0), 0, 0)];
        let selected = select_worker(1_024, &workers).unwrap();
        assert_eq!((selected.worker_index, selected.cost), (1, 1_024));

        let requests_free = AgenticCost {
            request_token_equivalent: 0,
        };
        let selected = requests_free.select_worker(1_024, &workers).unwrap();
        assert_eq!((selected.worker_index, selected.cost), (0, 512));
    }

    #[test]
    fn unknown_hits_stay_eligible_without_cache_credit() {
        let unknown = candidate(4, None, 0, 0);
        assert_eq!(
            score(128, &unknown),
            score(128, &candidate(4, Some(0), 0, 0))
        );
        let busy = candidate(9, Some(128), 1_000, 0);
        let selected = select_worker(128, &[unknown, busy]).unwrap();
        assert_eq!((selected.worker_index, selected.uncached_tokens), (4, 128));
    }

    #[test]
    fn hit_credit_never_exceeds_the_prompt() {
        let worker = candidate(0, Some(1_000), 42, 1);
        for prompt in [0, 128, 1_000] {
            let selected = score(prompt, &worker);
            assert_eq!((selected.uncached_tokens, selected.cost), (0, 554));
        }
    }

    #[test]
    fn large_counters_preserve_order_without_overflow() {
        let workers = [
            candidate(0, Some(0), usize::MAX, usize::MAX),
            candidate(1, Some(0), usize::MAX - 1, usize::MAX),
        ];
        let selected = select_worker(usize::MAX, &workers).unwrap();
        assert_eq!(selected.worker_index, 1);
        assert_eq!(selected.cost, usize::MAX as u128 * 514 - 1);
    }

    #[test]
    fn ties_only_choose_minimum_cost_workers_without_list_order_bias() {
        let workers = [
            candidate(2, None, 0, 0),
            candidate(5, Some(512), 0, 1),
            candidate(8, Some(0), 0, 0),
            candidate(11, Some(0), 1, 0),
        ];
        let cost = AgenticCost::default();
        let mut rng = StdRng::seed_from_u64(17);
        let mut counts = [0; 3];
        for _ in 0..3_000 {
            let selected = cost.select_with_rng(1_024, &workers, &mut rng).unwrap();
            assert_eq!(selected.cost, 1_024);
            match selected.worker_index {
                2 => counts[0] += 1,
                5 => counts[1] += 1,
                8 => counts[2] += 1,
                other => panic!("non-minimum worker selected: {other}"),
            }
        }
        assert!(
            counts.iter().all(|&count| (800..1_200).contains(&count)),
            "{counts:?}"
        );
    }

    #[test]
    fn empty_candidates_return_no_selection() {
        assert_eq!(select_worker(128, &[]), None);
    }

    /// The KV event index's hits by worker URL (unlisted workers are not indexed),
    /// recording the prompts routed to each worker.
    #[derive(Debug, Default)]
    struct Index {
        hits: Vec<(&'static str, u32)>,
        routed: std::sync::Mutex<Vec<(usize, String)>>,
    }

    impl PrefixMatchProvider for Index {
        fn match_workers(&self, _tokens: &[u32], workers: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            let hit = |w: &Arc<dyn Worker>| self.hits.iter().find(|(url, _)| *url == w.url());
            workers
                .iter()
                .map(|w| hit(w).map(|&(_, hit)| hit))
                .collect()
        }

        fn on_routed(&self, tokens: &[u32], worker: &dyn Worker) {
            let routed = (tokens.len(), worker.url().to_string());
            self.routed.lock().unwrap().push(routed);
        }
    }

    fn index(hits: Vec<(&'static str, u32)>) -> Arc<Index> {
        Arc::new(Index {
            hits,
            ..Index::default()
        })
    }

    fn agentic(request_token_equivalent: u64, index: Option<&Arc<Index>>) -> AgenticPolicy {
        let cost = AgenticCost {
            request_token_equivalent,
        };
        let config = AgenticConfig {
            cost,
            eviction_interval_secs: 0,
            ..AgenticConfig::default()
        };
        let policy = AgenticPolicy::new(config);
        if let Some(index) = index {
            policy.set_kv_events(index.clone());
        }
        policy
    }

    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";

    fn workers() -> Vec<Arc<dyn Worker>> {
        [A, B]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect()
    }

    type Outcome = (Option<usize>, &'static str, Option<f64>, Option<usize>);

    async fn decide(p: &AgenticPolicy, w: &[Arc<dyn Worker>], i: &SelectWorkerInfo<'_>) -> Outcome {
        let (selected, d) = p.select_with_decision(w, i).await;
        (selected, d.branch, d.estimated_hit_ratio, d.prefill_tokens)
    }

    #[tokio::test]
    async fn kv_event_hits_hold_until_queued_prefill_outweighs_them() {
        let workers = workers();
        let index = index(vec![(A, 96), (B, 0)]);
        let policy = agentic(512, Some(&index));
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert!(policy.wants_tokens() && policy.needs_request_text());
        let hit = (Some(0), "hit", Some(0.96), Some(4));
        assert_eq!(decide(&policy, &workers, &info).await, hit);

        // 97 queued plus 4 to prefill on a, against all 100 on b.
        workers[0].charge_prefill_tokens(97);
        let relocate = (Some(1), "relocate", Some(0.0), Some(100));
        assert_eq!(decide(&policy, &workers, &info).await, relocate);
        workers[0].release_prefill_tokens(97);

        // One in-flight request weighs 512 tokens unless configured otherwise.
        workers[0].increment_load();
        assert_eq!(decide(&policy, &workers, &info).await.0, Some(1));
        let requests_free = agentic(0, Some(&index));
        assert_eq!(decide(&requests_free, &workers, &info).await, hit);

        // Every placement predicts its tokens on the selected worker.
        let routed = index.routed.lock().unwrap().clone();
        let routed: Vec<_> = routed.iter().map(|(t, url)| (*t, url.as_str())).collect();
        assert_eq!(routed, [(100, A), (100, B), (100, B), (100, A)]);

        // Unhealthy workers are never selected.
        workers[0].set_healthy(false);
        let miss = (Some(1), "miss", Some(0.0), Some(100));
        assert_eq!(decide(&policy, &workers, &info).await, miss);
        workers[1].set_healthy(false);
        let none = (None, NO_HEALTHY_WORKERS, None, None);
        assert_eq!(decide(&policy, &workers, &info).await, none);
    }

    #[tokio::test]
    async fn requests_the_index_cannot_match_fall_back_to_the_text_tree() {
        let workers = workers();
        workers[0].increment_load();
        let text = "x".repeat(400);
        let longer = format!("{text}{}", "y".repeat(400));
        let no_tokens = |text| SelectWorkerInfo {
            request_text: Some(text),
            ..Default::default()
        };

        // Without tokens the prompt is estimated at 4 bytes a token; the first request
        // lands by load and teaches the tree, and the next turn is credited its prefix.
        let policy = agentic(512, None);
        let first = (Some(1), "text_miss", None, Some(100));
        assert_eq!(decide(&policy, &workers, &no_tokens(&text)).await, first);
        let next = (Some(1), "text_hit", Some(0.5), Some(100));
        assert_eq!(decide(&policy, &workers, &no_tokens(&longer)).await, next);

        // Media, and tokens without indexed events, use the tree too, with the tokens
        // as the prompt; only the request without media predicts its blocks.
        let index = index(vec![]);
        let policy = agentic(512, Some(&index));
        let tokens: Vec<u32> = (0..10).collect();
        let media = SelectWorkerInfo {
            tokens: Some(&tokens),
            request_text: Some(&text),
            carries_media: true,
            ..Default::default()
        };
        let media_miss = (Some(1), "text_miss", None, Some(10));
        assert_eq!(decide(&policy, &workers, &media).await, media_miss);
        let unindexed = SelectWorkerInfo {
            carries_media: false,
            ..media.clone()
        };
        let tree_hit = (Some(1), "text_hit", Some(1.0), Some(0));
        assert_eq!(decide(&policy, &workers, &unindexed).await, tree_hit);
        assert_eq!(index.routed.lock().unwrap().len(), 1);

        let nothing = (Some(1), "unmatched", None, Some(0));
        assert_eq!(
            decide(&policy, &workers, &Default::default()).await,
            nothing
        );
    }
}
