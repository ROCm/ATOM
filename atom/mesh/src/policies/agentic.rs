//! `agentic`: places each request on the worker (`url@rank`) with the lowest
//! prefill cost, weighing the prefix the worker's GPU KV cache holds against the
//! prefill queued on it. [`AgenticCost`] scores snapshots and owns no state;
//! [`AgenticPolicy`] feeds it live hits and load and reports the prefill charge.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::Mutex;
use rand::{seq::IteratorRandom, Rng};

use super::{
    get_healthy_worker_indices, normalize_model_key, tree::Tree, utils::PeriodicTask,
    LoadBalancingPolicy, RouteDecision, SelectWorkerInfo, NO_HEALTHY_WORKERS,
};
use crate::{core::Worker, kv_index::PrefixMatchProvider};

/// Text bytes standing in for one token when a request's tokens are unknown.
const TEXT_BYTES_PER_TOKEN: usize = 4;

/// The agentic cost's weights:
///
/// ```text
/// cost = inflight_prefill_tokens + uncached_tokens
///      + request_token_equivalent * inflight_requests
///      - stay_margin, on a worker whose device cache holds the prompt's prefix
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgenticCost {
    /// Cost of one in-flight request, in tokens. The engine's `least_tokens`
    /// DP balancer weighs requests by `ATOM_DP_LB_REQ_EQUIV`; keep the two
    /// equal so the router and the engine rank a rank's load the same way.
    /// Default: 512.
    pub request_token_equivalent: u64,
    /// Tokens of prefill queued ahead that a request accepts on the worker whose
    /// device cache holds its prefix ([`Candidate::holds_prefix`]) before it is
    /// placed elsewhere. A request that leaves that worker has the prefix loaded
    /// from the store or computed again, and the queue it leaves is mostly other
    /// requests' short suffixes. Default: 0, the cost alone decides.
    pub stay_margin: u64,
}

impl Default for AgenticCost {
    fn default() -> Self {
        Self {
            request_token_equivalent: 512,
            stay_margin: 0,
        }
    }
}

/// One eligible worker's inputs for this request, at `url@rank` granularity.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// Index in the caller's worker list, preserved even after health filtering.
    pub worker_index: usize,
    /// Leading prompt tokens the worker's GPU KV cache is estimated to hold, with
    /// those its engine will copy in from its share group counted at the share
    /// weight and those it will load from a store at the store's weight; a load
    /// takes at least the store's load cost off the sum.
    /// `None` means unknown and receives no cache credit; `Some(0)` is a known
    /// miss. Both stay eligible.
    pub cached_tokens: Option<usize>,
    /// Tokens assigned to this worker whose prefill has not completed.
    pub inflight_prefill_tokens: usize,
    /// Requests assigned to this worker that have not completed.
    pub inflight_requests: usize,
    /// Its device cache holds the longest prefix of the prompt any candidate's does,
    /// and that is at least half the prompt: the worker [`AgenticCost::stay_margin`]
    /// keeps the request on.
    pub holds_prefix: bool,
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
        let cost = candidate.inflight_prefill_tokens as u128
            + uncached_tokens as u128
            + u128::from(self.request_token_equivalent) * candidate.inflight_requests as u128;
        let margin = if candidate.holds_prefix {
            u128::from(self.stay_margin)
        } else {
            0
        };
        Selection {
            worker_index: candidate.worker_index,
            uncached_tokens,
            cost: cost.saturating_sub(margin),
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
    /// Characters each worker keeps in the text tree after an eviction, as
    /// `cache_aware`'s `max_tree_size`: the cap is per worker, not per tree.
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

/// The prompts [`StoreShare`] remembers of each size, give or take a half.
const SHARE_WINDOW: u64 = 64;

/// What a store held, while it answered, of the tokens of a prompt that no device did:
/// both summed over the last prompts, apart for each size of that gap, a power of two
/// to the next. A turn's few tokens are new to the store, and a conversation no device
/// holds any longer is nearly all in it.
#[derive(Debug, Default)]
struct StoreShare(Mutex<[(u64, u64); u32::BITS as usize]>);

impl StoreShare {
    /// A store held `held` of the `gap` tokens of a prompt that no device did.
    fn learn(&self, gap: u32, held: u32) {
        let Some(size) = gap.checked_ilog2() else {
            return;
        };
        let (held_sum, gap_sum) = &mut self.0.lock()[size as usize];
        *held_sum += u64::from(held.min(gap));
        *gap_sum += u64::from(gap);
        if *gap_sum >= SHARE_WINDOW << size {
            (*held_sum, *gap_sum) = (*held_sum / 2, *gap_sum / 2);
        }
    }

    /// The tokens a store that does not answer is taken to hold of the `gap` no device
    /// does: none until one has answered for a gap of that size.
    fn of(&self, gap: u32) -> u32 {
        let Some(size) = gap.checked_ilog2() else {
            return 0;
        };
        let (held_sum, gap_sum) = self.0.lock()[size as usize];
        match gap_sum {
            0 => 0,
            _ => (u128::from(gap) * u128::from(held_sum) / u128::from(gap_sum)) as u32,
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
/// healthy candidate holding the longest text match.
#[derive(Debug)]
pub struct AgenticPolicy {
    cost: AgenticCost,
    kv_events: OnceLock<Arc<dyn PrefixMatchProvider>>,
    store_share: StoreShare,
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
            store_share: StoreShare::default(),
            trees,
            _eviction_task: eviction_task,
        }
    }

    /// Credits cache hits from the KV event index from now on.
    pub(crate) fn set_kv_events(&self, provider: Arc<dyn PrefixMatchProvider>) {
        let _ = self.kv_events.set(provider);
    }

    /// Forgets a removed worker's text in every tree.
    pub(crate) fn remove_worker_by_url(&self, url: &str) {
        for tree in self.trees.iter() {
            tree.remove_tenant(url);
        }
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

/// Each worker's hit in `tree`: the healthy candidate holding the longest text match
/// holds its tokens, estimated from its bytes, and every other worker none.
fn text_hits(
    tree: &Tree,
    text: &str,
    workers: &[Arc<dyn Worker>],
    healthy: &[usize],
) -> Vec<Option<usize>> {
    let candidates: Vec<&str> = healthy.iter().map(|&idx| workers[idx].url()).collect();
    let matched = tree.prefix_match_among(text, &candidates);
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
        // The workers whose device cache holds the longest prefix of the prompt, when
        // that is at least half of it: the ones `stay_margin` keeps the request on.
        let holders: Vec<bool> = match &index_hits {
            Some(device) => {
                let held = |idx: &usize| device[*idx];
                let longest = healthy.iter().filter_map(held).max().unwrap_or(0);
                let enough = longest > 0 && 2 * longest as usize >= prompt_tokens;
                (device.iter())
                    .map(|hit| enough && *hit == Some(longest))
                    .collect()
            }
            None => vec![false; workers.len()],
        };
        // In a share group, `pooled` is what each worker's device cache holds once its
        // engine has copied in the prefix the other ranks of its group hold; with a store,
        // `held` is what its engine then reports cached, loads included. The hits count a
        // copy at the share weight, and a load at the store's weight and at no less than
        // its load cost, which ranks the workers; the one selected is estimated and
        // charged by what it will hold.
        let shared = match (&index_hits, matchable.zip(provider)) {
            (Some(device), Some((tokens, provider))) => {
                provider.shared_tokens(tokens, workers, device)
            }
            _ => None,
        };
        let copies = shared.is_some();
        let (index_hits, held, copiers) = match (index_hits, matchable.zip(provider)) {
            (Some(device), Some((tokens, provider))) if copies || provider.store_weight() > 0.0 => {
                let (share_weight, weight) = (provider.share_weight(), provider.store_weight());
                let pooled = shared.unwrap_or_else(|| device.clone());
                let copied = |device: u32, pooled: u32| {
                    (share_weight * f64::from(pooled.saturating_sub(device))).round() as u32
                };
                let copiers: Vec<bool> = device
                    .iter()
                    .zip(&pooled)
                    .map(|(device, pooled)| pooled > device)
                    .collect();
                if weight <= 0.0 {
                    let credited = device.iter().zip(&pooled).map(|(&device, &pooled)| {
                        let (device, pooled) = (device?, pooled?);
                        Some(device + copied(device, pooled))
                    });
                    let credited = credited.collect::<Vec<_>>();
                    (Some(credited), Some(pooled), Some(copiers))
                } else {
                    let load_cost = provider.store_load_cost();
                    let cached = provider.cached_tokens(tokens, workers, &pooled).await;
                    // A store that does not answer (its lookup timed out, or its breaker
                    // is open) is taken to hold the longest prefix on a device, which the
                    // engine that computed it offloads: a device hit then weighs what it
                    // does while the store answers. It is not taken to hold the prompt, or
                    // a burst of cold prompts would be charged nothing and pile onto the
                    // few workers that seem free.
                    let known = pooled.iter().flatten().copied().max().unwrap_or(0);
                    let held: Vec<_> = pooled
                        .iter()
                        .zip(&cached)
                        .map(|(&pooled, &cached)| Some(cached.unwrap_or(known).max(pooled?)))
                        .collect();
                    let credited = (device.iter().zip(&pooled).zip(&held)).map(
                        |((&device, &pooled), &held)| {
                            let (device, pooled, held) = (device?, pooled?, held?);
                            let weighed = device
                                + copied(device, pooled)
                                + (weight * f64::from(held - pooled)).round() as u32;
                            // The engine admits a loaded prefix about as much later whatever
                            // its size. Priced by its size alone, a load of a turn's few
                            // tokens is nearly free, and the next turn leaves the worker that
                            // holds them all for one that held the conversation earlier and
                            // has a slightly shorter queue.
                            let loads = held > pooled;
                            let cost = if loads { load_cost } else { 0 };
                            Some(weighed.min(held.saturating_sub(cost)))
                        },
                    );
                    let credited = credited.collect::<Vec<_>>();
                    // Past that prefix a silent store is taken to hold what it held of such
                    // prompts while it answered, for the charge alone: every worker would
                    // load it alike. A conversation no device holds any longer is nearly
                    // all in the store. Charged whole, a burst of them weighs a hundred
                    // times what it computes, and once the store answers again the requests
                    // charged what they compute all go to the few workers charged least.
                    let prompt = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
                    let silent = |(pooled, cached): (&Option<u32>, &Option<u32>)| {
                        pooled.is_some() && cached.is_none()
                    };
                    if !pooled.iter().zip(&cached).any(silent) {
                        let stored = cached.iter().flatten().copied().max().unwrap_or(0);
                        let (gap, past) =
                            (prompt.saturating_sub(known), stored.saturating_sub(known));
                        self.store_share.learn(gap, past);
                    }
                    let reported = held.iter().zip(&cached).map(|(&held, cached)| {
                        let held = held?;
                        let past = match cached {
                            Some(_) => 0,
                            None => self.store_share.of(prompt.saturating_sub(held)),
                        };
                        Some(held + past)
                    });
                    let reported = reported.collect::<Vec<_>>();
                    (Some(credited), Some(reported), Some(copiers))
                }
            }
            (hits, _) => (hits, None, None),
        };
        let tree = text.map(|_| self.tree(workers[healthy[0]].as_ref()));
        let (source, hits) = match (index_hits, text.zip(tree.as_deref())) {
            (Some(hits), _) => (
                Hits::Index,
                hits.into_iter()
                    .map(|hit| hit.map(|tokens| tokens as usize))
                    .collect(),
            ),
            (None, Some((text, tree))) => (Hits::Text, text_hits(tree, text, workers, &healthy)),
            (None, None) => (Hits::Unmatched, vec![None; workers.len()]),
        };

        let candidates: Vec<Candidate> = healthy
            .iter()
            .map(|&idx| Candidate {
                worker_index: idx,
                cached_tokens: hits[idx].map(|cached| cached.min(prompt_tokens)),
                inflight_prefill_tokens: workers[idx].prefill_tokens(),
                inflight_requests: workers[idx].load(),
                holds_prefix: holders[idx],
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
        // The selected worker computes what neither its device holds nor its engine
        // loads, and its queue is charged that alone: a load waits on the store while
        // the GPU serves others. The engine's share_cache balancer charges the same.
        let cached = held
            .and_then(|held| held[selection.worker_index])
            .map_or(reused, |held| (held as usize).min(prompt_tokens));
        let ratio =
            |tokens: usize| (prompt_tokens > 0).then(|| tokens as f64 / prompt_tokens as f64);
        let selected_known = hits[selection.worker_index].is_some();
        // The selected worker copies part of what it will hold in from its share group.
        let copies_in = copiers.is_some_and(|copiers| copiers[selection.worker_index]);
        let (branch, estimated_hit_ratio) = match source {
            Hits::Index if longest == 0 => ("miss", ratio(cached).filter(|_| selected_known)),
            Hits::Index if copies_in => ("shared", ratio(cached)),
            Hits::Index if reused == longest => ("hit", ratio(cached)),
            Hits::Index => ("relocate", ratio(cached).filter(|_| selected_known)),
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
            prefill_tokens: Some(prompt_tokens - cached),
            reservation_id: None,
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
            holds_prefix: false,
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
            ..AgenticCost::default()
        };
        let selected = requests_free.select_worker(1_024, &workers).unwrap();
        assert_eq!((selected.worker_index, selected.cost), (0, 512));
    }

    #[test]
    fn the_stay_margin_keeps_a_request_where_its_prefix_is() {
        // a holds 900 of the 1000 tokens behind 5000 queued; b holds none and is idle.
        let holder = Candidate {
            holds_prefix: true,
            ..candidate(0, Some(900), 5_000, 0)
        };
        let workers = [holder, candidate(1, Some(0), 0, 0)];
        let cost = |stay_margin| AgenticCost {
            stay_margin,
            ..AgenticCost::default()
        };
        let selected = |margin| cost(margin).select_worker(1_000, &workers).unwrap();
        // 5100 on a against 1000 on b: b, until the margin covers their difference.
        assert_eq!(selected(0).worker_index, 1);
        assert_eq!(selected(4_000).worker_index, 1);
        let kept = selected(4_200);
        assert_eq!(
            (kept.worker_index, kept.uncached_tokens, kept.cost),
            (0, 100, 900)
        );
        // The margin comes off the cost alone, and never below zero.
        assert_eq!(selected(1 << 40).cost, 0);
        // A worker that does not hold the prefix gets none.
        let workers = [
            candidate(0, Some(900), 5_000, 0),
            candidate(1, Some(0), 0, 0),
        ];
        let selected = cost(1 << 40).select_worker(1_000, &workers).unwrap();
        assert_eq!(selected.worker_index, 1);
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
            ..AgenticCost::default()
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

    /// Each worker's device hit and what its engine reports cached once it loads from a
    /// store (None when the store does not answer), the weight of those loads, and what
    /// a load costs at least.
    #[derive(Debug)]
    struct Stored(Mutex<Vec<(u32, Option<u32>)>>, f64, u32);

    #[async_trait]
    impl PrefixMatchProvider for Stored {
        fn match_workers(&self, _tokens: &[u32], _: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            let hits = self.0.lock();
            hits.iter().map(|&(device, _)| Some(device)).collect()
        }

        async fn cached_tokens(
            &self,
            _: &[u32],
            _: &[Arc<dyn Worker>],
            device: &[Option<u32>],
        ) -> Vec<Option<u32>> {
            assert_eq!(device, self.match_workers(&[], &[]));
            let hits = self.0.lock();
            hits.iter().map(|&(_, cached)| cached).collect()
        }

        fn store_weight(&self) -> f64 {
            self.1
        }

        fn store_load_cost(&self) -> u32 {
            self.2
        }
    }

    /// Each worker's device hit, what its device holds once its engine has copied in from
    /// its share group, and what its engine reports cached once it has loaded from a
    /// store as well; the weight of those copies and of those loads.
    #[derive(Debug)]
    struct Pooled(Vec<(u32, u32, u32)>, f64, f64);

    #[async_trait]
    impl PrefixMatchProvider for Pooled {
        fn match_workers(&self, _tokens: &[u32], _: &[Arc<dyn Worker>]) -> Vec<Option<u32>> {
            self.0.iter().map(|&(device, ..)| Some(device)).collect()
        }

        fn shared_tokens(
            &self,
            _: &[u32],
            _: &[Arc<dyn Worker>],
            device: &[Option<u32>],
        ) -> Option<Vec<Option<u32>>> {
            assert_eq!(device, self.match_workers(&[], &[]));
            (self.1 > 0.0).then(|| self.0.iter().map(|&(_, pooled, _)| Some(pooled)).collect())
        }

        fn share_weight(&self) -> f64 {
            self.1
        }

        async fn cached_tokens(
            &self,
            _: &[u32],
            _: &[Arc<dyn Worker>],
            device: &[Option<u32>],
        ) -> Vec<Option<u32>> {
            // The store is asked past what a device will hold, its copies included.
            let asked = |&(device, pooled, _): &(u32, u32, u32)| {
                Some(if self.1 > 0.0 { pooled } else { device })
            };
            assert_eq!(device, self.0.iter().map(asked).collect::<Vec<_>>());
            self.0.iter().map(|&(.., cached)| Some(cached)).collect()
        }

        fn store_weight(&self) -> f64 {
            self.2
        }
    }

    #[tokio::test]
    async fn tokens_a_worker_copies_in_from_its_share_group_count_at_the_share_weight() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // a's device holds 60 of the 100 tokens; b's 20, and its share group 80. The weight
        // says which of them leads; b, once chosen, is estimated to report the 80 it will
        // hold and charged the 20 it computes, whatever the weight.
        for (weight, want) in [
            (0.0, (Some(0), "hit", Some(0.6), Some(40))),
            (0.5, (Some(0), "hit", Some(0.6), Some(40))),
            (0.9, (Some(1), "shared", Some(0.8), Some(20))),
            (1.0, (Some(1), "shared", Some(0.8), Some(20))),
        ] {
            let policy = agentic(512, None);
            let pooled = Pooled(vec![(60, 60, 60), (20, 80, 80)], weight, 0.0);
            policy.set_kv_events(Arc::new(pooled));
            assert_eq!(
                decide(&policy, &workers, &info).await,
                want,
                "weight {weight}"
            );
        }
    }

    #[tokio::test]
    async fn a_store_is_asked_past_what_a_share_group_holds() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // a's device holds 40 of the 100 tokens. b's holds none, its share group 50, and
        // its store the 40 after those: b's copies and loads count at their weights, and
        // b, once chosen, is estimated to report 90 and charged the 10 it computes.
        let b_leads = (Some(1), "shared", Some(0.9), Some(10));
        for (share, store, want) in [
            (0.9, 0.5, b_leads),
            (0.9, 0.1, b_leads),
            (0.5, 0.1, (Some(0), "hit", Some(0.4), Some(60))),
            // Without share groups the store is asked past the device alone.
            (0.0, 0.5, (Some(1), "hit", Some(0.9), Some(10))),
        ] {
            let policy = agentic(512, None);
            let pooled = Pooled(vec![(40, 40, 40), (0, 50, 90)], share, store);
            policy.set_kv_events(Arc::new(pooled));
            assert_eq!(
                decide(&policy, &workers, &info).await,
                want,
                "share weight {share}, store weight {store}"
            );
        }
    }

    #[tokio::test]
    async fn tokens_a_worker_loads_from_its_store_count_at_the_store_weight() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // a's device holds 40 of the 100 tokens; b's none, but its store 90. The weight
        // says which of them leads; b, once chosen, is estimated to report the 90 it
        // loads and charged the 10 it computes, whatever the weight.
        for (weight, b_cached, want) in [
            (0.0, Some(90), (Some(0), "hit", Some(0.4), Some(60))),
            (0.3, Some(90), (Some(0), "hit", Some(0.4), Some(60))),
            (0.5, Some(90), (Some(1), "hit", Some(0.9), Some(10))),
            (1.0, Some(90), (Some(1), "hit", Some(0.9), Some(10))),
        ] {
            let policy = agentic(512, None);
            policy.set_kv_events(Arc::new(Stored(
                vec![(40, Some(40)), (0, b_cached)].into(),
                weight,
                0,
            )));
            assert_eq!(
                decide(&policy, &workers, &info).await,
                want,
                "weight {weight}"
            );
        }
    }

    #[tokio::test]
    async fn a_store_load_costs_no_less_than_the_store_load_cost() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // b's device holds 98 of the 100 tokens, as it ran the turn before; a's 90, from
        // an earlier turn, and its store the 8 more. Either is estimated to report 98
        // and charged the 2 it computes.
        let stored =
            |load_cost| Stored(vec![(90, Some(98)), (98, Some(98))].into(), 0.5, load_cost);
        let stays = (Some(1), "hit", Some(0.98), Some(2));
        let moves = (Some(0), "relocate", Some(0.98), Some(2));
        // By its size the load costs 4 tokens: 5 queued on b send the turn to a.
        let policy = agentic(512, None);
        policy.set_kv_events(Arc::new(stored(0)));
        assert_eq!(decide(&policy, &workers, &info).await, stays);
        workers[1].charge_prefill_tokens(5);
        assert_eq!(decide(&policy, &workers, &info).await, moves);
        // At 50 tokens a load, the turn stays until b's queue is over 50 ahead of a's.
        let policy = agentic(512, None);
        policy.set_kv_events(Arc::new(stored(50)));
        workers[1].charge_prefill_tokens(44);
        assert_eq!(decide(&policy, &workers, &info).await, stays);
        workers[1].charge_prefill_tokens(2);
        assert_eq!(decide(&policy, &workers, &info).await, moves);
        // A load costs no more than computing all the worker would hold.
        let policy = agentic(512, None);
        policy.set_kv_events(Arc::new(stored(1000)));
        workers[1].charge_prefill_tokens(46);
        assert_eq!(decide(&policy, &workers, &info).await, stays);
        workers[1].charge_prefill_tokens(2);
        assert_eq!(decide(&policy, &workers, &info).await, moves);
    }

    #[tokio::test]
    async fn a_store_that_does_not_answer_is_taken_to_hold_what_a_device_does() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        // a's device holds 40 of the 100 tokens, b's none, and the store both read
        // does not answer: a keeps its lead, and is charged the 60 it computes.
        let policy = agentic(512, None);
        let silent = |device: [u32; 2]| device.map(|device| (device, None)).to_vec();
        policy.set_kv_events(Arc::new(Stored(silent([40, 0]).into(), 0.5, 0)));
        let hit = (Some(0), "hit", Some(0.4), Some(60));
        assert_eq!(decide(&policy, &workers, &info).await, hit);
        // b, chosen over a's queue, is taken to load those 40 and charged as much.
        workers[0].charge_prefill_tokens(1000);
        let relocate = (Some(1), "relocate", Some(0.4), Some(60));
        assert_eq!(decide(&policy, &workers, &info).await, relocate);
        workers[0].release_prefill_tokens(1000);
        // With nothing on a device, the one chosen is charged the whole prompt.
        let policy = agentic(512, None);
        let stored = Arc::new(Stored(silent([0, 0]).into(), 0.5, 0));
        policy.set_kv_events(stored.clone());
        // Either worker is chosen: they cost the same.
        let charged = || async {
            let (_, branch, ratio, charge) = decide(&policy, &workers, &info).await;
            (branch, ratio, charge)
        };
        assert_eq!(charged().await, ("miss", Some(0.0), Some(100)));
        // Unless the store held 90 of such a prompt when it last answered: the one
        // chosen is then charged the 10 it would compute, as it was while it answered.
        *stored.0.lock() = vec![(0, Some(90)), (0, Some(90))];
        assert_eq!(charged().await, ("hit", Some(0.9), Some(10)));
        *stored.0.lock() = silent([0, 0]);
        assert_eq!(charged().await, ("miss", Some(0.9), Some(10)));
        // Of a turn's few tokens past what a device holds it never held any.
        *stored.0.lock() = silent([98, 0]);
        let turn = (Some(0), "hit", Some(0.98), Some(2));
        assert_eq!(decide(&policy, &workers, &info).await, turn);
    }

    #[test]
    fn a_store_share_follows_the_last_prompts_of_a_size() {
        let share = StoreShare::default();
        assert_eq!(share.of(100_000), 0);
        // By the size of what no device holds: 65536 to 131071 tokens here.
        share.learn(100_000, 97_000);
        let of = |gaps: [u32; 4]| gaps.map(|gap| share.of(gap));
        assert_eq!(
            of([80_000, 131_071, 4_000, 131_072]),
            [77_600, 127_138, 0, 0]
        );
        // A store that then holds none of them is soon taken to hold nearly none.
        for _ in 0..4 * SHARE_WINDOW {
            share.learn(100_000, 0);
        }
        assert!(share.of(100_000) < 100, "{}", share.of(100_000));
        assert_eq!((share.of(0), share.of(u32::MAX)), (0, 0));
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
    async fn the_stay_margin_holds_a_request_on_the_worker_with_its_prefix() {
        let workers = workers();
        let tokens: Vec<u32> = (0..100).collect();
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        let (held, short) = (index(vec![(A, 96), (B, 0)]), index(vec![(A, 40), (B, 0)]));
        let with_margin = |index: &Arc<Index>, stay_margin| {
            let cost = AgenticCost {
                stay_margin,
                ..AgenticCost::default()
            };
            let policy = AgenticPolicy::new(AgenticConfig {
                cost,
                eviction_interval_secs: 0,
                ..AgenticConfig::default()
            });
            policy.set_kv_events(index.clone());
            policy
        };
        // a holds 96 of the 100 tokens behind 1000 queued: 1004 on a against 100 on b.
        workers[0].charge_prefill_tokens(1_000);
        let relocate = (Some(1), "relocate", Some(0.0), Some(100));
        let hit = (Some(0), "hit", Some(0.96), Some(4));
        let policy = with_margin(&held, 0);
        assert_eq!(decide(&policy, &workers, &info).await, relocate);
        let policy = with_margin(&held, 900);
        assert_eq!(decide(&policy, &workers, &info).await, relocate);
        let policy = with_margin(&held, 905);
        assert_eq!(decide(&policy, &workers, &info).await, hit);
        // A prefix under half the prompt is not worth a queue: no margin for it.
        let policy = with_margin(&short, 1 << 30);
        assert_eq!(decide(&policy, &workers, &info).await, relocate);
        workers[0].release_prefill_tokens(1_000);
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
        // Worker 1 placed the shared prefix first; worker 0, sharing it, is credited it.
        workers[1].set_healthy(false);
        let [joins, asks] = ["z", "w"].map(|tail| format!("{text}{}", tail.repeat(400)));
        let joined = (Some(0), "text_miss", None, Some(200));
        assert_eq!(decide(&policy, &workers, &no_tokens(&joins)).await, joined);
        let shared = (Some(0), "text_hit", Some(0.5), Some(100));
        assert_eq!(decide(&policy, &workers, &no_tokens(&asks)).await, shared);
        workers[1].set_healthy(true);

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
