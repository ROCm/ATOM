//! Agentic prefill cost over externally supplied cache hits and worker load.
//!
//! The caller filters eligible workers and obtains their reusable prefix lengths
//! from the shared prefix-match provider. This module only scores those snapshots
//! and selects a worker; it owns no cache index or load counters.
//!
//! Router/CLI integration follows joint validation with the KV-event provider.

use rand::{seq::IteratorRandom, Rng};

/// ATOM's default `ATOM_DP_LB_REQ_EQUIV`, expressed in tokens per request.
const REQUEST_TOKEN_EQUIVALENT: u128 = 512;

/// One eligible worker's inputs for this request, at `url@rank` granularity.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// Index in the caller's worker list, preserved even after health filtering.
    pub worker_index: usize,
    /// Leading tokens the unified hit provider says this worker can reuse.
    /// `None` means unknown and receives no cache credit. `Some(0)` is a known
    /// miss; neither case consults historical routing state here.
    /// Lower-tier hits must already be limited to the worker's usable load path.
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

/// Score one candidate using the default agentic cost function.
pub fn score(prompt_tokens: usize, candidate: &Candidate) -> Selection {
    let uncached_tokens = prompt_tokens.saturating_sub(candidate.cached_tokens.unwrap_or(0));
    Selection {
        worker_index: candidate.worker_index,
        uncached_tokens,
        cost: candidate.inflight_prefill_tokens as u128
            + uncached_tokens as u128
            + REQUEST_TOKEN_EQUIVALENT * candidate.inflight_requests as u128,
    }
}

/// Select the lowest-cost eligible worker, breaking ties uniformly at random.
///
/// An empty candidate list returns `None`. The caller remains responsible for
/// worker health, session affinity, rank mapping and charging the final placement.
pub fn select_worker(prompt_tokens: usize, candidates: &[Candidate]) -> Option<Selection> {
    select_with_rng(prompt_tokens, candidates, &mut rand::rng())
}

fn select_with_rng(
    prompt_tokens: usize,
    candidates: &[Candidate],
    rng: &mut impl Rng,
) -> Option<Selection> {
    let scored = candidates
        .iter()
        .map(|candidate| score(prompt_tokens, candidate));
    let best = scored.clone().map(|selection| selection.cost).min()?;
    scored.filter(|selection| selection.cost == best).choose(rng)
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    fn candidate(index: usize, cached: Option<usize>, queued: usize, requests: usize) -> Candidate {
        Candidate {
            worker_index: index,
            cached_tokens: cached,
            inflight_prefill_tokens: queued,
            inflight_requests: requests,
        }
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
    }

    #[test]
    fn unknown_hits_stay_eligible_without_cache_credit() {
        let unknown = candidate(4, None, 0, 0);
        assert_eq!(score(128, &unknown), score(128, &candidate(4, Some(0), 0, 0)));
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
        let mut rng = StdRng::seed_from_u64(17);
        let mut counts = [0; 3];
        for _ in 0..3_000 {
            let selected = select_with_rng(1_024, &workers, &mut rng).unwrap();
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
}
