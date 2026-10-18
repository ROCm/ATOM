//! DP rank placement and in-flight load, kept the way CoreManager keeps them
//! (`engine_core_mgr.py`, `_select_dp_rank_for_seq_locked` and friends).
//!
//! A sequence is charged one request and its prompt tokens when placed. The
//! tokens are released at its first output (prefill is done), the request when
//! it ends. With session affinity a session sticks to the rank it first landed
//! on, and is charged only the growth of its prompt.

use std::collections::HashMap;

use super::spec::{BalanceSpec, BalanceStrategy};

pub(super) struct Balancer {
    spec: BalanceSpec,
    alive: Vec<bool>,
    reqs: Vec<u64>,
    tokens: Vec<u64>,
    cursor: usize,
    session_owner: HashMap<String, usize>,
    session_prompt: HashMap<String, u64>,
    charges: HashMap<u64, Charge>,
}

struct Charge {
    rank: usize,
    tokens: u64,
}

impl Balancer {
    pub fn new(spec: BalanceSpec, ranks: usize) -> Self {
        Self {
            spec,
            alive: vec![true; ranks],
            reqs: vec![0; ranks],
            tokens: vec![0; ranks],
            cursor: 0,
            session_owner: HashMap::new(),
            session_prompt: HashMap::new(),
            charges: HashMap::new(),
        }
    }

    /// Place sequence `seq` and charge its rank; `None` when no rank is alive.
    pub fn admit(&mut self, seq: u64, prompt_tokens: u64, session: Option<&str>) -> Option<usize> {
        let session = session.filter(|_| self.spec.session_affinity);
        let rank = match session {
            Some(id) => match self.session_owner.get(id) {
                Some(&owner) if self.alive[owner] => owner,
                _ => {
                    let owner = self.place_session(id)?;
                    self.session_owner.insert(id.to_string(), owner);
                    self.session_prompt.remove(id);
                    owner
                }
            },
            None => self.place()?,
        };
        let mut tokens = prompt_tokens;
        if let Some(id) = session {
            if let Some(previous) = self.session_prompt.insert(id.to_string(), prompt_tokens) {
                tokens = prompt_tokens.saturating_sub(previous);
            }
        }
        self.reqs[rank] += 1;
        self.tokens[rank] += tokens;
        self.charges.insert(seq, Charge { rank, tokens });
        Some(rank)
    }

    pub fn prefill_done(&mut self, seq: u64) {
        if let Some(charge) = self.charges.get_mut(&seq) {
            self.tokens[charge.rank] -= charge.tokens;
            charge.tokens = 0;
        }
    }

    pub fn release(&mut self, seq: u64) {
        if let Some(charge) = self.charges.remove(&seq) {
            self.reqs[charge.rank] -= 1;
            self.tokens[charge.rank] -= charge.tokens;
        }
    }

    pub fn mark_dead(&mut self, rank: usize) {
        self.alive[rank] = false;
    }

    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&alive| alive).count()
    }

    /// In-flight requests and queued prefill tokens, per rank.
    pub fn loads(&self) -> (Vec<u64>, Vec<u64>) {
        (self.reqs.clone(), self.tokens.clone())
    }

    fn load(&self, rank: usize) -> u64 {
        self.tokens[rank] + self.spec.req_equiv * self.reqs[rank]
    }

    fn place(&mut self) -> Option<usize> {
        let n = self.alive.len();
        let offset = self.cursor % n;
        self.cursor += 1;
        let mut ranks = (0..n).map(|i| (offset + i) % n).filter(|&r| self.alive[r]);
        match self.spec.strategy {
            BalanceStrategy::RoundRobin => ranks.next(),
            BalanceStrategy::LeastRequests => ranks.min_by_key(|&r| (self.reqs[r], self.tokens[r])),
            BalanceStrategy::LeastTokens => ranks.min_by_key(|&r| self.load(r)),
        }
    }

    /// The lightest rank for a new session; a stable hash breaks ties.
    fn place_session(&self, session: &str) -> Option<usize> {
        (0..self.alive.len())
            .filter(|&r| self.alive[r])
            .min_by_key(|&r| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(session.as_bytes());
                hasher.update(&(r as u32).to_le_bytes());
                let tie = u64::from_le_bytes(hasher.finalize().as_bytes()[..8].try_into().unwrap());
                (self.load(r), std::cmp::Reverse(tie))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn balancer(strategy: BalanceStrategy, ranks: usize, affinity: bool) -> Balancer {
        Balancer::new(
            BalanceSpec {
                strategy,
                req_equiv: 100,
                session_affinity: affinity,
            },
            ranks,
        )
    }

    #[test]
    fn least_requests_spreads_and_releases() {
        let mut b = balancer(BalanceStrategy::LeastRequests, 2, false);
        let first = b.admit(0, 10, None).unwrap();
        let second = b.admit(1, 10, None).unwrap();
        assert_ne!(first, second);
        b.release(0);
        assert_eq!(b.admit(2, 10, None), Some(first));
        b.release(0);
        assert_eq!(b.reqs, vec![1, 1]);
    }

    #[test]
    fn prefill_done_releases_tokens_only() {
        let mut b = balancer(BalanceStrategy::LeastTokens, 1, false);
        b.admit(7, 50, None);
        b.prefill_done(7);
        assert_eq!((b.reqs[0], b.tokens[0]), (1, 0));
        b.release(7);
        assert_eq!((b.reqs[0], b.tokens[0]), (0, 0));
    }

    #[test]
    fn session_sticks_and_is_charged_its_growth() {
        let mut b = balancer(BalanceStrategy::LeastRequests, 4, true);
        let owner = b.admit(0, 100, Some("s")).unwrap();
        for seq in 1..4 {
            b.admit(seq, 0, None);
        }
        assert_eq!(b.admit(9, 130, Some("s")), Some(owner));
        assert_eq!(b.charges[&9].tokens, 30);
    }

    #[test]
    fn dead_ranks_are_skipped() {
        let mut b = balancer(BalanceStrategy::RoundRobin, 2, true);
        b.mark_dead(0);
        assert!((0..4).all(|seq| b.admit(seq, 1, None) == Some(1)));
        assert_eq!(b.admit(9, 1, Some("s")), Some(1));
        b.mark_dead(1);
        assert_eq!(b.admit(10, 1, None), None);
    }
}
