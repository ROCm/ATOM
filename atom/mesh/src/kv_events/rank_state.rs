//! Per-rank admission and recovery state. Ranks are independent: a broken stream only
//! affects its own `url@rank` holder.
//!
//! No engine offers snapshots yet, so the snapshot branches of the admission table wait
//! for one; a gap that replay cannot fill clears the holder.

use std::{
    collections::VecDeque,
    mem,
    time::{Duration, Instant},
};

use serde::Serialize;

/// Live batches held while a replay is pending; one more gives the gap up.
const MAX_HELD: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    #[default]
    Bootstrap,
    Live,
    Recovering,
    Stale,
}

/// Whether the holder's index entries match the engine. `Partial` results are a lower
/// bound and get merged with the predicted source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    Exact,
    #[default]
    Partial,
}

/// What admitting a batch asks the caller to do, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action<B> {
    Apply(B),
    /// Clear the holder and apply the batch from its un-tiered `AllBlocksCleared` on.
    ApplyFromClear(B),
    /// A duplicate of an applied batch.
    Skip,
    RequestReplay {
        from: u64,
    },
    ClearHolder,
}

/// What the caller knows about a batch before applying it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchMeta {
    pub seq: u64,
    pub ts: f64,
    /// Came from a replay reply rather than the live stream.
    pub replayed: bool,
    /// Holds an un-tiered `AllBlocksCleared`.
    pub clears: bool,
}

/// From the source's caps: `replay`, and `stale_after` only if it declares a heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankPolicy {
    pub replay: bool,
    pub timeout: Duration,
    pub stale_after: Option<Duration>,
}

/// Admission table of one rank (§3.3.5). `B` is the decoded batch, handed back in an
/// `Apply` action once it is in order.
#[derive(Debug, Clone)]
pub struct RankState<B> {
    policy: RankPolicy,
    expected: u64,
    trust: Trust,
    phase: Phase,
    last_ts: Option<f64>,
    last_seen: Option<Instant>,
    /// Set while a replay is pending.
    deadline: Option<Instant>,
    held: VecDeque<(BatchMeta, B)>,
    /// Batches were lost since the last bootstrap or clear.
    lost: bool,
    /// Trust before going stale.
    prior: Trust,
}

impl<B> RankState<B> {
    pub fn new(policy: RankPolicy) -> Self {
        Self {
            policy,
            expected: 0,
            trust: Trust::Partial,
            phase: Phase::Bootstrap,
            last_ts: None,
            last_seen: None,
            deadline: None,
            held: VecDeque::new(),
            lost: false,
            prior: Trust::Partial,
        }
    }

    /// Next sequence number the rank expects.
    pub fn expected(&self) -> u64 {
        self.expected
    }

    pub fn trust(&self) -> Trust {
        self.trust
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Timestamp of the newest applied batch.
    pub fn last_ts(&self) -> Option<f64> {
        self.last_ts
    }

    pub fn admit(&mut self, meta: BatchMeta, batch: B, now: Instant) -> Vec<Action<B>> {
        // The replay end sentinel, never a batch.
        if meta.seq == u64::MAX {
            return vec![Action::Skip];
        }
        let mut out = Vec::new();
        self.last_seen = Some(now);
        if self.phase == Phase::Stale {
            self.phase = Phase::Live;
            self.trust = self.prior;
        }
        self.route(meta, batch, &mut out, now);
        out
    }

    /// The replay reply ended.
    pub fn replay_finished(&mut self, now: Instant) -> Vec<Action<B>> {
        let mut out = Vec::new();
        if self.deadline.is_some() {
            self.resume(&mut out, now);
        }
        out
    }

    /// Replay deadline and staleness.
    pub fn tick(&mut self, now: Instant) -> Vec<Action<B>> {
        let mut out = Vec::new();
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            self.resume(&mut out, now);
        } else if let (Phase::Live, Some(after), Some(seen)) =
            (self.phase, self.policy.stale_after, self.last_seen)
        {
            if now.duration_since(seen) > after {
                self.prior = self.trust;
                self.trust = Trust::Partial;
                self.phase = Phase::Stale;
            }
        }
        out
    }

    /// The engine restarted: a new incarnation, or the worker recovered from unhealthy.
    pub fn restart(&mut self) -> Vec<Action<B>> {
        *self = Self::new(self.policy);
        vec![Action::ClearHolder]
    }

    /// Live batches wait while a replay is pending; a live clear ends the replay.
    fn route(&mut self, meta: BatchMeta, batch: B, out: &mut Vec<Action<B>>, now: Instant) {
        if meta.clears && !meta.replayed && !self.duplicate(&meta) {
            self.deadline = None;
            self.held.clear();
            self.phase = Phase::Live;
        }
        if self.deadline.is_none() || meta.replayed {
            return self.step(meta, batch, out, now);
        }
        self.held.push_back((meta, batch));
        if self.held.len() > MAX_HELD {
            self.resume(out, now);
        }
    }

    fn step(&mut self, meta: BatchMeta, batch: B, out: &mut Vec<Action<B>>, now: Instant) {
        if self.phase == Phase::Bootstrap && self.deadline.is_none() {
            return self.first(meta, batch, out, now);
        }
        if self.duplicate(&meta) {
            return out.push(Action::Skip);
        }
        let behind = meta.seq < self.expected;
        if meta.clears {
            self.trust = Trust::Exact;
            self.lost = false;
            self.accept(meta);
            return out.push(Action::ApplyFromClear(batch));
        }
        if behind {
            // The engine restarted and its first batches were missed.
            *self = Self {
                last_seen: self.last_seen,
                ..Self::new(self.policy)
            };
            out.push(Action::ClearHolder);
            return self.first(meta, batch, out, now);
        }
        if meta.seq == self.expected {
            self.accept(meta);
            return out.push(Action::Apply(batch));
        }
        if self.policy.replay && !meta.replayed {
            self.phase = Phase::Recovering;
            return self.wait(meta, batch, out, now);
        }
        // The replay window no longer reaches `expected`, or the source cannot replay.
        self.lose(out);
        self.accept(meta);
        out.push(Action::Apply(batch));
    }

    fn first(&mut self, meta: BatchMeta, batch: B, out: &mut Vec<Action<B>>, now: Instant) {
        if meta.clears || meta.seq == 0 {
            self.phase = Phase::Live;
            self.trust = Trust::Exact;
            self.accept(meta);
            return out.push(if meta.clears {
                Action::ApplyFromClear(batch)
            } else {
                Action::Apply(batch)
            });
        }
        if self.policy.replay {
            return self.wait(meta, batch, out, now);
        }
        self.phase = Phase::Live;
        self.accept(meta);
        out.push(Action::Apply(batch));
    }

    /// Hold the batch and replay what is missing before it.
    fn wait(&mut self, meta: BatchMeta, batch: B, out: &mut Vec<Action<B>>, now: Instant) {
        self.deadline = Some(now + self.policy.timeout);
        self.held.push_back((meta, batch));
        out.push(Action::RequestReplay {
            from: self.expected,
        });
    }

    /// End the pending replay and admit the held batches. A gap still open before them is
    /// given up.
    fn resume(&mut self, out: &mut Vec<Action<B>>, now: Instant) {
        self.deadline = None;
        let held = mem::take(&mut self.held);
        if let Some((first, _)) = held.front() {
            if first.seq > self.expected && !first.clears {
                self.lose(out);
                self.expected = first.seq;
            }
        }
        if self.phase == Phase::Bootstrap && !self.lost {
            self.trust = Trust::Exact;
        }
        self.phase = Phase::Live;
        for (meta, batch) in held {
            self.route(meta, batch, out, now);
        }
    }

    fn duplicate(&self, meta: &BatchMeta) -> bool {
        meta.seq < self.expected && self.last_ts.is_none_or(|ts| meta.ts <= ts)
    }

    fn lose(&mut self, out: &mut Vec<Action<B>>) {
        if self.last_ts.is_some() {
            out.push(Action::ClearHolder);
        }
        self.trust = Trust::Partial;
        self.lost = true;
    }

    fn accept(&mut self, meta: BatchMeta) {
        self.expected = meta.seq + 1;
        self.last_ts = Some(self.last_ts.map_or(meta.ts, |ts| ts.max(meta.ts)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One case per line: name | policy | steps | final phase and trust. Steps: `L3` is live
    /// batch 3 with ts 3, `R3` the same replayed, `LC3`/`RC3` hold an un-tiered clear, `@9`
    /// sets the ts; `E` ends the replay, `T5` lets five seconds pass, `X` restarts the
    /// engine, `?live exact` checks the state. `-> ...` lists the expected actions.
    const TABLE: &str = "
        first batch is seq 0              | replay    | L0 -> apply 0 | live exact
        first batch replays from seq 0    | replay    | L2 -> replay 0; ?bootstrap partial; R0 -> apply 0; R1 -> apply 1; R2 -> apply 2; E -> skip | live exact
        first batch, replay window gone   | replay    | L5 -> replay 0; R3 -> apply 3; R4 -> apply 4; E -> apply 5 | live partial
        first batch, replay times out     | replay    | L5 -> replay 0; T5 -> apply 5 | live partial
        first batch without replay        | none      | L5 -> apply 5 | live partial
        first batch holds a clear         | replay    | LC7 -> from_clear 7 | live exact
        duplicates are skipped            | replay    | L0 -> apply 0; L1 -> apply 1; R0 -> skip; L1 -> skip | live exact
        clear makes a gap irrelevant      | none      | L0 -> apply 0; L3 -> clear, apply 3; LC4 -> from_clear 4; L5 -> apply 5 | live exact
        restart, first batches missed     | replay    | L0 -> apply 0; L1 -> apply 1; L2 -> apply 2; L1@9 -> clear, replay 0; R0@8 -> apply 0; R1@9 -> apply 1; E -> skip | live exact
        restart with the startup clear    | replay    | L0 -> apply 0; L1 -> apply 1; LC0@9 -> from_clear 0 | live exact
        contiguous                        | replay    | L0 -> apply 0; L1 -> apply 1; L2 -> apply 2 | live exact
        gap filled by replay              | replay    | L0 -> apply 0; L3 -> replay 1; L4; ?recovering exact; R1 -> apply 1; R2 -> apply 2; R3 -> apply 3; E -> skip, apply 4 | live exact
        replay timeout gives the gap up   | replay    | L0 -> apply 0; L3 -> replay 1; L4; T4; T1 -> clear, apply 3, apply 4; E | live partial
        replay window gone                | replay    | L0 -> apply 0; L4 -> replay 1; R3 -> clear, apply 3; R4 -> apply 4; E -> skip | live partial
        gap without replay                | none      | L0 -> apply 0; L3 -> clear, apply 3; L4 -> apply 4 | live partial
        stale until the next batch        | heartbeat | L0 -> apply 0; T31; ?stale partial; L1 -> apply 1 | live exact
        stale, then a gap                 | heartbeat | L0 -> apply 0; T31; L2 -> replay 1 | recovering exact
        never stale without a heartbeat   | replay    | L0 -> apply 0; T60 | live exact
        out of order                      | replay    | L0 -> apply 0; L2 -> replay 1; L1; R1 -> apply 1; R2 -> apply 2; E -> skip, skip | live exact
        old clear in a replay             | replay    | L0 -> apply 0; L1 -> apply 1; L2 -> apply 2; L5 -> replay 3; RC1 -> skip; R3 -> apply 3; R4 -> apply 4; E -> apply 5 | live exact
        held batch opens another gap      | replay    | L0 -> apply 0; L2 -> replay 1; L5; R1 -> apply 1; R2 -> apply 2; E -> skip, replay 3; R3 -> apply 3; R4 -> apply 4; E -> apply 5 | live exact
        new incarnation                   | replay    | L0 -> apply 0; X -> clear; ?bootstrap partial; L0@9 -> apply 0 | live exact
        live clear ends a pending replay  | replay    | L0 -> apply 0; L3 -> replay 1; LC4 -> from_clear 4; E; L5 -> apply 5 | live exact
        restart keeps the staleness clock | heartbeat | L0 -> apply 0; L1 -> apply 1; L0@9 -> clear, apply 0; T31 | stale partial
        an older ts is not a restart      | replay    | L0@5 -> apply 0; L1@3 -> apply 1; R0@4 -> skip | live exact
        the replay end sentinel is no batch | replay  | L0 -> apply 0; L18446744073709551615 -> skip; L1 -> apply 1 | live exact
    ";

    fn policy(name: &str) -> RankPolicy {
        RankPolicy {
            replay: name != "none",
            timeout: Duration::from_secs(5),
            stale_after: (name == "heartbeat").then_some(Duration::from_secs(30)),
        }
    }

    fn describe(state: &RankState<u64>) -> String {
        format!("{:?} {:?}", state.phase(), state.trust()).to_lowercase()
    }

    fn render(actions: &[Action<u64>]) -> String {
        let actions: Vec<String> = actions
            .iter()
            .map(|action| match action {
                Action::Apply(seq) => format!("apply {seq}"),
                Action::ApplyFromClear(seq) => format!("from_clear {seq}"),
                Action::Skip => "skip".to_string(),
                Action::RequestReplay { from } => format!("replay {from}"),
                Action::ClearHolder => "clear".to_string(),
            })
            .collect();
        actions.join(", ")
    }

    fn run(state: &mut RankState<u64>, script: &str, now: &mut Instant) {
        for step in script.split(';').map(str::trim) {
            let (event, want) = step.split_once(" -> ").unwrap_or((step, ""));
            let actions = match &event[..1] {
                "E" => state.replay_finished(*now),
                "X" => state.restart(),
                "T" => {
                    *now += Duration::from_secs(event[1..].parse().unwrap());
                    state.tick(*now)
                }
                "?" => {
                    assert_eq!(describe(state), event[1..], "{script}: {step}");
                    Vec::new()
                }
                _ => {
                    let clears = event[1..].starts_with('C');
                    let digits = &event[1 + usize::from(clears)..];
                    let (seq, ts) = digits.split_once('@').unwrap_or((digits, digits));
                    let meta = BatchMeta {
                        seq: seq.parse().unwrap(),
                        ts: ts.parse().unwrap(),
                        replayed: event.starts_with('R'),
                        clears,
                    };
                    state.admit(meta, meta.seq, *now)
                }
            };
            assert_eq!(render(&actions), want, "{script}: {step}");
        }
    }

    #[test]
    fn admission_follows_the_table() {
        for case in TABLE.lines().map(str::trim).filter(|line| !line.is_empty()) {
            let [name, policy_name, script, end] =
                case.split('|').map(str::trim).collect::<Vec<_>>()[..]
            else {
                panic!("malformed case: {case}");
            };
            let mut state = RankState::new(policy(policy_name));
            run(&mut state, script, &mut Instant::now());
            assert_eq!(describe(&state), end, "{name}");
        }
    }

    #[test]
    fn too_many_held_batches_give_the_gap_up() {
        let mut state = RankState::new(policy("replay"));
        let mut now = Instant::now();
        run(&mut state, "L0 -> apply 0; L2 -> replay 1", &mut now);
        let last = MAX_HELD as u64 + 2;
        let held: Vec<String> = (3..last).map(|seq| format!("L{seq}")).collect();
        run(&mut state, &held.join("; "), &mut now);
        let applied: Vec<String> = (2..=last).map(|seq| format!("apply {seq}")).collect();
        run(
            &mut state,
            &format!("L{last} -> clear, {}", applied.join(", ")),
            &mut now,
        );
        assert_eq!(
            (describe(&state), state.expected()),
            ("live partial".to_string(), last + 1)
        );
    }
}
