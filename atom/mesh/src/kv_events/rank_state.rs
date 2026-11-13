//! Admission of a rank's batches by sequence number. Ranks are independent: a broken
//! stream affects only its own holder. Batches missing before one that arrives are
//! replayed by the engine if it can; when they cannot be had, the holder is cleared and
//! holds a lower bound of the engine's cache until the next `AllBlocksCleared`. The first
//! batch after the socket lost its connection starts the holder over, as at a stream's
//! start: the engine may have restarted meanwhile.

use serde::Serialize;

/// Whether a holder matches its engine's cache (`Exact`) or is a lower bound of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    Exact,
    #[default]
    Partial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    Apply,
    /// Clear the holder, then apply.
    ClearAndApply,
    /// Already applied: a batch both replayed and live.
    Duplicate,
    /// The batches from `from` up to this one are missing: replay them first, or give
    /// them up with [`RankState::lose`].
    Missing {
        from: u64,
    },
}

#[derive(Debug, Clone, Default)]
pub struct RankState {
    /// The seq after the last batch applied; None before a stream's first.
    next: Option<u64>,
    /// `ts` of the last batch applied.
    ts: f64,
    trust: Trust,
    /// The holder keeps blocks of the engine's earlier run, or of an earlier connection.
    stale: bool,
}

impl RankState {
    pub fn trust(&self) -> Trust {
        self.trust
    }

    /// The socket lost its connection: an engine that restarted meanwhile may go on at the
    /// very seq that would follow, so the next batch starts the holder over.
    pub fn disconnected(&mut self) {
        if self.next.take().is_some() {
            self.stale = true;
        }
    }

    /// The seq after the last batch applied.
    pub fn next(&self) -> Option<u64> {
        self.next
    }

    /// `clears`: the batch holds an `AllBlocksCleared`, after which nothing missed matters.
    pub fn admit(&mut self, seq: u64, ts: f64, clears: bool) -> Admit {
        if self.next.is_some_and(|next| seq < next) {
            if ts <= self.ts {
                return Admit::Duplicate;
            }
            // A restarted engine starts its stream over.
            self.next = None;
            self.stale = true;
        }
        // seq 0 of an ATOM stream is only the publisher's startup AllBlocksCleared, so a
        // stream joined at seq 1 has missed nothing.
        let (from, complete) = match self.next {
            None => (0, seq <= 1),
            Some(next) => (next, seq == next),
        };
        if !complete && !clears {
            return Admit::Missing { from };
        }
        let reset = std::mem::take(&mut self.stale) || (!complete && self.next.is_some());
        let trust = if clears || self.next.is_none() {
            Trust::Exact
        } else {
            self.trust
        };
        self.applied(seq, ts, trust);
        if reset {
            Admit::ClearAndApply
        } else {
            Admit::Apply
        }
    }

    /// Gives up the batches missing before `seq`, the holder then a lower bound of the
    /// engine's cache.
    pub fn lose(&mut self, seq: u64, ts: f64) -> Admit {
        let reset = std::mem::take(&mut self.stale) || self.next.is_some();
        self.applied(seq, ts, Trust::Partial);
        if reset {
            Admit::ClearAndApply
        } else {
            Admit::Apply
        }
    }

    fn applied(&mut self, seq: u64, ts: f64, trust: Trust) {
        self.next = Some(seq.wrapping_add(1));
        self.ts = ts;
        self.trust = trust;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Batches in arrival order, each with its admission: `C` marks one that clears, `@`
    /// its `ts` (else its seq), `D` a lost connection. `missing N` holds a batch while those
    /// from N are replayed; a third word gives them up instead, with what that admits. Then
    /// the trust after.
    const TABLE: &str = r#"
        joined at the startup clear      | 0 apply                                  | exact
        joined at the first batch        | 1 apply                                  | exact
        joined later                     | 5 missing 0 apply                        | partial
        joined later at a clear          | 5C apply                                 | exact
        joined later, replayed from 0    | 3 missing 0, 0C apply, 1 apply, 2 apply, 3 apply, 3 dup | exact
        joined later, replay window past | 3 missing 0, 2 missing 0 apply, 3 apply, 3 dup | partial
        in order                         | 0 apply, 1 apply, 2 apply                | exact
        lost batches                     | 0 apply, 3 missing 1 clear, 4 apply      | partial
        lost batches, replayed           | 0 apply, 3 missing 1, 1 apply, 2 apply, 3 apply, 3 dup, 4 apply | exact
        lost batches, replay window past | 0 apply, 4 missing 1, 2 missing 1 clear, 3 apply, 4 apply, 4 dup | partial
        lost batches, replay cut short   | 0 apply, 4 missing 1, 1 apply, 4 missing 2 clear | partial
        a clear after lost batches       | 0 apply, 3 missing 1 clear, 4C apply     | exact
        a clear across lost batches      | 0 apply, 3C clear                        | exact
        joined later, then in order      | 3 missing 0 apply, 4 apply, 5 apply      | partial
        restart                          | 0 apply, 1 apply, 0C@2 clear             | exact
        restart, startup clear missed    | 0 apply, 1 apply, 2 apply, 1@3 clear     | exact
        restart, more missed             | 0 apply, 5 missing 1 clear, 6 apply, 3@7 missing 0 clear | partial
        restart, replayed from 0         | 0 apply, 1 apply, 2 apply, 3 apply, 2@12 missing 0, 0C@10 clear, 1@11 apply, 2@12 apply, 2@12 dup | exact
        restart at the seq that follows  | 0 apply, 1 apply, 2 apply, D, 3 missing 0 clear | partial
        the same, replayed from 0        | 0 apply, 1 apply, 2 apply, D, 3@10 missing 0, 0C@7 clear, 1@8 apply, 2@9 apply, 3@10 apply | exact
        restart seen at its first batch  | 0 apply, 1 apply, D, 1 clear             | exact
        restart seen at its clear        | 0 apply, 3 missing 1 clear, D, 0C clear  | exact
        connection lost before a batch   | D, 3 missing 0 apply                     | partial
    "#;

    fn name(admit: Admit) -> String {
        match admit {
            Admit::Apply => "apply".into(),
            Admit::ClearAndApply => "clear".into(),
            Admit::Duplicate => "dup".into(),
            Admit::Missing { from } => format!("missing {from}"),
        }
    }

    #[test]
    fn batches_are_admitted_by_sequence() {
        for row in TABLE.lines().filter(|row| !row.trim().is_empty()) {
            let [row_name, steps, trust] = row.split('|').map(str::trim).collect::<Vec<_>>()[..]
            else {
                panic!("bad row {row:?}");
            };
            let mut state = RankState::default();
            for step in steps.split(", ") {
                if step == "D" {
                    state.disconnected();
                    continue;
                }
                let (batch, want) = step.split_once(' ').unwrap();
                let (batch, ts) = batch.split_once('@').unwrap_or((batch, ""));
                let clears = batch.ends_with('C');
                let seq: u64 = batch.trim_end_matches('C').parse().unwrap();
                let ts = ts.parse().unwrap_or(seq as f64);
                let mut got = name(state.admit(seq, ts, clears));
                if got.starts_with("missing") && want.split(' ').count() == 3 {
                    got = format!("{got} {}", name(state.lose(seq, ts)));
                }
                assert_eq!(got, want, "{row_name}: {step}");
            }
            assert_eq!(
                format!("{:?}", state.trust()).to_lowercase(),
                trust,
                "{row_name}"
            );
        }
    }
}
