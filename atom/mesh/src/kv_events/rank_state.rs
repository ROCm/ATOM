//! Admission of a rank's batches by sequence number. Ranks are independent: a broken
//! stream affects only its own holder. A batch that does not follow the last one, or the
//! first one after the socket lost its connection, means batches may have been lost or
//! the engine restarted, so the holder is cleared. It then holds a lower bound of the
//! engine's cache, unless it starts over from the stream's first batch, until the next
//! `AllBlocksCleared`.

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
}

#[derive(Debug, Clone, Default)]
pub struct RankState {
    next: Option<u64>,
    trust: Trust,
    /// The holder keeps what an earlier connection carried.
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

    /// `clears`: the batch holds an `AllBlocksCleared`, after which nothing missed matters.
    pub fn admit(&mut self, seq: u64, clears: bool) -> Admit {
        let stale = std::mem::take(&mut self.stale);
        // seq 0 of an ATOM stream is only the publisher's startup AllBlocksCleared, so a
        // stream joined at seq 1 has missed nothing.
        let (admit, complete) = match self.next {
            None if stale => (Admit::ClearAndApply, seq <= 1),
            None => (Admit::Apply, seq <= 1),
            Some(next) if seq == next => (Admit::Apply, self.trust == Trust::Exact),
            Some(next) if seq < next => (Admit::ClearAndApply, seq <= 1),
            Some(_) => (Admit::ClearAndApply, false),
        };
        self.trust = if clears || complete {
            Trust::Exact
        } else {
            Trust::Partial
        };
        self.next = Some(seq.wrapping_add(1));
        admit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Batches in arrival order, `C` marking one that clears, each with the admission
    /// expected, `D` a lost connection, then the trust after the last.
    const TABLE: &str = r#"
        joined at the startup clear      | 0 apply                        | exact
        joined at the first batch        | 1 apply                        | exact
        joined later                     | 5 apply                        | partial
        joined later at a clear          | 5C apply                       | exact
        in order                         | 0 apply, 1 apply, 2 apply      | exact
        lost batches                     | 0 apply, 3 clear, 4 apply      | partial
        a clear after lost batches       | 0 apply, 3 clear, 4C apply     | exact
        joined later, then in order      | 3 apply, 4 apply, 5 apply      | partial
        restart                          | 0 apply, 1 apply, 0C clear     | exact
        restart, startup clear missed    | 0 apply, 1 apply, 2 apply, 1 clear | exact
        restart, more missed             | 0 apply, 5 clear, 6 apply, 3 clear | partial
        restart at the seq that follows  | 0 apply, 1 apply, 2 apply, 3 apply, D, 4 clear | partial
        restart seen at its first batch  | 0 apply, 1 apply, D, 1 clear   | exact
        restart seen at its clear        | 0 apply, 3 clear, D, 0C clear  | exact
        connection lost before a batch   | D, 3 apply                     | partial
    "#;

    #[test]
    fn batches_are_admitted_by_sequence() {
        for row in TABLE.lines().filter(|row| !row.trim().is_empty()) {
            let [name, steps, trust] = row.split('|').map(str::trim).collect::<Vec<_>>()[..] else {
                panic!("bad row {row:?}");
            };
            let mut state = RankState::default();
            for step in steps.split(", ") {
                if step == "D" {
                    state.disconnected();
                    continue;
                }
                let (batch, want) = step.split_once(' ').unwrap();
                let clears = batch.ends_with('C');
                let seq = batch.trim_end_matches('C').parse().unwrap();
                let admit = match state.admit(seq, clears) {
                    Admit::Apply => "apply",
                    Admit::ClearAndApply => "clear",
                };
                assert_eq!(admit, want, "{name}: {step}");
            }
            assert_eq!(
                format!("{:?}", state.trust()).to_lowercase(),
                trust,
                "{name}"
            );
        }
    }
}
