//! Per-rank admission and recovery state. Ranks are independent: a broken stream only
//! affects its own `url@rank` holder.

use serde::Serialize;

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

/// What admitting one batch asks the caller to do, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Apply,
    /// Clear the holder and apply the batch from its un-tiered `AllBlocksCleared` on.
    ApplyFromClear,
    Skip,
    /// Hold the live batch until the pending replay closes the gap.
    Buffer,
    RequestReplay {
        from: u64,
    },
    RequestSnapshot,
    ClearHolder,
}

#[derive(Debug, Clone, Default)]
pub struct RankState {
    expected: u64,
    trust: Trust,
    phase: Phase,
    last_ts: Option<f64>,
}

impl RankState {
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
}
