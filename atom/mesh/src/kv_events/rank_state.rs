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
