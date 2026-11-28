//! The router's index of the prefixes each worker rank's device cache holds, fed by the
//! rank's KV events and predicted from the prompts routed to it, and the matches
//! `cache_aware` routes by.
//!
//! One partition per model, pool and block size, each rank on a lane of its own. The partition's
//! writer thread alone applies the ranks' admitted batches (`partition`); requests walk the
//! index concurrently (`index`). Keys chain hashes of the blocks' tokens (`hash`), so the
//! engines' own block hashes are only translated, never compared.

mod hash;
mod index;
mod partition;
#[cfg(test)]
mod tests;

use std::{fmt::Debug, sync::Arc};

pub(crate) use partition::{match_groups, match_lanes, Holder, Lane, Partition, Predictor};

use crate::{
    core::Worker,
    kv_tiers::{TierWeights, TieredHit},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixHash(u64);

/// A rank's lane in its partition: one of the 64 bits of the index's masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HolderId(u8);

/// What policies read of the index.
pub(crate) trait PrefixMatchProvider: Send + Sync + Debug {
    /// What the index knows of the leading tokens of `tokens` at each worker, in the order
    /// of `workers`; `None` for a worker it knows nothing of. `device` is what the
    /// worker's device cache holds, `Some(0)` when none of them: by its KV events when
    /// they are exact, raised to the prompts routed to it when they are partial, and by
    /// those prompts alone when it has none indexed. `shared` is what its engine will then
    /// copy in from the pools of the other ranks of its share group: 0 where none can,
    /// and where `weights` counts the tier for nothing, as the groups are then left out
    /// of the walk. `stored` is not the index's to tell, and `Some(0)`. The walk
    /// read-locks the shards of the partitions it reads.
    fn tiered_hits(
        &self,
        tokens: &[u32],
        workers: &[Arc<dyn Worker>],
        weights: &TierWeights,
    ) -> Vec<Option<TieredHit>>;

    /// Predicts that `worker` will hold `tokens`, routed to it, without waiting for the
    /// index's writer.
    fn on_routed(&self, _tokens: &[u32], _worker: &dyn Worker) {}
}
