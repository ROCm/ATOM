//! The router's index of the prefixes each worker rank's device cache holds, fed by the
//! rank's KV events, and the matches `cache_aware` routes by.
//!
//! One partition per model, pool and block size, each rank on a lane of its own. The partition's
//! writer task alone applies the ranks' admitted batches (`partition`); requests walk the
//! index concurrently (`index`). Keys chain hashes of the blocks' tokens (`hash`), so the
//! engines' own block hashes are only translated, never compared.

mod hash;
mod index;
mod partition;
#[cfg(test)]
mod tests;

use std::{fmt::Debug, sync::Arc};

pub(crate) use partition::{match_holders, Holder, Partition};

use crate::core::Worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PrefixHash(u64);

/// A rank's lane in its partition: one of the 64 bits of the index's masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HolderId(u8);

/// What policies read of the index.
pub(crate) trait PrefixMatchProvider: Send + Sync + Debug {
    /// The leading tokens of `tokens` each worker's device cache holds by its KV events, in
    /// the order of `workers`: `Some(0)` for a worker that holds none of them, `None` for
    /// one whose events are not indexed. A rank whose trust is partial may hold more. The
    /// walk read-locks the shards of the partitions it reads.
    fn match_workers(&self, tokens: &[u32], workers: &[Arc<dyn Worker>]) -> Vec<Option<u32>>;
}
