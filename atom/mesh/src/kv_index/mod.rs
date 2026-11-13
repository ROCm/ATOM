//! Router-side prefix index fed by KV events, and the prefix-match query used by policies.
//!
//! One index per `(model, pool, block_size)` partition. Keys are chained prefix hashes
//! `p_0 = c_0`, `p_i = XXH3(p_{i-1} || c_i)` over per-block content hashes, so blocks from
//! engines with different hash algorithms land on the same key.

pub mod hash;
pub mod index;
pub mod provider;
#[cfg(test)]
mod tests;

use serde::Serialize;

pub use self::{
    index::{HolderTable, PrefixIndex, MAX_HOLDERS},
    provider::{CacheHit, HitSource, LoadPath, LowerHit, PrefixMatchProvider, PrefixRequest},
};
use crate::kv_events::Tier;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixHash(pub u64);

/// Compact id of one `url@rank` holder within the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HolderId(pub u32);

/// A store or remove whose engine hash is already translated to a prefix hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexOp {
    Store { hash: PrefixHash, tier: Tier },
    Remove { hash: PrefixHash, tier: Tier },
}

#[derive(Debug, Clone, Copy)]
pub struct MatchQuery<'a> {
    pub prefix: &'a [PrefixHash],
    pub holders: &'a [HolderId],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HolderMatch {
    pub holder: HolderId,
    /// Cumulative prefix blocks found at Device, at Host or above, and at Disk or above.
    pub blocks: [u32; 3],
    /// Leading blocks present in the local lower tiers (Host, Disk), ignoring Device.
    pub offload: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct IndexStats {
    /// Distinct prefix hashes.
    pub entries: u64,
    /// Holder blocks per tier: Device, Host, Disk.
    pub blocks: [u64; 3],
    pub holders: u32,
    pub approx_bytes: u64,
}

/// One partition. A single writer task applies events; queries read concurrently.
pub trait KvIndex: Send + Sync {
    fn apply(&self, holder: HolderId, ops: &[IndexOp]);
    /// `tier` None clears every tier of the holder.
    fn clear(&self, holder: HolderId, tier: Option<Tier>);
    fn remove_holder(&self, holder: HolderId);
    /// Walks the prefix from block 0 and stops once every holder has broken off.
    /// Holders without a match are omitted.
    fn find_matches(&self, query: &MatchQuery<'_>) -> Vec<HolderMatch>;
    fn stats(&self) -> IndexStats;
}
