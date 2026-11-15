//! KV events as the router keeps them, whatever the engine's wire format.

use serde::Serialize;

/// A block hash as the engine computes it: u64 as is, i64 bit-cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineHash(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvEvent {
    Stored { hashes: Vec<EngineHash> },
    Removed { hashes: Vec<EngineHash> },
    Cleared,
}

/// Why an event was left out, as `/kv_events/sources` counts them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// Not an event of a documented type and shape.
    Decode,
    /// Another tier than the device's (`GPU`), the only one followed.
    Tier,
    /// Another KV cache group than the full-attention group 0.
    Group,
    /// LoRA or media-seeded blocks, which a prompt's tokens alone cannot hash to.
    Namespace,
    /// Another block size than the source reports.
    BlockSize,
}
