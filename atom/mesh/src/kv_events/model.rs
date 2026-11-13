//! KV events as the router keeps them, whatever the engine's wire format.

use std::sync::Arc;

use serde::Serialize;

/// A block hash as the engine computes it: u64 as is, i64 bit-cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineHash(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredBlock {
    pub hash: EngineHash,
    /// The block's `block_size` tokens.
    pub tokens: Arc<[u32]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvEvent {
    /// `blocks` continue the chain that ends at `parent`, or start one.
    Stored {
        parent: Option<EngineHash>,
        blocks: Vec<StoredBlock>,
    },
    Removed {
        hashes: Vec<EngineHash>,
    },
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
    /// A store whose parent block the rank's holder does not hold, as when the parent was
    /// stored before the subscription.
    UnknownParent,
    /// A store past the blocks one holder may index.
    Capacity,
}
