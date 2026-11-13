//! Canonical KV events, independent of the engine wire format.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

/// Engine block hash: u64 as is, i64 bit-cast, byte strings by their low 64 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EngineHash(pub u64);

/// Engine cache tiers, fastest load first. Shared storage is not one: it is looked up per
/// request (`kv_storage`), never indexed from events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Device,
    Host,
    Disk,
}

/// Value of a source's `tier_map`: a canonical engine tier, or `drop` for blocks that
/// belong to shared storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierTarget {
    Device,
    Host,
    Disk,
    Drop,
}

impl TierTarget {
    pub fn tier(self) -> Option<Tier> {
        match self {
            TierTarget::Device => Some(Tier::Device),
            TierTarget::Host => Some(Tier::Host),
            TierTarget::Disk => Some(Tier::Disk),
            TierTarget::Drop => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    Full,
    SlidingWindow(u32),
    Mamba,
    Unknown,
}

/// KV cache group of hybrid models. v1 indexes only `Full` groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Group {
    pub idx: u32,
    pub kind: GroupKind,
}

impl Default for Group {
    fn default() -> Self {
        Self {
            idx: 0,
            kind: GroupKind::Full,
        }
    }
}

/// Request identity beyond tokens. LoRA and salt only seed the first block's content
/// hash; `opaque` marks identities the router cannot model (e.g. multimodal).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Namespace {
    pub lora: Option<Arc<str>>,
    pub salt: Option<Arc<str>>,
    pub opaque: bool,
}

/// `tokens` is None when the event carries no tokens; such blocks are resolved through
/// the holder's reverse table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredBlock {
    pub hash: EngineHash,
    pub tokens: Option<Arc<[u32]>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KvEvent {
    Stored {
        parent: Option<EngineHash>,
        blocks: Vec<StoredBlock>,
        block_size: u32,
        tier: Tier,
        group: Group,
        ns: Namespace,
    },
    Removed {
        hashes: Vec<EngineHash>,
        tier: Tier,
        group: Group,
    },
    /// `tier` None clears every tier of the holder.
    Cleared { tier: Option<Tier> },
}

/// Why an event or batch was dropped; the `reason` label of `mesh_kv_events_rejected_total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DropReason {
    Decode,
    UnknownTier,
    Storage,
    BlockSize,
    UnknownParent,
    Unresolved,
    Namespace,
    RankMismatch,
}
