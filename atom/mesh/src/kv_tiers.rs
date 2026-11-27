//! The tiers a worker's engine takes a prompt's KV from, with `--kv-events`, nearest
//! first: its device cache, the device pools of its share group, and the store it reads.
//! What a token of each counts for in a policy's score is one `TierWeights`, in the
//! policy's config.

/// What a token of each tier below a worker's device cache counts for in a policy's score,
/// against one its device cache holds, which counts 1. A tier that counts 0 is not asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierWeights {
    /// A token the engine will copy in from the device pools of its share group:
    /// `--kv-share-weight`.
    pub shared: f64,
    /// A token the engine will load from the store it reads: `--kv-storage-weight`.
    pub stored: f64,
}

impl Default for TierWeights {
    fn default() -> Self {
        Self {
            shared: 0.9,
            stored: 0.0,
        }
    }
}
