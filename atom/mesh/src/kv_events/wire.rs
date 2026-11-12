//! Payload decoding into canonical events.
//!
//! All engines publish vLLM's msgpack format: a batch is `[ts, events, data_parallel_rank]`
//! and each event is a map tagged by `type` (`BlockStored`, `BlockRemoved`,
//! `AllBlocksCleared`); array-encoded events are accepted too and unknown fields are ignored.
//! A block's `extra_keys` entry holds, untagged: the LoRA name, `[mm_identifier, offset]`
//! pairs, then the cache salt. Any pair marks the block multimodal. ATOM (PR-1) sends one
//! `[[seed_hex, 0]]` entry per block of a multimodal request and never sends LoRA or salt.

use super::model::{DropReason, KvEvent};

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBatch {
    pub ts: f64,
    pub dp_rank: Option<u32>,
    pub events: Vec<KvEvent>,
    /// One entry per event or block left out of `events`.
    pub dropped: Vec<DropReason>,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("malformed batch: {0}")]
    Malformed(String),
    /// Strict mode only: the batch decoded but something was dropped.
    #[error("strict decoding dropped {0:?}")]
    Strict(DropReason),
}

pub trait WireDecoder: Send + Sync {
    fn decode(&self, payload: &[u8]) -> Result<DecodedBatch, DecodeError>;
}
