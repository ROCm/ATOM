//! Payload decoding into canonical events.
//!
//! All engines publish vLLM's msgpack format: a batch is `[ts, events, data_parallel_rank]`.
//! vLLM and SGLang encode each event as a map tagged by `type` (`BlockStored`,
//! `BlockRemoved`, `AllBlocksCleared`); ATOM does so from ROCm/ATOM#3275 (unmerged), while
//! ATOM main still encodes events `array_like`, tag first. Both forms are accepted and
//! unknown fields are ignored.
//! A block's `extra_keys` entry holds, untagged: the LoRA name, `[mm_identifier, offset]`
//! pairs, then the cache salt. Any pair marks the block multimodal. ATOM with #3275 sends
//! one `[[seed_hex, 0]]` entry per block of a multimodal request and never sends LoRA or
//! salt; ATOM main sends no `extra_keys`.

use std::sync::Arc;

use rmpv::Value;

use super::{
    model::{DropReason, EngineHash, Group, GroupKind, KvEvent, Namespace, StoredBlock, Tier},
    source::{TierMap, WireProfile},
};

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

/// Field order of array-encoded events after the tag; only the prefix that vLLM, SGLang
/// and ATOM share.
const STORED_FIELDS: &[&str] = &[
    "block_hashes",
    "parent_block_hash",
    "token_ids",
    "block_size",
    "lora_id",
    "medium",
];
const REMOVED_FIELDS: &[&str] = &["block_hashes", "medium"];
const BATCH_FIELDS: &[&str] = &["ts", "events", "data_parallel_rank"];

/// Decoder for the `vllm`, `sglang` and `atom` profiles. They differ only in SGLang's
/// `cache_salt` field and EAGLE token pairs.
#[derive(Debug, Clone, Default)]
pub struct VllmWireDecoder {
    profile: WireProfile,
    tier_map: TierMap,
    strict: bool,
}

impl VllmWireDecoder {
    pub fn new(profile: WireProfile, tier_map: TierMap, strict: bool) -> Self {
        Self {
            profile,
            tier_map,
            strict,
        }
    }

    fn event(&self, event: &Value, dropped: &mut Vec<DropReason>) -> Result<KvEvent, DropReason> {
        let (tag, fields) = Fields::event(event).ok_or(DropReason::Decode)?;
        match tag {
            "BlockStored" => self.stored(&fields, dropped),
            "BlockRemoved" => Ok(KvEvent::Removed {
                hashes: hashes(&fields)?,
                tier: self.tier(&fields)?,
                group: group(&fields),
            }),
            "AllBlocksCleared" => Ok(KvEvent::Cleared {
                tier: match fields.get("medium") {
                    Some(_) => Some(self.tier(&fields)?),
                    None => None,
                },
            }),
            _ => Err(DropReason::Decode),
        }
    }

    fn stored(
        &self,
        fields: &Fields,
        dropped: &mut Vec<DropReason>,
    ) -> Result<KvEvent, DropReason> {
        let tier = self.tier(fields)?;
        let hashes = hashes(fields)?;
        let parent = fields
            .get("parent_block_hash")
            .map(|hash| engine_hash(hash).ok_or(DropReason::Decode))
            .transpose()?;
        let block_size = fields
            .get("block_size")
            .and_then(as_u32)
            .ok_or(DropReason::Decode)?;
        // Placeholder events (vLLM's OffloadingConnector) carry no tokens at all.
        let tokens = self.tokens(fields)?;
        let size = block_size as usize;
        if !tokens.is_empty() && (size == 0 || tokens.len() != hashes.len() * size) {
            return Err(DropReason::Decode);
        }
        let mut chunks = tokens.chunks(size.max(1)).map(Arc::from);
        let mut blocks: Vec<StoredBlock> = hashes
            .into_iter()
            .map(|hash| StoredBlock {
                hash,
                tokens: chunks.next(),
            })
            .collect();

        let lora = optional_str(fields, "lora_name")?;
        let extra_keys = match fields.get("extra_keys") {
            Some(keys) => keys.as_array().ok_or(DropReason::Decode)?.as_slice(),
            None => &[],
        };
        // A block whose keys are not all strings (multimodal pairs, embedding digests)
        // depends on content the router cannot hash, and so does every later block of the
        // chain.
        let modelled = |keys: &Value| {
            keys.as_array()
                .is_none_or(|keys| keys.iter().all(Value::is_str))
        };
        if let Some(cut) = extra_keys.iter().position(|keys| !modelled(keys)) {
            if cut == 0 {
                return Err(DropReason::Namespace);
            }
            if cut < blocks.len() {
                blocks.truncate(cut);
                dropped.push(DropReason::Namespace);
            }
        }
        let salt = match self.profile {
            WireProfile::Sglang => optional_str(fields, "cache_salt")?,
            WireProfile::Vllm | WireProfile::Atom => extra_keys
                .first()
                .and_then(Value::as_array)
                .and_then(|keys| {
                    keys.iter()
                        .filter_map(Value::as_str)
                        .find(|key| Some(*key) != lora)
                }),
        };
        Ok(KvEvent::Stored {
            parent,
            blocks,
            block_size,
            tier,
            group: group(fields),
            ns: Namespace {
                lora: lora.map(Arc::from),
                salt: salt.map(Arc::from),
                opaque: false,
            },
        })
    }

    /// SGLang with EAGLE publishes `[t_j, t_{j+1}]` pairs; the page's token is the first.
    fn tokens(&self, fields: &Fields) -> Result<Vec<u32>, DropReason> {
        let tokens = fields
            .get("token_ids")
            .and_then(Value::as_array)
            .ok_or(DropReason::Decode)?;
        tokens
            .iter()
            .map(|token| {
                let token = match (token, self.profile) {
                    (Value::Array(pair), WireProfile::Sglang) if pair.len() == 2 => &pair[0],
                    _ => token,
                };
                as_u32(token).ok_or(DropReason::Decode)
            })
            .collect()
    }

    /// Tier names match exactly and unknown names fail closed: a guessed tier would
    /// record pool objects as the worker's own, usually with no removal to undo it.
    fn tier(&self, fields: &Fields) -> Result<Tier, DropReason> {
        let remote = fields.get("locality").and_then(Value::as_str) == Some("REMOTE");
        if remote || fields.get("ownership").is_some() {
            return Err(DropReason::Storage);
        }
        let Some(medium) = fields.get("medium") else {
            return Ok(Tier::Device);
        };
        let medium = medium.as_str().ok_or(DropReason::Decode)?;
        if let Some(target) = self.tier_map.get(medium) {
            return target.tier().ok_or(DropReason::Storage);
        }
        match medium {
            "GPU" | "DEVICE" => Ok(Tier::Device),
            "CPU" | "HOST" | "HOST_PINNED" | "CPU_PINNED" => Ok(Tier::Host),
            "DISK" | "STORAGE" => Ok(Tier::Disk),
            "EXTERNAL" => Err(DropReason::Storage),
            _ => Err(DropReason::UnknownTier),
        }
    }
}

impl WireDecoder for VllmWireDecoder {
    fn decode(&self, payload: &[u8]) -> Result<DecodedBatch, DecodeError> {
        let malformed = |reason: &str| DecodeError::Malformed(reason.to_string());
        let batch = rmpv::decode::read_value(&mut &*payload)
            .map_err(|e| DecodeError::Malformed(e.to_string()))?;
        let Value::Array(items) = &batch else {
            return Err(malformed("batch is not an array"));
        };
        let fields = Fields::Array(items, BATCH_FIELDS);
        let ts = fields
            .get("ts")
            .and_then(Value::as_f64)
            .filter(|ts| ts.is_finite())
            .ok_or_else(|| malformed("missing or non-finite ts"))?;
        let events = fields
            .get("events")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("missing events"))?;
        let dp_rank = fields
            .get("data_parallel_rank")
            .map(|rank| as_u32(rank).ok_or_else(|| malformed("bad data_parallel_rank")))
            .transpose()?;

        let mut decoded = DecodedBatch {
            ts,
            dp_rank,
            events: Vec::with_capacity(events.len()),
            dropped: Vec::new(),
        };
        for event in events {
            match self.event(event, &mut decoded.dropped) {
                Ok(event) => decoded.events.push(event),
                Err(reason) => decoded.dropped.push(reason),
            }
        }
        let nonconforming =
            |reason: &&DropReason| matches!(reason, DropReason::Decode | DropReason::UnknownTier);
        match decoded.dropped.iter().find(nonconforming) {
            Some(&reason) if self.strict => Err(DecodeError::Strict(reason)),
            _ => Ok(decoded),
        }
    }
}

/// A struct encoded as a msgpack map, or positionally (msgspec `array_like`).
enum Fields<'a> {
    Map(&'a [(Value, Value)]),
    Array(&'a [Value], &'static [&'static str]),
}

impl<'a> Fields<'a> {
    fn event(event: &'a Value) -> Option<(&'a str, Self)> {
        match event {
            Value::Map(pairs) => {
                let fields = Fields::Map(pairs);
                Some((fields.get("type")?.as_str()?, fields))
            }
            Value::Array(items) => {
                let (tag, rest) = items.split_first()?;
                let tag = tag.as_str()?;
                let names = match tag {
                    "BlockStored" => STORED_FIELDS,
                    "BlockRemoved" => REMOVED_FIELDS,
                    _ => &[],
                };
                Some((tag, Fields::Array(rest, names)))
            }
            _ => None,
        }
    }

    /// A field that is absent or nil is None.
    fn get(&self, name: &str) -> Option<&'a Value> {
        let value = match self {
            Fields::Map(pairs) => pairs
                .iter()
                .find(|(key, _)| key.as_str() == Some(name))
                .map(|(_, value)| value),
            Fields::Array(items, names) => names
                .iter()
                .position(|field| *field == name)
                .and_then(|i| items.get(i)),
        };
        value.filter(|value| !value.is_nil())
    }
}

fn hashes(fields: &Fields) -> Result<Vec<EngineHash>, DropReason> {
    fields
        .get("block_hashes")
        .and_then(Value::as_array)
        .ok_or(DropReason::Decode)?
        .iter()
        .map(|hash| engine_hash(hash).ok_or(DropReason::Decode))
        .collect()
}

/// Byte strings keep their low 64 bits, the integer vLLM publishes for the same digest.
fn engine_hash(value: &Value) -> Option<EngineHash> {
    match value {
        Value::Integer(n) => n.as_u64().or_else(|| n.as_i64().map(|n| n as u64)),
        Value::Binary(bytes) => Some(
            bytes[bytes.len().saturating_sub(8)..]
                .iter()
                .fold(0, |hash, &byte| hash << 8 | u64::from(byte)),
        ),
        _ => None,
    }
    .map(EngineHash)
}

/// Removed events carry only `group_idx`; the kind comes with the group's Stored events.
fn group(fields: &Fields) -> Group {
    let window = fields
        .get("kv_cache_spec_sliding_window")
        .and_then(as_u32)
        .unwrap_or(0);
    let kind = match fields.get("kv_cache_spec_kind").and_then(Value::as_str) {
        None | Some("full_attention" | "mla_attention") => GroupKind::Full,
        Some("sliding_window" | "sliding_window_mla") => GroupKind::SlidingWindow(window),
        Some("mamba") => GroupKind::Mamba,
        Some(_) => GroupKind::Unknown,
    };
    Group {
        idx: fields.get("group_idx").and_then(as_u32).unwrap_or(0),
        kind,
    }
}

fn optional_str<'a>(fields: &Fields<'a>, name: &str) -> Result<Option<&'a str>, DropReason> {
    fields
        .get(name)
        .map(|value| value.as_str().ok_or(DropReason::Decode))
        .transpose()
}

fn as_u32(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests;
