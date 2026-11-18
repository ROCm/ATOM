//! ATOM's KV event payloads (ROCm/ATOM#3275), encoded as vLLM encodes its own: a batch
//! is the msgpack array `[ts, events, data_parallel_rank]`, each event a map tagged by
//! `type`, and fields not read here are ignored. Only the full-attention group 0 is
//! followed: on the device, and on the host when the source reports an in-process
//! LMCache, whose chunks ATOM stores and removes as blocks of medium `CPU`. Any other
//! event shape, such as the arrays ATOM sent before #3275, is dropped as `Decode`.

use rmpv::Value;

use super::model::{DropReason, EngineHash, KvEvent, StoredBlock, Tier};

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBatch {
    pub ts: f64,
    pub dp_rank: Option<u32>,
    /// The device's events.
    pub events: Vec<KvEvent>,
    /// The host's events, when it is followed.
    pub host: Vec<KvEvent>,
    /// Why each event missing from both was left out.
    pub dropped: Vec<DropReason>,
}

/// A payload that is not a batch at all.
#[derive(Debug, thiserror::Error)]
#[error("malformed KV event batch: {0}")]
pub struct DecodeError(String);

/// Decodes a payload from a source whose blocks hold `block_size` tokens, at least 1, and
/// whose host chunks, if followed, `host_block_size`.
pub fn decode(
    payload: &[u8],
    block_size: u32,
    host_block_size: Option<u32>,
) -> Result<DecodedBatch, DecodeError> {
    let malformed = |what: &str| DecodeError(what.into());
    let batch = rmpv::decode::read_value(&mut &*payload).map_err(|e| DecodeError(e.to_string()))?;
    let Value::Array(items) = batch else {
        return Err(malformed("not an array"));
    };
    let ts = items
        .first()
        .and_then(Value::as_f64)
        .filter(|ts| ts.is_finite())
        .ok_or_else(|| malformed("no ts"))?;
    let events = items
        .get(1)
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("no events"))?;
    let dp_rank = match items.get(2) {
        None | Some(Value::Nil) => None,
        Some(rank) => Some(as_u32(rank).ok_or_else(|| malformed("bad data_parallel_rank"))?),
    };
    let mut batch = DecodedBatch {
        ts,
        dp_rank,
        events: Vec::with_capacity(events.len()),
        host: Vec::new(),
        dropped: Vec::new(),
    };
    for event in events {
        match decode_event(event, block_size, host_block_size) {
            Ok((Some(Tier::Device), event)) => batch.events.push(event),
            Ok((Some(Tier::Host), event)) => batch.host.push(event),
            // A clear without a medium clears every tier.
            Ok((None, event)) => {
                if host_block_size.is_some() {
                    batch.host.push(event.clone());
                }
                batch.events.push(event);
            }
            Err(reason) => batch.dropped.push(reason),
        }
    }
    Ok(batch)
}

/// The event and its tier; no tier for a clear of every tier.
fn decode_event(
    event: &Value,
    block_size: u32,
    host_block_size: Option<u32>,
) -> Result<(Option<Tier>, KvEvent), DropReason> {
    let Value::Map(fields) = event else {
        return Err(DropReason::Decode);
    };
    let fields = Fields(fields);
    let tier = || tier(&fields, host_block_size.is_some());
    match fields.get("type").and_then(Value::as_str) {
        Some("BlockStored") => {
            let tier = tier()?.unwrap_or(Tier::Device);
            let size = match tier {
                Tier::Device => block_size,
                Tier::Host => host_block_size.ok_or(DropReason::Tier)?,
            };
            Ok((Some(tier), stored(&fields, size)?))
        }
        Some("BlockRemoved") => {
            let tier = tier()?.unwrap_or(Tier::Device);
            full_attention(&fields)?;
            let hashes = hashes(&fields)?;
            Ok((Some(tier), KvEvent::Removed { hashes }))
        }
        Some("AllBlocksCleared") => Ok((tier()?, KvEvent::Cleared)),
        _ => Err(DropReason::Decode),
    }
}

fn stored(fields: &Fields, block_size: u32) -> Result<KvEvent, DropReason> {
    full_attention(fields)?;
    // ATOM marks every block of a multimodal prompt with the request's media seed.
    let marked = match fields.get("extra_keys") {
        None => false,
        Some(Value::Array(keys)) => keys.iter().any(|key| !key.is_nil()),
        Some(_) => return Err(DropReason::Decode),
    };
    if marked || fields.get("lora_id").is_some() || fields.get("lora_name").is_some() {
        return Err(DropReason::Namespace);
    }
    let size = fields.get("block_size").and_then(as_u32);
    if size.ok_or(DropReason::Decode)? != block_size {
        return Err(DropReason::BlockSize);
    }
    let hashes = hashes(fields)?;
    let parent = fields
        .get("parent_block_hash")
        .map(|hash| engine_hash(hash).ok_or(DropReason::Decode))
        .transpose()?;
    let tokens = fields
        .get("token_ids")
        .and_then(Value::as_array)
        .ok_or(DropReason::Decode)?
        .iter()
        .map(|token| as_u32(token).ok_or(DropReason::Decode))
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.len() != hashes.len() * block_size as usize {
        return Err(DropReason::Decode);
    }
    let blocks = hashes
        .into_iter()
        .zip(tokens.chunks_exact(block_size as usize))
        .map(|(hash, tokens)| StoredBlock {
            hash,
            tokens: tokens.into(),
        })
        .collect();
    Ok(KvEvent::Stored { parent, blocks })
}

/// The tier the event's medium names: the device's `GPU`, or the host's `CPU` when the
/// host is followed; none without a medium.
fn tier(fields: &Fields, host: bool) -> Result<Option<Tier>, DropReason> {
    match fields.get("medium").map(Value::as_str) {
        None => Ok(None),
        Some(Some("GPU")) => Ok(Some(Tier::Device)),
        Some(Some("CPU")) if host => Ok(Some(Tier::Host)),
        Some(Some(_)) => Err(DropReason::Tier),
        Some(None) => Err(DropReason::Decode),
    }
}

/// ATOM leaves both fields out for the full-attention group 0.
fn full_attention(fields: &Fields) -> Result<(), DropReason> {
    let group = match fields.get("group_idx") {
        None => 0,
        Some(group) => as_u32(group).ok_or(DropReason::Decode)?,
    };
    let kind = match fields.get("kv_cache_spec_kind") {
        None => None,
        Some(kind) => Some(kind.as_str().ok_or(DropReason::Decode)?),
    };
    match (group, kind) {
        (0, None | Some("full_attention")) => Ok(()),
        _ => Err(DropReason::Group),
    }
}

fn hashes(fields: &Fields) -> Result<Vec<EngineHash>, DropReason> {
    let hashes = fields.get("block_hashes").and_then(Value::as_array);
    hashes
        .ok_or(DropReason::Decode)?
        .iter()
        .map(|hash| engine_hash(hash).ok_or(DropReason::Decode))
        .collect()
}

/// ATOM's xxh64 hashes are u64; negative ones, as vLLM's can be, are bit-cast.
fn engine_hash(value: &Value) -> Option<EngineHash> {
    let Value::Integer(n) = value else {
        return None;
    };
    n.as_u64()
        .or_else(|| n.as_i64().map(|n| n as u64))
        .map(EngineHash)
}

fn as_u32(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|n| u32::try_from(n).ok())
}

/// An event's fields by name, an absent field and a nil one alike.
struct Fields<'a>(&'a [(Value, Value)]);

impl<'a> Fields<'a> {
    fn get(&self, name: &str) -> Option<&'a Value> {
        let (_, value) = self.0.iter().find(|(key, _)| key.as_str() == Some(name))?;
        Some(value).filter(|value| !value.is_nil())
    }
}

#[cfg(test)]
pub(crate) mod tests;
