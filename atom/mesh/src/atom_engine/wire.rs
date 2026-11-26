//! The engine wire protocol: the msgpack messages
//! `atom/model_engine/zmq_protocol/messages.py` defines (the spec), one frame
//! each.
//!
//! A message is the array `[kind, body]`. Every body is a struct encoded as
//! the array of its fields in declaration order -- `rmp-serde`'s default and
//! msgspec's `array_like`, so the structs below mirror the Python ones field
//! for field. Fields are only ever appended, each with a default: a reader
//! fills a missing trailing field and skips one it does not know
//! (`positional!`), so either side may be a version ahead.

use std::{collections::HashMap, fmt};

use serde::{
    de::{DeserializeOwned, IgnoredAny, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_bytes::ByteBuf;

/// The version this side speaks; a Core's `Ready` must name the same one.
pub const PROTOCOL_VERSION: u32 = 2;

/// `EngineCoreRequestType`, as the wire's kind.
pub mod kind {
    pub const ADD: u8 = 0;
    pub const ABORT: u8 = 1;
    pub const SHUTDOWN: u8 = 5;
    pub const STREAM: u8 = 6;
    pub const READY: u8 = 7;
}

/// `Completion::reject_code`.
pub mod reject {
    pub const FINISHED: u8 = 0;
    pub const INVALID: u8 = 1;
    pub const OVERLOADED: u8 = 2;
    pub const INTERNAL: u8 = 3;
}

/// `Deserialize` for a wire struct, read as the array of its fields: a field
/// the array ends before keeps its default, and elements past the last field
/// (a later version's) are skipped. `rmp-serde`'s derived reader refuses both.
macro_rules! positional {
    ($name:ident { $($field:ident),* $(,)? }) => {
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct Fields;
                impl<'de> Visitor<'de> for Fields {
                    type Value = $name;
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str(concat!("the array of ", stringify!($name), "'s fields"))
                    }
                    #[allow(unused_mut)]
                    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<$name, A::Error> {
                        let mut out = $name::default();
                        $(
                            match seq.next_element()? {
                                Some(value) => out.$field = value,
                                None => return Ok(out),
                            }
                        )*
                        while seq.next_element::<IgnoredAny>()?.is_some() {}
                        Ok(out)
                    }
                }
                deserializer.deserialize_seq(Fields)
            }
        }
    };
}

/// A field this client neither reads nor fills: skipped when it arrives,
/// nil when it is sent, so the fields after it keep their places.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Unread;

impl Serialize for Unread {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit()
    }
}

impl<'de> Deserialize<'de> for Unread {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        IgnoredAny::deserialize(deserializer).map(|_| Unread)
    }
}

/// `messages.RequiredChannel`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct RequiredChannel {
    pub after: Vec<i32>,
    pub opening: Vec<i32>,
    pub opens_at: u32,
    pub end_ids: Vec<i32>,
}

positional!(RequiredChannel {
    after,
    opening,
    opens_at,
    end_ids
});

/// `messages.Sampling`: `SamplingParams`' fields, in its order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: i64,
    pub top_p: f64,
    pub max_tokens: i64,
    pub ignore_eos: bool,
    pub stop_strings: Option<Vec<String>>,
    pub n: u32,
    pub logprobs: Option<bool>,
    pub required_channel: Option<RequiredChannel>,
    pub top_logprobs: u32,
    pub json_schema: Option<String>,
    pub structured_tool_format: Option<String>,
    pub structured_reasoning_end: String,
    pub max_tokens_is_default: bool,
}

positional!(Sampling {
    temperature,
    top_k,
    top_p,
    max_tokens,
    ignore_eos,
    stop_strings,
    n,
    logprobs,
    required_channel,
    top_logprobs,
    json_schema,
    structured_tool_format,
    structured_reasoning_end,
    max_tokens_is_default
});

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            max_tokens: 64,
            ignore_eos: false,
            stop_strings: None,
            n: 1,
            logprobs: None,
            required_channel: None,
            top_logprobs: 0,
            json_schema: None,
            structured_tool_format: None,
            structured_reasoning_end: String::new(),
            max_tokens_is_default: false,
        }
    }
}

/// `messages.SubmitItem`: one sequence to admit. The fields only Python
/// produces (KV-transfer params, media, hash chain, trace) stay `None` here.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SubmitItem {
    pub seq_id: u64,
    /// int32 token ids, little-endian.
    pub prompt: ByteBuf,
    pub sampling: Sampling,
    pub stop_token_sequences: Vec<Vec<i32>>,
    pub kv_transfer_params: Unread,
    pub multimodal: Unread,
    pub multimodal_buffers: Unread,
    pub mrope_position_delta: i64,
    pub cache_seed: Option<i64>,
    pub needs_independent_noise: bool,
    pub parent_request_id: Option<String>,
    pub sibling_index: u32,
    pub external_request_id: Option<String>,
    pub data_parallel_rank: Option<i64>,
    pub dp_session_id: Option<String>,
    pub dp_parent_session_id: Option<String>,
    pub arrive_time: f64,
    pub prompt_hash_chain: Unread,
    pub trace: Unread,
    pub dp_est_cached_tokens: u64,
    pub refused: Option<String>,
}

positional!(SubmitItem {
    seq_id,
    prompt,
    sampling,
    stop_token_sequences,
    kv_transfer_params,
    multimodal,
    multimodal_buffers,
    mrope_position_delta,
    cache_seed,
    needs_independent_noise,
    parent_request_id,
    sibling_index,
    external_request_id,
    data_parallel_rank,
    dp_session_id,
    dp_parent_session_id,
    arrive_time,
    prompt_hash_chain,
    trace,
    dp_est_cached_tokens,
    refused
});

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Submit {
    pub items: Vec<SubmitItem>,
}

positional!(Submit { items });

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Abort {
    pub seq_ids: Vec<u64>,
}

positional!(Abort { seq_ids });

/// SHUTDOWN, either way.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Empty {}

positional!(Empty {});

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ready {
    pub protocol_version: u32,
    pub max_pool_tokens: Option<u64>,
    pub weight_syncs_unavailable: HashMap<String, String>,
}

positional!(Ready {
    protocol_version,
    max_pool_tokens,
    weight_syncs_unavailable
});

impl Default for Ready {
    fn default() -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            max_pool_tokens: None,
            weight_syncs_unavailable: HashMap::new(),
        }
    }
}

/// `messages.StreamItem`: one sequence's step. What this client does not
/// read (KV-transfer params, logprob records) is skipped.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StreamItem {
    pub seq_id: u64,
    pub output_tokens: Vec<u32>,
    pub finished: bool,
    pub finish_reason: Option<String>,
    pub kv_transfer_params_output: Unread,
    pub num_cached_tokens: u64,
    pub logprobs: Unread,
}

positional!(StreamItem {
    seq_id,
    output_tokens,
    finished,
    finish_reason,
    kv_transfer_params_output,
    num_cached_tokens,
    logprobs
});

#[derive(Debug, Clone, Default, Serialize)]
pub struct Stream {
    pub items: Vec<StreamItem>,
}

positional!(Stream { items });

/// `messages.Completion`: a finished sequence, or one the Core refused.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Completion {
    pub seq_id: u64,
    /// int32, little-endian.
    pub completion_tokens: ByteBuf,
    /// float64, little-endian, one per completion token.
    pub logprobs_bytes: Option<ByteBuf>,
    pub return_logprobs: bool,
    pub num_prompt_tokens: u32,
    pub num_cached_tokens: u64,
    pub arrive_time: f64,
    pub first_token_time: f64,
    pub leave_reason: String,
    pub prefill_completed: bool,
    pub json_schema: Option<String>,
    pub structured_tool_format: Option<String>,
    pub reject_code: u8,
    pub leave_time: f64,
}

positional!(Completion {
    seq_id,
    completion_tokens,
    logprobs_bytes,
    return_logprobs,
    num_prompt_tokens,
    num_cached_tokens,
    arrive_time,
    first_token_time,
    leave_reason,
    prefill_completed,
    json_schema,
    structured_tool_format,
    reject_code,
    leave_time
});

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Completions {
    pub items: Vec<Completion>,
}

positional!(Completions { items });

/// One message for a Core.
pub fn encode<T: Serialize>(kind: u8, body: &T) -> Vec<u8> {
    rmp_serde::to_vec(&(kind, body)).expect("a wire struct always encodes")
}

/// int32 token ids, little-endian: a prompt.
pub fn token_bytes(tokens: &[u32]) -> ByteBuf {
    ByteBuf::from(
        tokens
            .iter()
            .flat_map(|t| (*t as i32).to_le_bytes())
            .collect::<Vec<u8>>(),
    )
}

/// int32 tokens back out of `bytes`.
pub fn tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("an int32 buffer must be a multiple of 4 bytes".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
        .collect())
}

/// float64 values out of `bytes`.
pub fn floats(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err("a float64 buffer must be a multiple of 8 bytes".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// What a Core sent this client.
#[derive(Debug)]
pub enum Inbound {
    Ready(Ready),
    Stream(Stream),
    Completions(Completions),
    Stopped,
    /// A kind only the Python client reads (metrics, utility replies, ...).
    Ignored(u8),
}

/// `[kind, body]`: the kind, and the bytes of the body.
pub fn split(message: &[u8]) -> Result<(u8, &[u8]), String> {
    let mut rest = message;
    let len = rmp::decode::read_array_len(&mut rest).map_err(|e| e.to_string())?;
    if len != 2 {
        return Err(format!("a message is [kind, body], not {len} items"));
    }
    let kind: u8 = rmp::decode::read_int(&mut rest).map_err(|e| e.to_string())?;
    Ok((kind, rest))
}

pub fn body<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    rmp_serde::from_slice(bytes).map_err(|e| e.to_string())
}

impl Inbound {
    pub fn decode(message: &[u8]) -> Result<Self, String> {
        let (kind, rest) = split(message)?;
        Ok(match kind {
            kind::READY => Self::Ready(body(rest)?),
            kind::STREAM => Self::Stream(body(rest)?),
            kind::ADD => Self::Completions(body(rest)?),
            kind::SHUTDOWN => Self::Stopped,
            other => Self::Ignored(other),
        })
    }
}
