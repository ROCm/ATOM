//! Transport-neutral response rendering: TokenHandle<TokenChunk> + ResponseContext → axum::Response.

use serde::Serialize;
use serde_json::{json, Value};

pub mod chat_aggregator;
pub mod chat_streaming;
pub mod generate_aggregator;
pub mod generate_streaming;
pub(crate) mod logprob_conversion;

#[cfg(test)]
mod tests;

/// `body` as JSON, its `usage` given the `prompt_tokens_details` that the protocol
/// crate's chat `Usage` has no field for.
fn with_cached_tokens(body: &impl Serialize, cached_tokens: u32) -> serde_json::Result<Value> {
    let mut value = serde_json::to_value(body)?;
    if let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) {
        let details = json!({ "cached_tokens": cached_tokens });
        usage.insert("prompt_tokens_details".to_string(), details);
    }
    Ok(value)
}
