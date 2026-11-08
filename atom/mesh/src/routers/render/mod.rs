//! Transport-neutral response rendering: TokenHandle<TokenChunk> + ResponseContext → axum::Response.

pub mod chat_aggregator;
pub mod chat_streaming;
pub mod completion;
pub mod generate_aggregator;
pub mod generate_streaming;
pub(crate) mod logprob_conversion;

#[cfg(test)]
mod tests;

/// `usage.prompt_tokens_details.cached_tokens`, which the protocol crate's
/// `Usage` has no field for, onto a serialized response.
pub(crate) fn with_cached_tokens(
    mut response: serde_json::Value,
    cached: u32,
) -> serde_json::Value {
    if let Some(usage) = response.get_mut("usage").and_then(|u| u.as_object_mut()) {
        usage.insert(
            "prompt_tokens_details".into(),
            serde_json::json!({ "cached_tokens": cached }),
        );
    }
    response
}
