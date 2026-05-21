//! Transport-neutral response rendering: TokenHandle<TokenChunk> + ResponseContext → axum::Response.

pub mod chat_aggregator;
pub mod chat_streaming;
pub mod generate_aggregator;
pub mod generate_streaming;
pub(crate) mod logprob_conversion;

#[cfg(test)]
mod tests;
