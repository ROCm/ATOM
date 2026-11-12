//! Transport-neutral response rendering: TokenHandle<TokenChunk> + ResponseContext → axum::Response.

pub mod chat_aggregator;
pub mod chat_streaming;
pub mod generate_aggregator;
pub mod generate_streaming;
pub(crate) mod logprob_conversion;

#[cfg(test)]
mod tests;

use serde::Serialize;

use crate::protocols::{
    chat::{ChatCompletionResponse, ChatCompletionStreamResponse},
    common::{PromptTokenUsageInfo, Usage},
};

/// A response that carries a `usage`.
pub(crate) trait WithUsage: Serialize {
    fn take_usage(&mut self) -> Option<Usage>;
}

impl WithUsage for ChatCompletionResponse {
    fn take_usage(&mut self) -> Option<Usage> {
        self.usage.take()
    }
}

impl WithUsage for ChatCompletionStreamResponse {
    fn take_usage(&mut self) -> Option<Usage> {
        self.usage.take()
    }
}

/// A response with `usage.prompt_tokens_details.cached_tokens`, which the
/// protocol crate's `Usage` has no field for; serializes in one pass.
#[derive(Serialize)]
pub(crate) struct WithCachedTokens<T> {
    #[serde(flatten)]
    response: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageWithCachedTokens>,
}

#[derive(Serialize)]
struct UsageWithCachedTokens {
    #[serde(flatten)]
    usage: Usage,
    prompt_tokens_details: PromptTokenUsageInfo,
}

pub(crate) fn with_cached_tokens<T: WithUsage>(
    mut response: T,
    cached: u32,
) -> WithCachedTokens<T> {
    let usage = response.take_usage().map(|usage| UsageWithCachedTokens {
        usage,
        prompt_tokens_details: PromptTokenUsageInfo {
            cached_tokens: cached,
        },
    });
    WithCachedTokens { response, usage }
}
