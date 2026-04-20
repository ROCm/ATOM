//! Non-streaming execution for Regular Responses API
//!
//! This module handles non-streaming request execution:
//! - `route_responses_internal` - Core execution orchestration

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use super::{common::load_conversation_history, conversions};
use crate::{
    protocols::responses::{ResponsesRequest, ResponsesResponse},
    routers::{
        error,
        grpc::common::responses::{persist_response_if_needed, ResponsesContext},
    },
};

/// Internal implementation for non-streaming responses
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Converts to chat format
/// 3. Executes chat pipeline
/// 4. Converts back to responses format
/// 5. Persists to storage
pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(ctx, &request).await?;

    // 2. Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(&modified_request).map_err(|e| {
        error!(
            function = "route_responses_internal",
            error = %e,
            "Failed to convert ResponsesRequest to ChatCompletionRequest"
        );
        error::bad_request(
            "convert_request_failed",
            format!("Failed to convert request: {}", e),
        )
    })?;

    // 3. Execute chat pipeline
    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await?;

    // 4. Convert ChatCompletionResponse → ResponsesResponse
    let responses_response = conversions::chat_to_responses(&chat_response, &request, response_id)
        .map_err(|e| {
            error!(
                function = "route_responses_internal",
                error = %e,
                "Failed to convert ChatCompletionResponse to ResponsesResponse"
            );
            error::internal_error(
                "convert_to_responses_format_failed",
                format!("Failed to convert to responses format: {}", e),
            )
        })?;

    // 5. Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &request,
    )
    .await;

    Ok(responses_response)
}
