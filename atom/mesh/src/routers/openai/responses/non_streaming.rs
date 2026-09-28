//! Non-streaming execution for /v1/responses.

use std::sync::Arc;

use axum::response::Response;
use tracing::error;

use super::{
    context::ResponsesContext, conversation::load_conversation_history, conversions,
    persistence::persist_response_if_needed,
};
use crate::{
    protocols::responses::{ResponsesRequest, ResponsesResponse},
    routers::comm::error,
};

pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    let modified_request = load_conversation_history(ctx, &request).await?;

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

    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            headers,
            model_id,
            ctx.components.clone(),
        )
        .await?;

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
