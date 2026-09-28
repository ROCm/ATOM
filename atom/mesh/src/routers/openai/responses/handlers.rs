//! Entry handler for POST /v1/responses. Dispatches to streaming or
//! synchronous execution; background mode is not supported.

use std::sync::Arc;

use axum::{
    http,
    response::{IntoResponse, Response},
};
use uuid::Uuid;

use super::{
    context::ResponsesContext, conversation::load_conversation_history, conversions, non_streaming,
    streaming,
};
use crate::{protocols::responses::ResponsesRequest, routers::comm::error};

pub(crate) async fn route_responses(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    let is_background = request.background.unwrap_or(false);
    if is_background {
        return error::bad_request(
            "unsupported_parameter",
            "Background mode is not supported. Please set 'background' to false or omit it.",
        );
    }

    let is_streaming = request.stream.unwrap_or(false);
    if is_streaming {
        route_responses_streaming(ctx, request, headers, model_id).await
    } else {
        let response_id = Some(format!("resp_{}", Uuid::new_v4()));
        route_responses_sync(ctx, request, headers, model_id, response_id).await
    }
}

async fn route_responses_sync(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
    response_id: Option<String>,
) -> Response {
    match non_streaming::route_responses_internal(ctx, request, headers, model_id, response_id)
        .await
    {
        Ok(responses_response) => axum::Json(responses_response).into_response(),
        Err(response) => response,
    }
}

async fn route_responses_streaming(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    headers: Option<http::HeaderMap>,
    model_id: Option<String>,
) -> Response {
    let modified_request = match load_conversation_history(ctx, &request).await {
        Ok(req) => req,
        Err(response) => return response,
    };

    let chat_request = match conversions::responses_to_chat(&modified_request) {
        Ok(req) => Arc::new(req),
        Err(e) => {
            return error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {}", e),
            );
        }
    };

    streaming::convert_chat_stream_to_responses_stream(
        ctx,
        chat_request,
        headers,
        model_id,
        &request,
    )
    .await
}
