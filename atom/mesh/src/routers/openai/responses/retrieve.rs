//! GET /v1/responses/{id} and POST /v1/responses/{id}/cancel handlers.

use axum::response::{IntoResponse, Response};
use data_connector::ResponseId;

use super::context::ResponsesContext;
use crate::routers::comm::error;

pub(crate) async fn get_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => axum::Json(stored_response.raw_response).into_response(),
        Ok(None) => error::not_found(
            "response_not_found",
            format!("Response with id '{}' not found", response_id),
        ),
        Err(e) => error::internal_error(
            "retrieve_response_failed",
            format!("Failed to retrieve response: {}", e),
        ),
    }
}

pub(crate) async fn cancel_response_impl(ctx: &ResponsesContext, response_id: &str) -> Response {
    let resp_id = ResponseId::from(response_id);

    match ctx.response_storage.get_response(&resp_id).await {
        Ok(Some(stored_response)) => {
            let current_status = stored_response
                .raw_response
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            match current_status {
                "completed" => error::bad_request(
                    "response_already_completed",
                    "Cannot cancel completed response",
                ),
                "failed" => {
                    error::bad_request("response_already_failed", "Cannot cancel failed response")
                }
                _ => error::bad_request(
                    "cancellation_not_supported",
                    "Background mode is not supported. Synchronous and streaming responses cannot be cancelled.",
                ),
            }
        }
        Ok(None) => error::not_found(
            "response_not_found",
            format!("Response with id '{}' not found", response_id),
        ),
        Err(e) => error::internal_error(
            "retrieve_response_failed",
            format!("Failed to retrieve response: {}", e),
        ),
    }
}
