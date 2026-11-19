//! Token HTTP API.
//!
//! `/inference/v1/generate` and `/inference/v1/abort_requests` pass token ids
//! through. They do not render a template, tokenize, or detokenize.

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::json;

use super::{MeshBackend, TokenBackend};
use crate::server::AppState;
use crate::token_path::{AbortRequests, TokenGenerateEvent, TokenGenerateRequest, TokenPathError};

pub async fn generate(
    State(state): State<std::sync::Arc<AppState>>,
    Json(request): Json<TokenGenerateRequest>,
) -> Response {
    let stream = request.stream;
    let backend = MeshBackend::from_context(&state.context);
    match backend.generate(request).await {
        Ok(events) => {
            if stream {
                streaming_response(events)
            } else {
                collect_response(events).await
            }
        }
        Err(error) => error_response(error),
    }
}

pub async fn abort_requests(
    State(state): State<std::sync::Arc<AppState>>,
    Json(body): Json<AbortRequests>,
) -> Response {
    let backend = MeshBackend::from_context(&state.context);
    match backend.abort(&body.request_ids).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(error) => error_response(error),
    }
}

fn streaming_response(events: crate::token_path::TokenStream) -> Response {
    let body = Body::from_stream(events.into_inner().map(|item| {
        let line = match item {
            Ok(event) => format!(
                "data: {}\n\n",
                serde_json::to_string(&event).unwrap_or_default()
            ),
            Err(error) => format!("data: {}\n\n", error_json(&error)),
        };
        Ok::<Bytes, std::io::Error>(Bytes::from(line))
    }));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn collect_response(events: crate::token_path::TokenStream) -> Response {
    let mut collected: Vec<TokenGenerateEvent> = Vec::new();
    let mut stream = events.into_inner();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => collected.push(event),
            Err(error) => return error_response(error),
        }
    }
    Json(collected).into_response()
}

fn error_response(error: TokenPathError) -> Response {
    let status = error.status_code();
    let body = error_json(&error);
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap_or_else(|_| status.into_response())
}

fn error_json(error: &TokenPathError) -> String {
    json!({
        "error": {
            "type": status_name(error.status_code()),
            "code": error.code(),
            "message": error.to_string(),
        }
    })
    .to_string()
}

fn status_name(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        502 => "server_error",
        503 => "server_error",
        _ => "server_error",
    }
}
