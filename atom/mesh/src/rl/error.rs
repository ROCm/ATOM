//! Errors of the RL control plane. Each maps to one HTTP status and a JSON
//! body `{"error": <code>, "message": <text>, ...context}`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlError {
    #[error("missing or wrong `Authorization: Bearer` token")]
    Unauthorized,
    #[error("worker `{0}` not found")]
    WorkerNotFound(String),
}

impl RlError {
    /// Stable machine-readable code for the JSON body.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::WorkerNotFound(_) => "worker_not_found",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::WorkerNotFound(_) => StatusCode::NOT_FOUND,
        }
    }

    /// JSON body with the code, message and per-variant context.
    pub fn to_json(&self) -> Value {
        let mut body = json!({ "error": self.code(), "message": self.to_string() });
        match self {
            Self::Unauthorized => {}
            Self::WorkerNotFound(id) => body["id"] = json!(id),
        }
        body
    }
}

impl IntoResponse for RlError {
    fn into_response(self) -> Response {
        (self.status(), Json(self.to_json())).into_response()
    }
}
