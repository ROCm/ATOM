//! Builds real Atomesh HTTP requests from fixture data.
//!
//! The harness should exercise Atomesh's public API surface, not call router
//! internals directly. `VirtualRequest` is the small adapter from
//! `MockTestCase` JSON into an Axum request that can enter `server::build_app`.

use axum::{body::Body, extract::Request, http::header::CONTENT_TYPE};
use serde_json::{Map, Value};

use super::mock_test_case::MockTestCase;

/// Client-facing request generated from a fixture.
#[derive(Clone, Debug)]
pub struct VirtualRequest {
    pub endpoint: String,
    pub body: Value,
}

impl VirtualRequest {
    /// Convert fixture request data into the shape expected by Atomesh routes.
    pub fn from_case(case: &MockTestCase) -> Self {
        let mut body = match case.request.clone() {
            Value::Object(map) => map,
            _ => Map::new(),
        };

        if matches!(
            case.endpoint.as_str(),
            "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
        ) {
            // OpenAI-compatible routes require a model; fixtures may omit it
            // when they want to reuse the top-level `model` field.
            body.entry("model".to_string())
                .or_insert_with(|| Value::String(case.model.clone()));
        }

        Self {
            endpoint: case.endpoint.clone(),
            body: Value::Object(body),
        }
    }

    /// Turn the virtual request into an Axum request for `oneshot`.
    pub fn into_axum_request(self) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(self.endpoint)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&self.body).unwrap()))
            .unwrap()
    }
}
