//! Router implementations

use std::fmt::Debug;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};

use crate::protocols::{
    chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    responses::ResponsesRequest,
};

pub mod atom_standalone;
pub mod comm;
pub mod factory;
pub mod grpc;
pub mod http_pd_router;
pub mod http_router;
pub mod ingress;
pub mod openai;
pub mod prepare;
pub mod render;
pub mod router_manager;
#[cfg(test)]
pub(crate) mod test_mocks;
pub mod token_handle;

pub use factory::RouterFactory;

/// Core trait for all router implementations
///
/// This trait provides a unified interface for routing requests,
/// regardless of whether it's a regular router or PD router.
#[async_trait]
pub trait RouterTrait: Send + Sync + Debug {
    /// Get a reference to self as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Gracefully release router-owned resources during server shutdown.
    async fn shutdown(&self) {}

    /// Raw ingress dispatch. Native backends retain their existing typed API handling.
    async fn route_inference(
        &self,
        request: ingress::InferenceEnvelope,
        _app: &std::sync::Arc<crate::app_context::AppContext>,
    ) -> Response {
        use prepare::inference::ParsedInference;
        let headers = Some(&request.headers);
        let model = request.metadata.model.as_deref();
        match &request.parsed {
            ParsedInference::Chat(body) => self.route_chat(headers, body, model).await,
            ParsedInference::Completion(body) => self.route_completion(headers, body, model).await,
            ParsedInference::Generate(body) => self.route_generate(headers, body, model).await,
            ParsedInference::Responses(body) => {
                use crate::protocols::validated::Normalizable;
                use validator::Validate;
                match serde_json::from_value::<ResponsesRequest>(body.clone()) {
                    Ok(mut body) => {
                        body.normalize();
                        if let Err(err) = body.validate() {
                            return comm::error::IngressError::invalid(err.to_string())
                                .response(request.uri.path());
                        }
                        self.route_responses(headers, &body, model).await
                    }
                    Err(err) => comm::error::IngressError::invalid(err.to_string())
                        .response(request.uri.path()),
                }
            }
            ParsedInference::Messages(_) => comm::error::IngressError::new(
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_api",
                "Messages requires an HTTP backend",
            )
            .response(request.uri.path()),
        }
    }

    /// Route a health generate request
    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not implemented",
        )
            .into_response()
    }

    /// Get server information
    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Server info not implemented").into_response()
    }

    /// Get available models
    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get models not implemented").into_response()
    }

    /// Get model information
    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Get model info not implemented",
        )
            .into_response()
    }

    /// Route a generate request
    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not implemented",
        )
            .into_response()
    }

    /// Route a chat completion request
    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response;

    /// Route a completion request
    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented",
        )
            .into_response()
    }

    /// Route a responses request
    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented",
        )
            .into_response()
    }

    /// Retrieve a stored/background response by id
    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _query: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED, "Get response not implemented").into_response()
    }

    /// Cancel a background response by id
    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response not implemented",
        )
            .into_response()
    }

    /// Delete a response by id
    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses delete endpoint not implemented",
        )
            .into_response()
    }

    /// List input items of a response by id
    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _query: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses list input items endpoint not implemented",
        )
            .into_response()
    }

    /// Get router type name
    fn router_type(&self) -> &'static str;

    /// Check if this is a PD router
    fn is_pd_mode(&self) -> bool {
        self.router_type() == "pd"
    }
}
