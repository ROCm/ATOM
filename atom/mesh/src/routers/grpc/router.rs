//! gRPC router for regular (single-worker) chat/generate/completion paths.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use tracing::debug;

use super::{
    completion_adapter::{
        completion_to_generate, wrap_generate_response_as_completion,
        wrap_streaming_generate_as_completion,
    },
    pipeline::Pipeline,
};
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{is_retryable_status, RetryExecutor, WorkerRegistry, UNKNOWN_MODEL_ID},
    observability::metrics::{metrics_labels, MeshMetrics},
    protocols::{
        chat::ChatCompletionRequest,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        comm::error,
        openai::responses::{
            context::ResponsesContext,
            handlers as responses_handlers,
            retrieve::{cancel_response_impl, get_response_impl},
        },
        RouterTrait,
    },
};

#[derive(Clone)]
pub struct GrpcRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: Pipeline,
    app_context: Arc<AppContext>,
    responses_context: ResponsesContext,
    retry_config: RetryConfig,
}

impl GrpcRouter {
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        if ctx.reasoning_parser_factory.is_none() {
            return Err("gRPC router requires reasoning parser factory".to_string());
        }
        if ctx.tool_parser_factory.is_none() {
            return Err("gRPC router requires tool parser factory".to_string());
        }

        let worker_registry = ctx.worker_registry.clone();
        let pipeline = Pipeline::new_regular(worker_registry.clone(), ctx.policy_registry.clone());
        let app_context = ctx.clone();

        let responses_context = ResponsesContext::new(
            Arc::new(pipeline.clone()),
            app_context.clone(),
            ctx.response_storage.clone(),
            ctx.conversation_storage.clone(),
            ctx.conversation_item_storage.clone(),
        );

        Ok(GrpcRouter {
            worker_registry,
            pipeline,
            app_context,
            responses_context,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
        );

        let pipeline = &self.pipeline;

        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.app_context.clone();

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_chat(request, headers, model_id, components)
                        .await
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                MeshMetrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
        )
        .await
    }

    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing generate request for model: {}",
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        let request = Arc::new(body.clone());
        let headers_cloned = headers.cloned();
        let model_id_cloned = model_id.map(|s| s.to_string());
        let components = self.app_context.clone();
        let pipeline = &self.pipeline;

        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let request = Arc::clone(&request);
                let headers = headers_cloned.clone();
                let model_id = model_id_cloned.clone();
                let components = Arc::clone(&components);
                async move {
                    pipeline
                        .execute_generate(request, headers, model_id, components)
                        .await
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                MeshMetrics::record_worker_retry(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::ENDPOINT_GENERATE,
                );
            },
        )
        .await
    }

    async fn route_responses_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        responses_handlers::route_responses(
            &self.responses_context,
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(|s| s.to_string()),
        )
        .await
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_impl(headers, body, model_id).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let synthetic = match completion_to_generate(body) {
            Ok(g) => g,
            Err(msg) => return error::bad_request("completion_unsupported_field", msg),
        };

        let is_stream = body.stream;
        debug!(
            "Routing /v1/completions via synthetic generate (stream={}) for model: {}",
            is_stream,
            model_id.unwrap_or(UNKNOWN_MODEL_ID)
        );

        let upstream = self
            .route_generate_impl(headers, &synthetic, model_id)
            .await;

        if is_stream {
            wrap_streaming_generate_as_completion(upstream, body.model.clone()).await
        } else {
            wrap_generate_response_as_completion(upstream, body.model.clone()).await
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_responses_impl(headers, body, model_id).await
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        get_response_impl(&self.responses_context, response_id).await
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, response_id: &str) -> Response {
        cancel_response_impl(&self.responses_context, response_id).await
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }
}
