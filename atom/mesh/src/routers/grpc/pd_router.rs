//! gRPC router for prefill/decode (PD) dual-dispatch chat/generate paths.

use std::sync::Arc;

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use tracing::debug;

#[cfg(feature = "grpc-client")]
use super::engine::atom::pair::AtomPairs;
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
    core::{
        is_retryable_status, ConnectionMode, RetryExecutor, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
    observability::metrics::{metrics_labels, MeshMetrics},
    protocols::{
        chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
    },
    routers::{comm::error, prepare, RouterTrait, SentChatBody},
};

#[derive(Clone)]
pub struct GrpcPDRouter {
    worker_registry: Arc<WorkerRegistry>,
    pipeline: Pipeline,
    app_context: Arc<AppContext>,
    retry_config: RetryConfig,
}

impl GrpcPDRouter {
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        if ctx.reasoning_parser_factory.is_none() {
            return Err("gRPC PD router requires reasoning parser factory".to_string());
        }
        if ctx.tool_parser_factory.is_none() {
            return Err("gRPC PD router requires tool parser factory".to_string());
        }

        let worker_registry = ctx.worker_registry.clone();
        let pipeline = Pipeline::new_pd(worker_registry.clone(), ctx.policy_registry.clone());
        #[cfg(feature = "grpc-client")]
        let pipeline = pipeline.with_atom_pairs(
            AtomPairs::new(
                worker_registry.clone(),
                ctx.router_config.atom_pd_rank_mapping_policy,
            )
            .await?,
        );

        Ok(GrpcPDRouter {
            worker_registry,
            pipeline,
            app_context: ctx.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    async fn route_generate_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing generate request for model: {} (PD mode)",
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
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                MeshMetrics::record_worker_retry(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_GENERATE,
                );
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_GENERATE,
                );
            },
        )
        .await
    }

    /// Route a chat, its prompt rendered from its body as `sent` where the
    /// caller has it (`prepare::sent_chat_body`).
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
        sent: Option<SentChatBody>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {} (PD mode)",
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
                let sent = sent.clone();
                async move {
                    pipeline
                        .execute_chat_with(request, headers, model_id, components, sent)
                        .await
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                MeshMetrics::record_worker_retry(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_CHAT,
                );
                MeshMetrics::record_worker_retry(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_CHAT,
                );
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    metrics_labels::ENDPOINT_CHAT,
                );
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    metrics_labels::ENDPOINT_CHAT,
                );
            },
        )
        .await
    }
}

impl std::fmt::Debug for GrpcPDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefill_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(ConnectionMode::Grpc { port: None }),
            None,
            false,
        );
        let decode_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Decode),
            Some(ConnectionMode::Grpc { port: None }),
            None,
            false,
        );
        f.debug_struct("GrpcPDRouter")
            .field("prefill_workers_count", &prefill_workers.len())
            .field("decode_workers_count", &decode_workers.len())
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcPDRouter {
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

    fn sent_chat_body(
        &self,
        sent: &[u8],
        model_id: Option<&str>,
    ) -> Result<Option<SentChatBody>, Response> {
        prepare::sent_chat_body(sent, model_id, &self.app_context)
    }

    async fn route_chat_as_sent(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        sent: Option<SentChatBody>,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id, sent).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id, None).await
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

    fn router_type(&self) -> &'static str {
        "grpc_pd"
    }
}
