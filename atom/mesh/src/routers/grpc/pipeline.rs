//! Transport-neutral 4-step gRPC pipeline: prepare → plan → engine.dispatch → render.

use std::{sync::Arc, time::Instant};

use axum::response::Response;
use http::HeaderMap;
use tracing::error;

use super::engine::{Dispatcher, GrpcEngine};
use crate::{
    app_context::AppContext,
    core::{
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor},
        },
        WorkerLoadGuard, WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    observability::metrics::{bool_to_static_str, metrics_labels, MeshMetrics},
    policies::PolicyRegistry,
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse},
        generate::GenerateRequest,
    },
    routers::{
        comm::{
            error, metrics_utils::error_type_from_status,
            placement_response::placement_err_to_response,
        },
        prepare::{self, generation_payload::GenerationPayload},
        render,
        token_handle::engine_error::EngineError,
    },
};

#[derive(Clone)]
pub(crate) struct Pipeline {
    planner: Arc<dyn PdPlanner>,
    engine: Arc<dyn Dispatcher>,
    backend_label: &'static str,
}

#[cfg(test)]
impl Pipeline {
    pub(crate) fn with_injected(
        planner: Arc<dyn PdPlanner>,
        engine: Arc<dyn Dispatcher>,
        backend_label: &'static str,
    ) -> Self {
        Self {
            planner,
            engine,
            backend_label,
        }
    }
}

impl Pipeline {
    pub fn new_regular(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_label(
            worker_registry,
            policy_registry,
            metrics_labels::BACKEND_REGULAR,
        )
    }

    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_label(worker_registry, policy_registry, metrics_labels::BACKEND_PD)
    }

    fn with_label(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        backend_label: &'static str,
    ) -> Self {
        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry)),
            Arc::new(PolicyRegistryAdapter::new(policy_registry)),
        ));
        Self {
            planner,
            engine: Arc::new(GrpcEngine::new()),
            backend_label,
        }
    }

    pub async fn execute_chat(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Response {
        let start = Instant::now();
        let model = req.model.clone();
        let streaming = req.stream;
        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        let backend = PipelineBackend { pipeline: self };
        let response = match crate::entrypoints::openai::run_chat(
            &backend,
            req,
            headers,
            model_id,
            &components,
            self.backend_label,
        )
        .await
        {
            Ok(response) => response,
            Err(response) => return self.record_chat_err(&model, start, response),
        };

        MeshMetrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model,
            metrics_labels::ENDPOINT_CHAT,
            start.elapsed(),
        );
        response
    }

    pub async fn execute_generate(
        &self,
        req: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Response {
        let start = Instant::now();
        let model_for_metric = model_id
            .clone()
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());
        let streaming = req.stream;
        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model_for_metric,
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        let backend = PipelineBackend { pipeline: self };
        let response = match crate::entrypoints::openai::run_generate(
            &backend,
            req,
            headers,
            model_id,
            &components,
            self.backend_label,
        )
        .await
        {
            Ok(response) => response,
            Err(response) => return self.record_generate_err(&model_for_metric, start, response),
        };

        MeshMetrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            &model_for_metric,
            metrics_labels::ENDPOINT_GENERATE,
            start.elapsed(),
        );
        response
    }

    pub async fn execute_chat_for_responses(
        &self,
        req: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<AppContext>,
    ) -> Result<ChatCompletionResponse, Response> {
        if req.stream {
            return Err(error::bad_request(
                "streaming_not_supported",
                "Streaming is not supported in this context".to_string(),
            ));
        }

        let (payload, mut resp_ctx) = prepare::prepare_chat(req, headers, model_id, &components)?;
        let stream = self
            .dispatch_tokens(crate::entrypoints::backend::TokenRequest::from_prepared(
                payload,
                &mut resp_ctx,
            ))
            .await
            .map_err(crate::entrypoints::backend::BackendError::into_response)?;
        // The load guards ride in `_finish` until the response is built.
        let (handle, _finish) = stream.into_parts();
        render::chat_aggregator::process_typed(handle, resp_ctx).await
    }

    pub(crate) async fn dispatch_tokens(
        &self,
        request: crate::entrypoints::backend::TokenRequest,
    ) -> Result<crate::entrypoints::backend::TokenStream, crate::entrypoints::backend::BackendError>
    {
        use crate::entrypoints::backend::{TokenRequest, TokenStream};
        let TokenRequest {
            mut payload,
            model,
            headers,
        } = request;
        let placement = self
            .plan_tokens(&payload, model.as_deref(), headers.as_ref())
            .await
            .map_err(backend_error_from_response)?;
        let guards = make_load_guards(&placement, headers.as_ref());
        let handle = self
            .engine
            .dispatch(&placement, &mut payload)
            .await
            .map_err(|error| backend_error_from_response(engine_err_to_response(error)))?;
        Ok(TokenStream::with_finish(handle, guards))
    }

    async fn plan_tokens(
        &self,
        payload: &GenerationPayload,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<PlacementPlan, Response> {
        let tokens = if payload.token_ids.is_empty() {
            None
        } else {
            Some(payload.token_ids.as_slice())
        };
        let text = if payload.text.is_empty() {
            None
        } else {
            Some(payload.text.as_str())
        };
        let descriptor = RequestDescriptor {
            model_id,
            protocol: Some(Protocol::Grpc),
            text,
            tokens,
            headers,
            stream: payload.stream,
            carries_media: false,
        };
        self.planner
            .plan(&descriptor)
            .await
            .map_err(|error| placement_err_to_response_log(error, model_id))
    }

    fn record_chat_err(&self, model: &str, _start: Instant, resp: Response) -> Response {
        MeshMetrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            model,
            metrics_labels::ENDPOINT_CHAT,
            error_type_from_status(resp.status()),
        );
        resp
    }

    fn record_generate_err(&self, model: &str, _start: Instant, resp: Response) -> Response {
        MeshMetrics::record_router_error(
            metrics_labels::ROUTER_GRPC,
            self.backend_label,
            metrics_labels::CONNECTION_GRPC,
            model,
            metrics_labels::ENDPOINT_GENERATE,
            error_type_from_status(resp.status()),
        );
        resp
    }
}

struct PipelineBackend<'a> {
    pipeline: &'a Pipeline,
}

#[async_trait::async_trait]
impl crate::entrypoints::backend::Backend for PipelineBackend<'_> {
    async fn generate(
        &self,
        request: crate::entrypoints::backend::TokenRequest,
    ) -> Result<crate::entrypoints::backend::TokenStream, crate::entrypoints::backend::BackendError>
    {
        self.pipeline.dispatch_tokens(request).await
    }

    async fn abort(
        &self,
        _request_id: &str,
    ) -> Result<(), crate::entrypoints::backend::BackendError> {
        Err(crate::entrypoints::backend::BackendError::unsupported(
            "the grpc pipeline does not abort an accepted request",
        ))
    }
}

fn backend_error_from_response(response: Response) -> crate::entrypoints::backend::BackendError {
    let status = response.status();
    let (code, message) = response
        .extensions()
        .get::<error::MeshLocalError>()
        .map(error::MeshLocalError::code_and_message)
        .unwrap_or_else(|| {
            (
                "backend_error".to_string(),
                status
                    .canonical_reason()
                    .unwrap_or("backend error")
                    .to_string(),
            )
        });
    crate::entrypoints::backend::BackendError::new(status, code, message)
}

fn make_load_guards(plan: &PlacementPlan, headers: Option<&HeaderMap>) -> Vec<WorkerLoadGuard> {
    match plan {
        PlacementPlan::Single { worker, .. } => {
            vec![WorkerLoadGuard::new(worker.clone(), headers)]
        }
        PlacementPlan::Pair {
            prefill, decode, ..
        } => vec![
            WorkerLoadGuard::new(prefill.clone(), headers),
            WorkerLoadGuard::new(decode.clone(), headers),
        ],
    }
}

fn placement_err_to_response_log(err: PlacementError, model_id: Option<&str>) -> Response {
    error!(
        function = "Pipeline::plan_for",
        model_id = %model_id.unwrap_or(UNKNOWN_MODEL_ID),
        error = %err,
        "Placement planner returned error"
    );
    placement_err_to_response(err, model_id)
}

fn engine_err_to_response(err: EngineError) -> Response {
    match err {
        EngineError::Transport(s) => {
            error::service_unavailable("engine_transport_error", format!("transport error: {}", s))
        }
        EngineError::Prefill(m) => {
            error::internal_error("engine_prefill_error", format!("prefill error: {}", m))
        }
        EngineError::DecodeError(m) => {
            error::internal_error("engine_decode_error", format!("decode error: {}", m))
        }
        EngineError::PrefillEarlyClose => error::internal_error(
            "engine_prefill_early_close",
            "prefill stream closed without Complete",
        ),
        EngineError::DecodeIncomplete => error::internal_error(
            "engine_decode_incomplete",
            "decode stream closed without Complete",
        ),
        EngineError::ConnectionAcquireFailed(r) => {
            error::service_unavailable("engine_connection_failed", r)
        }
        EngineError::RequestBuildFailed(r) => error::bad_request("engine_request_build_failed", r),
    }
}
