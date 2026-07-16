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
        AttachedBody, WorkerLoadGuard, WorkerRegistry, UNKNOWN_MODEL_ID,
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
        prepare::{self, generation_payload::GenerationPayload, response_context::ResponseContext},
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

        let (mut payload, resp_ctx) =
            match prepare::prepare_chat(req, headers, model_id, &components) {
                Ok(t) => t,
                Err(e) => return self.record_chat_err(&model, start, e),
            };
        let placement = match self.plan_for(&payload, &resp_ctx).await {
            Ok(p) => p,
            Err(e) => return self.record_chat_err(&model, start, e),
        };
        let guards = make_load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = match self.engine.dispatch(&placement, &mut payload).await {
            Ok(s) => s,
            Err(e) => return self.record_chat_err(&model, start, engine_err_to_response(e)),
        };

        let is_stream = resp_ctx.original.is_streaming();
        let response = if is_stream {
            render::chat_streaming::process(stream, resp_ctx, self.backend_label)
        } else {
            render::chat_aggregator::process(stream, resp_ctx).await
        };
        let response = AttachedBody::wrap_response(response, guards);

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

        let (mut payload, resp_ctx) =
            match prepare::prepare_generate(req, headers, model_id, &components) {
                Ok(t) => t,
                Err(e) => return self.record_generate_err(&model_for_metric, start, e),
            };
        let placement = match self.plan_for(&payload, &resp_ctx).await {
            Ok(p) => p,
            Err(e) => return self.record_generate_err(&model_for_metric, start, e),
        };
        let guards = make_load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = match self.engine.dispatch(&placement, &mut payload).await {
            Ok(s) => s,
            Err(e) => {
                return self.record_generate_err(
                    &model_for_metric,
                    start,
                    engine_err_to_response(e),
                )
            }
        };

        let is_stream = resp_ctx.original.is_streaming();
        let response = if is_stream {
            render::generate_streaming::process(stream, resp_ctx, self.backend_label)
        } else {
            render::generate_aggregator::process(stream, resp_ctx).await
        };
        let response = AttachedBody::wrap_response(response, guards);

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

        let (mut payload, resp_ctx) = prepare::prepare_chat(req, headers, model_id, &components)?;
        let placement = self.plan_for(&payload, &resp_ctx).await?;
        let _guards = make_load_guards(&placement, resp_ctx.headers.as_ref());
        let stream = self
            .engine
            .dispatch(&placement, &mut payload)
            .await
            .map_err(engine_err_to_response)?;

        render::chat_aggregator::process_typed(stream, resp_ctx).await
    }

    async fn plan_for(
        &self,
        payload: &GenerationPayload,
        resp_ctx: &ResponseContext,
    ) -> Result<PlacementPlan, Response> {
        let tokens = if payload.token_ids.is_empty() {
            None
        } else {
            Some(payload.token_ids.as_slice())
        };
        let descriptor = RequestDescriptor {
            model_id: resp_ctx.model_id.as_deref(),
            protocol: Some(Protocol::Grpc),
            text: resp_ctx.original_text.as_deref(),
            tokens,
            headers: resp_ctx.headers.as_ref(),
            stream: resp_ctx.original.is_streaming(),
        };
        self.planner
            .plan(&descriptor)
            .await
            .map_err(|e| placement_err_to_response_log(e, resp_ctx.model_id.as_deref()))
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
