//! Staged `RequestPipeline` retained for the PD router until Part G switches
//! it to the new `Pipeline`. The regular gRPC router has already moved off
//! this code path.

use std::{sync::Arc, time::Instant};

use axum::response::{IntoResponse, Response};
use http::HeaderMap;
use tracing::error;

use super::{
    common::stages::*,
    context::*,
    regular::{processor, stages::*, streaming},
    utils::error_type_from_status,
};
use crate::{
    core::{
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::PdPlanner,
        },
        WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    observability::metrics::{bool_to_static_str, metrics_labels, Metrics},
    policies::PolicyRegistry,
    protocols::{chat::ChatCompletionRequest, generate::GenerateRequest},
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::error,
    tool_parser::ParserFactory as ToolParserFactory,
};

#[derive(Clone)]
pub(crate) struct RequestPipeline {
    stages: Arc<Vec<Box<dyn PipelineStage>>>,
    backend_type: &'static str,
}

impl RequestPipeline {
    pub fn new_pd(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        let processor = processor::ResponseProcessor::new(
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            configured_tool_parser.clone(),
            configured_reasoning_parser.clone(),
        );

        let streaming_processor = Arc::new(streaming::StreamingProcessor::new(
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
            metrics_labels::BACKEND_PD,
        ));

        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry)),
            Arc::new(PolicyRegistryAdapter::new(policy_registry)),
        ));

        let stages: Vec<Box<dyn PipelineStage>> = vec![
            Box::new(PreparationStage::new()),
            Box::new(WorkerSelectionStage::new(planner)),
            Box::new(ClientAcquisitionStage),
            Box::new(RequestBuildingStage::new(true)),
            Box::new(DispatchMetadataStage),
            Box::new(RequestExecutionStage::new(ExecutionMode::DualDispatch)),
            Box::new(ResponseProcessingStage::new(processor, streaming_processor)),
        ];

        Self {
            stages: Arc::new(stages),
            backend_type: metrics_labels::BACKEND_PD,
        }
    }

    pub async fn execute_chat(
        &self,
        request: Arc<ChatCompletionRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let start = Instant::now();
        let request_for_metrics = Arc::clone(&request);
        let streaming = request.stream;

        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            &request_for_metrics.model,
            metrics_labels::ENDPOINT_CHAT,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_chat(request, headers, model_id, components);

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        &request_for_metrics.model,
                        metrics_labels::ENDPOINT_CHAT,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Chat(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(FinalResponse::Generate(_)) => {
                error!(
                    function = "execute_chat",
                    "Wrong response type: expected Chat, got Generate"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_chat",
                    "No response produced by pipeline"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    &request_for_metrics.model,
                    metrics_labels::ENDPOINT_CHAT,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }

    pub async fn execute_generate(
        &self,
        request: Arc<GenerateRequest>,
        headers: Option<HeaderMap>,
        model_id: Option<String>,
        components: Arc<SharedComponents>,
    ) -> Response {
        let start = Instant::now();
        let streaming = request.stream;

        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            self.backend_type,
            metrics_labels::CONNECTION_GRPC,
            model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            metrics_labels::ENDPOINT_GENERATE,
            bool_to_static_str(streaming),
        );

        let mut ctx = RequestContext::for_generate(request, headers, model_id.clone(), components);

        for stage in self.stages.iter() {
            match stage.execute(&mut ctx).await {
                Ok(Some(response)) => {
                    Metrics::record_router_duration(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_GENERATE,
                        start.elapsed(),
                    );
                    return response;
                }
                Ok(None) => continue,
                Err(response) => {
                    Metrics::record_router_error(
                        metrics_labels::ROUTER_GRPC,
                        self.backend_type,
                        metrics_labels::CONNECTION_GRPC,
                        model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                        metrics_labels::ENDPOINT_GENERATE,
                        error_type_from_status(response.status()),
                    );
                    error!(
                        "Stage {} failed with status {}",
                        stage.name(),
                        response.status()
                    );
                    return response;
                }
            }
        }

        match ctx.state.response.final_response {
            Some(FinalResponse::Generate(response)) => {
                Metrics::record_router_duration(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    start.elapsed(),
                );
                axum::Json(response).into_response()
            }
            Some(FinalResponse::Chat(_)) => {
                error!(
                    function = "execute_generate",
                    "Wrong response type: expected Generate, got Chat"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("wrong_response_type", "Internal error: wrong response type")
            }
            None => {
                error!(
                    function = "execute_generate",
                    "No response produced by pipeline"
                );
                Metrics::record_router_error(
                    metrics_labels::ROUTER_GRPC,
                    self.backend_type,
                    metrics_labels::CONNECTION_GRPC,
                    model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
                    metrics_labels::ENDPOINT_GENERATE,
                    metrics_labels::ERROR_INTERNAL,
                );
                error::internal_error("no_response_produced", "No response produced")
            }
        }
    }
}
