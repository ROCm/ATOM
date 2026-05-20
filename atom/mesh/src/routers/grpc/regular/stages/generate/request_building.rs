//! Generate request building stage: Build proto GenerateRequest for generate requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext, WorkerSelection},
        engine::payload_to_proto::{to_sglang_proto, to_vllm_proto},
        engine::proto_stream_wrapper::ProtoGenerateRequest,
        engine::worker_client_cache::GrpcClient,
    },
    prepare::build_generate_payload,
};

pub(crate) struct GenerateRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl GenerateRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for GenerateRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "GenerateRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let generate_request = ctx.generate_request_arc();

        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let request_id = generate_request
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

        let payload = build_generate_payload(
            request_id,
            &generate_request,
            prep.original_text.clone(),
            prep.token_ids.clone(),
        );

        let mut proto_request = match builder_client {
            GrpcClient::Sglang(_) => {
                ProtoGenerateRequest::Sglang(Box::new(to_sglang_proto(&payload)))
            }
            GrpcClient::Vllm(_) => ProtoGenerateRequest::Vllm(Box::new(to_vllm_proto(&payload))),
        };

        if self.inject_pd_metadata {
            if let WorkerSelection::Dual { prefill, .. } = ctx.state.workers.as_ref().unwrap() {
                helpers::inject_bootstrap_metadata(&mut proto_request, prefill);
            }
        }

        ctx.state.proto_request = Some(
            crate::routers::grpc::engine::proto_stream_wrapper::ProtoRequest::Generate(
                proto_request,
            ),
        );
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "GenerateRequestBuilding"
    }
}
