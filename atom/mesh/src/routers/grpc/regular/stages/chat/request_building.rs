//! Chat request building stage: Build proto GenerateRequest for chat requests

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;
use uuid::Uuid;

use crate::routers::{
    error,
    grpc::{
        client::GrpcClient,
        common::stages::{helpers, PipelineStage},
        context::{ClientSelection, RequestContext, WorkerSelection},
        engine::payload_to_proto::{to_sglang_proto, to_vllm_proto},
        proto_wrapper::ProtoGenerateRequest,
    },
    prepare::build_chat_payload,
};

pub(crate) struct ChatRequestBuildingStage {
    inject_pd_metadata: bool,
}

impl ChatRequestBuildingStage {
    pub fn new(inject_pd_metadata: bool) -> Self {
        Self { inject_pd_metadata }
    }
}

#[async_trait]
impl PipelineStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Preparation not completed"
            );
            error::internal_error("preparation_not_completed", "Preparation not completed")
        })?;

        let clients = ctx.state.clients.as_ref().ok_or_else(|| {
            error!(
                function = "ChatRequestBuildingStage::execute",
                "Client acquisition not completed"
            );
            error::internal_error(
                "client_acquisition_not_completed",
                "Client acquisition not completed",
            )
        })?;

        let chat_request = ctx.chat_request_arc();

        let builder_client = match clients {
            ClientSelection::Single { client } => client,
            ClientSelection::Dual { prefill, .. } => prefill,
        };

        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let body_ref = prep.filtered_request.as_ref().unwrap_or(&chat_request);
        let processed_text = prep
            .processed_messages
            .as_ref()
            .ok_or_else(|| {
                error!(
                    function = "ChatRequestBuildingStage::execute",
                    "Chat preparation missing processed_messages"
                );
                error::internal_error(
                    "processed_messages_missing",
                    "Chat preparation missing processed_messages",
                )
            })?
            .text
            .clone();

        let payload = build_chat_payload(
            request_id,
            &chat_request,
            body_ref,
            &processed_text,
            prep.token_ids.clone(),
            prep.tool_constraints.clone(),
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
            crate::routers::grpc::proto_wrapper::ProtoRequest::Generate(proto_request),
        );
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}
