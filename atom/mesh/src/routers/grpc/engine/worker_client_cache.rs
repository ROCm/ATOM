//! Unified gRPC client wrapper for SGLang and vLLM backends, plus the
//! per-worker client lookup used by dispatch.

use std::sync::Arc;

use axum::response::Response;
use mesh_grpc::{SglangSchedulerClient, VllmEngineClient};
use tracing::error;

use crate::core::Worker;
use crate::routers::comm::error;
use crate::routers::grpc::engine::proto_stream_wrapper::{
    ProtoEmbedRequest, ProtoEmbedResponse, ProtoGenerateRequest, ProtoStream,
};

#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
}

impl GrpcClient {
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            Self::Vllm(_) => panic!("Expected SGLang client, got vLLM"),
        }
    }

    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            Self::Vllm(_) => panic!("Expected SGLang client, got vLLM"),
        }
    }

    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            Self::Sglang(_) => panic!("Expected vLLM client, got SGLang"),
        }
    }

    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            Self::Sglang(_) => panic!("Expected vLLM client, got SGLang"),
        }
    }

    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match runtime_type {
            "sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),
            "vllm" => Ok(Self::Vllm(VllmEngineClient::connect(url).await?)),
            _ => Err(format!("Unknown runtime type: {}", runtime_type).into()),
        }
    }

    pub async fn health_check(
        &self,
    ) -> Result<HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Vllm(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    pub async fn get_model_info(
        &self,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Sglang(Box::new(info)))
            }
            Self::Vllm(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Vllm(info))
            }
        }
    }

    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Vllm(stream))
            }
            _ => panic!("Mismatched client and request types"),
        }
    }

    pub async fn embed(
        &mut self,
        req: ProtoEmbedRequest,
    ) -> Result<ProtoEmbedResponse, Box<dyn std::error::Error + Send + Sync>> {
        match (self, req) {
            (Self::Sglang(client), ProtoEmbedRequest::Sglang(boxed_req)) => {
                let resp = client.embed(*boxed_req).await?;
                Ok(ProtoEmbedResponse::Sglang(resp))
            }
            _ => panic!("Mismatched client and request types or unsupported embedding backend"),
        }
    }
}

pub enum ModelInfo {
    Sglang(Box<mesh_grpc::sglang_proto::GetModelInfoResponse>),
    Vllm(mesh_grpc::vllm_proto::GetModelInfoResponse),
}

impl ModelInfo {
    /// Project the proto into a label map, dropping zero/empty/false fields.
    pub fn to_labels(&self) -> std::collections::HashMap<String, String> {
        let mut labels = std::collections::HashMap::new();

        let value = match self {
            ModelInfo::Sglang(info) => serde_json::to_value(info).ok(),
            ModelInfo::Vllm(info) => serde_json::to_value(info).ok(),
        };

        if let Some(serde_json::Value::Object(obj)) = value {
            for (key, val) in obj {
                match val {
                    serde_json::Value::String(s) if !s.is_empty() => {
                        labels.insert(key, s);
                    }
                    serde_json::Value::Number(n) if n.as_i64().unwrap_or(0) > 0 => {
                        labels.insert(key, n.to_string());
                    }
                    serde_json::Value::Bool(true) => {
                        labels.insert(key, "true".to_string());
                    }
                    serde_json::Value::Array(arr) if !arr.is_empty() => {
                        if let Ok(json_str) = serde_json::to_string(&arr) {
                            labels.insert(key, json_str);
                        }
                    }
                    _ => {}
                }
            }
        }

        labels
    }
}

pub(crate) async fn get_grpc_client_from_worker(
    worker: &Arc<dyn Worker>,
) -> Result<GrpcClient, Response> {
    let client_arc = worker
        .get_grpc_client()
        .await
        .map_err(|e| {
            error!(
                function = "get_grpc_client_from_worker",
                error = %e,
                "Failed to get gRPC client from worker"
            );
            error::internal_error(
                "get_grpc_client_failed",
                format!("Failed to get gRPC client: {}", e),
            )
        })?
        .ok_or_else(|| {
            error!(
                function = "get_grpc_client_from_worker",
                "Selected worker not configured for gRPC"
            );
            error::internal_error(
                "worker_not_configured_for_grpc",
                "Selected worker is not configured for gRPC",
            )
        })?;

    Ok((*client_arc).clone())
}
