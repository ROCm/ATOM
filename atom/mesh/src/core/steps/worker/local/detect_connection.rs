//! Connection mode detection step.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::strip_protocol;
use crate::{
    core::{steps::workflow_data::LocalWorkerWorkflowData, ConnectionMode},
    routers::grpc::engine::worker_client_cache::GrpcClient,
};

/// Try HTTP health check.
async fn try_http_health_check(
    url: &str,
    timeout_secs: u64,
    client: &Client,
) -> Result<(), String> {
    let is_https = url.starts_with("https://");
    let protocol = if is_https { "https" } else { "http" };
    let clean_url = strip_protocol(url);
    let health_url = format!("{}://{}/health", protocol, clean_url);

    client
        .get(&health_url)
        .timeout(Duration::from_secs(timeout_secs))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("Health check failed: {}", e))?;

    Ok(())
}

/// Starts the message of a refusal, which retrying cannot change.
const REFUSED: &str = "Refused";

/// Perform gRPC health check with runtime type; `Ok(Some(reason))` refuses the worker.
async fn do_grpc_health_check(
    grpc_url: &str,
    timeout_secs: u64,
    runtime_type: &str,
) -> Result<Option<String>, String> {
    let connect_future = GrpcClient::connect(grpc_url, runtime_type);
    let client = tokio::time::timeout(Duration::from_secs(timeout_secs), connect_future)
        .await
        .map_err(|_| "gRPC connection timeout".to_string())?
        .map_err(|e| format!("gRPC connection failed: {}", e))?;

    let health_future = client.health_check();
    tokio::time::timeout(Duration::from_secs(timeout_secs), health_future)
        .await
        .map_err(|_| "gRPC health check timeout".to_string())?
        .map_err(|e| format!("gRPC health check failed: {}", e))?;

    tokio::time::timeout(Duration::from_secs(timeout_secs), client.protocol_refusal())
        .await
        .map_err(|_| "gRPC server info timeout".to_string())?
        .map_err(|e| format!("gRPC server info failed: {}", e))
}

/// Try gRPC health check (SGLang, then vLLM, then ATOM if not specified).
async fn try_grpc_health_check(
    url: &str,
    timeout_secs: u64,
    runtime_type: Option<&str>,
) -> Result<Option<String>, String> {
    let grpc_url = if url.starts_with("grpc://") {
        url.to_string()
    } else {
        format!("grpc://{}", strip_protocol(url))
    };

    match runtime_type {
        Some(runtime) => do_grpc_health_check(&grpc_url, timeout_secs, runtime).await,
        None => {
            for runtime in ["sglang", "vllm"] {
                if let Ok(refusal) = do_grpc_health_check(&grpc_url, timeout_secs, runtime).await {
                    return Ok(refusal);
                }
            }
            do_grpc_health_check(&grpc_url, timeout_secs, "atom")
                .await
                .map_err(|e| format!("gRPC failed (tried SGLang, vLLM and ATOM): {}", e))
        }
    }
}

/// Step 1: Detect connection mode by probing HTTP and gRPC.
pub struct DetectConnectionModeStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DetectConnectionModeStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!(
            "Detecting connection mode for {} (timeout: {}s, max_attempts: {})",
            config.url, config.health_check_timeout_secs, config.max_connection_attempts
        );

        // Try both protocols in parallel
        let url = config.url.clone();
        let timeout = config.health_check_timeout_secs;
        let client = &app_context.client;
        let runtime_type = config.runtime.as_deref();

        let (http_result, grpc_result) = tokio::join!(
            try_http_health_check(&url, timeout, client),
            try_grpc_health_check(&url, timeout, runtime_type)
        );

        let connection_mode = match (http_result, grpc_result) {
            (Ok(_), _) => {
                debug!("{} detected as HTTP", config.url);
                ConnectionMode::Http
            }
            (_, Ok(None)) => {
                debug!("{} detected as gRPC", config.url);
                ConnectionMode::Grpc { port: None }
            }
            (_, Ok(Some(refusal))) => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!("{REFUSED} {}: {}", config.url, refusal),
                });
            }
            (Err(http_err), Err(grpc_err)) => {
                return Err(WorkflowError::StepFailed {
                    step_id: StepId::new("detect_connection_mode"),
                    message: format!(
                        "Both HTTP and gRPC health checks failed for {}: HTTP: {}, gRPC: {}",
                        config.url, http_err, grpc_err
                    ),
                });
            }
        };

        context.data.connection_mode = Some(connection_mode);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        !matches!(error, WorkflowError::StepFailed { message, .. } if message.starts_with(REFUSED))
    }
}
