//! Data Parallel (DP) information discovery step.

use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::{
    create_worker::determine_runtime_type,
    discover_metadata::{get_openai_model_id, get_server_info, ServerInfo},
};
use crate::core::{
    steps::workflow_data::LocalWorkerWorkflowData, worker::RuntimeType, UNKNOWN_MODEL_ID,
};

static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

/// DP (Data Parallel) information for a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DpInfo {
    pub dp_size: usize,
    pub model_id: String,
}

#[derive(Debug, Deserialize)]
struct AtomKvTransferInfo {
    dp_size: Option<usize>,
}

async fn get_atom_dp_size(url: &str, api_key: Option<&str>) -> Result<usize, String> {
    let base_url = url.trim_end_matches('/');
    let endpoint = format!("{}/kv_transfer_info", base_url);
    let mut req = HTTP_CLIENT.get(&endpoint);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", endpoint, e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Server returned status {} from {}",
            response.status(),
            endpoint
        ));
    }

    let info = response
        .json::<AtomKvTransferInfo>()
        .await
        .map_err(|e| format!("Failed to parse response from {}: {}", endpoint, e))?;

    info.dp_size
        .ok_or_else(|| format!("No dp_size in response from {}", endpoint))
}

/// The `/server_info` body that an ATOM gRPC worker's GetServerInfo carries.
#[cfg(feature = "grpc-client")]
async fn get_atom_grpc_server_info(url: &str) -> Result<ServerInfo, String> {
    let grpc_url = format!("grpc://{}", super::strip_protocol(url));
    let client = crate::routers::grpc::engine::atom::AtomClient::connect(&grpc_url)
        .await
        .map_err(|e| format!("Failed to connect to {}: {}", grpc_url, e))?;
    let response = client
        .get_server_info()
        .await
        .map_err(|e| format!("GetServerInfo failed at {}: {}", grpc_url, e))?;
    serde_json::from_str(&response.server_info_json)
        .map_err(|e| format!("Failed to parse GetServerInfo from {}: {}", grpc_url, e))
}

/// The model id `/server_info` names: its model id, served name or model path's last part.
fn server_info_model_id(info: ServerInfo) -> Option<String> {
    info.model_id
        .filter(|s| !s.is_empty())
        .or(info.served_model_name.filter(|s| !s.is_empty()))
        .or_else(|| {
            info.model_path
                .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
        })
}

/// Get DP info for a worker URL; an ATOM gRPC worker's comes from GetServerInfo.
#[cfg_attr(not(feature = "grpc-client"), allow(unused_variables))]
pub async fn get_dp_info(
    url: &str,
    api_key: Option<&str>,
    atom_grpc: bool,
) -> Result<DpInfo, String> {
    #[cfg(feature = "grpc-client")]
    if atom_grpc {
        let info = get_atom_grpc_server_info(url).await?;
        let dp_size = info
            .dp_size
            .ok_or_else(|| format!("No dp_size in GetServerInfo from {}", url))?;
        let model_id = server_info_model_id(info).unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());
        return Ok(DpInfo { dp_size, model_id });
    }

    let info = match get_server_info(url, api_key).await {
        Ok(info) => Some(info),
        Err(e) => {
            debug!(
                "Unable to fetch /server_info for DP discovery from {}; trying ATOM /kv_transfer_info fallback: {}",
                url, e
            );
            None
        }
    };

    let dp_size = if let Some(dp_size) = info.as_ref().and_then(|info| info.dp_size) {
        dp_size
    } else {
        let dp_size = get_atom_dp_size(url, api_key).await.map_err(|e| {
            format!(
                "No dp_size in /server_info response from {} and ATOM /kv_transfer_info fallback failed: {}",
                url, e
            )
        })?;
        warn!(
            "Using ATOM /kv_transfer_info dp_size={} for DP-aware worker discovery at {}",
            dp_size, url
        );
        dp_size
    };

    let model_id = if let Some(model_id) = info.and_then(server_info_model_id) {
        model_id
    } else {
        get_openai_model_id(url, api_key)
            .await
            .unwrap_or_else(|_| UNKNOWN_MODEL_ID.to_string())
    };

    Ok(DpInfo { dp_size, model_id })
}

/// Step 2b: Discover DP (Data Parallel) information (only for DP-aware workers).
pub struct DiscoverDPInfoStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for DiscoverDPInfoStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;

        if !config.dp_aware {
            debug!(
                "Worker {} is not DP-aware, skipping DP discovery",
                config.url
            );
            return Ok(StepResult::Success);
        }

        debug!("Discovering DP info for {} (DP-aware)", config.url);

        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let atom_grpc =
            determine_runtime_type(connection_mode, &context.data, config) == RuntimeType::Atom;
        let dp_info = get_dp_info(&config.url, config.api_key.as_deref(), atom_grpc)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("discover_dp_info"),
                message: format!("Failed to get DP info: {}", e),
            })?;

        debug!(
            "Discovered DP size {} for {} (model: {})",
            dp_info.dp_size, config.url, dp_info.model_id
        );

        context.data.dp_info = Some(dp_info);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true
    }
}
