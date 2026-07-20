//! Data Parallel (DP) information discovery step.

use std::time::Duration;

use async_trait::async_trait;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde::Deserialize;
use tracing::{debug, warn};
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use super::discover_metadata::{get_openai_model_id, get_server_info};
use crate::core::{steps::workflow_data::LocalWorkerWorkflowData, UNKNOWN_MODEL_ID};

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

/// Get DP info for a worker URL.
pub async fn get_dp_info(url: &str, api_key: Option<&str>) -> Result<DpInfo, String> {
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

    let model_id = if let Some(model_id) = info
        .and_then(|info| {
            info.model_id
                .filter(|s| !s.is_empty())
                .or(info.served_model_name.filter(|s| !s.is_empty()))
                .or_else(|| {
                    info.model_path
                        .and_then(|path| path.split('/').next_back().map(|s| s.to_string()))
                })
        }) {
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

        let dp_info = get_dp_info(&config.url, config.api_key.as_deref())
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
