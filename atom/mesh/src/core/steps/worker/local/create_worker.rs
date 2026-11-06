//! Local worker creation step.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::debug;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    core::{
        circuit_breaker::CircuitBreakerConfig,
        steps::workflow_data::LocalWorkerWorkflowData,
        worker::{HealthConfig, RuntimeType, WorkerType},
        BasicWorkerBuilder, ConnectionMode, DPAwareWorkerBuilder, Worker, UNKNOWN_MODEL_ID,
    },
    protocols::worker_spec::WorkerConfigRequest,
};

/// Step 3: Create worker object(s) with merged configuration + metadata.
///
/// This step:
/// 1. Merges discovered labels with config labels
/// 2. Determines the model ID from various sources
/// 3. Builds worker(s) - either single worker or multiple DP-aware workers
/// 4. Outputs unified `workers: Vec<Arc<dyn Worker>>` for downstream steps
pub struct CreateLocalWorkerStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for CreateLocalWorkerStep {
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
        let connection_mode =
            context.data.connection_mode.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("connection_mode".to_string())
            })?;
        let discovered_labels = &context.data.discovered_labels;

        // Check if worker already exists
        if app_context
            .worker_registry
            .get_by_url(&config.url)
            .is_some()
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: format!("Worker {} already exists", config.url),
            });
        }

        // Build labels from config
        let mut config_labels = config.labels.clone();
        if let Some(priority) = config.priority {
            config_labels.insert("priority".to_string(), priority.to_string());
        }
        if let Some(cost) = config.cost {
            config_labels.insert("cost".to_string(), cost.to_string());
        }

        // Merge: discovered labels first, then config labels (config takes precedence)
        let mut final_labels = discovered_labels.clone();
        for (key, value) in &config_labels {
            final_labels.insert(key.clone(), value.clone());
        }

        // Determine model_id: config > served_model_name > model_path > UNKNOWN_MODEL_ID
        let model_id = config
            .model_id
            .clone()
            .or_else(|| final_labels.get("served_model_name").cloned())
            .or_else(|| final_labels.get("model_path").cloned())
            .unwrap_or_else(|| UNKNOWN_MODEL_ID.to_string());

        if model_id != UNKNOWN_MODEL_ID {
            debug!("Using model_id: {}", model_id);
        }

        debug!(
            "Creating worker {} with {} discovered + {} config = {} final labels",
            config.url,
            discovered_labels.len(),
            config_labels.len(),
            final_labels.len()
        );

        // Parse worker type
        let worker_type = parse_worker_type(config);

        // Get runtime type (for gRPC workers)
        let runtime_type = determine_runtime_type(connection_mode, &context.data, config);

        // Build circuit breaker config
        let circuit_breaker_config = build_circuit_breaker_config(app_context);

        // Build health config
        let health_config = build_health_config(app_context, config);

        // Normalize URL
        let normalized_url = normalize_url(&config.url, connection_mode);

        if normalized_url != config.url {
            debug!(
                "Normalized worker URL: {} -> {} ({:?})",
                config.url, normalized_url, connection_mode
            );
        }

        if app_context.router_config.atom_pd_prefill_backend_routing
            && matches!(worker_type, WorkerType::Prefill { .. })
            && !config.dp_aware
        {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("create_worker"),
                message: "Backend-routed P services require DP discovery".into(),
            });
        }
        // Create workers - always output as Vec for unified downstream handling
        let workers = if config.dp_aware {
            create_dp_aware_workers(
                &context.data,
                &normalized_url,
                &model_id,
                worker_type,
                connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                config,
                &final_labels,
            )?
        } else {
            create_single_worker(
                &normalized_url,
                &model_id,
                worker_type,
                connection_mode,
                runtime_type,
                circuit_breaker_config,
                health_config,
                config,
                &final_labels,
            )
        };

        // Update workflow data
        context.data.actual_workers = Some(workers);
        context.data.final_labels = final_labels;
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

fn parse_worker_type(config: &WorkerConfigRequest) -> WorkerType {
    config
        .worker_type
        .as_ref()
        .map(|t| match t.as_str() {
            "prefill" => WorkerType::Prefill {
                bootstrap_port: config.bootstrap_port,
            },
            "decode" => WorkerType::Decode,
            _ => WorkerType::Regular,
        })
        .unwrap_or(WorkerType::Regular)
}

fn determine_runtime_type(
    connection_mode: &ConnectionMode,
    data: &LocalWorkerWorkflowData,
    config: &WorkerConfigRequest,
) -> RuntimeType {
    if !matches!(connection_mode, ConnectionMode::Grpc { .. }) {
        return RuntimeType::Sglang;
    }

    if let Some(ref detected_runtime) = data.detected_runtime_type {
        match detected_runtime.as_str() {
            "vllm" => RuntimeType::Vllm,
            _ => RuntimeType::Sglang,
        }
    } else if let Some(ref runtime) = config.runtime {
        match runtime.as_str() {
            "vllm" => RuntimeType::Vllm,
            _ => RuntimeType::Sglang,
        }
    } else {
        RuntimeType::Sglang
    }
}

fn build_circuit_breaker_config(app_context: &AppContext) -> CircuitBreakerConfig {
    let cfg = app_context.router_config.effective_circuit_breaker_config();
    CircuitBreakerConfig {
        failure_threshold: cfg.failure_threshold,
        success_threshold: cfg.success_threshold,
        timeout_duration: Duration::from_secs(cfg.timeout_duration_secs),
        window_duration: Duration::from_secs(cfg.window_duration_secs),
    }
}

fn build_health_config(app_context: &AppContext, config: &WorkerConfigRequest) -> HealthConfig {
    let cfg = &app_context.router_config.health_check;
    HealthConfig {
        timeout_secs: cfg.timeout_secs,
        check_interval_secs: cfg.check_interval_secs,
        endpoint: cfg.endpoint.clone(),
        failure_threshold: cfg.failure_threshold,
        success_threshold: cfg.success_threshold,
        disable_health_check: cfg.disable_health_check || config.disable_health_check,
    }
}

fn normalize_url(url: &str, connection_mode: &ConnectionMode) -> String {
    if url.starts_with("http://") || url.starts_with("https://") || url.starts_with("grpc://") {
        url.to_string()
    } else {
        match connection_mode {
            ConnectionMode::Http => format!("http://{}", url),
            ConnectionMode::Grpc { .. } => format!("grpc://{}", url),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn create_dp_aware_workers(
    data: &LocalWorkerWorkflowData,
    normalized_url: &str,
    model_id: &str,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthConfig,
    config: &WorkerConfigRequest,
    final_labels: &HashMap<String, String>,
) -> Result<Vec<Arc<dyn Worker>>, WorkflowError> {
    let dp_info = data
        .dp_info
        .as_ref()
        .ok_or_else(|| WorkflowError::ContextValueNotFound("dp_info".to_string()))?;

    let backend_routing = matches!(worker_type, WorkerType::Prefill { .. })
        && data
            .app_context
            .as_ref()
            .is_some_and(|ctx| ctx.router_config.atom_pd_prefill_backend_routing);
    let count = if backend_routing { 1 } else { dp_info.dp_size };
    if dp_info.dp_size == 0 {
        return Err(WorkflowError::ContextValueNotFound(
            "dp_size must be positive".into(),
        ));
    }
    debug!(
        "Creating {} workers for {} (dp_size: {}, backend_routing: {})",
        count, normalized_url, dp_info.dp_size, backend_routing
    );
    let mut workers = Vec::with_capacity(count);
    for rank in 0..count {
        let mut builder =
            DPAwareWorkerBuilder::new(normalized_url.to_string(), rank, dp_info.dp_size)
                .backend_rank_selection(backend_routing)
                .model_id(model_id)
                .worker_type(worker_type.clone())
                .connection_mode(connection_mode.clone())
                .runtime_type(runtime_type.clone())
                .circuit_breaker_config(circuit_breaker_config.clone())
                .health_config(health_config.clone());

        if let Some(ref api_key) = config.api_key {
            builder = builder.api_key(api_key.clone());
        }
        if let Some(ref tokenizer_path) = config.tokenizer_path {
            builder = builder.tokenizer_path(tokenizer_path.clone());
        }
        if let Some(ref reasoning_parser) = config.reasoning_parser {
            builder = builder.reasoning_parser(reasoning_parser.clone());
        }
        if let Some(ref tool_parser) = config.tool_parser {
            builder = builder.tool_parser(tool_parser.clone());
        }
        if let Some(ref chat_template) = config.chat_template {
            builder = builder.chat_template(chat_template.clone());
        }
        if !final_labels.is_empty() {
            builder = builder.labels(final_labels.clone());
        }

        let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
        if health_config.disable_health_check {
            worker.set_healthy(true);
        } else {
            worker.set_healthy(false);
        }
        debug!(
            "Created DP-aware worker {} (dp_size: {}, {:?})",
            worker.url(),
            dp_info.dp_size,
            connection_mode
        );
        workers.push(worker);
    }

    Ok(workers)
}

#[allow(clippy::too_many_arguments)]
fn create_single_worker(
    normalized_url: &str,
    model_id: &str,
    worker_type: WorkerType,
    connection_mode: &ConnectionMode,
    runtime_type: RuntimeType,
    circuit_breaker_config: CircuitBreakerConfig,
    health_config: HealthConfig,
    config: &WorkerConfigRequest,
    final_labels: &HashMap<String, String>,
) -> Vec<Arc<dyn Worker>> {
    let health_check_disabled = health_config.disable_health_check;

    let mut builder = BasicWorkerBuilder::new(normalized_url.to_string())
        .model_id(model_id)
        .worker_type(worker_type)
        .connection_mode(connection_mode.clone())
        .runtime_type(runtime_type)
        .circuit_breaker_config(circuit_breaker_config)
        .health_config(health_config);

    if let Some(ref api_key) = config.api_key {
        builder = builder.api_key(api_key.clone());
    }
    if let Some(ref tokenizer_path) = config.tokenizer_path {
        builder = builder.tokenizer_path(tokenizer_path.clone());
    }
    if let Some(ref reasoning_parser) = config.reasoning_parser {
        builder = builder.reasoning_parser(reasoning_parser.clone());
    }
    if let Some(ref tool_parser) = config.tool_parser {
        builder = builder.tool_parser(tool_parser.clone());
    }
    if let Some(ref chat_template) = config.chat_template {
        builder = builder.chat_template(chat_template.clone());
    }
    if !final_labels.is_empty() {
        builder = builder.labels(final_labels.clone());
    }

    let worker = Arc::new(builder.build()) as Arc<dyn Worker>;
    if health_check_disabled {
        worker.set_healthy(true);
    } else {
        worker.set_healthy(false);
    }

    debug!(
        "Created worker object for {} ({:?}) with {} labels",
        normalized_url,
        connection_mode,
        final_labels.len()
    );

    vec![worker]
}

#[cfg(test)]
mod backend_routing_tests {
    use super::*;
    use crate::config::{BackendType, RouterConfig, RoutingMode};
    use crate::core::steps::worker::local::{create_local_worker_workflow_data, DpInfo};

    #[tokio::test]
    async fn backend_prefill_discovery_registers_one_service_but_all_decode_ranks() {
        let mut router = RouterConfig::default();
        router.backend = BackendType::Atom;
        router.dp_aware = true;
        router.atom_pd_prefill_backend_routing = true;
        router.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
        let app = Arc::new(AppContext::from_config(router, 5).await.unwrap());
        let config: WorkerConfigRequest =
            serde_json::from_value(serde_json::json!({"url":"http://p:80","dp_aware":true}))
                .unwrap();
        let mut data = create_local_worker_workflow_data(config.clone(), app.clone());
        data.dp_info = Some(DpInfo {
            dp_size: 4,
            model_id: "model".into(),
        });
        for role in [
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            WorkerType::Decode,
        ] {
            let workers = create_dp_aware_workers(
                &data,
                "http://p:80",
                "model",
                role.clone(),
                &ConnectionMode::Http,
                RuntimeType::Sglang,
                build_circuit_breaker_config(&app),
                HealthConfig::default(),
                &config,
                &HashMap::new(),
            )
            .unwrap();
            if matches!(role, WorkerType::Prefill { .. }) {
                assert_eq!(workers.len(), 1);
                assert_eq!(workers[0].url(), "http://p:80");
                assert_eq!(workers[0].dp_rank(), None);
                assert_eq!(workers[0].dp_size(), Some(4));
            } else {
                assert_eq!(workers.len(), 4);
                for (rank, w) in workers.iter().enumerate() {
                    assert_eq!(w.dp_rank(), Some(rank));
                }
            }
        }
    }
}
