//! Unified worker registration step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;
use wfaas::{
    StepExecutor, StepId, StepResult, WorkflowContext, WorkflowData, WorkflowError, WorkflowResult,
};

use crate::{
    core::{
        steps::{
            tokenizer_registration::admit_model_template, workflow_data::WorkerRegistrationData,
        },
        worker::RuntimeType,
    },
    observability::metrics::MeshMetrics,
};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
/// Works with any workflow data type that implements `WorkerRegistrationData`.
pub struct RegisterWorkersStep;

#[async_trait]
impl<D: WorkerRegistrationData + WorkflowData> StepExecutor<D> for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext<D>) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .get_app_context()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        let workers = context
            .data
            .get_actual_workers()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;

        // Admission and registration in one step: of two registrations with different
        // templates, the second sees the first's workers.
        let admitting = app_context.template_admission.lock();
        let atom = workers
            .iter()
            .find(|w| w.metadata().runtime_type == RuntimeType::Atom);
        if let Some(atom) = atom {
            let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();
            let advertised = atom.metadata().labels.get("chat_template");
            let (model, file) = (atom.model_id(), atom.chat_template());
            let advertised = advertised.map(String::as_str);
            let admitted = admit_model_template(&app_context, model, &urls, file, advertised);
            admitted.map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("register_workers"),
                message: format!("Refused {}: {e}", atom.url()),
            })?;
        }

        let mut worker_ids = Vec::with_capacity(workers.len());

        for worker in workers.iter() {
            let worker_id = app_context.worker_registry.register(Arc::clone(worker));
            debug!(
                "Registered worker {} (model: {}) with ID {:?}",
                worker.url(),
                worker.model_id(),
                worker_id
            );
            worker_ids.push(worker_id);
        }
        drop(admitting);

        // Collect unique worker configurations to avoid redundant metric updates
        let unique_configs: HashSet<_> = workers
            .iter()
            .map(|w| {
                let meta = w.metadata();
                (
                    meta.worker_type.clone(),
                    meta.connection_mode.clone(),
                    w.model_id().to_string(),
                )
            })
            .collect();

        // Update Layer 3 worker pool size metrics per unique type/connection/model
        for (worker_type, connection_mode, model_id) in unique_configs {
            // Get labels before moving values into get_workers_filtered
            let worker_type_label = worker_type.as_metric_label();
            let connection_mode_label = connection_mode.as_metric_label();

            let pool_size = app_context
                .worker_registry
                .get_workers_filtered(
                    Some(&model_id),
                    Some(worker_type),
                    Some(connection_mode),
                    None,
                    false,
                )
                .len();

            MeshMetrics::set_worker_pool_size(
                worker_type_label,
                connection_mode_label,
                &model_id,
                pool_size,
            );
        }

        // Note: worker_ids are stored for potential future use but not persisted
        // as they are internal registry identifiers
        debug!(
            "Registered {} workers with IDs: {:?}",
            worker_ids.len(),
            worker_ids
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
