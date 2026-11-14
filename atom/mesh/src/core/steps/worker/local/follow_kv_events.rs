//! With `--kv-events`, the KV event stream of each worker rank is followed from its
//! registration to its removal, at the endpoint the worker reports for the rank at
//! `/kv_event_sources` (ROCm/ATOM#3275).

use async_trait::async_trait;
use tracing::warn;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    core::{
        steps::workflow_data::{LocalWorkerWorkflowData, WorkerRemovalWorkflowData},
        ConnectionMode,
    },
    kv_events::source,
};

/// Follows every rank just registered. A worker whose sources cannot be read still serves.
pub struct FollowKvEventsStep;

#[async_trait]
impl StepExecutor<LocalWorkerWorkflowData> for FollowKvEventsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<LocalWorkerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let Some(monitor) = &app_context.kv_event_monitor else {
            return Ok(StepResult::Skip);
        };
        let workers = context
            .data
            .actual_workers
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers".to_string()))?;
        let Some(first) = workers
            .first()
            .filter(|worker| matches!(worker.connection_mode(), ConnectionMode::Http))
        else {
            return Ok(StepResult::Skip);
        };
        let api_key = first.api_key().as_deref();
        let sources = source::fetch(&app_context.client, first.base_url(), api_key)
            .await
            .map_err(|message| {
                warn!(worker = first.base_url(), error = %message, "Cannot read KV event sources");
                WorkflowError::StepFailed {
                    step_id: StepId::new("follow_kv_events"),
                    message,
                }
            })?;
        for worker in workers {
            // Each subscription follows one rank: a DP-aware worker's own, or the only
            // rank of a worker that is not.
            let source = match worker.dp_rank().filter(|_| worker.is_dp_aware()) {
                Some(rank) => sources.get_key_value(&(rank as u32)),
                None if sources.len() == 1 => sources.first_key_value(),
                None => {
                    warn!(
                        worker = worker.url(),
                        ranks = sources.len(),
                        "Register the worker DP-aware to follow the KV events of each rank"
                    );
                    continue;
                }
            };
            match source {
                Some((&rank, source)) => {
                    monitor.attach(worker.url(), worker.model_id(), rank, source.clone());
                }
                None => warn!(
                    worker = worker.url(),
                    rank = worker.dp_rank(),
                    "No KV events reported for the rank"
                ),
            }
        }
        Ok(StepResult::Success)
    }
}

pub struct StopFollowingKvEventsStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for StopFollowingKvEventsStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        if let Some(monitor) = &app_context.kv_event_monitor {
            for url in &context.data.worker_urls {
                monitor.detach(url);
            }
        }
        Ok(StepResult::Success)
    }
}
