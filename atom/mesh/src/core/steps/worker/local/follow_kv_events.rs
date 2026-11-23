//! With `--kv-events`, the KV event stream of each worker rank is followed from its
//! registration to its removal, at the endpoint the worker reports for the rank at
//! `/kv_event_sources` (ROCm/ATOM#3275), or an ATOM gRPC worker in GetServerInfo, and
//! the rank reads meanwhile the store it reports offloading to.

use std::collections::BTreeMap;

use async_trait::async_trait;
use tracing::warn;
use wfaas::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult};

use crate::{
    app_context::AppContext,
    core::{
        steps::workflow_data::{LocalWorkerWorkflowData, WorkerRemovalWorkflowData},
        worker::RuntimeType,
        ConnectionMode, Worker,
    },
    kv_events::source::{self, KvEventSource},
};

/// Follows every rank just registered, and predicts the prompts routed to every worker. A
/// worker whose sources cannot be read still serves.
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
        for worker in workers {
            monitor.register(worker.url());
        }
        let Some(first) = workers.first() else {
            return Ok(StepResult::Skip);
        };
        let Some(sources) = read_sources(app_context, first.as_ref()).await else {
            warn!(
                worker = first.url(),
                "Not following KV events: a gRPC worker reports its sources only if it is ATOM's"
            );
            return Ok(StepResult::Skip);
        };
        let sources = sources.map_err(|message| {
            warn!(worker = first.base_url(), error = %message, "Cannot read KV event sources");
            WorkflowError::StepFailed {
                step_id: StepId::new("follow_kv_events"),
                message,
            }
        })?;
        // The answer is all the ranks read now: a store it leaves out is no longer theirs.
        if let Some(storage) = &app_context.kv_storage {
            for worker in workers {
                storage.remove(worker.url());
            }
        }
        if sources.is_empty() {
            warn!(
                worker = first.base_url(),
                "The engine has KV events off (ATOM_KV_EVENTS_ENABLE), or its publisher is null"
            );
            return Ok(StepResult::Skip);
        }
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
                    let pool = worker.worker_type().as_metric_label();
                    monitor.attach(worker.url(), worker.model_id(), pool, rank, source.clone());
                    if let (Some(storage), Some(offload)) =
                        (&app_context.kv_storage, &source.offload)
                    {
                        storage.add(worker.url(), worker.model_id(), offload);
                    }
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

/// Every rank's source as `worker` reports them: at `/kv_event_sources`, or in an ATOM
/// gRPC worker's GetServerInfo. None for another gRPC worker, which reports none.
async fn read_sources(
    app_context: &AppContext,
    worker: &dyn Worker,
) -> Option<Result<BTreeMap<u32, KvEventSource>, String>> {
    match worker.connection_mode() {
        ConnectionMode::Http => {
            let api_key = worker.api_key().as_deref();
            Some(source::fetch(&app_context.client, worker.base_url(), api_key).await)
        }
        #[cfg(feature = "grpc-client")]
        ConnectionMode::Grpc { .. } if worker.metadata().runtime_type == RuntimeType::Atom => {
            Some(atom_grpc_sources(worker.base_url()).await)
        }
        ConnectionMode::Grpc { .. } => None,
    }
}

#[cfg(feature = "grpc-client")]
async fn atom_grpc_sources(url: &str) -> Result<BTreeMap<u32, KvEventSource>, String> {
    let (info, grpc_url) = super::discover_dp::atom_grpc_get_server_info(url).await?;
    let sources = serde_json::from_str(&info.kv_event_sources_json)
        .map_err(|e| format!("{grpc_url}: GetServerInfo's kv_event_sources_json: {e}"))?;
    source::parse(sources, &grpc_url).map_err(|e| format!("{grpc_url}: {e}"))
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
        if let Some(storage) = &app_context.kv_storage {
            for url in &context.data.worker_urls {
                storage.remove(url);
            }
        }
        Ok(StepResult::Success)
    }
}
