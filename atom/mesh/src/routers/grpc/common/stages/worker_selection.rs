use std::sync::Arc;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use super::PipelineStage;
use crate::{
    core::{
        placement::{
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor},
        },
        UNKNOWN_MODEL_ID,
    },
    routers::{
        error,
        grpc::context::{RequestContext, WorkerSelection},
    },
};

pub(crate) struct WorkerSelectionStage {
    planner: Arc<dyn PdPlanner>,
}

impl WorkerSelectionStage {
    pub fn new(planner: Arc<dyn PdPlanner>) -> Self {
        Self { planner }
    }
}

#[async_trait]
impl PipelineStage for WorkerSelectionStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let prep = ctx.state.preparation.as_ref().ok_or_else(|| {
            error!(
                function = "WorkerSelectionStage::execute",
                "Preparation stage not completed"
            );
            error::internal_error(
                "preparation_stage_not_completed",
                "Preparation stage not completed",
            )
        })?;

        let tokens = if prep.token_ids.is_empty() {
            None
        } else {
            Some(prep.token_ids.as_slice())
        };

        let descriptor = RequestDescriptor {
            model_id: ctx.input.model_id.as_deref(),
            protocol: Some(Protocol::Grpc),
            text: prep.original_text.as_deref(),
            tokens,
            headers: ctx.input.headers.as_ref(),
            stream: ctx.is_streaming(),
            return_logprob: false,
        };

        match self.planner.plan(&descriptor).await {
            Ok(plan) => {
                ctx.state.workers = Some(plan_to_worker_selection(plan));
                Ok(None)
            }
            Err(err) => {
                let model = ctx.input.model_id.as_deref().unwrap_or(UNKNOWN_MODEL_ID);
                error!(
                    function = "WorkerSelectionStage::execute",
                    model_id = %model,
                    error = %err,
                    "Placement planner returned error"
                );
                Err(placement_err_to_response(err, ctx.input.model_id.as_deref()))
            }
        }
    }

    fn name(&self) -> &'static str {
        "WorkerSelection"
    }
}

pub(crate) fn plan_to_worker_selection(plan: PlacementPlan) -> WorkerSelection {
    match plan {
        PlacementPlan::Single { worker, .. } => WorkerSelection::Single { worker },
        PlacementPlan::Pair {
            prefill, decode, ..
        } => WorkerSelection::Dual { prefill, decode },
    }
}

pub(crate) fn placement_err_to_response(err: PlacementError, model_id: Option<&str>) -> Response {
    let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
    let (code, message) = match &err {
        PlacementError::NoWorkers => (
            "no_workers",
            format!("No workers in registry (model: {})", model),
        ),
        PlacementError::NoAvailableWorkers => (
            "no_available_workers",
            format!("No available workers for model: {}", model),
        ),
        PlacementError::NoPrefillWorkers => (
            "no_prefill_workers",
            format!("No available prefill workers for model: {}", model),
        ),
        PlacementError::NoDecodeWorkers => (
            "no_decode_workers",
            format!("No available decode workers for model: {}", model),
        ),
        PlacementError::PolicyReturnedNone => (
            "policy_returned_none",
            format!("Load balancing policy returned no worker for model: {}", model),
        ),
        PlacementError::ModelNotFound { model_id } => (
            "model_not_found",
            format!("Model not found: {}", model_id),
        ),
    };
    error::service_unavailable(code, message)
}
