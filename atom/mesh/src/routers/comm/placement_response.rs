use axum::response::Response;

use crate::core::{placement::types::PlacementError, UNKNOWN_MODEL_ID};
use crate::routers::comm::error;

pub fn placement_err_to_response(err: PlacementError, model_id: Option<&str>) -> Response {
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
            format!(
                "Load balancing policy returned no worker for model: {}",
                model
            ),
        ),
        PlacementError::ModelNotFound { model_id } => {
            ("model_not_found", format!("Model not found: {}", model_id))
        }
    };
    error::service_unavailable(code, message)
}
