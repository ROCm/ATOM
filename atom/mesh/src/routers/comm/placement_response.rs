use axum::response::Response;

use crate::core::placement::types::PlacementError;
use crate::routers::comm::error;

pub fn placement_err_to_response(err: PlacementError, model_id: Option<&str>) -> Response {
    let error = error::IngressError::placement(err, model_id);
    error::service_unavailable(error.code, error.message)
}
