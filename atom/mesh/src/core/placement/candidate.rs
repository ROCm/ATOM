use std::sync::Arc;

use super::traits::WorkerSource;
use crate::core::{ConnectionMode, Worker, WorkerType};

pub fn filter_candidates(
    source: &dyn WorkerSource,
    model_id: Option<&str>,
    worker_type: Option<WorkerType>,
    connection_mode: Option<ConnectionMode>,
) -> Vec<Arc<dyn Worker>> {
    let _ = (source, model_id, worker_type, connection_mode);
    todo!()
}
