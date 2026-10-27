use super::error::ProcessingError;
pub(super) use crate::core::admission::AdmissionLease;
use crate::{app_context::AppContext, core::admission::AdmissionController};
use std::sync::Arc;

pub(super) struct Admission(Arc<AdmissionController>);
impl Admission {
    pub fn new(app: &AppContext) -> Self {
        Self(app.admission.clone())
    }
    pub async fn acquire(&self) -> Result<AdmissionLease, ProcessingError> {
        self.0
            .acquire("ext_proc")
            .await
            .map_err(|error| ProcessingError::new(error.status, error.code, error.message))
    }
}
