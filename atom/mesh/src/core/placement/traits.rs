use std::sync::Arc;

use async_trait::async_trait;

use super::types::{PlacementError, PlacementPlan, RequestDescriptor};
use crate::core::{ConnectionMode, HashRing, Worker, WorkerType};
use crate::policies::LoadBalancingPolicy;

pub trait WorkerSource: Send + Sync {
    /// `worker_type` is matched by variant: `Prefill { bootstrap_port: _ }` returns
    /// any prefill regardless of bootstrap_port. Other variants match exactly.
    /// Adapters wrapping `WorkerRegistry` (which uses strict `PartialEq` on the full
    /// `WorkerType` value) must dispatch Prefill queries through
    /// `WorkerRegistry::get_prefill_workers()` to honor this contract.
    fn workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>>;

    fn hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>>;
}

pub trait PolicySource: Send + Sync {
    fn regular_policy(&self, model_id: Option<&str>) -> Arc<dyn LoadBalancingPolicy>;
    fn prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy>;
    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy>;

    fn pd_needs_request_text(&self) -> bool {
        self.prefill_policy().needs_request_text() || self.decode_policy().needs_request_text()
    }
}

#[async_trait]
pub trait PdPlanner: Send + Sync {
    async fn plan(&self, req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError>;
}
