use std::sync::Arc;

use super::traits::{PolicySource, WorkerSource};
use crate::core::{ConnectionMode, HashRing, Worker, WorkerRegistry, WorkerType};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};

pub struct WorkerRegistryAdapter {
    registry: Arc<WorkerRegistry>,
}

impl WorkerRegistryAdapter {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        Self { registry }
    }
}

impl WorkerSource for WorkerRegistryAdapter {
    fn workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>> {
        match worker_type {
            Some(WorkerType::Prefill { .. }) => {
                let pool: Vec<Arc<dyn Worker>> = match model_id {
                    Some(m) => self.registry.get_by_model(m).iter().cloned().collect(),
                    None => self.registry.get_prefill_workers(),
                };
                pool.into_iter()
                    .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                    .filter(|w| match &connection_mode {
                        Some(cm) => w.connection_mode().matches(cm),
                        None => true,
                    })
                    .collect()
            }
            other => {
                self.registry
                    .get_workers_filtered(model_id, other, connection_mode, None, false)
            }
        }
    }

    fn hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.registry.get_hash_ring(model_id)
    }
}

pub struct PolicyRegistryAdapter {
    registry: Arc<PolicyRegistry>,
}

impl PolicyRegistryAdapter {
    pub fn new(registry: Arc<PolicyRegistry>) -> Self {
        Self { registry }
    }
}

impl PolicySource for PolicyRegistryAdapter {
    fn regular_policy(&self, model_id: Option<&str>) -> Arc<dyn LoadBalancingPolicy> {
        match model_id {
            Some(m) => self.registry.get_policy_or_default(m),
            None => self.registry.get_default_policy(),
        }
    }

    fn prefill_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.registry.get_prefill_policy()
    }

    fn decode_policy(&self) -> Arc<dyn LoadBalancingPolicy> {
        self.registry.get_decode_policy()
    }
}
