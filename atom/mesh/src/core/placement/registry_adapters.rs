use std::sync::Arc;

use super::traits::{PolicySource, WorkerSource};
use crate::core::{ConnectionMode, HashRing, Worker, WorkerRegistry, WorkerType};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};

pub struct WorkerRegistryAdapter {
    registry: Arc<WorkerRegistry>,
    candidates: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkerRegistryAdapter {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        Self {
            registry,
            candidates: None,
        }
    }
    pub fn with_candidates(
        registry: Arc<WorkerRegistry>,
        candidates: Vec<Arc<dyn Worker>>,
    ) -> Self {
        Self {
            registry,
            candidates: Some(candidates),
        }
    }
}

impl WorkerSource for WorkerRegistryAdapter {
    fn workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
    ) -> Vec<Arc<dyn Worker>> {
        let pool = match &self.candidates {
            Some(candidates) => candidates.clone(),
            None => match model_id {
                Some(model) => self.registry.get_by_model(model).iter().cloned().collect(),
                None => self.registry.get_all(),
            },
        };
        pool.into_iter()
            .filter(|worker| {
                model_id.is_none_or(|model| worker.model_id() == model)
                    && worker_type.as_ref().is_none_or(|kind| {
                        std::mem::discriminant(worker.worker_type()) == std::mem::discriminant(kind)
                    })
                    && connection_mode
                        .as_ref()
                        .is_none_or(|mode| worker.connection_mode().matches(mode))
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::placement::test_support::{
        make_prefill_http, make_regular_grpc, make_regular_http,
    };
    #[test]
    fn restricted_candidates_share_model_type_and_transport_filters() {
        let registry = Arc::new(WorkerRegistry::new());
        let workers = vec![
            make_regular_http("http://r", "m"),
            make_prefill_http("http://p", "m", Some(8001)),
            make_regular_grpc("http://g", "m"),
            make_regular_http("http://other", "other"),
        ];
        for worker in &workers {
            registry.register(worker.clone());
        }
        let all = WorkerRegistryAdapter::new(registry.clone());
        let snapshot =
            WorkerRegistryAdapter::with_candidates(registry.clone(), workers[..2].to_vec());
        for source in [&all, &snapshot] {
            assert_eq!(
                source.workers_filtered(
                    Some("m"),
                    Some(WorkerType::Prefill {
                        bootstrap_port: None
                    }),
                    Some(ConnectionMode::Http)
                )[0]
                .url(),
                "http://p"
            );
            assert_eq!(
                source.workers_filtered(
                    Some("m"),
                    Some(WorkerType::Regular),
                    Some(ConnectionMode::Http)
                )[0]
                .url(),
                "http://r"
            );
        }
        assert!(snapshot
            .workers_filtered(Some("other"), None, None)
            .is_empty());
        assert!(snapshot
            .workers_filtered(None, None, Some(ConnectionMode::Grpc { port: None }))
            .is_empty());
        assert!(WorkerRegistryAdapter::with_candidates(registry, vec![])
            .workers_filtered(None, None, None)
            .is_empty());
    }
}
