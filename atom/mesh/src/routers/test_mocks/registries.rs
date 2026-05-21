use std::sync::Arc;

use crate::config::types::PolicyConfig;
use crate::core::worker::Worker;
use crate::core::WorkerRegistry;
use crate::policies::PolicyRegistry;

pub(crate) fn worker_registry(workers: Vec<Arc<dyn Worker>>) -> Arc<WorkerRegistry> {
    let registry = WorkerRegistry::new();
    for w in workers {
        registry.register(w);
    }
    Arc::new(registry)
}

pub(crate) fn policy_registry() -> Arc<PolicyRegistry> {
    Arc::new(PolicyRegistry::new(PolicyConfig::Random))
}
