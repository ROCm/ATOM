//! Typed workflow engines collection
//!
//! This module provides a collection of typed workflow engines for different workflow types.
//! Each workflow type has its own engine with compile-time type safety.

use std::sync::Arc;

use wfaas::{EventSubscriber, InMemoryStore, WorkflowEngine};

use super::{
    create_local_worker_workflow, create_tokenizer_registration_workflow,
    create_worker_removal_workflow, create_worker_update_workflow, LocalWorkerWorkflowData,
    TokenizerWorkflowData, WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
};
use crate::config::RouterConfig;

/// Type alias for local worker workflow engine
pub type LocalWorkerEngine =
    WorkflowEngine<LocalWorkerWorkflowData, InMemoryStore<LocalWorkerWorkflowData>>;

/// Type alias for worker removal workflow engine
pub type WorkerRemovalEngine =
    WorkflowEngine<WorkerRemovalWorkflowData, InMemoryStore<WorkerRemovalWorkflowData>>;

/// Type alias for worker update workflow engine
pub type WorkerUpdateEngine =
    WorkflowEngine<WorkerUpdateWorkflowData, InMemoryStore<WorkerUpdateWorkflowData>>;

/// Type alias for tokenizer registration workflow engine
pub type TokenizerEngine =
    WorkflowEngine<TokenizerWorkflowData, InMemoryStore<TokenizerWorkflowData>>;

/// Collection of typed workflow engines
///
/// Each workflow type has its own engine with compile-time type safety.
/// This replaces the old `WorkflowEngine<AnyWorkflowData, ...>` approach.
#[derive(Clone, Debug)]
pub struct WorkflowEngines {
    /// Engine for local worker registration workflows
    pub local_worker: Arc<LocalWorkerEngine>,
    /// Engine for worker removal workflows
    pub worker_removal: Arc<WorkerRemovalEngine>,
    /// Engine for worker update workflows
    pub worker_update: Arc<WorkerUpdateEngine>,
    /// Engine for tokenizer registration workflows
    pub tokenizer: Arc<TokenizerEngine>,
}

impl WorkflowEngines {
    /// Create and initialize all workflow engines with their workflow definitions
    pub fn new(router_config: &RouterConfig) -> Self {
        // Create local worker engine
        let local_worker = WorkflowEngine::new();
        local_worker
            .register_workflow(create_local_worker_workflow(router_config))
            .expect("local_worker_registration workflow should be valid");

        // Create worker removal engine
        let worker_removal = WorkflowEngine::new();
        worker_removal
            .register_workflow(create_worker_removal_workflow())
            .expect("worker_removal workflow should be valid");

        // Create worker update engine
        let worker_update = WorkflowEngine::new();
        worker_update
            .register_workflow(create_worker_update_workflow())
            .expect("worker_update workflow should be valid");

        // Create tokenizer engine
        let tokenizer = WorkflowEngine::new();
        tokenizer
            .register_workflow(create_tokenizer_registration_workflow())
            .expect("tokenizer_registration workflow should be valid");

        Self {
            local_worker: Arc::new(local_worker),
            worker_removal: Arc::new(worker_removal),
            worker_update: Arc::new(worker_update),
            tokenizer: Arc::new(tokenizer),
        }
    }

    /// Subscribe an event subscriber to all workflow engines
    pub async fn subscribe_all<S: EventSubscriber + 'static>(&self, subscriber: Arc<S>) {
        self.local_worker
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_removal
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_update
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.tokenizer.event_bus().subscribe(subscriber).await;
    }
}
