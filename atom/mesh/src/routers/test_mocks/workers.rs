use std::sync::Arc;

use crate::core::worker::{ConnectionMode, Worker, WorkerType};
use crate::core::worker_builder::BasicWorkerBuilder;

pub(crate) struct MockWorkerConfig {
    pub url: String,
    pub worker_type: WorkerType,
    pub connection_mode: ConnectionMode,
    pub api_key: Option<String>,
    pub healthy: bool,
}

impl Default for MockWorkerConfig {
    fn default() -> Self {
        Self {
            url: "http://test-worker:8000".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Grpc { port: Some(50051) },
            api_key: None,
            healthy: true,
        }
    }
}

pub(crate) fn mock_grpc_worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
    let cfg = MockWorkerConfig {
        url: url.to_string(),
        worker_type,
        connection_mode: ConnectionMode::Grpc { port: Some(50051) },
        ..Default::default()
    };
    build_worker(cfg)
}

pub(crate) fn mock_http_only_worker(url: &str) -> Arc<dyn Worker> {
    let cfg = MockWorkerConfig {
        url: url.to_string(),
        connection_mode: ConnectionMode::Http,
        ..Default::default()
    };
    build_worker(cfg)
}

fn build_worker(cfg: MockWorkerConfig) -> Arc<dyn Worker> {
    let mut builder = BasicWorkerBuilder::new(cfg.url)
        .worker_type(cfg.worker_type)
        .connection_mode(cfg.connection_mode);
    if let Some(key) = cfg.api_key {
        builder = builder.api_key(key);
    }
    let worker = builder.build();
    worker.set_healthy(cfg.healthy);
    Arc::new(worker)
}
