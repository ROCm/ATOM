//! Managed virtual worker pools for fixture-driven Atomesh tests.
//!
//! A pool starts multiple virtual worker instances on one host with ports
//! allocated from a base port. Each instance still owns its own server task and
//! request log, while the pool provides centralized URL collection and shutdown.

use futures_util::future::join_all;

use super::{ConnectionModeFixture, MockCase, VirtualGrpcWorker, VirtualWorker};

#[derive(Clone, Debug)]
pub struct VirtualWorkerPoolConfig {
    pub host: String,
    pub base_port: u16,
}

impl VirtualWorkerPoolConfig {
    pub fn new(host: impl Into<String>, base_port: u16) -> Self {
        Self {
            host: host.into(),
            base_port,
        }
    }

    fn port_for(&self, index: usize) -> Result<u16, Box<dyn std::error::Error>> {
        let offset = u16::try_from(index)
            .map_err(|_| format!("virtual worker index {} exceeds u16 port range", index))?;
        self.base_port
            .checked_add(offset)
            .ok_or_else(|| format!("virtual worker port overflow at index {}", index).into())
    }
}

#[derive(Clone, Debug)]
pub struct VirtualWorkerSpec {
    pub case: MockCase,
    pub connection_mode: ConnectionModeFixture,
}

impl VirtualWorkerSpec {
    pub fn from_case(case: MockCase) -> Self {
        Self {
            connection_mode: case.route.connection_mode.clone(),
            case,
        }
    }
}

pub enum VirtualWorkerInstance {
    Http(VirtualWorker),
    Grpc(VirtualGrpcWorker),
}

impl VirtualWorkerInstance {
    pub fn url(&self) -> String {
        match self {
            Self::Http(worker) => worker.url.clone().unwrap(),
            Self::Grpc(worker) => worker.url.clone().unwrap(),
        }
    }

    pub fn request_log(&self) -> Vec<String> {
        match self {
            Self::Http(worker) => worker.request_log(),
            Self::Grpc(worker) => worker.request_log(),
        }
    }

    pub async fn stop(&mut self) {
        match self {
            Self::Http(worker) => worker.stop().await,
            Self::Grpc(worker) => worker.stop().await,
        }
    }
}

pub struct VirtualWorkerPool {
    workers: Vec<VirtualWorkerInstance>,
}

impl VirtualWorkerPool {
    pub async fn start(
        config: VirtualWorkerPoolConfig,
        specs: Vec<VirtualWorkerSpec>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut start_tasks = Vec::with_capacity(specs.len());

        for (index, spec) in specs.into_iter().enumerate() {
            let host = config.host.clone();
            let port = config.port_for(index)?;
            start_tasks.push(start_worker_instance(spec, host, port));
        }

        let results = join_all(start_tasks).await;
        let mut workers = Vec::with_capacity(results.len());
        let mut first_error: Option<Box<dyn std::error::Error>> = None;

        for result in results {
            match result {
                Ok(worker) if first_error.is_none() => workers.push(worker),
                Ok(mut worker) => worker.stop().await,
                Err(error) if first_error.is_none() => first_error = Some(error),
                Err(_) => {}
            }
        }

        if let Some(error) = first_error {
            stop_workers(&mut workers).await;
            return Err(error);
        }

        Ok(Self { workers })
    }

    pub fn urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(VirtualWorkerInstance::url)
            .collect()
    }

    pub fn request_logs(&self) -> Vec<String> {
        self.workers
            .iter()
            .flat_map(VirtualWorkerInstance::request_log)
            .collect()
    }

    pub async fn stop(&mut self) {
        stop_workers(&mut self.workers).await;
    }
}

impl Drop for VirtualWorkerPool {
    fn drop(&mut self) {
        // Individual workers also send best-effort shutdown from their Drop
        // impls. Explicit `stop().await` remains preferred because it waits for
        // the server tasks to exit.
    }
}

async fn start_worker_instance(
    spec: VirtualWorkerSpec,
    host: String,
    port: u16,
) -> Result<VirtualWorkerInstance, Box<dyn std::error::Error>> {
    match spec.connection_mode {
        ConnectionModeFixture::Http => {
            let mut worker = VirtualWorker::new(spec.case);
            worker.start_on(&host, port).await?;
            Ok(VirtualWorkerInstance::Http(worker))
        }
        ConnectionModeFixture::Grpc => {
            let mut worker = VirtualGrpcWorker::new(spec.case)?;
            worker.start_on(&host, port).await?;
            Ok(VirtualWorkerInstance::Grpc(worker))
        }
    }
}

async fn stop_workers(workers: &mut [VirtualWorkerInstance]) {
    for worker in workers.iter_mut().rev() {
        worker.stop().await;
    }
}
