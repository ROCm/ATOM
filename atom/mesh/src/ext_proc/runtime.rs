use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::{
    net::TcpListener,
    sync::{mpsc, watch},
    task::JoinHandle,
};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{
    transport::{Certificate, Identity, Server, ServerTlsConfig},
    Request, Response, Status,
};

use crate::{
    app_context::AppContext,
    core::{ConnectionMode, WorkerType},
};

use super::{
    executor::PdExecutor,
    pb,
    proto::grpc::health::v1::{
        self as health,
        health_server::{Health, HealthServer},
    },
    service::ExtProcService,
};

type RuntimeError = Box<dyn std::error::Error + Send + Sync>;

/// Owns the listener, readiness and bounded shutdown of the ext-proc service.
pub struct ExtProcRuntime {
    pub address: SocketAddr,
    stop: watch::Sender<bool>,
    draining: Arc<AtomicBool>,
    force: watch::Sender<bool>,
    task: JoinHandle<Result<(), RuntimeError>>,
}

impl ExtProcRuntime {
    pub async fn start(app: Arc<AppContext>) -> Result<Self, RuntimeError> {
        let config = app.router_config.ext_proc.clone();
        config.validate(&app.router_config)?;
        let listener = TcpListener::bind(config.listen).await?;
        let address = listener.local_addr()?;
        let (stop, mut stopped) = watch::channel(false);
        let (force, forced) = watch::channel(false);
        let executor = if app.router_config.mode.is_pd_mode() {
            Some(PdExecutor::bind(&app).await?)
        } else {
            None
        };
        let draining = Arc::new(AtomicBool::new(false));
        let processor =
            pb::external_processor_server::ExternalProcessorServer::new(ExtProcService::new(
                app.clone(),
                draining.clone(),
                forced.clone(),
                executor.as_ref().map(|(executor, _)| executor.clone()),
            ))
            .max_decoding_message_size(config.max_message_bytes)
            .max_encoding_message_size(config.max_message_bytes);
        let health = HealthServer::new(HealthService {
            app,
            draining: draining.clone(),
        });
        let mut server = Server::builder();
        if let (Some(cert), Some(key)) = (&config.tls_cert, &config.tls_key) {
            let _ = rustls::crypto::ring::default_provider().install_default();
            let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(
                tokio::fs::read(cert).await?,
                tokio::fs::read(key).await?,
            ));
            if let Some(ca) = &config.client_ca {
                tls = tls.client_ca_root(Certificate::from_pem(tokio::fs::read(ca).await?));
            }
            server = server.tls_config(tls)?;
        }
        let executor_stop = stopped.clone();
        let mut deadline_stop = stopped.clone();
        let force_task = force.clone();
        let task = tokio::spawn(async move {
            let grpc = server
                .add_service(processor)
                .add_service(health)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    let _ = stopped.wait_for(|v| *v).await;
                });
            let executor = async move {
                if let Some((executor, listener)) = executor {
                    let mut forced = forced;
                    tokio::select! {
                        result = executor.serve(listener, executor_stop) => result?,
                        _ = async { let _ = forced.wait_for(|v| *v).await; } => {},
                    }
                }
                Ok::<(), RuntimeError>(())
            };
            let services = async {
                tokio::try_join!(
                    async { grpc.await.map_err(|e| Box::new(e) as RuntimeError) },
                    executor
                )?;
                Ok(())
            };
            tokio::pin!(services);
            tokio::select! {
                result = &mut services => result,
                _ = async {
                    let _ = deadline_stop.wait_for(|v| *v).await;
                    tokio::time::sleep(Duration::from_secs(config.drain_timeout_secs)).await;
                } => {
                    let _ = force_task.send(true);
                    // Give session guards a chance to release before stopping transport.
                    match tokio::time::timeout(Duration::from_secs(1), &mut services).await {
                        Ok(result) => result,
                        Err(_) => Ok(()),
                    }
                }
            }
        });
        tracing::info!(%address, "ext-proc listener started");
        Ok(Self {
            address,
            stop,
            draining,
            force,
            task,
        })
    }

    pub fn shutdown_handle(&self) -> impl FnOnce() + Send + 'static {
        let stop = self.stop.clone();
        let draining = self.draining.clone();
        move || {
            draining.store(true, Ordering::Release);
            let _ = stop.send(true);
        }
    }

    pub async fn shutdown(mut self) -> Result<(), RuntimeError> {
        (self.shutdown_handle())();
        self.wait().await
    }

    /// Wait for shutdown or a listener failure. Safe to cancel in `select!`.
    pub async fn wait(&mut self) -> Result<(), RuntimeError> {
        (&mut self.task).await??;
        if !self.draining.load(Ordering::Acquire) {
            return Err(std::io::Error::other("ext-proc listener exited unexpectedly").into());
        }
        Ok(())
    }
}

impl Drop for ExtProcRuntime {
    fn drop(&mut self) {
        self.draining.store(true, Ordering::Release);
        let _ = self.stop.send(true);
        let _ = self.force.send(true);
        self.task.abort();
    }
}

struct HealthService {
    app: Arc<AppContext>,
    draining: Arc<AtomicBool>,
}

impl HealthService {
    fn status(&self, name: &str) -> Result<health::HealthCheckResponse, Status> {
        if !matches!(name, "" | "envoy.service.ext_proc.v3.ExternalProcessor") {
            return Err(Status::not_found("unknown service"));
        }
        let eligible: Vec<_> = self
            .app
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|w| {
                w.is_available()
                    && matches!(w.connection_mode(), ConnectionMode::Http)
                    && (!super::request::RequestEnvelope::needs_tokens(&self.app, w.model_id())
                        || self.app.tokenizer_registry.get(w.model_id()).is_some())
            })
            .collect();
        let available = !self.draining.load(Ordering::Acquire)
            && if self.app.router_config.mode.is_pd_mode() {
                eligible.iter().any(|p| {
                    matches!(p.worker_type(), WorkerType::Prefill { .. })
                        && eligible.iter().any(|d| {
                            matches!(d.worker_type(), WorkerType::Decode)
                                && p.model_id() == d.model_id()
                        })
                })
            } else {
                eligible
                    .iter()
                    .any(|w| matches!(w.worker_type(), WorkerType::Regular))
            };
        Ok(health::HealthCheckResponse {
            status: if available { 1 } else { 2 },
        })
    }
}

#[tonic::async_trait]
impl Health for HealthService {
    async fn check(
        &self,
        request: Request<health::HealthCheckRequest>,
    ) -> Result<Response<health::HealthCheckResponse>, Status> {
        Ok(Response::new(self.status(&request.into_inner().service)?))
    }
    type WatchStream = ReceiverStream<Result<health::HealthCheckResponse, Status>>;
    async fn watch(
        &self,
        request: Request<health::HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let name = request.into_inner().service;
        let service = Self {
            app: self.app.clone(),
            draining: self.draining.clone(),
        };
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let mut previous = None;
            loop {
                let status = service
                    .status(&name)
                    .unwrap_or(health::HealthCheckResponse { status: 3 });
                if previous != Some(status.status) {
                    previous = Some(status.status);
                    if tx.send(Ok(status)).await.is_err() {
                        break;
                    }
                }
                if service.draining.load(Ordering::Acquire) && previous == Some(2) {
                    break;
                }
                tokio::select! { _ = tx.closed() => break, _ = tokio::time::sleep(Duration::from_millis(250)) => {} }
            }
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
