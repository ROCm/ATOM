use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use futures_util::StreamExt;
use tokio::{
    net::TcpListener,
    sync::{watch, Semaphore},
    task::JoinHandle,
};
use tokio_stream::wrappers::{TcpListenerStream, WatchStream};
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
    completed: Option<Result<(), String>>,
}

impl ExtProcRuntime {
    pub async fn start(app: Arc<AppContext>) -> Result<Self, RuntimeError> {
        let config = app.router_config.ext_proc.clone();
        config.validate(&app.router_config)?;
        let listener = TcpListener::bind(config.listen).await.map_err(|error| {
            std::io::Error::new(
                error.kind(),
                format!("ext-proc listener {}: {error}", config.listen),
            )
        })?;
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
        let (health_tx, health_status) =
            watch::channel(HealthService::current_status(&app, &draining));
        let health = HealthServer::new(HealthService {
            app: app.clone(),
            draining: draining.clone(),
            status: health_status,
            // A separate bounded pool keeps health subscribers from occupying inference slots.
            watchers: Arc::new(Semaphore::new(config.max_streams)),
        });
        let mut health_stop = stopped.clone();
        let health_draining = draining.clone();
        let health_updates = async move {
            let mut tick = tokio::time::interval(Duration::from_millis(250));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = health_stop.wait_for(|v| *v) => break,
                    _ = tick.tick() => {
                        let status = HealthService::current_status(&app, &health_draining);
                        health_tx.send_if_modified(|previous| {
                            if *previous == status { false } else { *previous = status; true }
                        });
                    }
                }
            }
            health_tx.send_replace(2);
            // Closing the publisher also terminates unknown-service subscriptions.
        };
        let mut server = Server::builder();
        if let (Some(cert), Some(key)) = (&config.tls_cert, &config.tls_key) {
            let material = crate::core::tls::TlsMaterial::load(cert, key).await?;
            let mut tls = ServerTlsConfig::new().identity(Identity::from_pem(
                material.certificate,
                material.private_key,
            ));
            if let Some(ca) = &config.client_ca {
                tls = tls.client_ca_root(Certificate::from_pem(
                    crate::core::tls::certificate(ca).await?,
                ));
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
                    executor,
                    async {
                        health_updates.await;
                        Ok::<(), RuntimeError>(())
                    }
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
            completed: None,
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
        if self.completed.is_none() {
            // Borrow the handle while pending: canceling this await must not lose it.
            let result = match (&mut self.task).await {
                Ok(Ok(())) if self.draining.load(Ordering::Acquire) => Ok(()),
                Ok(Ok(())) => Err("ext-proc listener exited unexpectedly".to_owned()),
                Ok(Err(error)) => Err(error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            self.completed = Some(result);
        }
        self.completed
            .as_ref()
            .unwrap()
            .clone()
            .map_err(|error| std::io::Error::other(error).into())
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
    status: watch::Receiver<i32>,
    watchers: Arc<Semaphore>,
}

impl HealthService {
    fn status(&self, name: &str) -> Result<health::HealthCheckResponse, Status> {
        if !matches!(name, "" | "envoy.service.ext_proc.v3.ExternalProcessor") {
            return Err(Status::not_found("unknown service"));
        }
        Ok(health::HealthCheckResponse {
            status: Self::current_status(&self.app, &self.draining),
        })
    }

    fn current_status(app: &AppContext, draining: &AtomicBool) -> i32 {
        let eligible: Vec<_> = app
            .worker_registry
            .get_all()
            .into_iter()
            .filter(|w| {
                w.is_available()
                    && matches!(w.connection_mode(), ConnectionMode::Http)
                    && (!super::request::RequestEnvelope::needs_tokens(app, Some(w.model_id()))
                        || app.tokenizer_registry.get(w.model_id()).is_some())
            })
            .collect();
        let available = !draining.load(Ordering::Acquire)
            && if app.router_config.mode.is_pd_mode() {
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
        if available {
            1
        } else {
            2
        }
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
    type WatchStream = Pin<
        Box<dyn futures_util::Stream<Item = Result<health::HealthCheckResponse, Status>> + Send>,
    >;
    async fn watch(
        &self,
        request: Request<health::HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let permit = self
            .watchers
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("health watch limit exceeded"))?;
        let unknown = !matches!(
            request.into_inner().service.as_str(),
            "" | "envoy.service.ext_proc.v3.ExternalProcessor"
        );
        let updates = WatchStream::new(self.status.clone());
        let stream = futures_util::stream::unfold(
            (updates, permit, None),
            move |(mut updates, permit, mut previous)| async move {
                while let Some(status) = updates.next().await {
                    let status = if unknown { 3 } else { status };
                    if previous == Some(status) {
                        continue;
                    }
                    previous = Some(status);
                    return Some((
                        Ok(health::HealthCheckResponse { status }),
                        (updates, permit, previous),
                    ));
                }
                None
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}
