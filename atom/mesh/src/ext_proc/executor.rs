use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    response::{IntoResponse, Response},
    Router,
};
use dashmap::DashMap;
use futures_util::StreamExt;
use http::StatusCode;
use tokio::{net::TcpListener, sync::watch, time::Instant};

use crate::{
    app_context::AppContext,
    core::placement::types::PlacementPlan,
    routers::{
        http_pd_router::{PDRouter, ReservedPair},
        prepare::inference::InferenceMetadata,
    },
};

use super::{error::ProcessingError, request::RequestEnvelope};

/// Executes an already selected PD pair with the existing backend adapters.
pub(super) struct PdExecutor {
    router: PDRouter,
    policies: Arc<crate::policies::PolicyRegistry>,
    pending: Arc<DashMap<String, Execution>>,
    pub address: SocketAddr,
    max_body: usize,
    reservation_timeout: Duration,
}

#[derive(Clone)]
struct Execution {
    placement: Arc<ReservedPair>,
    metadata: InferenceMetadata,
    body_hash: blake3::Hash,
    path: String,
    canceled: watch::Receiver<bool>,
    accepted: watch::Sender<bool>,
    receiving: Arc<AtomicBool>,
    deadline: Instant,
}

pub(super) struct ExecutionLease {
    pub id: String,
    pending: Arc<DashMap<String, Execution>>,
    cancel: watch::Sender<bool>,
    _placement: Arc<ReservedPair>,
    accepted: watch::Receiver<bool>,
    deadline: Instant,
    policies: Vec<(String, Arc<dyn crate::policies::LoadBalancingPolicy>)>,
}

impl ExecutionLease {
    pub fn expiration(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        let mut accepted = self.accepted.clone();
        let deadline = self.deadline;
        let pending = self.pending.clone();
        let id = self.id.clone();
        Box::pin(async move {
            tokio::select! {
                _ = accepted.wait_for(|accepted| *accepted) => {},
                _ = tokio::time::sleep_until(deadline) => {
                    // Removing the reservation and accepting it use the same map
                    // operation. Exactly one side can win at the deadline.
                    if pending.remove(&id).is_some() { return; }
                },
            }
            std::future::pending::<()>().await;
        })
    }
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        self.pending.remove(&self.id);
        let _ = self.cancel.send(true);
        self._placement.cancel();
        if !*self.accepted.borrow() {
            for (url, policy) in &self.policies {
                policy.on_request_complete(url, false);
            }
        }
    }
}

impl PdExecutor {
    pub const HEADER: &'static str = "x-mesh-execution-id";

    pub async fn bind(
        app: &Arc<AppContext>,
    ) -> Result<(Arc<Self>, TcpListener), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(app.router_config.ext_proc.executor_listen).await?;
        let address = app
            .router_config
            .ext_proc
            .executor_advertise
            .unwrap_or(listener.local_addr()?);
        let mut router = PDRouter::new(app).await.map_err(std::io::Error::other)?;
        router.retry_config.max_retries = 1;
        Ok((
            Arc::new(Self {
                router,
                policies: app.policy_registry.clone(),
                pending: Arc::new(DashMap::new()),
                address,
                max_body: app.router_config.ext_proc.max_body_bytes,
                reservation_timeout: Duration::from_secs(
                    app.router_config.ext_proc.reservation_timeout_secs,
                ),
            }),
            listener,
        ))
    }

    pub fn reserve(
        &self,
        plan: PlacementPlan,
        request: &RequestEnvelope,
        metadata: &InferenceMetadata,
    ) -> Result<ExecutionLease, ProcessingError> {
        let placement = self
            .router
            .reserve_pair(plan, Some(&request.headers))
            .map_err(|_| ProcessingError::invalid("PD execution requires a pair"))?;
        let policies = vec![
            (
                placement.prefill.url().to_owned(),
                self.policies.get_prefill_policy(),
            ),
            (
                placement.decode.url().to_owned(),
                self.policies.get_decode_policy(),
            ),
        ];
        let id = uuid::Uuid::new_v4().to_string();
        let (cancel, canceled) = watch::channel(false);
        let (accepted_tx, accepted) = watch::channel(false);
        let deadline = Instant::now() + self.reservation_timeout;
        self.pending.insert(
            id.clone(),
            Execution {
                placement: placement.clone(),
                metadata: metadata.execution_metadata(),
                body_hash: blake3::hash(&request.raw),
                path: request.path.clone(),
                canceled,
                accepted: accepted_tx,
                receiving: Arc::new(AtomicBool::new(false)),
                deadline,
            },
        );
        Ok(ExecutionLease {
            id,
            pending: self.pending.clone(),
            cancel,
            _placement: placement,
            policies,
            accepted,
            deadline,
        })
    }

    pub async fn serve(
        self: Arc<Self>,
        listener: TcpListener,
        mut stop: watch::Receiver<bool>,
    ) -> std::io::Result<()> {
        let app = Router::new().fallback(Self::handle).with_state(self);
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = stop.wait_for(|v| *v).await;
            })
            .await
    }

    async fn handle(State(executor): State<Arc<Self>>, request: Request) -> Response {
        let (mut parts, body) = request.into_parts();
        let Some(id) = parts
            .headers
            .remove(Self::HEADER)
            .and_then(|v| v.to_str().ok().map(str::to_owned))
        else {
            return (StatusCode::FORBIDDEN, "execution lease required").into_response();
        };
        let Some(execution) = executor.pending.get(&id).map(|entry| entry.clone()) else {
            return (StatusCode::FORBIDDEN, "unknown or consumed execution lease").into_response();
        };
        // Only one upload can claim a reservation, including invalid uploads.
        // Keep the entry until validation so its deadline can still end the session.
        if execution.receiving.swap(true, Ordering::AcqRel) {
            return (StatusCode::FORBIDDEN, "execution lease already claimed").into_response();
        }
        let mut canceled = execution.canceled;
        let operation = async {
            let body =
                tokio::time::timeout_at(execution.deadline, to_bytes(body, executor.max_body))
                    .await
                    .map_err(|_| {
                        ProcessingError::new(
                            504,
                            "execution_lease_timeout",
                            "execution reservation expired while receiving request",
                        )
                    })?
                    .map_err(|_| {
                        ProcessingError::new(413, "body_too_large", "executor body limit exceeded")
                    })?;
            if parts.method != http::Method::POST
                || parts.uri.path() != execution.path
                || blake3::hash(&body) != execution.body_hash
            {
                return Err(ProcessingError::invalid(
                    "request does not match its execution lease",
                ));
            }
            let body = serde_json::from_slice(&body)?;
            // Publish acceptance while holding the entry lock. Expiry and Drop
            // use the same lock, so they cannot observe a removed but unaccepted lease.
            match executor.pending.entry(id.clone()) {
                dashmap::mapref::entry::Entry::Occupied(entry)
                    if Instant::now() < execution.deadline =>
                {
                    let _ = entry.get().accepted.send(true);
                    entry.remove();
                }
                _ => {
                    return Err(ProcessingError::new(
                        403,
                        "execution_lease_expired",
                        "execution lease expired or was already accepted",
                    ))
                }
            }
            parts.headers.remove(super::mutation::Mutation::DESTINATION);
            Ok(executor
                .router
                .execute_placement(
                    &parts.headers,
                    body,
                    &execution.metadata,
                    execution.placement,
                )
                .await)
        };
        let response = tokio::select! {
            result = operation => match result {
                Ok(response) => response,
                Err(error) => return (StatusCode::from_u16(error.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),error.message).into_response(),
            },
            _ = async { let _ = canceled.wait_for(|v| *v).await; } => return StatusCode::REQUEST_TIMEOUT.into_response(),
        };
        let (parts, body) = response.into_parts();
        let stream = futures_util::stream::unfold(
            Some((body.into_data_stream(), canceled)),
            |state| async move {
                let (mut stream, mut canceled) = state?;
                tokio::select! {
                    next = stream.next() => next.map(|bytes| (bytes, Some((stream,canceled)))),
                    _ = async { let _ = canceled.wait_for(|v| *v).await; } => Some((Err(axum::Error::new(std::io::Error::new(std::io::ErrorKind::Interrupted, "execution canceled"))), None)),
                }
            },
        );
        Response::from_parts(parts, Body::from_stream(stream))
    }
}
