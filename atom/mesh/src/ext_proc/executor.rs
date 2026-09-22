use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    response::{IntoResponse, Response},
    Router,
};
use dashmap::DashMap;
use futures_util::StreamExt;
use http::StatusCode;
use tokio::{net::TcpListener, sync::watch};

use crate::{
    app_context::AppContext,
    core::{
        placement::{
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, RequestDescriptor},
        },
        WorkerLoadGuard,
    },
    routers::http_pd_router::PDRouter,
};

use super::{error::ProcessingError, request::RequestEnvelope};

/// Executes an already selected PD pair with the existing backend adapters.
pub(super) struct PdExecutor {
    router: PDRouter,
    policies: Arc<crate::policies::PolicyRegistry>,
    pending: Arc<DashMap<String, Execution>>,
    pub address: SocketAddr,
    max_body: usize,
}

struct Execution {
    plan: Arc<PlacementPlan>,
    body_hash: blake3::Hash,
    path: String,
    canceled: watch::Receiver<bool>,
}

pub(super) struct ExecutionLease {
    pub id: String,
    pending: Arc<DashMap<String, Execution>>,
    cancel: watch::Sender<bool>,
    _load: Vec<WorkerLoadGuard>,
    policies: Vec<(String, Arc<dyn crate::policies::LoadBalancingPolicy>)>,
}

impl ExecutionLease {
    pub fn complete(&self, success: bool) {
        for (url, policy) in &self.policies {
            policy.on_request_complete(url, success);
        }
    }
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        self.pending.remove(&self.id);
        let _ = self.cancel.send(true);
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
            }),
            listener,
        ))
    }

    pub fn reserve(&self, plan: PlacementPlan, request: &RequestEnvelope) -> ExecutionLease {
        let plan = self.router.finalize_external_placement(plan);
        let load = match &plan {
            PlacementPlan::Pair {
                prefill, decode, ..
            } => vec![
                WorkerLoadGuard::new(prefill.clone(), Some(&request.headers)),
                WorkerLoadGuard::new(decode.clone(), Some(&request.headers)),
            ],
            _ => Vec::new(),
        };
        let policies = match &plan {
            PlacementPlan::Pair {
                prefill, decode, ..
            } => vec![
                (prefill.url().to_owned(), self.policies.get_prefill_policy()),
                (decode.url().to_owned(), self.policies.get_decode_policy()),
            ],
            _ => Vec::new(),
        };
        let id = uuid::Uuid::new_v4().to_string();
        let (cancel, canceled) = watch::channel(false);
        self.pending.insert(
            id.clone(),
            Execution {
                plan: Arc::new(plan),
                body_hash: blake3::hash(&request.raw),
                path: request.path.clone(),
                canceled,
            },
        );
        ExecutionLease {
            id,
            pending: self.pending.clone(),
            cancel,
            _load: load,
            policies,
        }
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
        let Some((_, execution)) = executor.pending.remove(&id) else {
            return (StatusCode::FORBIDDEN, "unknown or consumed execution lease").into_response();
        };
        let mut canceled = execution.canceled;
        let operation = async {
            let body = to_bytes(body, executor.max_body).await.map_err(|_| {
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
            parts.headers.remove(super::mutation::Mutation::DESTINATION);
            let router = executor
                .router
                .with_external_placement(Arc::new(PinnedPlanner(execution.plan)));
            let body = serde_json::from_slice(&body)?;
            Ok(router
                .execute_external(&parts.headers, &execution.path, body)
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
            (body.into_data_stream(), canceled),
            |(mut stream, mut canceled)| async move {
                tokio::select! {
                    next = stream.next() => next.map(|bytes| (bytes,(stream,canceled))),
                    _ = async { let _ = canceled.wait_for(|v| *v).await; } => None,
                }
            },
        );
        Response::from_parts(parts, Body::from_stream(stream))
    }
}

struct PinnedPlanner(Arc<PlacementPlan>);

#[async_trait::async_trait]
impl PdPlanner for PinnedPlanner {
    async fn plan(
        &self,
        _request: &RequestDescriptor<'_>,
    ) -> Result<PlacementPlan, PlacementError> {
        match self.0.as_ref() {
            PlacementPlan::Pair {
                prefill,
                decode,
                prefill_policy,
                decode_policy,
            } => Ok(PlacementPlan::Pair {
                prefill: prefill.clone(),
                decode: decode.clone(),
                prefill_policy,
                decode_policy,
            }),
            _ => Err(PlacementError::NoPrefillWorkers),
        }
    }
}
