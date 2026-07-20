//! Public health and metrics routes owned by the metrics subsystem.
//!
//! The factory currently accepts the full `AppState` to preserve route behavior
//! during the refactor. A future cleanup can replace it with a narrower state
//! containing only router health, worker registry, router config, and HTTP
//! client access.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;

use crate::{
    config::RoutingMode, core::WorkerType,
    observability::metrics::engine_metrics::collect_engine_metrics, server::AppState,
};

pub struct MetricsRouteFactory;

impl MetricsRouteFactory {
    pub fn get(&self, _state: Arc<AppState>) -> Router<Arc<AppState>> {
        Router::new()
            .route("/liveness", get(liveness))
            .route("/readiness", get(readiness))
            .route("/health", get(health))
            .route("/health_generate", get(health_generate))
            .route("/engine_metrics", get(engine_metrics))
    }
}

async fn liveness() -> Response {
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.context.worker_registry.get_all();
    let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

    let is_ready = match &state.context.router_config.mode {
        RoutingMode::PrefillDecode { .. } => {
            let has_prefill = healthy_workers
                .iter()
                .any(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }));
            let has_decode = healthy_workers
                .iter()
                .any(|w| matches!(w.worker_type(), WorkerType::Decode));
            has_prefill && has_decode
        }
        RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
    };

    if is_ready {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "healthy_workers": healthy_workers.len(),
                "total_workers": workers.len()
            })),
        )
            .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not ready",
                "reason": "insufficient healthy workers"
            })),
        )
            .into_response()
    }
}

async fn health(_state: State<Arc<AppState>>) -> Response {
    liveness().await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    collect_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}
