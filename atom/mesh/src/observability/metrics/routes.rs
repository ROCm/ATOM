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

/// A failed preparation pool never recovers, so report the process as not
/// live and let the orchestrator restart it rather than only draining traffic.
async fn liveness(State(state): State<Arc<AppState>>) -> Response {
    if state.context.prepare_pool.is_failed() {
        return prepare_pool_unavailable();
    }
    (StatusCode::OK, "OK").into_response()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    if state.context.prepare_pool.is_failed() {
        return prepare_pool_unavailable();
    }
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

async fn health(state: State<Arc<AppState>>) -> Response {
    liveness(state).await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    if state.context.prepare_pool.is_failed() {
        return prepare_pool_unavailable();
    }
    state.router.health_generate(req).await
}

fn prepare_pool_unavailable() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "status": "unavailable",
            "reason": "request preparation worker failed"
        })),
    )
        .into_response()
}

async fn engine_metrics(State(state): State<Arc<AppState>>) -> Response {
    collect_engine_metrics(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        app_context::AppContext,
        core::prepare_pool::{PoolConfig, PreparePoolRuntime},
    };

    #[tokio::test]
    async fn liveness_and_health_fail_after_the_preparation_pool_fails() {
        struct PanickingOutput;
        impl Drop for PanickingOutput {
            fn drop(&mut self) {
                panic!("expected output destructor panic");
            }
        }
        let mut pool = PreparePoolRuntime::new(PoolConfig {
            workers: 1,
            ..crate::config::RouterConfig::default()
                .resolved_prepare_pool()
                .unwrap()
        })
        .unwrap();
        let context = Arc::new(
            AppContext::from_config(Default::default(), 5, pool.handle())
                .await
                .unwrap(),
        );
        let router = Arc::new(
            crate::routers::http_router::Router::new(&context)
                .await
                .unwrap(),
        );
        let state = Arc::new(AppState {
            router,
            context,
            router_manager: None,
        });
        let app = MetricsRouteFactory
            .get(state.clone())
            .with_state(state.clone());
        let status = |path: &'static str| {
            let app = app.clone();
            async move {
                app.oneshot(Request::get(path).body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status()
            }
        };
        for path in ["/liveness", "/health"] {
            assert_eq!(status(path).await, StatusCode::OK, "{path}");
        }

        // Drop the ticket before the job returns so the worker itself drops the
        // panicking output, outside the user-work panic guard.
        let handle = pool.handle();
        let (started_tx, started) = tokio::sync::oneshot::channel();
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let ticket = handle
            .submit(handle.prepare_deadline(), move |_| {
                started_tx.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                PanickingOutput
            })
            .await
            .unwrap();
        started.await.unwrap();
        drop(ticket);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !handle.is_failed() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        for path in ["/liveness", "/health"] {
            assert_eq!(
                status(path).await,
                StatusCode::SERVICE_UNAVAILABLE,
                "{path}"
            );
        }
        pool.shutdown().await;
    }
}
