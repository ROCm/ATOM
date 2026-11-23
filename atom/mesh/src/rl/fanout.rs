//! `GET|POST /v1/rl/engine/{*path}?selector=`: one engine call fanned out to
//! every worker the selector matches. Not `WorkerManager::fan_out`: this one
//! sends the caller's body, uses the RL control timeout and calls a DP group
//! once, on its base URL, with no retry, circuit breaker or load accounting.

use std::{collections::HashMap, sync::Arc, time::Instant};

use axum::{
    body::Bytes,
    extract::{Path, Query, RawQuery, State},
    http::{HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::{stream, StreamExt};
use serde::Deserialize;
use tracing::info;

use super::{
    discovery::{collapse, RlWorkerInfo},
    error::RlError,
    protocol::{RlCallOutcome, RlFailedCall, RlFanoutResponse},
    proxy::{call_worker, ProxyRequest},
    selector::{merged_labels, Selector},
};
use crate::server::AppState;

#[derive(Deserialize)]
pub(super) struct FanoutQuery {
    selector: Option<String>,
}

/// One worker a fan-out calls, with its own API key from the registry.
#[derive(Debug, Clone)]
pub struct FanoutTarget {
    pub info: RlWorkerInfo,
    pub api_key: Option<String>,
}

/// The registry's workers matching `selector`, a DP group as its lowest rank,
/// sorted by base URL.
pub fn resolve_targets(state: &AppState, selector: &Selector) -> Vec<FanoutTarget> {
    let backend = state.context.router_config.backend;
    let mut api_keys = HashMap::new();
    let workers = state
        .context
        .worker_registry
        .get_all_with_ids()
        .iter()
        .map(|(id, worker)| {
            api_keys.insert(id.as_str().to_string(), worker.api_key().clone());
            RlWorkerInfo::new(id, worker.as_ref(), backend)
        })
        .collect();
    collapse(workers)
        .into_iter()
        .map(|(info, _)| info)
        .filter(|info| selector.matches(&merged_labels(info)))
        .map(|info| FanoutTarget {
            api_key: api_keys.remove(&info.id).flatten(),
            info,
        })
        .collect()
}

/// 200 when no call failed, else 207: a partial success is never a success.
pub fn fanout_status(report: &RlFanoutResponse) -> StatusCode {
    if report.failed.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::MULTI_STATUS
    }
}

/// Send `req` to every target, at most `--rl-fanout-concurrency` at once, on
/// the caller's task: a client disconnect drops the calls still in flight.
pub async fn run_fanout(
    state: &AppState,
    targets: Vec<FanoutTarget>,
    req: &ProxyRequest,
) -> RlFanoutResponse {
    let total = targets.len();
    let concurrency = state.context.router_config.rl.fanout_concurrency.max(1);
    let outcomes: Vec<(RlWorkerInfo, Result<RlCallOutcome, RlError>)> = stream::iter(targets)
        .map(|target| async move {
            let result = call_worker(state, &target.info, target.api_key.as_deref(), req).await;
            (target.info, result)
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    let mut report = RlFanoutResponse {
        total,
        ..RlFanoutResponse::default()
    };
    for (worker, result) in outcomes {
        match result {
            Ok(outcome) if (200..300).contains(&outcome.status) => {
                report.succeeded += 1;
                report.results.insert(worker.id, outcome);
            }
            Ok(outcome) => {
                report.failed.push(RlFailedCall {
                    worker_id: worker.id.clone(),
                    url: worker.url,
                    error: "upstream_error".to_string(),
                    message: format!("HTTP {}", outcome.status),
                    status: Some(outcome.status),
                    connection_mode: None,
                });
                report.results.insert(worker.id, outcome);
            }
            Err(e) => {
                let connection_mode = match &e {
                    RlError::NoControlEndpoint {
                        connection_mode, ..
                    } => Some(connection_mode.clone()),
                    _ => None,
                };
                report.failed.push(RlFailedCall {
                    worker_id: worker.id,
                    url: worker.url,
                    error: e.code().to_string(),
                    message: e.to_string(),
                    status: None,
                    connection_mode,
                });
            }
        }
    }
    report.failed.sort_by(|a, b| a.worker_id.cmp(&b.worker_id));
    report
}

/// The route: the fan-out report, or a 400 before anything is sent.
pub(super) async fn fanout_handler(
    State(state): State<Arc<AppState>>,
    Path(raw_path): Path<String>,
    method: Method,
    Query(FanoutQuery { selector }): Query<FanoutQuery>,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let started = Instant::now();
    let selector = match selector.as_deref().map(str::trim) {
        Some(s) if !s.is_empty() => match Selector::parse(s) {
            Ok(selector) => selector,
            Err(e) => return e.into_response(),
        },
        _ => return RlError::SelectorRequired.into_response(),
    };
    let req =
        match ProxyRequest::from_parts(method, &raw_path, raw_query.as_deref(), &headers, body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
    let targets = resolve_targets(&state, &selector);
    if targets.is_empty() {
        return RlError::NoWorkersMatch(selector.source().to_string()).into_response();
    }
    let report = run_fanout(&state, targets, &req).await;
    info!(
        path = %req.path, selector = %selector.source(), total = report.total,
        succeeded = report.succeeded, failed = report.failed.len(),
        latency_ms = started.elapsed().as_millis() as u64, "rl.fanout"
    );
    (fanout_status(&report), Json(report)).into_response()
}
