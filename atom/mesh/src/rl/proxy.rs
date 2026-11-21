//! `GET|POST /v1/rl/workers/{id}/engine/{*path}`: one engine route of one
//! worker, called verbatim.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Bytes,
    extract::{Path, RawQuery, State},
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        HeaderMap, HeaderName, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;
use tracing::info;

use super::{
    discovery::RlWorkerInfo,
    error::RlError,
    path::{passthrough_query, validate_engine_path},
    protocol::RlCallOutcome,
};
use crate::{core::worker_registry::WorkerId, server::AppState};

/// Largest engine response body kept in an outcome, in bytes.
pub const BODY_CAP: usize = 1 << 20;

/// Request headers copied to the engine. The caller's `authorization` is not
/// one: it carries the `--rl-api-key` bearer the router checked; the engine
/// gets the worker's own key instead.
const FORWARDED_HEADERS: &[&str] = &["x-request-id", "traceparent", "tracestate"];

/// A caller's request, ready to replay against any worker.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    pub method: Method,
    /// The validated engine path, without a leading `/`.
    pub path: String,
    pub query: Option<String>,
    pub content_type: Option<HeaderValue>,
    pub forward: Vec<(HeaderName, HeaderValue)>,
    pub body: Bytes,
}

impl ProxyRequest {
    pub fn from_parts(
        method: Method,
        raw_path: &str,
        raw_query: Option<&str>,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Self, RlError> {
        let path = validate_engine_path(raw_path)?;
        let content_type = headers
            .get(CONTENT_TYPE)
            .cloned()
            .or_else(|| (!body.is_empty()).then(|| HeaderValue::from_static("application/json")));
        let forward = FORWARDED_HEADERS
            .iter()
            .filter_map(|name| {
                let name = HeaderName::from_static(name);
                headers.get(&name).map(|v| (name, v.clone()))
            })
            .collect();
        Ok(Self {
            method,
            path,
            query: passthrough_query(raw_query),
            content_type,
            forward,
            body,
        })
    }

    fn url_for(&self, base: &str) -> String {
        let base = base.trim_end_matches('/');
        match &self.query {
            Some(q) => format!("{base}/{}?{q}", self.path),
            None => format!("{base}/{}", self.path),
        }
    }
}

/// The engine's body as JSON when its content type says JSON and it parses
/// whole, otherwise as text; text when `cut`, since cut bytes do not parse.
fn parse_body(content_type: Option<&HeaderValue>, bytes: &[u8], cut: bool) -> Value {
    let is_json = content_type
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.contains("json"));
    if is_json && !cut {
        if let Ok(v) = serde_json::from_slice(bytes) {
            return v;
        }
    }
    Value::String(String::from_utf8_lossy(bytes).into_owned())
}

/// Send `req` to `worker`'s control endpoint, with `api_key` (the worker's
/// own) as bearer and `--rl-control-timeout-secs` as the call's deadline.
/// `Err` only when the call did not complete: an engine 4xx or 5xx is an
/// outcome carrying that status.
pub async fn call_worker(
    state: &AppState,
    worker: &RlWorkerInfo,
    api_key: Option<&str>,
    req: &ProxyRequest,
) -> Result<RlCallOutcome, RlError> {
    let Some(base) = worker.control_url.as_deref() else {
        return Err(RlError::NoControlEndpoint {
            worker_id: worker.id.clone(),
            url: worker.url.clone(),
            connection_mode: worker.connection_mode.clone(),
            message: "only HTTP workers take RL control calls".to_string(),
        });
    };
    let timeout_secs = state.context.router_config.rl.control_timeout_secs;
    // A per-request timeout replaces the client's request timeout, which a
    // refit can outlive.
    let mut builder = state
        .context
        .client
        .request(req.method.clone(), req.url_for(base))
        .timeout(Duration::from_secs(timeout_secs));
    if let Some(ct) = &req.content_type {
        builder = builder.header(CONTENT_TYPE, ct.clone());
    }
    for (name, value) in &req.forward {
        builder = builder.header(name.clone(), value.clone());
    }
    if let Some(key) = api_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }
    if !req.body.is_empty() {
        builder = builder.body(req.body.clone());
    }

    let started = Instant::now();
    let mut response = builder.send().await.map_err(|e| {
        if e.is_timeout() {
            RlError::UpstreamTimeout {
                worker_id: worker.id.clone(),
                url: worker.url.clone(),
                timeout_secs,
            }
        } else {
            RlError::UpstreamUnreachable {
                worker_id: worker.id.clone(),
                url: worker.url.clone(),
                message: e.to_string(),
            }
        }
    })?;
    let status = response.status().as_u16();
    let content_type = response.headers().get(CONTENT_TYPE).cloned();
    // Read at most BODY_CAP bytes rather than buffer an unbounded body. The
    // status already says whether the engine applied the call, so a failed
    // read keeps it and is reported beside it, not as an unreachable worker.
    let mut collected = Vec::new();
    let mut truncated = false;
    let mut body_error = None;
    loop {
        let chunk = match response.chunk().await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(e) => {
                body_error = Some(format!("reading response body: {e}"));
                break;
            }
        };
        let room = BODY_CAP - collected.len();
        if chunk.len() > room {
            collected.extend_from_slice(&chunk[..room]);
            truncated = true;
            break;
        }
        collected.extend_from_slice(&chunk);
    }
    let latency_ms = started.elapsed().as_millis() as u64;
    info!(
        worker_id = %worker.id, url = %worker.url, method = %req.method,
        path = %req.path, status, latency_ms,
        body_error = body_error.as_deref(), "rl.proxy"
    );
    Ok(RlCallOutcome {
        worker_id: worker.id.clone(),
        url: worker.url.clone(),
        status,
        latency_ms,
        body: parse_body(
            content_type.as_ref(),
            &collected,
            truncated || body_error.is_some(),
        ),
        body_truncated: truncated,
        body_error,
    })
}

/// The route: the outcome under the engine's own status, or the error.
pub(super) async fn proxy_handler(
    State(state): State<Arc<AppState>>,
    Path((id, raw_path)): Path<(String, String)>,
    method: Method,
    RawQuery(raw_query): RawQuery,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req =
        match ProxyRequest::from_parts(method, &raw_path, raw_query.as_deref(), &headers, body) {
            Ok(req) => req,
            Err(e) => return e.into_response(),
        };
    let worker_id = WorkerId::from_string(id);
    let Some(worker) = state.context.worker_registry.get(&worker_id) else {
        return RlError::WorkerNotFound(worker_id.as_str().to_string()).into_response();
    };
    let info = RlWorkerInfo::new(
        &worker_id,
        worker.as_ref(),
        state.context.router_config.backend,
    );
    match call_worker(&state, &info, worker.api_key().as_deref(), &req).await {
        Ok(outcome) => {
            let status = StatusCode::from_u16(outcome.status).unwrap_or(StatusCode::BAD_GATEWAY);
            (status, Json(outcome)).into_response()
        }
        Err(e) => e.into_response(),
    }
}
