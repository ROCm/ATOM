//! `GET /v1/rl/workers` and `GET /v1/rl/workers/{id}`: the registry's workers
//! as SMG lists them, with DP ranks collapsed into one row per group.

use std::{collections::HashMap, sync::Arc};

use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response},
    Json,
};

use super::{
    capability::capabilities_for,
    error::RlError,
    protocol::{RlWorkerEntry, RlWorkersResponse, RL_PROTOCOL_VERSION},
};
use crate::{
    config::BackendType,
    core::{worker_registry::WorkerId, ConnectionMode, Worker},
    server::AppState,
};

/// A snapshot of one registered worker, as the RL control plane sees it.
#[derive(Debug, Clone)]
pub struct RlWorkerInfo {
    /// Registry id.
    pub id: String,
    /// `Worker::url()`; carries an `@<rank>` suffix for a DP-aware worker.
    pub url: String,
    pub base_url: String,
    /// The router's `--backend` for an HTTP worker, the runtime type for a
    /// gRPC one.
    pub engine: String,
    pub model_id: String,
    /// `regular`, `prefill` or `decode`.
    pub worker_type: String,
    /// `http` or `grpc`.
    pub connection_mode: String,
    /// `ready` or `not_ready`.
    pub health: String,
    pub is_dp_aware: bool,
    pub dp_rank: usize,
    /// The worker's labels: discovered metadata merged with caller labels.
    pub labels: HashMap<String, String>,
    /// Where control calls go: the worker itself for an HTTP worker; a gRPC
    /// worker has none.
    pub control_url: Option<String>,
}

fn backend_name(backend: BackendType) -> &'static str {
    match backend {
        BackendType::Sglang => "sglang",
        BackendType::Vllm => "vllm",
        BackendType::Atom => "atom",
    }
}

impl RlWorkerInfo {
    pub fn new(id: &WorkerId, worker: &dyn Worker, backend: BackendType) -> Self {
        let metadata = worker.metadata();
        let (engine, connection_mode, control_url) = match worker.connection_mode() {
            ConnectionMode::Http => (
                backend_name(backend).to_string(),
                "http",
                Some(worker.base_url().to_string()),
            ),
            ConnectionMode::Grpc { .. } => (metadata.runtime_type.to_string(), "grpc", None),
        };
        Self {
            id: id.as_str().to_string(),
            url: worker.url().to_string(),
            base_url: worker.base_url().to_string(),
            engine,
            model_id: worker.model_id().to_string(),
            worker_type: worker.worker_type().as_metric_label().to_string(),
            connection_mode: connection_mode.to_string(),
            health: if worker.is_healthy() {
                "ready"
            } else {
                "not_ready"
            }
            .to_string(),
            is_dp_aware: worker.is_dp_aware(),
            dp_rank: worker.dp_rank().unwrap_or(0),
            labels: metadata.labels.clone(),
            control_url,
        }
    }
}

/// Every registered worker.
pub fn registry_workers(state: &AppState) -> Vec<RlWorkerInfo> {
    let backend = state.context.router_config.backend;
    state
        .context
        .worker_registry
        .get_all_with_ids()
        .iter()
        .map(|(id, worker)| RlWorkerInfo::new(id, worker.as_ref(), backend))
        .collect()
}

fn int_label(labels: &HashMap<String, String>, key: &str) -> Option<u64> {
    labels.get(key).and_then(|v| v.trim().parse().ok())
}

/// Collapse DP-aware ranks sharing a `base_url` into one row (the lowest rank
/// is kept) and sort rows by `base_url`, then `url`. The count is the number
/// of ranks. A non-DP worker whose `url` equals a DP group's `base_url` stays
/// its own row.
pub fn collapse(workers: Vec<RlWorkerInfo>) -> Vec<(RlWorkerInfo, usize)> {
    let mut groups: HashMap<(bool, String), (RlWorkerInfo, usize)> = HashMap::new();
    for w in workers {
        let key = if w.is_dp_aware {
            (true, w.base_url.clone())
        } else {
            (false, w.url.clone())
        };
        match groups.get_mut(&key) {
            None => {
                groups.insert(key, (w, 1));
            }
            Some((kept, n)) => {
                *n += 1;
                if w.dp_rank < kept.dp_rank {
                    *kept = w;
                }
            }
        }
    }
    let mut rows: Vec<(RlWorkerInfo, usize)> = groups.into_values().collect();
    rows.sort_by(|a, b| a.0.base_url.cmp(&b.0.base_url).then(a.0.url.cmp(&b.0.url)));
    rows
}

/// The worker's labels plus the keys a selector can match on; those keys
/// shadow labels of the same name.
pub fn merged_labels(info: &RlWorkerInfo) -> HashMap<String, String> {
    let mut m = info.labels.clone();
    m.insert("id".to_string(), info.id.clone());
    m.insert("url".to_string(), info.url.clone());
    m.insert("base_url".to_string(), info.base_url.clone());
    m.insert("engine".to_string(), info.engine.clone());
    m.insert("model".to_string(), info.model_id.clone());
    m.insert("worker_type".to_string(), info.worker_type.clone());
    m.insert("connection_mode".to_string(), info.connection_mode.clone());
    m.insert("health".to_string(), info.health.clone());
    m
}

/// One row of `GET /v1/rl/workers` for `info`, with `dp_ranks` collapsed into it.
pub fn entry(info: &RlWorkerInfo, dp_ranks: usize) -> RlWorkerEntry {
    RlWorkerEntry {
        id: info.id.clone(),
        url: info.url.clone(),
        base_url: info.base_url.clone(),
        engine: info.engine.clone(),
        engine_version: info.labels.get("version").cloned(),
        model_id: info.model_id.clone(),
        worker_type: info.worker_type.clone(),
        connection_mode: info.connection_mode.clone(),
        control_url: info.control_url.clone(),
        tp_size: int_label(&info.labels, "tp_size"),
        dp_size: int_label(&info.labels, "dp_size"),
        pp_size: int_label(&info.labels, "pp_size"),
        dp_ranks,
        role: info.labels.get("role").cloned(),
        health: info.health.clone(),
        weight_version: info.labels.get("weight_version").cloned(),
        labels: info.labels.clone(),
        capabilities: capabilities_for(&info.engine, &info.labels),
    }
}

pub(super) async fn list_workers(State(state): State<Arc<AppState>>) -> Response {
    let workers: Vec<RlWorkerEntry> = collapse(registry_workers(&state))
        .iter()
        .map(|(w, n)| entry(w, *n))
        .collect();
    let total = workers.len();
    Json(RlWorkersResponse {
        protocol_version: RL_PROTOCOL_VERSION,
        workers,
        total,
    })
    .into_response()
}

pub(super) async fn get_worker(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    let worker_id = WorkerId::from_string(id);
    let Some(worker) = state.context.worker_registry.get(&worker_id) else {
        return RlError::WorkerNotFound(worker_id.as_str().to_string()).into_response();
    };
    let backend = state.context.router_config.backend;
    let info = RlWorkerInfo::new(&worker_id, worker.as_ref(), backend);
    let ranks = if info.is_dp_aware {
        state
            .context
            .worker_registry
            .get_all()
            .iter()
            .filter(|other| other.is_dp_aware() && other.base_url() == info.base_url)
            .count()
    } else {
        1
    };
    Json(entry(&info, ranks)).into_response()
}
