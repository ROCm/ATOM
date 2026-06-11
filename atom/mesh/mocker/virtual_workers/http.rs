//! Axum-based virtual backend worker for Atomesh integration tests.
//!
//! This worker exposes the same HTTP endpoints Atomesh expects from an
//! inference backend, but it does not load a model or run inference. It matches
//! forwarded requests against fixture cases and returns the fixture's mock
//! response, which keeps tests deterministic and GPU-free.

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::{get, post},
    Router,
};
use futures_util::stream;
use serde_json::{json, Value};
use tokio::sync::oneshot;

use super::{MockCase, ReplayCaseStore};

#[derive(Clone)]
struct VirtualWorkerState {
    replay_case_store: Arc<ReplayCaseStore>,
    request_log: Arc<Mutex<Vec<String>>>,
}

pub struct VirtualWorker {
    replay_case_store: Arc<ReplayCaseStore>,
    request_log: Arc<Mutex<Vec<String>>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    pub url: Option<String>,
}

impl VirtualWorker {
    /// Create a worker that can replay a single fixture case.
    pub fn new(case: MockCase) -> Self {
        Self::with_replay_case_store(ReplayCaseStore::new(vec![case]))
    }

    /// Create a worker that can replay multiple fixture cases.
    pub fn with_replay_case_store(replay_case_store: ReplayCaseStore) -> Self {
        Self {
            replay_case_store: Arc::new(replay_case_store),
            request_log: Arc::new(Mutex::new(Vec::new())),
            shutdown_tx: None,
            shutdown_handle: None,
            url: None,
        }
    }

    /// Start the HTTP worker on a random local port and wait for readiness.
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        self.start_with_bind_addr(("127.0.0.1", 0)).await
    }

    /// Start the HTTP worker on a deterministic host and port.
    pub async fn start_on(
        &mut self,
        host: &str,
        port: u16,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.start_with_bind_addr((host, port)).await
    }

    async fn start_with_bind_addr(
        &mut self,
        bind_addr: (&str, u16),
    ) -> Result<String, Box<dyn std::error::Error>> {
        let listener = tokio::net::TcpListener::bind(bind_addr).await?;
        let addr = listener.local_addr()?;
        self.start_with_listener(listener, addr).await
    }

    async fn start_with_listener(
        &mut self,
        listener: tokio::net::TcpListener,
        addr: SocketAddr,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("http://{}", addr);
        let state = VirtualWorkerState {
            replay_case_store: Arc::clone(&self.replay_case_store),
            request_log: Arc::clone(&self.request_log),
        };

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/health_generate", get(health_generate_handler))
            .route("/get_server_info", get(server_info_handler))
            .route("/get_model_info", get(model_info_handler))
            .route("/v1/models", get(models_handler))
            .route("/generate", post(generate_handler))
            .route("/v1/chat/completions", post(chat_handler))
            .route("/v1/completions", post(completion_handler))
            .route("/v1/responses", post(responses_handler))
            .with_state(state);

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
            if let Err(error) = server.await {
                eprintln!("virtual worker error on {}: {}", addr, error);
            }
        });

        self.shutdown_tx = Some(shutdown_tx);
        self.shutdown_handle = Some(handle);
        self.url = Some(url.clone());
        wait_until_ready(&url).await?;
        Ok(url)
    }

    pub fn request_log(&self) -> Vec<String> {
        self.request_log.lock().unwrap().clone()
    }

    /// Stop the worker and wait briefly for the server task to exit.
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for VirtualWorker {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

async fn wait_until_ready(url: &str) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(3);

    loop {
        if let Ok(response) = client.get(format!("{}/health", url)).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }

        if tokio::time::Instant::now() > deadline {
            return Err(format!("virtual worker at {} did not become ready", url).into());
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
}

async fn health_handler(State(state): State<VirtualWorkerState>) -> Response {
    state
        .request_log
        .lock()
        .unwrap()
        .push("/health".to_string());
    Json(json!({ "status": "healthy" })).into_response()
}

async fn health_generate_handler(State(state): State<VirtualWorkerState>) -> Response {
    state
        .request_log
        .lock()
        .unwrap()
        .push("/health_generate".to_string());
    Json(json!({ "status": "ok", "queue_length": 0 })).into_response()
}

async fn server_info_handler(State(state): State<VirtualWorkerState>) -> Response {
    state
        .request_log
        .lock()
        .unwrap()
        .push("/get_server_info".to_string());
    Json(json!({
        "model_path": model(&state),
        "tokenizer_path": "virtual-tokenizer",
        "context_length": 32768,
        "version": "virtual-worker"
    }))
    .into_response()
}

async fn model_info_handler(State(state): State<VirtualWorkerState>) -> Response {
    state
        .request_log
        .lock()
        .unwrap()
        .push("/get_model_info".to_string());
    Json(json!({
        "model_path": model(&state),
        "is_generation": true
    }))
    .into_response()
}

async fn models_handler(State(state): State<VirtualWorkerState>) -> Response {
    state
        .request_log
        .lock()
        .unwrap()
        .push("/v1/models".to_string());
    Json(json!({
        "object": "list",
        "data": [{
            "id": model(&state),
            "object": "model",
            "owned_by": "virtual-worker"
        }]
    }))
    .into_response()
}

async fn generate_handler(
    State(state): State<VirtualWorkerState>,
    Json(body): Json<Value>,
) -> Response {
    replay("/generate", state, body).await
}

async fn chat_handler(
    State(state): State<VirtualWorkerState>,
    Json(body): Json<Value>,
) -> Response {
    replay("/v1/chat/completions", state, body).await
}

async fn completion_handler(
    State(state): State<VirtualWorkerState>,
    Json(body): Json<Value>,
) -> Response {
    replay("/v1/completions", state, body).await
}

async fn responses_handler(
    State(state): State<VirtualWorkerState>,
    Json(body): Json<Value>,
) -> Response {
    replay("/v1/responses", state, body).await
}

async fn replay(endpoint: &str, state: VirtualWorkerState, body: Value) -> Response {
    state.request_log.lock().unwrap().push(endpoint.to_string());

    // A miss means Atomesh sent a request this fixture set does not describe,
    // which should be visible as a backend error in the calling test.
    let Some(case) = state.replay_case_store.match_request(endpoint, &body) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("no replay fixture matched {}", endpoint)
            })),
        )
            .into_response();
    };

    if case.is_streaming() {
        return streaming_response(case);
    }

    let status = StatusCode::from_u16(case.expected_response.status)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(case.expected_response.body.clone())).into_response()
}

fn streaming_response(case: &MockCase) -> Response {
    let body = case.expected_response.body.clone();
    let mut events = match body {
        Value::Array(items) => items
            .into_iter()
            .map(|item| Ok::<_, Infallible>(Event::default().data(item.to_string())))
            .collect::<Vec<_>>(),
        body => vec![Ok(Event::default().data(body.to_string()))],
    };
    events.push(Ok(Event::default().data("[DONE]")));

    Sse::new(stream::iter(events))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn model(state: &VirtualWorkerState) -> String {
    state
        .replay_case_store
        .first()
        .map(|case| case.model.clone())
        .unwrap_or_else(|| "virtual-model".to_string())
}
