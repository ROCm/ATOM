//! Real Envoy contract tests. Run explicitly with Docker available.

use std::{
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use futures_util::StreamExt;
use mesh::{
    app_context::AppContext,
    config::RouterConfig,
    core::{BasicWorkerBuilder, Worker},
    ext_proc::ExtProcRuntime,
};
use serde_json::{json, Value};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

const ENVOY_IMAGE: &str = "envoyproxy/envoy:v1.37.0";
const ENVOY_CONFIG: &str = include_str!("fixtures/ext-proc/envoy.yaml");

struct Envoy {
    name: String,
    child: Child,
    _config: tempfile::TempDir,
    url: String,
}

impl Envoy {
    async fn start(epp_port: u16) -> Self {
        Self::with_config(epp_port, ENVOY_CONFIG).await
    }

    async fn with_config(epp_port: u16, template: &str) -> Self {
        let config = tempfile::tempdir().unwrap();
        let path = config.path().join("envoy.yaml");
        std::fs::write(
            &path,
            template
                .replace("port_value: 8080", "port_value: CLIENT_PORT")
                .replace("port_value: 9002", "port_value: PROCESSOR_PORT")
                .replace("CLIENT_PORT", "0")
                .replace("PROCESSOR_PORT", &epp_port.to_string())
                + "\nadmin:\n  address:\n    socket_address: {address: 127.0.0.1, port_value: 0}\n",
        )
        .unwrap();
        let validation = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--network",
                "host",
                "--user",
                "0",
                "--env",
                "ENVOY_UID=0",
                "-v",
            ])
            .arg(format!("{}:/etc/envoy/envoy.yaml:ro", path.display()))
            .args([
                ENVOY_IMAGE,
                "-c",
                "/etc/envoy/envoy.yaml",
                "--mode",
                "validate",
            ])
            .output()
            .unwrap();
        assert!(
            validation.status.success(),
            "Envoy config rejected: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        let name = format!("atomesh-extproc-{}", uuid::Uuid::new_v4());
        let child = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--network",
                "host",
                "--user",
                "0",
                "--env",
                "ENVOY_UID=0",
                "--name",
                &name,
                "-v",
            ])
            .arg(format!("{}:/etc/envoy", config.path().display()))
            .args([
                ENVOY_IMAGE,
                "-c",
                "/etc/envoy/envoy.yaml",
                "--disable-hot-restart",
                "--admin-address-path",
                "/etc/envoy/admin-address.txt",
                "--concurrency",
                "2",
                "--log-level",
                "error",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut envoy = Self {
            name,
            child,
            _config: config,
            url: String::new(),
        };
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                assert!(
                    envoy.child.try_wait().unwrap().is_none(),
                    "Envoy exited during startup"
                );
                if let Ok(admin) =
                    std::fs::read_to_string(envoy._config.path().join("admin-address.txt"))
                {
                    let client = reqwest::Client::builder()
                        .timeout(Duration::from_secs(1))
                        .build()
                        .unwrap();
                    if let Ok(response) = client
                        .get(format!("http://{}/listeners?format=json", admin.trim()))
                        .send()
                        .await
                    {
                        if let Ok(listeners) = response.json::<Value>().await {
                            if let Some(port) = listeners["listener_statuses"]
                                .as_array()
                                .and_then(|items| {
                                    items.iter().find(|item| item["name"] == "inference")
                                })
                                .and_then(|item| {
                                    item["local_address"]["socket_address"]["port_value"].as_u64()
                                })
                            {
                                if port > 0 {
                                    envoy.url = format!("http://127.0.0.1:{port}");
                                    break;
                                }
                            }
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        envoy
    }

    async fn raw(&self, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(self.url.trim_start_matches("http://"))
            .await
            .unwrap();
        stream.write_all(request).await.unwrap();
        let mut response = Vec::new();
        tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut response))
            .await
            .unwrap()
            .unwrap();
        String::from_utf8(response).unwrap()
    }

    async fn chunked(&self, path: &str, body: &[u8], trailers: bool) -> String {
        self.chunked_with_headers(path, body, trailers, "").await
    }

    async fn chunked_with_headers(
        &self,
        path: &str,
        body: &[u8],
        trailers: bool,
        headers: &str,
    ) -> String {
        let mut request = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nTE: trailers\r\nConnection: close\r\n{headers}{}\r\n",
            if trailers { "Trailer: x-request-id-span, x-request-checksum, x-mesh-execution-id, cookie, authorization, x-request-id\r\n" } else { "" }).into_bytes();
        for chunk in body.chunks(7) {
            request.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            request.extend_from_slice(chunk);
            request.extend_from_slice(b"\r\n");
        }
        if trailers {
            let value = uuid::Uuid::new_v4();
            request.extend_from_slice(format!("0\r\nx-request-id-span: original\r\nx-request-checksum: original\r\nx-mesh-execution-id: {value}\r\ncookie: {value}\r\nauthorization: Bearer {value}\r\nx-request-id: {value}\r\n\r\n").as_bytes());
        } else {
            request.extend_from_slice(b"0\r\n\r\n");
        }
        self.raw(&request).await
    }
}

impl Drop for Envoy {
    fn drop(&mut self) {
        let _ = Command::new("docker")
            .args(["rm", "-f", &self.name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = self.child.wait();
    }
}

struct Backend {
    calls: AtomicUsize,
}

impl Backend {
    async fn handle(State(state): State<Arc<Self>>, Json(body): Json<Value>) -> Response {
        state.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(body["vendor_extension"], 42);
        if body["stream"] == true {
            let stream = futures_util::stream::unfold(0, |step| async move {
                if step > 0 {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                }
                Some((
                    Ok::<_, std::convert::Infallible>(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                    ),
                    step + 1,
                ))
            });
            (
                [("content-type", "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        } else {
            Json(json!({"choices":[{"message":{"role":"assistant","content":"hello"}}]}))
                .into_response()
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_routes_once_preserves_body_and_cleans_up_sse_cancel() {
    let backend = Arc::new(Backend {
        calls: AtomicUsize::new(0),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = Router::new()
        .route("/v1/chat/completions", post(Backend::handle))
        .with_state(backend.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut config = RouterConfig::default();
    config.ext_proc.enabled = true;
    config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
    config.ext_proc.max_body_bytes = 1024;
    let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("http://{address}"))
            .model_id("test-model")
            .build(),
    );
    app.worker_registry.register(worker.clone());
    let runtime = ExtProcRuntime::start(app).await.unwrap();
    let envoy = Envoy::start(runtime.address.port()).await;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let url = format!("{}/v1/chat/completions", envoy.url);
    let mut body = json!({"model":"test-model","messages":[{"role":"user","content":"hello"}],"vendor_extension":42});
    let response = client
        .post(&url)
        .header("x-gateway-destination-endpoint", "127.0.0.1:1")
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let response = response.text().await.unwrap();
    assert_eq!(status, StatusCode::OK, "{response}");
    assert!(response.contains("hello"));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    body["stream"] = json!(true);
    let response = client
        .post(&url)
        .header("x-correlation-id", "streaming-id")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["x-request-id"], "streaming-id");
    let mut stream = response.bytes_stream();
    assert!(stream.next().await.unwrap().unwrap().starts_with(b"data:"));
    assert_eq!(worker.load(), 1);
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3), async {
        while worker.load() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
    assert_eq!(worker.circuit_breaker().total_successes(), 1);
    assert_eq!(worker.circuit_breaker().total_failures(), 0);
    let bad = client
        .post(&url)
        .header("content-type", "application/json")
        .body("broken")
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    let large = client
        .post(&url)
        .header("content-type", "application/json")
        .body("x".repeat(1025))
        .send()
        .await
        .unwrap();
    assert_eq!(large.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
    drop(envoy);
    runtime.shutdown().await.unwrap();
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_preserves_chunked_body_and_bidirectional_trailers() {
    use http_body::Frame;
    use http_body_util::{BodyExt, StreamBody};

    type Captured = Arc<std::sync::Mutex<Vec<(HeaderMap, Vec<u8>, HeaderMap)>>>;
    async fn handle(State(captured): State<Captured>, request: axum::extract::Request) -> Response {
        let (headers, body) = request.into_parts();
        let body = body.collect().await.unwrap();
        let trailers = body.trailers().cloned().unwrap_or_default();
        let bytes = body.to_bytes();
        let empty_response =
            serde_json::from_slice::<Value>(&bytes).unwrap()["empty_response"] == true;
        captured
            .lock()
            .unwrap()
            .push((headers.headers, bytes.to_vec(), trailers));
        let mut frames = Vec::new();
        if !empty_response {
            frames.push(Ok::<_, std::convert::Infallible>(Frame::data(
                bytes::Bytes::from_static(b"original response"),
            )));
        }
        frames.push(Ok(Frame::trailers(HeaderMap::from_iter([(
            http::HeaderName::from_static("x-response-checksum"),
            http::HeaderValue::from_static("original"),
        )]))));
        (
            [
                ("content-type", "text/plain"),
                ("trailer", "x-response-checksum"),
            ],
            Body::new(StreamBody::new(futures_util::stream::iter(frames))),
        )
            .into_response()
    }
    let captured: Captured = Default::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let worker_address = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/v1/chat/completions", post(handle))
        .with_state(captured.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mut config = RouterConfig::default();
    config.ext_proc.enabled = true;
    config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
    let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("http://{worker_address}"))
            .model_id("test-model")
            .build(),
    );
    app.worker_registry.register(worker.clone());
    let runtime = ExtProcRuntime::start(app).await.unwrap();
    let envoy = Envoy::start(runtime.address.port()).await;
    for (index, (trailers, empty_response)) in [(false, false), (true, false), (true, true)]
        .into_iter()
        .enumerate()
    {
        // Larger than one outgoing mesh body chunk, with unchanged JSON bytes.
        let body = serde_json::to_vec(&json!({"model":"test-model", "messages":[{"role":"user","content":"hello".repeat(13000)}], "empty_response":empty_response})).unwrap();
        let response = envoy.chunked("/v1/chat/completions", &body, trailers).await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(
            response.contains("x-response-checksum: original"),
            "{response}"
        );
        assert_eq!(response.contains("original response"), !empty_response);
        let calls = captured.lock().unwrap();
        let (headers, received, received_trailers) = &calls[index];
        assert!(!headers.contains_key("content-length"));
        assert_eq!(headers["transfer-encoding"], "chunked");
        assert_eq!(received, &body);
        assert_eq!(
            received_trailers
                .get("x-request-id-span")
                .map(|v| v.to_str().unwrap()),
            trailers.then_some("original")
        );
        assert!(!received_trailers.contains_key("x-request-checksum"));
        assert!(!received_trailers.contains_key("x-mesh-execution-id"));
        assert!(!received_trailers.contains_key("cookie"));
        assert!(!received_trailers.contains_key("authorization"));
        assert!(!received_trailers.contains_key("x-request-id"));
        if let Some(declared) = headers.get("trailer") {
            assert_eq!(declared, "x-request-id-span");
        }
    }
    drop(envoy);
    runtime.shutdown().await.unwrap();
    assert_eq!(worker.load(), 0);
    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_rejects_incompatible_modes_without_waiting_for_body() {
    let mut config = RouterConfig::default();
    config.ext_proc.enabled = true;
    config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
    let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let runtime = ExtProcRuntime::start(app).await.unwrap();
    for (field, mode) in [
        ("request_body_mode", "BUFFERED"),
        ("response_body_mode", "STREAMED"),
    ] {
        let template = ENVOY_CONFIG.replace(
            &format!("{field}: FULL_DUPLEX_STREAMED"),
            &format!("{field}: {mode}"),
        );
        let envoy = Envoy::with_config(runtime.address.port(), &template).await;
        // Deliberately withhold the body. Configuration rejection must not wait
        // for the upload or the normal 30-second mesh body timeout.
        let response = envoy.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n").await;
        assert!(response.starts_with("HTTP/1.1 500"), "{response}");
        assert!(
            response.contains("unsupported_processing_mode"),
            "{response}"
        );
        assert!(response.contains(&format!("{field}={mode}")), "{response}");
    }
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_local_errors_keep_the_original_status_and_body() {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let backend_calls = calls.clone();
    let backend = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let calls = backend_calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(10)).await;
                "late upstream response"
            }
        }),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, backend).await.unwrap();
    });
    let mut config = RouterConfig::default();
    config.ext_proc.enabled = true;
    config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
    let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("http://{address}"))
            .model_id("test-model")
            .build(),
    );
    app.worker_registry.register(worker.clone());
    let runtime = ExtProcRuntime::start(app).await.unwrap();
    let base = ENVOY_CONFIG.replace("stat_prefix: inference", "stat_prefix: inference\n                local_reply_config:\n                  body_format: {text_format: 'envoy-local:%RESPONSE_CODE%:%RESPONSE_CODE_DETAILS%'}");
    let buffer = "                  - name: envoy.filters.http.buffer\n                    typed_config:\n                      \"@type\": type.googleapis.com/envoy.extensions.filters.http.buffer.v3.Buffer\n                      max_request_bytes: 1\n";
    let template = base.replace(
        "                http_filters:\n",
        &format!("                http_filters:\n{buffer}"),
    );
    let envoy = Envoy::with_config(runtime.address.port(), &template).await;
    let response = envoy.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await;
    assert!(response.starts_with("HTTP/1.1 413"), "{response}");
    assert!(response.contains("envoy-local:413:"), "{response}");
    assert!(!response.contains("invalid_processing_sequence"));
    drop(envoy);
    // FULL_DUPLEX_STREAMED does not use ext_proc's per-message timeout.
    // Trigger an HCM idle timeout while Mesh is still collecting the upload.
    let template = base.replace("stream_idle_timeout: 300s", "stream_idle_timeout: 0.2s");
    let envoy = Envoy::with_config(runtime.address.port(), &template).await;
    let response = envoy.raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\nx-correlation-id: local-timeout\r\n\r\n{").await;
    assert!(response.starts_with("HTTP/1.1 408"), "{response}");
    assert!(
        response.contains("x-request-id: local-timeout"),
        "{response}"
    );
    assert!(response.contains("envoy-local:408:"), "{response}");
    assert!(!response.contains("invalid_processing_sequence"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        worker.circuit_breaker().total_failures(),
        0,
        "local upload errors must be neutral"
    );
    drop(envoy);
    let template = base.replace("timeout: 1800s", "timeout: 0.2s");
    let envoy = Envoy::with_config(runtime.address.port(), &template).await;
    let response = envoy
        .chunked(
            "/v1/chat/completions",
            br#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
            false,
        )
        .await;
    assert!(response.starts_with("HTTP/1.1 504"), "{response}");
    assert!(response.contains("envoy-local:504:"), "{response}");
    assert!(!response.contains("invalid_processing_sequence"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(envoy);
    runtime.shutdown().await.unwrap();
    assert_eq!(worker.load(), 0);
    assert_eq!(
        worker.circuit_breaker().total_failures(),
        1,
        "Envoy upstream timeout must count once"
    );
    server.abort();
}

struct PdBackend {
    kind: mesh::config::types::BackendType,
    calls: std::sync::Mutex<Vec<(bool, Value)>>,
    mode: AtomicUsize,
    active_bodies: Arc<AtomicUsize>,
}

struct ActiveBody(Arc<AtomicUsize>);
impl Drop for ActiveBody {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl PdBackend {
    async fn metadata() -> Json<Value> {
        Json(json!({"tp_size":4,"kv_role":"kv_producer"}))
    }

    async fn bootstrap() -> Json<Value> {
        Json(json!({"0":{"engine_id":"engine-0"}}))
    }

    async fn handle(
        State((backend, prefill)): State<(Arc<Self>, bool)>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response {
        use mesh::config::types::BackendType;
        assert!(!headers.contains_key("x-mesh-execution-id"));
        assert!(!headers.contains_key("cookie"));
        assert!(!headers.contains_key("x-client-internal"));
        assert!(!headers.contains_key("x-envoy-client-control"));
        if headers.contains_key("x-correlation-id") {
            assert_eq!(headers["x-request-id"], headers["x-correlation-id"]);
        }
        assert_eq!(body["vendor_extension"], 42);
        backend.calls.lock().unwrap().push((prefill, body.clone()));
        if prefill {
            if backend.mode.load(Ordering::SeqCst) == 1 {
                return Json(json!({"missing":"kv"})).into_response();
            }
            if backend.mode.load(Ordering::SeqCst) == 2 {
                return StatusCode::SERVICE_UNAVAILABLE.into_response();
            }
            if backend.kind == BackendType::Vllm && backend.mode.load(Ordering::SeqCst) == 3 {
                return backend.stream(false);
            }
            Json(json!({"kv_transfer_params":{"dp_rank":0,"marker":"from-prefill"}}))
                .into_response()
        } else {
            if backend.kind == BackendType::Atom {
                assert_eq!(body["kv_transfer_params"]["marker"], "from-prefill");
                assert_eq!(body["kv_transfer_params"]["remote_tp_size"], 4);
                assert_eq!(body["kv_transfer_params"]["remote_dp_size"], 1);
                assert_eq!(body["kv_transfer_params"]["remote_dp_rank"], 0);
                assert!(backend.calls.lock().unwrap().iter().any(|(p, _)| *p));
            }
            // A real decode also waits for KV transfer before producing output.
            tokio::time::sleep(Duration::from_millis(30)).await;
            if body["stream"] == true {
                backend.stream(true)
            } else {
                Json(json!({"choices":[{"text":"pd-result"}]})).into_response()
            }
        }
    }

    fn stream(&self, output: bool) -> Response {
        self.active_bodies.fetch_add(1, Ordering::SeqCst);
        let active = ActiveBody(self.active_bodies.clone());
        let stream = futures_util::stream::unfold((0, active), move |(step, active)| async move {
            if step > 0 || !output {
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
            Some((
                Ok::<_, std::convert::Infallible>(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"pd\"}}]}\n\n",
                ),
                (step + 1, active),
            ))
        });
        (
            [("content-type", "text/event-stream")],
            Body::from_stream(stream),
        )
            .into_response()
    }

    async fn verify(kind: mesh::config::types::BackendType) {
        use mesh::{config::RoutingMode, core::WorkerType};
        let backend = Arc::new(Self {
            kind,
            calls: Default::default(),
            mode: AtomicUsize::new(0),
            active_bodies: Arc::new(AtomicUsize::new(0)),
        });
        let mut workers = Vec::new();
        let mut servers = Vec::new();
        let mut config = RouterConfig::default();
        config.backend = kind;
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
        config.ext_proc.enabled = true;
        config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
        config.ext_proc.executor_listen = "127.0.0.1:0".parse().unwrap();
        let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
        for prefill in [true, false] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let router = Router::new()
                .route("/v1/chat/completions", post(Self::handle))
                .route("/v1/completions", post(Self::handle))
                .route("/generate", post(Self::handle))
                .route("/kv_transfer_info", axum::routing::get(Self::metadata))
                .route("/query", axum::routing::get(Self::bootstrap))
                .with_state((backend.clone(), prefill));
            servers.push(tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap();
            }));
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("http://{address}"))
                    .model_id("test-model")
                    .worker_type(if prefill {
                        WorkerType::Prefill {
                            bootstrap_port: Some(address.port()),
                        }
                    } else {
                        WorkerType::Decode
                    })
                    .build(),
            );
            app.worker_registry.register(worker.clone());
            workers.push(worker);
        }
        let runtime = ExtProcRuntime::start(app).await.unwrap();
        let envoy = Envoy::start(runtime.address.port()).await;
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let requests = [
            (
                "/v1/chat/completions",
                json!({"model":"test-model","messages":[{"role":"user","content":"hi"}],"vendor_extension":42}),
            ),
            (
                "/v1/completions",
                json!({"model":"test-model","prompt":"hi","vendor_extension":42}),
            ),
            (
                "/generate",
                json!({"model":"test-model","text":"hi","vendor_extension":42}),
            ),
            ("/generate", json!({"text":"hi","vendor_extension":42})),
            (
                "/generate",
                json!({"model":null,"text":"hi","vendor_extension":42}),
            ),
        ];
        for (index, (path, body)) in requests.iter().enumerate() {
            let text = if index == 0 {
                let response = client
                    .post(format!("{}{path}", envoy.url))
                    .header("x-correlation-id", "pd-correlation")
                    .header("cookie", uuid::Uuid::new_v4().to_string())
                    .header("x-client-internal", "untrusted")
                    .header("x-envoy-client-control", "untrusted")
                    .header("x-mesh-execution-id", "untrusted")
                    .json(body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.headers()["x-request-id"], "pd-correlation");
                let status = response.status();
                let text = response.text().await.unwrap();
                assert_eq!(status, StatusCode::OK, "{kind:?}: {text}");
                text
            } else {
                // The executor must accept the same decoded bytes regardless of
                // chunk boundaries or request trailers, without weakening its hash.
                let text = envoy
                    .chunked(path, &serde_json::to_vec(body).unwrap(), index == 2)
                    .await;
                assert!(text.starts_with("HTTP/1.1 200"), "{kind:?}: {text}");
                text
            };
            assert!(text.contains("pd-result"));
            let calls = backend.calls.lock().unwrap();
            for prefill in [true, false] {
                let (_, forwarded) = calls.iter().rev().find(|(p, _)| *p == prefill).unwrap();
                assert_eq!(forwarded.get("model"), body.get("model"));
            }
        }
        let calls = backend.calls.lock().unwrap().clone();
        assert_eq!(calls.iter().filter(|(p, _)| *p).count(), requests.len());
        assert_eq!(calls.iter().filter(|(p, _)| !*p).count(), requests.len());
        if kind == mesh::config::types::BackendType::Vllm {
            for (_, decode) in calls.iter().filter(|(p, _)| !*p) {
                assert_eq!(decode["kv_transfer_params"]["remote_engine_id"], "engine-0");
                assert!(calls.iter().any(|(p, body)| *p
                    && body["kv_transfer_params"]["transfer_id"]
                        == decode["kv_transfer_params"]["transfer_id"]));
            }
        }
        if kind == mesh::config::types::BackendType::Sglang {
            for (_, decode) in calls.iter().filter(|(p, _)| !*p) {
                assert!(calls
                    .iter()
                    .any(|(p, body)| *p && body["bootstrap_room"] == decode["bootstrap_room"]));
            }
        }
        if kind == mesh::config::types::BackendType::Atom {
            for mode in [1, 2] {
                backend.mode.store(mode, Ordering::SeqCst);
                let response = client
                    .post(format!("{}{}", envoy.url, requests[0].0))
                    .json(&requests[0].1)
                    .send()
                    .await
                    .unwrap();
                assert!(response.status().is_server_error());
                let _ = response.bytes().await.unwrap();
            }
            let calls = backend.calls.lock().unwrap();
            assert_eq!(calls.iter().filter(|(p, _)| *p).count(), requests.len() + 2);
            assert_eq!(
                calls.iter().filter(|(p, _)| !*p).count(),
                requests.len(),
                "failed prefill must not invoke decode or retry"
            );
        }
        backend.mode.store(3, Ordering::SeqCst);
        let mut body = requests[0].1.clone();
        body["stream"] = json!(true);
        let response = client
            .post(format!("{}{}", envoy.url, requests[0].0))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.bytes_stream();
        assert!(stream.next().await.unwrap().unwrap().starts_with(b"data:"));
        assert_eq!(
            workers[0].load(),
            usize::from(kind == mesh::config::types::BackendType::Vllm),
            "only the still-running vLLM prefill retains its reservation"
        );
        assert_eq!(
            workers[1].load(),
            1,
            "decode retains exactly one reservation"
        );
        drop(stream);
        tokio::time::timeout(Duration::from_secs(3), async {
            while workers.iter().any(|w| w.load() != 0)
                || backend.active_bodies.load(Ordering::SeqCst) != 0
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        drop(envoy);
        runtime.shutdown().await.unwrap();
        for server in servers {
            server.abort();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_atom_pd_executes_selected_pair_and_relays_kv() {
    PdBackend::verify(mesh::config::types::BackendType::Atom).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_vllm_pd_executes_selected_pair_and_cancels_prefill() {
    PdBackend::verify(mesh::config::types::BackendType::Vllm).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_sglang_pd_executes_selected_pair_and_cancels_decode() {
    PdBackend::verify(mesh::config::types::BackendType::Sglang).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires Docker and envoyproxy/envoy:v1.37.0"]
async fn real_envoy_preserves_correlation_and_filters_client_headers() {
    async fn echo(headers: HeaderMap, Json(_): Json<Value>) -> Response {
        let headers: std::collections::BTreeMap<_, Vec<_>> = headers
            .keys()
            .map(|name| {
                (
                    name.to_string(),
                    headers
                        .get_all(name)
                        .iter()
                        .map(|v| v.to_str().unwrap().to_owned())
                        .collect(),
                )
            })
            .collect();
        ([("x-request-id", "backend-id")], Json(headers)).into_response()
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, Router::new().route("/generate", post(echo)))
            .await
            .unwrap();
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let client_auth = format!("Bearer {}", uuid::Uuid::new_v4());
    let worker_key = uuid::Uuid::new_v4().to_string();
    for custom in [false, true] {
        let mut config = RouterConfig::default();
        config.ext_proc.enabled = true;
        config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
        config.request_id_headers =
            custom.then(|| vec!["x-customer-id".into(), "x-correlation-id".into()]);
        let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
        let mut worker =
            BasicWorkerBuilder::new(format!("http://{address}")).model_id("test-model");
        if custom {
            worker = worker.api_key(worker_key.clone());
        }
        app.worker_registry.register(Arc::new(worker.build()));
        let runtime = ExtProcRuntime::start(app).await.unwrap();
        let envoy = Envoy::start(runtime.address.port()).await;
        let url = format!("{}/generate", envoy.url);
        let response = client
            .post(&url)
            .header("x-correlation-id", "correlation")
            .header("x-customer-id", "customer")
            .header("x-trace-id", "lower-priority")
            .header("x-session-id", "session")
            .header(
                "traceparent",
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            )
            .header("authorization", &client_auth)
            .header("cookie", uuid::Uuid::new_v4().to_string())
            .header("x-client-internal", "untrusted")
            .header("x-envoy-client-control", "untrusted")
            .header("x-mesh-execution-id", "untrusted")
            .json(&json!({"text":"hello"}))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response_id = response.headers().get("x-request-id").cloned();
        let text = response.text().await.unwrap();
        assert_eq!(status, StatusCode::OK, "{text}");
        let expected = if custom { "customer" } else { "correlation" };
        assert_eq!(response_id.unwrap(), expected);
        let headers: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(headers["x-request-id"], json!([expected]));
        assert_eq!(headers["x-correlation-id"], json!(["correlation"]));
        assert_eq!(headers["x-session-id"], json!(["session"]));
        assert!(headers.get("traceparent").is_some());
        assert_eq!(headers["content-type"], json!(["application/json"]));
        let expected_auth = if custom {
            format!("Bearer {worker_key}")
        } else {
            client_auth.clone()
        };
        assert_eq!(headers["authorization"], json!([expected_auth]));
        for name in [
            "cookie",
            "x-client-internal",
            "x-envoy-client-control",
            "x-mesh-execution-id",
            "x-customer-id",
            "x-trace-id",
        ] {
            assert!(headers.get(name).is_none(), "forwarded {name}: {headers}");
        }
        let response = client
            .post(&url)
            .json(&json!({"text":"hello"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        assert!(id.starts_with("gnt-") && id.len() == 28, "{id}");
        assert_eq!(
            response.json::<Value>().await.unwrap()["x-request-id"],
            json!([id])
        );
        let response = client
            .post(&url)
            .header("x-correlation-id", "parse-error")
            .header("content-type", "application/json")
            .body("broken")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response.headers()["x-request-id"], "parse-error");
        // Also verify disallowed fields cannot enter through request trailers.
        let request_headers = format!(
            "x-correlation-id: trailer-id\r\ncookie: {}\r\nx-client-internal: untrusted\r\n",
            uuid::Uuid::new_v4()
        );
        let response = envoy
            .chunked_with_headers("/generate", br#"{"text":"hello"}"#, true, &request_headers)
            .await;
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("x-request-id: trailer-id"), "{response}");
        drop(envoy);
        runtime.shutdown().await.unwrap();
    }
    server.abort();
}
