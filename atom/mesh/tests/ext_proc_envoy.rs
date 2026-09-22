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
use tokio::net::TcpListener;

struct Envoy {
    name: String,
    child: Child,
    _config: tempfile::TempDir,
    url: String,
}

impl Envoy {
    async fn start(epp_port: u16) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let config = tempfile::tempdir().unwrap();
        let path = config.path().join("envoy.yaml");
        std::fs::write(
            &path,
            include_str!("fixtures/ext-proc/envoy.yaml")
                .replace("port_value: 8080", &format!("port_value: {port}"))
                .replace("port_value: 9002", &format!("port_value: {epp_port}")),
        )
        .unwrap();
        let name = format!("atomesh-extproc-{}", uuid::Uuid::new_v4());
        let child = Command::new("docker")
            .args([
                "run",
                "--rm",
                "--network",
                "host",
                "--user",
                "0",
                "--name",
                &name,
                "-v",
            ])
            .arg(format!("{}:/etc/envoy/envoy.yaml:ro", path.display()))
            .args([
                "envoyproxy/envoy:v1.37.0",
                "-c",
                "/etc/envoy/envoy.yaml",
                "--disable-hot-restart",
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
            url: format!("http://127.0.0.1:{port}"),
        };
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                assert!(
                    envoy.child.try_wait().unwrap().is_none(),
                    "Envoy exited during startup"
                );
                if tokio::net::TcpStream::connect(("127.0.0.1", port))
                    .await
                    .is_ok()
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .unwrap();
        envoy
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
    async fn handle(
        State(state): State<Arc<Self>>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Response {
        state.calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(body["vendor_extension"], 42);
        assert_eq!(headers["authorization"], "Bearer worker-key");
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
            .api_key("worker-key")
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
    let mut stream = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .unwrap()
        .bytes_stream();
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
        assert_eq!(headers["authorization"], "Bearer pd-worker-key");
        assert!(!headers.contains_key("x-mesh-execution-id"));
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
                    .api_key("pd-worker-key")
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
        ];
        for (path, body) in &requests {
            let response = client
                .post(format!("{}{path}", envoy.url))
                .json(body)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let text = response.text().await.unwrap();
            assert_eq!(status, StatusCode::OK, "{kind:?}: {text}");
            assert!(text.contains("pd-result"));
        }
        let calls = backend.calls.lock().unwrap().clone();
        assert_eq!(calls.iter().filter(|(p, _)| *p).count(), 3);
        assert_eq!(calls.iter().filter(|(p, _)| !*p).count(), 3);
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
            assert_eq!(calls.iter().filter(|(p, _)| *p).count(), 5);
            assert_eq!(
                calls.iter().filter(|(p, _)| !*p).count(),
                3,
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
        assert!(
            workers.iter().all(|w| w.load() == 1),
            "each chosen worker has exactly one lease"
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
