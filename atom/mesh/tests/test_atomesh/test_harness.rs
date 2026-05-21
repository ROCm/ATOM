//! End-to-end harness for fixture-driven Atomesh tests.
//!
//! `TestHarness` starts virtual backend workers, registers them through the
//! normal worker initialization path, builds the real Axum app, sends a
//! fixture-generated client request, and returns the observed response plus the
//! worker path for assertions.

use http_body_util::BodyExt;
use mesh::{
    config::{BackendType, RouterConfig},
    core::Job,
    routers::{RouterFactory, RouterTrait},
    tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry},
};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;

use crate::common::{create_test_context, create_test_context_with_parsers, test_app};

use super::{
    golden_assert::{assert_any_json_contains, GoldenAssert},
    mock_test_case::{BackendFixture, ConnectionModeFixture, MockTestCase, WorkerKindFixture},
    VirtualGrpcWorker, VirtualRequest, VirtualWorker,
};

/// Result observed after one fixture has gone through Atomesh.
#[derive(Debug)]
pub struct TestHarnessResult {
    pub status: u16,
    pub body: Value,
    pub stream_events: Vec<Value>,
    pub worker_path: Vec<String>,
    pub router_mode: String,
    pub connection_mode: String,
    pub policy: String,
    pub worker_urls: Vec<String>,
    pub registered_workers: usize,
    pub healthy_workers: usize,
}

impl TestHarnessResult {
    /// Assert HTTP status and stable response fields against the fixture.
    pub fn assert_response(&self, case: &MockTestCase) {
        if case.is_streaming() {
            assert_eq!(self.status, case.expected_response.status);
            assert_any_json_contains(&self.stream_events, &case.expected_response.body);
            return;
        }

        GoldenAssert {
            expected_status: case.expected_response.status,
            expected_body: case.expected_response.body.clone(),
        }
        .assert_response(self.status, &self.body);
    }

    /// Assert Atomesh actually routed to the expected virtual worker endpoint.
    pub fn assert_worker_path_contains(&self, endpoint: &str) {
        assert!(
            self.worker_path.iter().any(|path| path == endpoint),
            "worker path {:?} did not contain {}",
            self.worker_path,
            endpoint
        );
    }

    /// Assert Atomesh routed to the same virtual worker endpoint multiple times.
    pub fn assert_worker_path_count_at_least(&self, endpoint: &str, expected_count: usize) {
        let actual_count = self
            .worker_path
            .iter()
            .filter(|path| path.as_str() == endpoint)
            .count();
        assert!(
            actual_count >= expected_count,
            "worker path {:?} contained {} only {} times, expected at least {}",
            self.worker_path,
            endpoint,
            actual_count,
            expected_count
        );
    }

    /// Assert the runtime router and worker-pool state implied by the fixture.
    ///
    /// These checks protect the harness from silently exercising the wrong
    /// Atomesh branch. A response can be correct even if the fixture accidentally
    /// built a regular router instead of PD, or HTTP instead of gRPC; this
    /// assertion ties the observed runtime config and registry state back to the
    /// route metadata.
    pub fn assert_runtime_state(&self, case: &MockTestCase) {
        let expected_worker_count = match case.route.worker_kind {
            WorkerKindFixture::Regular => 1,
            WorkerKindFixture::PrefillDecode => 2,
        };

        match case.route.worker_kind {
            WorkerKindFixture::Regular => assert!(
                self.router_mode.starts_with("Regular"),
                "router mode {:?} did not match regular fixture {}",
                self.router_mode,
                case.name
            ),
            WorkerKindFixture::PrefillDecode => assert!(
                self.router_mode.starts_with("PrefillDecode"),
                "router mode {:?} did not match PD fixture {}",
                self.router_mode,
                case.name
            ),
        }

        match case.route.connection_mode {
            ConnectionModeFixture::Http => assert_eq!(self.connection_mode, "Http"),
            ConnectionModeFixture::Grpc => assert!(
                self.connection_mode.starts_with("Grpc"),
                "connection mode {:?} did not match gRPC fixture {}",
                self.connection_mode,
                case.name
            ),
        }

        assert_eq!(
            self.worker_urls.len(),
            expected_worker_count,
            "fixture {} expected {} worker URLs, got {:?}",
            case.name,
            expected_worker_count,
            self.worker_urls
        );
        assert_eq!(
            self.registered_workers, expected_worker_count,
            "fixture {} registered unexpected worker count",
            case.name
        );
        assert_eq!(
            self.healthy_workers, expected_worker_count,
            "fixture {} healthy worker count did not match expected pool size",
            case.name
        );
    }
}

/// Runs one fixture case through a real Atomesh app and virtual backend pool.
pub struct TestHarness {
    case: MockTestCase,
}

enum StartedVirtualWorker {
    Http(VirtualWorker),
    Grpc(VirtualGrpcWorker),
}

impl StartedVirtualWorker {
    fn url(&self) -> String {
        match self {
            Self::Http(worker) => worker.url.clone().unwrap(),
            Self::Grpc(worker) => worker.url.clone().unwrap(),
        }
    }

    fn request_log(&self) -> Vec<String> {
        match self {
            Self::Http(worker) => worker.request_log(),
            Self::Grpc(worker) => worker.request_log(),
        }
    }

    async fn stop(&mut self) {
        match self {
            Self::Http(worker) => worker.stop().await,
            Self::Grpc(worker) => worker.stop().await,
        }
    }
}

impl TestHarness {
    pub fn new(case: MockTestCase) -> Self {
        Self { case }
    }

    pub fn case(&self) -> &MockTestCase {
        &self.case
    }

    /// Execute the full fixture flow through Atomesh and virtual workers.
    pub async fn run(&self) -> Result<TestHarnessResult, Box<dyn std::error::Error>> {
        // Stage 1: start virtual backend workers from fixture route metadata.
        // Regular mode starts one worker; PD mode starts a prefill/decode pair.
        let mut workers = self.start_virtual_workers().await?;
        let worker_urls = workers
            .iter()
            .map(StartedVirtualWorker::url)
            .collect::<Vec<_>>();

        // Stage 2: build the same app context pieces Atomesh uses at runtime,
        // then submit the production worker-initialization job. This is what
        // triggers Atomesh's own worker health/model-info probes and registry
        // insertion instead of manually registering fake workers.
        let config = self.router_config(&worker_urls);
        let app_context = match self.case.route.connection_mode {
            ConnectionModeFixture::Http => create_test_context(config.clone()).await,
            ConnectionModeFixture::Grpc => {
                let app_context = create_test_context_with_parsers(config.clone()).await;
                register_mock_tokenizer(&app_context.tokenizer_registry, &self.case.model).await?;
                app_context
            }
        };
        initialize_workers(&app_context, &config, worker_urls.len()).await?;

        // Capture runtime router state from the actual config/context used by
        // RouterFactory, not from the fixture declaration.
        let router_mode = format!("{:?}", config.mode);
        let connection_mode = format!("{:?}", config.connection_mode);
        let policy = format!("{:?}", config.policy);
        let registered_workers = app_context.worker_registry.len();
        let healthy_workers = app_context
            .worker_registry
            .get_all()
            .iter()
            .filter(|worker| worker.is_healthy())
            .count();

        // Stage 3: create the real router and Axum app from the initialized
        // context. Requests below enter through `server::build_app`, not by
        // directly calling router internals.
        let router = RouterFactory::create_router(&app_context).await?;
        let router: Arc<dyn RouterTrait> = Arc::from(router);
        let app = test_app::create_test_app_with_context(router, app_context);

        // Stage 4: convert the fixture into a real Atomesh API request and send
        // it through the app. Atomesh will select workers and forward the
        // request to the virtual backend using the fixture's connection mode.
        let request = VirtualRequest::from_case(&self.case).into_axum_request();
        let response = app.oneshot(request).await?;

        // Stage 5: collect the response into assertion-friendly forms. For
        // streaming responses we keep parsed SSE JSON events in addition to the
        // raw response body fallback.
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await?.to_bytes();
        let response_text = String::from_utf8_lossy(&bytes).to_string();
        let stream_events = parse_sse_events(&response_text);
        let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(response_text));

        // Stage 6: collect the worker-side request log. This includes startup
        // and registration probes (`/health`, `/get_model_info`) plus the
        // business endpoint, so tests can assert routing shape when needed.
        let worker_path = workers
            .iter()
            .flat_map(StartedVirtualWorker::request_log)
            .collect::<Vec<_>>();

        // Stage 7: stop virtual workers before returning the result. Keeping
        // teardown inside the harness keeps individual fixture tests small.
        for worker in &mut workers {
            worker.stop().await;
        }

        Ok(TestHarnessResult {
            status,
            body,
            stream_events,
            worker_path,
            router_mode,
            connection_mode,
            policy,
            worker_urls,
            registered_workers,
            healthy_workers,
        })
    }

    async fn start_virtual_workers(
        &self,
    ) -> Result<Vec<StartedVirtualWorker>, Box<dyn std::error::Error>> {
        // The fixture's worker kind decides the topology under test.
        let worker_count = match self.case.route.worker_kind {
            WorkerKindFixture::Regular => 1,
            WorkerKindFixture::PrefillDecode => 2,
        };
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let worker = match self.case.route.connection_mode {
                ConnectionModeFixture::Http => {
                    let mut worker = VirtualWorker::new(self.case.clone());
                    worker.start().await?;
                    StartedVirtualWorker::Http(worker)
                }
                ConnectionModeFixture::Grpc => {
                    let mut worker = VirtualGrpcWorker::new(self.case.clone())?;
                    worker.start().await?;
                    StartedVirtualWorker::Grpc(worker)
                }
            };
            workers.push(worker);
        }

        Ok(workers)
    }

    fn router_config(&self, worker_urls: &[String]) -> RouterConfig {
        // Keep the harness config minimal and deterministic: HTTP transport,
        // round-robin policy, no retries, and random local router port.
        let mut builder = RouterConfig::builder()
            .backend(match self.case.route.backend {
                BackendFixture::Sglang => BackendType::Sglang,
                BackendFixture::Vllm => BackendType::Vllm,
            })
            .host("127.0.0.1")
            .port(portpicker::pick_unused_port().expect("no free port for test router"))
            .round_robin_policy()
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(30)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_retries();

        builder = match self.case.route.connection_mode {
            ConnectionModeFixture::Http => builder.http_connection(),
            ConnectionModeFixture::Grpc => builder.grpc_connection_default(),
        };

        builder = match self.case.route.worker_kind {
            WorkerKindFixture::Regular => builder.regular_mode(worker_urls.to_vec()),
            WorkerKindFixture::PrefillDecode => builder.prefill_decode_mode(
                vec![(worker_urls[0].clone(), None)],
                vec![worker_urls[1].clone()],
            ),
        };

        builder.build_unchecked()
    }
}

fn parse_sse_events(response_text: &str) -> Vec<Value> {
    // Axum's collected SSE body is line-oriented (`data: ...`). Ignore
    // terminal `[DONE]` markers and keep only JSON payloads for golden asserts.
    response_text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .collect()
}

async fn register_mock_tokenizer(
    tokenizer_registry: &Arc<TokenizerRegistry>,
    model: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_id = TokenizerRegistry::generate_id();
    tokenizer_registry
        .load(&tokenizer_id, model, "mock-tokenizer", || async {
            Ok(Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>)
        })
        .await?;
    Ok(())
}

async fn initialize_workers(
    app_context: &Arc<mesh::app_context::AppContext>,
    config: &RouterConfig,
    expected_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if expected_count == 0 {
        return Ok(());
    }

    // Worker initialization is intentionally performed through the job queue so
    // tests exercise the same registration workflow as server startup.
    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    job_queue
        .submit(Job::InitializeWorkersFromConfig {
            router_config: Box::new(config.clone()),
        })
        .await?;

    // Use the same registry readiness signal as production startup instead of
    // sleeping for a fixed duration.
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(10);
    loop {
        let healthy_workers = app_context
            .worker_registry
            .get_all()
            .iter()
            .filter(|worker| worker.is_healthy())
            .count();

        if healthy_workers >= expected_count {
            return Ok(());
        }

        if tokio::time::Instant::now() > deadline {
            return Err(format!(
                "timed out waiting for {} virtual workers, only {} ready",
                expected_count, healthy_workers
            )
            .into());
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }
}
