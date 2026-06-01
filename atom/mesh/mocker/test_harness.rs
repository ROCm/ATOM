//! End-to-end harness for fixture-driven Atomesh mocker tests.

use std::sync::Arc;

use http_body_util::BodyExt;
use mesh::{
    config::{BackendType, RouterConfig},
    core::Job,
    routers::{RouterFactory, RouterTrait},
    tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry},
};
use serde_json::Value;
use tower::ServiceExt;

use crate::{
    app_helpers::{
        create_mocker_app_with_context, create_mocker_context,
        create_mocker_context_with_parsers,
    },
    BackendFixture, ConnectionModeFixture, MockCase, VirtualRequest, VirtualResponse,
    VirtualWorkerPool, VirtualWorkerPoolConfig, VirtualWorkerSpec, WorkerKindFixture,
};

/// Result observed after one fixture has gone through Atomesh.
#[derive(Debug)]
pub struct TestHarnessResult {
    pub status: u16,
    pub body: Value,
    pub stream_events: Vec<Value>,
    pub response: VirtualResponse,
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
    pub fn assert_response(&self, case: &MockCase) {
        self.response.assert_matches(case).unwrap();
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
    pub fn assert_runtime_state(&self, case: &MockCase) {
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
            case.name,
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
    case: MockCase,
}

impl TestHarness {
    pub fn new(case: MockCase) -> Self {
        Self { case }
    }

    pub fn case(&self) -> &MockCase {
        &self.case
    }

    /// Execute the full fixture flow through Atomesh and virtual workers.
    pub async fn run(&self) -> Result<TestHarnessResult, Box<dyn std::error::Error>> {
        let mut workers = self.start_virtual_worker_pool().await?;
        let result = self.run_with_worker_pool(&workers).await;
        workers.stop().await;
        result
    }

    async fn run_with_worker_pool(
        &self,
        workers: &VirtualWorkerPool,
    ) -> Result<TestHarnessResult, Box<dyn std::error::Error>> {
        let worker_urls = workers.urls();

        let config = self.router_config(&worker_urls);
        let app_context = match self.case.route.connection_mode {
            ConnectionModeFixture::Http => create_mocker_context(config.clone()).await,
            ConnectionModeFixture::Grpc => {
                let app_context = create_mocker_context_with_parsers(config.clone()).await;
                register_mock_tokenizer(&app_context.tokenizer_registry, &self.case.model).await?;
                app_context
            }
        };
        initialize_workers(&app_context, &config, worker_urls.len()).await?;

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

        let router = RouterFactory::create_router(&app_context).await?;
        let router: Arc<dyn RouterTrait> = Arc::from(router);
        let app = create_mocker_app_with_context(router, app_context);

        let request = VirtualRequest::from_case(&self.case).into_axum_request();
        let response = app.oneshot(request).await?;

        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await?.to_bytes();
        let response_text = String::from_utf8_lossy(&bytes).to_string();
        let response = VirtualResponse::from_parts(status, response_text);

        Ok(TestHarnessResult {
            status,
            body: response.body.clone(),
            stream_events: response.stream_events.clone(),
            response,
            worker_path: workers.request_logs(),
            router_mode,
            connection_mode,
            policy,
            worker_urls,
            registered_workers,
            healthy_workers,
        })
    }

    async fn start_virtual_worker_pool(
        &self,
    ) -> Result<VirtualWorkerPool, Box<dyn std::error::Error>> {
        let worker_count = match self.case.route.worker_kind {
            WorkerKindFixture::Regular => 1,
            WorkerKindFixture::PrefillDecode => 2,
        };
        let base_port = pick_base_port(worker_count).expect("no free base port for worker pool");
        let specs = (0..worker_count)
            .map(|_| VirtualWorkerSpec::from_case(self.case.clone()))
            .collect();

        VirtualWorkerPool::start(
            VirtualWorkerPoolConfig::new("127.0.0.1", base_port),
            specs,
        )
        .await
    }

    fn router_config(&self, worker_urls: &[String]) -> RouterConfig {
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

fn pick_base_port(worker_count: usize) -> Option<u16> {
    for _ in 0..32 {
        let base_port = portpicker::pick_unused_port()?;
        let last_port =
            base_port.checked_add(u16::try_from(worker_count.saturating_sub(1)).ok()?)?;
        let listeners = (base_port..=last_port)
            .map(|port| std::net::TcpListener::bind(("127.0.0.1", port)))
            .collect::<Result<Vec<_>, _>>();
        if listeners.is_ok() {
            return Some(base_port);
        }
    }
    None
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

    let job_queue = app_context
        .worker_job_queue
        .get()
        .expect("JobQueue should be initialized");
    job_queue
        .submit(Job::InitializeWorkersFromConfig {
            router_config: Box::new(config.clone()),
        })
        .await?;

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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{MockCase, TestHarness, TestHarnessResult};

    fn fixture_path(file_name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(file_name)
    }

    fn load_fixture_case(file_name: &str) -> MockCase {
        MockCase::from_fixture(fixture_path(file_name)).unwrap()
    }

    fn log_case_result(case: &MockCase, result: &TestHarnessResult) {
        println!(
            concat!(
                "fixture_case={} endpoint={} status={} ",
                "router_mode={} connection_mode={} policy={} ",
                "worker_urls={:?} registered_workers={} healthy_workers={} ",
                "worker_path={:?} stream_events={}"
            ),
            case.name,
            case.endpoint,
            result.status,
            result.router_mode,
            result.connection_mode,
            result.policy,
            result.worker_urls,
            result.registered_workers,
            result.healthy_workers,
            result.worker_path,
            result.stream_events.len()
        );
    }

    fn assert_harness_runtime(case: &MockCase, result: &TestHarnessResult) {
        result.assert_runtime_state(case);
    }

    #[tokio::test]
    async fn test_atomesh_harness_http_regular_chat() {
        let case = load_fixture_case("http_regular_chat.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("/v1/chat/completions");
    }

    #[tokio::test]
    async fn test_atomesh_harness_http_regular_generate() {
        let case = load_fixture_case("http_regular_generate.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("/generate");
    }

    #[tokio::test]
    async fn test_atomesh_harness_http_regular_chat_streaming() {
        let case = load_fixture_case("http_regular_chat_streaming.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("/v1/chat/completions");
    }

    #[tokio::test]
    async fn test_atomesh_harness_http_regular_completion() {
        let case = load_fixture_case("http_regular_completion.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("/v1/completions");
    }

    #[tokio::test]
    async fn test_atomesh_harness_http_pd_chat() {
        let case = load_fixture_case("http_pd_chat.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_count_at_least("/v1/chat/completions", 2);
    }

    #[tokio::test]
    async fn test_atomesh_harness_grpc_regular_generate() {
        let case = load_fixture_case("grpc_regular_generate.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("generate");
    }

    #[tokio::test]
    async fn test_atomesh_harness_grpc_regular_generate_vllm() {
        let case = load_fixture_case("grpc_regular_generate_vllm.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_contains("generate");
    }

    #[tokio::test]
    async fn test_atomesh_harness_grpc_pd_generate() {
        let case = load_fixture_case("grpc_pd_generate.json");
        let harness = TestHarness::new(case.clone());

        let result = harness.run().await.unwrap();
        log_case_result(&case, &result);

        assert_harness_runtime(&case, &result);
        result.assert_response(&case);
        result.assert_worker_path_count_at_least("generate", 2);
    }
}
