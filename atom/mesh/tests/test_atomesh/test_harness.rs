//! End-to-end harness for fixture-driven Atomesh tests.
//!
//! `TestHarness` starts virtual backend workers, registers them through the
//! normal worker initialization path, builds the real Axum app, sends a
//! fixture-generated client request, and returns the observed response plus the
//! worker path for assertions.

use http_body_util::BodyExt;
use mesh::{
    config::RouterConfig,
    core::Job,
    routers::{RouterFactory, RouterTrait},
};
use serde_json::Value;
use tower::ServiceExt;

use crate::common::{create_test_context, test_app};

use super::{
    golden_assert::GoldenAssert,
    mock_test_case::{MockTestCase, WorkerKindFixture},
    VirtualRequest, VirtualWorker,
};

/// Result observed after one fixture has gone through Atomesh.
#[derive(Debug)]
pub struct TestHarnessResult {
    pub status: u16,
    pub body: Value,
    pub worker_path: Vec<String>,
}

impl TestHarnessResult {
    /// Assert HTTP status and stable response fields against the fixture.
    pub fn assert_response(&self, case: &MockTestCase) {
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
}

/// Runs one fixture case through a real Atomesh app and virtual backend pool.
pub struct TestHarness {
    case: MockTestCase,
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
        if !matches!(
            self.case.route.connection_mode,
            super::mock_test_case::ConnectionModeFixture::Http
        ) {
            return Err("TestHarness currently supports HTTP fixtures only".into());
        }

        let mut workers = self.start_virtual_workers().await?;
        let worker_urls = workers
            .iter()
            .map(|worker| worker.url.clone().unwrap())
            .collect::<Vec<_>>();
        let config = self.router_config(&worker_urls);
        let app_context = create_test_context(config.clone()).await;
        initialize_workers(&app_context, &config, worker_urls.len()).await?;
        let router = RouterFactory::create_router(&app_context).await?;
        let router: std::sync::Arc<dyn RouterTrait> = std::sync::Arc::from(router);
        let app = test_app::create_test_app_with_context(router, app_context);

        let request = VirtualRequest::from_case(&self.case).into_axum_request();
        let response = app.oneshot(request).await?;
        let status = response.status().as_u16();
        let bytes = response.into_body().collect().await?.to_bytes();
        let body = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()));

        let worker_path = workers
            .iter()
            .flat_map(VirtualWorker::request_log)
            .collect::<Vec<_>>();

        for worker in &mut workers {
            worker.stop().await;
        }

        Ok(TestHarnessResult {
            status,
            body,
            worker_path,
        })
    }

    async fn start_virtual_workers(
        &self,
    ) -> Result<Vec<VirtualWorker>, Box<dyn std::error::Error>> {
        let worker_count = match self.case.route.worker_kind {
            WorkerKindFixture::Regular => 1,
            WorkerKindFixture::PrefillDecode => 2,
        };
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let mut worker = VirtualWorker::new(self.case.clone());
            worker.start().await?;
            workers.push(worker);
        }

        Ok(workers)
    }

    fn router_config(&self, worker_urls: &[String]) -> RouterConfig {
        let mut builder = RouterConfig::builder()
            .host("127.0.0.1")
            .port(portpicker::pick_unused_port().expect("no free port for test router"))
            .round_robin_policy()
            .http_connection()
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(30)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .disable_retries();

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

async fn initialize_workers(
    app_context: &std::sync::Arc<mesh::app_context::AppContext>,
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
