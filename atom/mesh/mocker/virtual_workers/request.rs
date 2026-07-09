//! Builds and executes real Atomesh HTTP requests from fixture data.
//!
//! The harness should exercise Atomesh's public API surface, not call router
//! internals directly. `VirtualRequest` is the small adapter from
//! `MockCase` JSON into either an Axum request for in-process harness tests
//! or a real HTTP POST for the standalone mocker CLI.

use std::{
    future,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{body::Body, extract::Request, http::header::CONTENT_TYPE};
use serde_json::{Map, Value};
use tokio::sync::{mpsc, Mutex};

use super::{
    any_json_contains,
    req_metrics::{run_metrics_printer, VirtualRequestMetrics},
    GoldenAssert, MockCase,
};

type RequestResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtualRequestMode {
    Http,
    Grpc,
}

/// Atomesh endpoint category used to build request URLs consistently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum VirtualRequestEndpoint {
    Generate = 1,
    ChatCompletions = 2,
    Completions = 3,
    Responses = 4,
}

impl VirtualRequestEndpoint {
    pub fn from_path(endpoint: &str) -> Option<Self> {
        match endpoint {
            "/generate" => Some(Self::Generate),
            "/v1/chat/completions" => Some(Self::ChatCompletions),
            "/v1/completions" => Some(Self::Completions),
            "/v1/responses" => Some(Self::Responses),
            _ => None,
        }
    }

    pub fn from_code(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::Generate),
            2 => Some(Self::ChatCompletions),
            3 => Some(Self::Completions),
            4 => Some(Self::Responses),
            _ => None,
        }
    }

    pub fn path(&self) -> &'static str {
        match self {
            Self::Generate => "/generate",
            Self::ChatCompletions => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
            Self::Responses => "/v1/responses",
        }
    }

    fn requires_model(&self) -> bool {
        matches!(
            self,
            Self::ChatCompletions | Self::Completions | Self::Responses
        )
    }
}

/// Client-facing request generated from a fixture.
#[derive(Clone, Debug)]
pub struct VirtualRequest {
    pub endpoint: u16,
    pub body: Value,
}

impl VirtualRequest {
    /// Convert fixture request data into the shape expected by Atomesh routes.
    pub fn from_case(case: &MockCase) -> Self {
        let endpoint_type = VirtualRequestEndpoint::from_path(&case.endpoint)
            .unwrap_or_else(|| panic!("unsupported virtual request endpoint `{}`", case.endpoint));
        let mut body = match case.request.clone() {
            Value::Object(map) => map,
            _ => Map::new(),
        };

        if endpoint_type.requires_model() {
            // OpenAI-compatible routes require a model; fixtures may omit it
            // when they want to reuse the top-level `model` field.
            body.entry("model".to_string())
                .or_insert_with(|| Value::String(case.model.clone()));
        }

        Self {
            endpoint: endpoint_type as u16,
            body: Value::Object(body),
        }
    }

    /// Turn the virtual request into an Axum request for `oneshot`.
    pub fn into_axum_request(self) -> Request<Body> {
        let endpoint = self.endpoint_path();
        Request::builder()
            .method("POST")
            .uri(endpoint)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&self.body).unwrap()))
            .unwrap()
    }

    pub fn endpoint_type(&self) -> VirtualRequestEndpoint {
        VirtualRequestEndpoint::from_code(self.endpoint).unwrap_or_else(|| {
            panic!(
                "unsupported virtual request endpoint code `{}`",
                self.endpoint
            )
        })
    }

    pub fn endpoint_path(&self) -> &'static str {
        self.endpoint_type().path()
    }

    /// POST this virtual request to a running Atomesh HTTP endpoint.
    pub async fn post(
        &self,
        client: &reqwest::Client,
        base_url: &str,
    ) -> RequestResult<VirtualResponse> {
        self.post_with_host(client, base_url, None).await
    }

    /// POST this virtual request with an optional HTTP Host header override.
    pub async fn post_with_host(
        &self,
        client: &reqwest::Client,
        base_url: &str,
        host_header: Option<&str>,
    ) -> RequestResult<VirtualResponse> {
        let endpoint = self.endpoint_type();
        let url = build_post_url(base_url, endpoint.path());
        let mut request = client.post(url).json(&self.body);
        if let Some(host_header) = host_header {
            request = request.header(reqwest::header::HOST, host_header);
        }
        let response = request.send().await?;
        VirtualResponse::from_response(response).await
    }

    /// POST this request and validate the response against the fixture.
    pub async fn post_and_assert(
        client: &reqwest::Client,
        base_url: &str,
        case: &MockCase,
    ) -> RequestResult<VirtualResponse> {
        let request = Self::from_case(case);
        let response = request.post(client, base_url).await?;
        response.assert_matches(case)?;
        Ok(response)
    }
}

/// HTTP response observed from a real Atomesh POST.
#[derive(Clone, Debug)]
pub struct VirtualResponse {
    pub status: u16,
    pub body: Value,
    pub stream_events: Vec<Value>,
    pub text: String,
}

impl VirtualResponse {
    async fn from_response(response: reqwest::Response) -> RequestResult<Self> {
        let status = response.status().as_u16();
        let text = response.text().await?;
        Ok(Self::from_parts(status, text))
    }

    pub(crate) fn from_parts(status: u16, text: String) -> Self {
        let stream_events = parse_sse_events(&text);
        let body = serde_json::from_str(&text).unwrap_or_else(|_| Value::String(text.clone()));

        Self {
            status,
            body,
            stream_events,
            text,
        }
    }

    pub fn assert_matches(&self, case: &MockCase) -> RequestResult<()> {
        let result = if case.is_streaming() {
            if self.status != case.expected_response.status {
                Err(format!(
                    "expected status {}, got {} with body {}",
                    case.expected_response.status, self.status, self.body
                ))
            } else {
                any_json_contains(&self.stream_events, &case.expected_response.body)
            }
        } else {
            GoldenAssert {
                expected_status: case.expected_response.status,
                expected_body: case.expected_response.body.clone(),
            }
            .validate_response(self.status, &self.body)
        };

        result.map_err(|error| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("fixture {} failed response validation: {}", case.name, error),
            )) as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

#[derive(Clone, Debug)]
pub struct VirtualRequestPipelineConfig {
    pub base_url: String,
    pub mode: VirtualRequestMode,
    pub host_header: Option<String>,
    pub tls_ca_cert_path: Option<PathBuf>,
    pub tls_accept_invalid_certs: bool,
    pub duration: Option<Duration>,
    pub producer_threads: usize,
    pub consumer_threads: usize,
    pub queue_capacity: usize,
}

impl VirtualRequestPipelineConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            mode: VirtualRequestMode::Http,
            host_header: None,
            tls_ca_cert_path: None,
            tls_accept_invalid_certs: false,
            duration: None,
            producer_threads: 1,
            consumer_threads: 1,
            queue_capacity: 64,
        }
    }

    pub fn host_header(mut self, host_header: Option<impl Into<String>>) -> Self {
        self.host_header = host_header.map(Into::into);
        self
    }

    pub fn tls_ca_cert_path(mut self, path: Option<impl Into<PathBuf>>) -> Self {
        self.tls_ca_cert_path = path.map(Into::into);
        self
    }

    pub fn tls_accept_invalid_certs(mut self, accept_invalid_certs: bool) -> Self {
        self.tls_accept_invalid_certs = accept_invalid_certs;
        self
    }

    pub fn duration(mut self, duration: Option<Duration>) -> Self {
        self.duration = duration;
        self
    }

    pub fn mode(mut self, mode: VirtualRequestMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn producer_threads(mut self, producer_threads: usize) -> Self {
        self.producer_threads = producer_threads.max(1);
        self
    }

    pub fn consumer_threads(mut self, consumer_threads: usize) -> Self {
        self.consumer_threads = consumer_threads.max(1);
        self
    }

    pub fn queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.queue_capacity = queue_capacity.max(1);
        self
    }
}

#[derive(Clone, Debug)]
pub struct VirtualRequestPipelineResult {
    pub endpoint: VirtualRequestEndpoint,
    pub response: VirtualResponse,
}

#[derive(Clone)]
pub struct VirtualRequestPipeline {
    config: VirtualRequestPipelineConfig,
    client: reqwest::Client,
}

impl VirtualRequestPipeline {
    pub fn new(config: VirtualRequestPipelineConfig) -> Self {
        Self::try_new(config).expect("failed to build virtual request HTTP client")
    }

    pub fn try_new(config: VirtualRequestPipelineConfig) -> RequestResult<Self> {
        let client = build_client(&config)?;
        Ok(Self {
            config,
            client,
        })
    }

    /// Run fixture cases through a producer-consumer request pipeline.
    pub async fn run_cases(
        &self,
        cases: Vec<MockCase>,
    ) -> RequestResult<Vec<VirtualRequestPipelineResult>> {
        if cases.is_empty() {
            return Ok(Vec::new());
        }

        let cases = Arc::new(cases);
        let running = Arc::new(AtomicBool::new(true));
        let (job_tx, job_rx) = mpsc::channel::<VirtualRequestJob>(self.config.queue_capacity);
        let (result_tx, mut result_rx) =
            mpsc::channel::<RequestResult<VirtualRequestPipelineResult>>(
                self.config.queue_capacity,
            );
        let metrics = Arc::new(VirtualRequestMetrics::new());
        let metrics_handle = tokio::spawn(run_metrics_printer(
            Arc::clone(&metrics),
            Arc::clone(&running),
        ));

        let mut producer_handles = Vec::with_capacity(self.config.producer_threads);
        for producer_index in 0..self.config.producer_threads {
            let cases = Arc::clone(&cases);
            let running = Arc::clone(&running);
            let job_tx = job_tx.clone();
            producer_handles.push(tokio::spawn(async move {
                let mut case_index = producer_index;
                loop {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }

                    let case = cases[case_index % cases.len()].clone();
                    case_index += 1;
                    let request = VirtualRequest::from_case(&case);
                    if job_tx.send(VirtualRequestJob { case, request }).await.is_err() {
                        break;
                    }
                }
            }));
        }
        drop(job_tx);

        let job_rx = Arc::new(Mutex::new(job_rx));
        let mut consumer_handles = Vec::with_capacity(self.config.consumer_threads);
        for _ in 0..self.config.consumer_threads {
            let base_url = self.config.base_url.clone();
            let mode = self.config.mode;
            let host_header = self.config.host_header.clone();
            let client = self.client.clone();
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let metrics = Arc::clone(&metrics);
            consumer_handles.push(tokio::spawn(async move {
                loop {
                    let job = {
                        let mut job_rx = job_rx.lock().await;
                        job_rx.recv().await
                    };
                    let Some(job) = job else {
                        break;
                    };

                    let result =
                        execute_job(&client, mode, &base_url, host_header.as_deref(), job, &metrics)
                            .await;
                    if result_tx.send(result).await.is_err() {
                        break;
                    }
                }
            }));
        }
        drop(result_tx);

        let results = Vec::new();
        let mut first_error = None;
        let mut shutdown_requested = false;
        let duration = self.config.duration;
        let duration_sleep = async move {
            if let Some(duration) = duration {
                tokio::time::sleep(duration).await;
            } else {
                future::pending::<()>().await;
            }
        };
        tokio::pin!(duration_sleep);
        loop {
            tokio::select! {
                biased;
                _ = &mut duration_sleep => {
                    running.store(false, Ordering::Relaxed);
                    shutdown_requested = true;
                    break;
                }
                _ = tokio::signal::ctrl_c() => {
                    running.store(false, Ordering::Relaxed);
                    shutdown_requested = true;
                    break;
                }
                result = result_rx.recv() => {
                    let Some(result) = result else {
                        break;
                    };
                    match result {
                        Ok(_) => {}
                        Err(error) if first_error.is_none() => first_error = Some(error),
                        Err(_) => {}
                    }
                }
            }
        }

        running.store(false, Ordering::Relaxed);
        drop(result_rx);
        for handle in &producer_handles {
            handle.abort();
        }
        for handle in &consumer_handles {
            handle.abort();
        }

        for handle in producer_handles {
            let _ = handle.await;
        }
        for handle in consumer_handles {
            let _ = handle.await;
        }

        metrics.print();
        metrics_handle.abort();
        let _ = metrics_handle.await;
        if !shutdown_requested {
            if let Some(error) = first_error {
                return Err(error);
            }
        }
        Ok(results)
    }
}

fn build_client(config: &VirtualRequestPipelineConfig) -> RequestResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder();

    if config.tls_accept_invalid_certs {
        builder = builder.danger_accept_invalid_certs(true);
    }

    if let Some(path) = &config.tls_ca_cert_path {
        let pem = std::fs::read(path).map_err(|error| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "failed to read TLS CA certificate '{}': {}",
                    path.display(),
                    error
                ),
            )) as Box<dyn std::error::Error + Send + Sync>
        })?;
        let certificate = reqwest::Certificate::from_pem(&pem).map_err(|error| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "failed to parse TLS CA certificate '{}': {}",
                    path.display(),
                    error
                ),
            )) as Box<dyn std::error::Error + Send + Sync>
        })?;
        builder = builder.add_root_certificate(certificate);
    }

    builder
        .build()
        .map_err(|error| Box::new(error) as Box<dyn std::error::Error + Send + Sync>)
}

struct VirtualRequestJob {
    case: MockCase,
    request: VirtualRequest,
}

async fn execute_job(
    client: &reqwest::Client,
    mode: VirtualRequestMode,
    base_url: &str,
    host_header: Option<&str>,
    job: VirtualRequestJob,
    metrics: &VirtualRequestMetrics,
) -> RequestResult<VirtualRequestPipelineResult> {
    let endpoint_code = job.request.endpoint;
    let endpoint = job.request.endpoint_type();
    let started_at = Instant::now();
    let response_result = match mode {
        VirtualRequestMode::Http => {
            job.request
                .post_with_host(client, base_url, host_header)
                .await
        }
        VirtualRequestMode::Grpc => Err("grpc benchmark request mode is not implemented".into()),
    };
    let duration = started_at.elapsed();
    let response_result = response_result.and_then(|response| {
        response.assert_matches(&job.case)?;
        Ok(response)
    });
    metrics.record(endpoint_code, duration, response_result.is_ok());

    let response = response_result?;
    Ok(VirtualRequestPipelineResult {
        endpoint,
        response,
    })
}

fn build_post_url(base_url: &str, endpoint: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        endpoint.trim_start_matches('/')
    )
}

fn parse_sse_events(response_text: &str) -> Vec<Value> {
    response_text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .collect()
}

