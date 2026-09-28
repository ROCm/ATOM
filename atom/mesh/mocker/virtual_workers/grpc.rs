//! Tonic-backed virtual gRPC worker for Atomesh harness tests.
//!
//! This worker implements the backend-specific gRPC services expected by
//! Atomesh's gRPC routers. It remains fixture-driven like `VirtualWorker`, but
//! exercises the real gRPC client, worker registration, and router execution path.

use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures_util::{stream, Stream};
use mesh_grpc::{
    sglang_proto::{
        self as sglang,
        sglang_scheduler_server::{SglangScheduler, SglangSchedulerServer},
    },
    vllm_proto::{
        self as vllm,
        vllm_engine_server::{VllmEngine, VllmEngineServer},
    },
    SglangSchedulerClient, VllmEngineClient,
};
use serde_json::Value;
use tokio::sync::oneshot;
use tonic::{transport::Server, Request, Response, Status};

use super::{BackendFixture, ConnectionModeFixture, MockCase, WorkerKindFixture};

type SglangGenerateStream =
    Pin<Box<dyn Stream<Item = Result<sglang::GenerateResponse, Status>> + Send + 'static>>;
type VllmGenerateStream =
    Pin<Box<dyn Stream<Item = Result<vllm::GenerateResponse, Status>> + Send + 'static>>;

#[derive(Debug)]
pub struct VirtualGrpcWorker {
    case: MockCase,
    case_name: String,
    worker_kind: WorkerKindFixture,
    request_log: Arc<Mutex<Vec<String>>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    pub url: Option<String>,
}

impl VirtualGrpcWorker {
    /// Create a virtual gRPC worker for a single fixture case.
    pub fn new(case: MockCase) -> Result<Self, Box<dyn std::error::Error>> {
        if case.route.connection_mode != ConnectionModeFixture::Grpc {
            return Err("VirtualGrpcWorker requires a gRPC fixture".into());
        }

        Ok(Self {
            case: case.clone(),
            case_name: case.name.clone(),
            worker_kind: case.route.worker_kind.clone(),
            request_log: Arc::new(Mutex::new(Vec::new())),
            shutdown_tx: None,
            shutdown_handle: None,
            url: None,
        })
    }

    pub fn case_name(&self) -> &str {
        &self.case_name
    }

    pub fn worker_kind(&self) -> &WorkerKindFixture {
        &self.worker_kind
    }

    pub fn request_log(&self) -> Vec<String> {
        self.request_log.lock().unwrap().clone()
    }

    /// Start the tonic worker on a random local port and wait for readiness.
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let port = portpicker::pick_unused_port().expect("no free port for virtual gRPC worker");
        self.start_on("127.0.0.1", port).await
    }

    /// Start the tonic worker on a deterministic host and port.
    pub async fn start_on(
        &mut self,
        host: &str,
        port: u16,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
        self.start_with_addr(host, port, addr).await
    }

    async fn start_with_addr(
        &mut self,
        host: &str,
        port: u16,
        addr: SocketAddr,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("grpc://{}:{}", host, port);
        let service = VirtualGrpcServiceState {
            case: self.case.clone(),
            request_log: Arc::clone(&self.request_log),
        };

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let backend = self.case.route.backend.clone();
        let handle = match backend {
            BackendFixture::Sglang => tokio::spawn(async move {
                let server = Server::builder()
                    .add_service(SglangSchedulerServer::new(SglangVirtualGrpcService {
                        state: service,
                    }))
                    .serve_with_shutdown(addr, async move {
                        let _ = shutdown_rx.await;
                    });
                if let Err(error) = server.await {
                    eprintln!("virtual SGLang gRPC worker error on {}: {}", addr, error);
                }
            }),
            BackendFixture::Vllm => tokio::spawn(async move {
                let server = Server::builder()
                    .add_service(VllmEngineServer::new(VllmVirtualGrpcService {
                        state: service,
                    }))
                    .serve_with_shutdown(addr, async move {
                        let _ = shutdown_rx.await;
                    });
                if let Err(error) = server.await {
                    eprintln!("virtual vLLM gRPC worker error on {}: {}", addr, error);
                }
            }),
        };

        self.shutdown_tx = Some(shutdown_tx);
        self.shutdown_handle = Some(handle);
        self.url = Some(url.clone());
        wait_until_ready(&url, &self.case.route.backend).await?;
        Ok(url)
    }

    /// Stop the worker and wait briefly for the tonic server task to exit.
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for VirtualGrpcWorker {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

#[derive(Clone)]
struct VirtualGrpcServiceState {
    case: MockCase,
    request_log: Arc<Mutex<Vec<String>>>,
}

#[derive(Clone)]
struct SglangVirtualGrpcService {
    state: VirtualGrpcServiceState,
}

#[derive(Clone)]
struct VllmVirtualGrpcService {
    state: VirtualGrpcServiceState,
}

#[tonic::async_trait]
impl SglangScheduler for SglangVirtualGrpcService {
    type GenerateStream = SglangGenerateStream;

    async fn generate(
        &self,
        request: Request<sglang::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("generate".to_string());
        let request = request.into_inner();
        let complete = sglang::GenerateComplete {
            output_ids: expected_output_ids(&self.state.case.expected_response.body),
            finish_reason: "stop".to_string(),
            prompt_tokens: request
                .tokenized
                .as_ref()
                .map(|tokenized| tokenized.input_ids.len() as i32)
                .unwrap_or_default(),
            completion_tokens: 2,
            cached_tokens: 0,
            output_logprobs: None,
            all_hidden_states: Vec::new(),
            matched_stop: None,
            input_logprobs: None,
            index: 0,
        };
        let response = sglang::GenerateResponse {
            request_id: request.request_id,
            response: Some(sglang::generate_response::Response::Complete(complete)),
        };

        Ok(Response::new(Box::pin(stream::iter(vec![Ok(response)]))))
    }

    async fn embed(
        &self,
        _request: Request<sglang::EmbedRequest>,
    ) -> Result<Response<sglang::EmbedResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("embed".to_string());
        Err(Status::unimplemented(
            "virtual gRPC embed is not implemented",
        ))
    }

    async fn health_check(
        &self,
        _request: Request<sglang::HealthCheckRequest>,
    ) -> Result<Response<sglang::HealthCheckResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("health_check".to_string());
        Ok(Response::new(sglang::HealthCheckResponse {
            healthy: true,
            message: "healthy".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<sglang::AbortRequest>,
    ) -> Result<Response<sglang::AbortResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("abort".to_string());
        Ok(Response::new(sglang::AbortResponse {
            success: true,
            message: "aborted".to_string(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<sglang::GetModelInfoRequest>,
    ) -> Result<Response<sglang::GetModelInfoResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("get_model_info".to_string());
        Ok(Response::new(sglang::GetModelInfoResponse {
            model_path: self.state.case.model.clone(),
            tokenizer_path: "mock-tokenizer".to_string(),
            is_generation: true,
            preferred_sampling_params: String::new(),
            weight_version: "virtual".to_string(),
            served_model_name: self.state.case.model.clone(),
            max_context_length: 32768,
            vocab_size: 16,
            supports_vision: false,
            model_type: "virtual".to_string(),
            eos_token_ids: vec![999],
            pad_token_id: 0,
            bos_token_id: 1000,
            max_req_input_len: 32768,
            architectures: vec!["VirtualGrpcWorker".to_string()],
            id2label_json: String::new(),
            num_labels: 0,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<sglang::GetServerInfoRequest>,
    ) -> Result<Response<sglang::GetServerInfoResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("get_server_info".to_string());
        Ok(Response::new(sglang::GetServerInfoResponse {
            server_args: None,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            last_receive_timestamp: 0.0,
            uptime_seconds: 0.0,
            sglang_version: "virtual-worker".to_string(),
            server_type: "grpc".to_string(),
            start_time: None,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<sglang::GetLoadsRequest>,
    ) -> Result<Response<sglang::GetLoadsResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("get_loads".to_string());
        Ok(Response::new(sglang::GetLoadsResponse::default()))
    }
}

#[tonic::async_trait]
impl VllmEngine for VllmVirtualGrpcService {
    type GenerateStream = VllmGenerateStream;

    async fn generate(
        &self,
        request: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("generate".to_string());
        let request = request.into_inner();
        let complete = vllm::GenerateComplete {
            output_ids: expected_output_ids(&self.state.case.expected_response.body),
            finish_reason: "stop".to_string(),
            prompt_tokens: match request.input {
                Some(vllm::generate_request::Input::Tokenized(tokenized)) => {
                    tokenized.input_ids.len() as u32
                }
                _ => 0,
            },
            completion_tokens: 2,
            cached_tokens: 0,
        };
        let response = vllm::GenerateResponse {
            response: Some(vllm::generate_response::Response::Complete(complete)),
        };

        Ok(Response::new(Box::pin(stream::iter(vec![Ok(response)]))))
    }

    async fn embed(
        &self,
        _request: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("embed".to_string());
        Err(Status::unimplemented(
            "virtual vLLM embed is not implemented",
        ))
    }

    async fn health_check(
        &self,
        _request: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("health_check".to_string());
        Ok(Response::new(vllm::HealthCheckResponse {
            healthy: true,
            message: "healthy".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("abort".to_string());
        Ok(Response::new(vllm::AbortResponse {}))
    }

    async fn get_model_info(
        &self,
        _request: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("get_model_info".to_string());
        Ok(Response::new(vllm::GetModelInfoResponse {
            model_path: self.state.case.model.clone(),
            is_generation: true,
            max_context_length: 32768,
            vocab_size: 16,
            supports_vision: false,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        self.state
            .request_log
            .lock()
            .unwrap()
            .push("get_server_info".to_string());
        Ok(Response::new(vllm::GetServerInfoResponse {
            active_requests: 0,
            is_paused: false,
            last_receive_timestamp: 0.0,
            uptime_seconds: 0.0,
            server_type: "vllm-grpc".to_string(),
        }))
    }
}

async fn wait_until_ready(
    url: &str,
    backend: &BackendFixture,
) -> Result<(), Box<dyn std::error::Error>> {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(3);

    loop {
        match backend {
            BackendFixture::Sglang => {
                if let Ok(client) = SglangSchedulerClient::connect(url).await {
                    if let Ok(response) = client.health_check().await {
                        if response.healthy {
                            return Ok(());
                        }
                    }
                }
            }
            BackendFixture::Vllm => {
                if let Ok(client) = VllmEngineClient::connect(url).await {
                    if let Ok(response) = client.health_check().await {
                        if response.healthy {
                            return Ok(());
                        }
                    }
                }
            }
        }

        if tokio::time::Instant::now() > deadline {
            return Err(format!("virtual gRPC worker at {} did not become ready", url).into());
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
}

// The production gRPC path returns token ids, then Atomesh decodes them through
// the test tokenizer. Keep this mapping tiny and explicit so fixture text drives
// the observed API response without depending on a real tokenizer model.
fn expected_output_ids(body: &Value) -> Vec<u32> {
    let text = body
        .as_array()
        .and_then(|items| items.first())
        .and_then(|item| item.get("text"))
        .and_then(Value::as_str)
        .or_else(|| body.get("text").and_then(Value::as_str))
        .unwrap_or("Hello world");

    text.split_whitespace()
        .filter_map(|token| match token {
            "Hello" => Some(1),
            "world" => Some(2),
            "test" => Some(3),
            "token" => Some(4),
            _ => None,
        })
        .collect()
}
