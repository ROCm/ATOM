//! ATOM workers against an in-process `AtomEngine`: the request a payload becomes,
//! the stream's `TokenChunk`s, an error's HTTP status, and cancel on drop.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use http::StatusCode;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::{server::TcpIncoming, Server};
use tonic::{Request, Response, Status, Streaming};

use super::request::generate_request;
use crate::atom_engine::grpc::{
    atom_engine_server::{AtomEngine, AtomEngineServer},
    generate_response, AbortRequest, AbortResponse, Frames, GenerateComplete, GenerateError,
    GenerateRequest, GenerateResponse, GenerateStreamChunk, GetModelInfoRequest,
    GetModelInfoResponse, GetServerInfoRequest, GetServerInfoResponse, HealthCheckRequest,
    HealthCheckResponse, PrepareStructuredOutputRequest, PrepareStructuredOutputResponse,
};
use crate::atom_engine::wire::pb::{
    stop_strings, SamplingParams, StopStrings, StringList, TokenIds,
};
use crate::core::placement::PlacementPlan;
use crate::core::worker::{ConnectionMode, RuntimeType, Worker};
use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder};
use crate::observability::metrics::metrics_labels;
use crate::protocols::generate::GenerateRequest as GenerateBody;
use crate::routers::grpc::engine::{worker_client_cache::GrpcClient, GrpcEngine};
use crate::routers::prepare::{build_generate_payload, generation_payload::GenerationPayload};
use crate::routers::test_mocks;
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::{FinishReason, MatchedStop, TokenChunk};

type Reply = Result<GenerateResponse, Status>;

/// Each Generate call sends the next scripted replies, then ends its stream,
/// or with `hold` keeps it open and its sender in `held`.
#[derive(Default)]
struct FakeAtom {
    scripts: Mutex<VecDeque<Vec<GenerateResponse>>>,
    hold: bool,
    requests: Mutex<Vec<GenerateRequest>>,
    held: Mutex<Option<mpsc::Sender<Reply>>>,
}

impl FakeAtom {
    fn new(scripts: Vec<Vec<GenerateResponse>>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            ..Default::default()
        })
    }
}

#[tonic::async_trait]
impl AtomEngine for FakeAtom {
    type GenerateStream = ReceiverStream<Reply>;
    type SessionStream = ReceiverStream<Result<Frames, Status>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        self.requests.lock().unwrap().push(request.into_inner());
        let replies = self.scripts.lock().unwrap().pop_front().unwrap_or_default();
        let (tx, rx) = mpsc::channel(replies.len() + 1);
        for reply in replies {
            tx.try_send(Ok(reply)).unwrap();
        }
        if self.hold {
            *self.held.lock().unwrap() = Some(tx);
        }
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn abort(&self, _: Request<AbortRequest>) -> Result<Response<AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }

    async fn get_model_info(
        &self,
        _: Request<GetModelInfoRequest>,
    ) -> Result<Response<GetModelInfoResponse>, Status> {
        Ok(Response::new(GetModelInfoResponse {
            model_path: "/models/m".into(),
            tokenizer_path: "/models/m-tokenizer".into(),
            served_model_name: "m".into(),
            max_context_length: 4096,
            eos_token_ids: vec![2, 7],
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _: Request<GetServerInfoRequest>,
    ) -> Result<Response<GetServerInfoResponse>, Status> {
        Ok(Response::new(GetServerInfoResponse {
            protocol_version: 1,
            kv_transfer_info_json: r#"{"tp_size":1}"#.into(),
            server_info_json: r#"{"dp_size":2}"#.into(),
        }))
    }

    async fn health_check(
        &self,
        _: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: false,
            message: "engine core died".into(),
        }))
    }

    async fn session(
        &self,
        _: Request<Streaming<Frames>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        Err(Status::unimplemented("session"))
    }

    async fn prepare_structured_output(
        &self,
        _: Request<PrepareStructuredOutputRequest>,
    ) -> Result<Response<PrepareStructuredOutputResponse>, Status> {
        Err(Status::unimplemented("prepare_structured_output"))
    }
}

/// Serves `fake` on a free local port and returns its `grpc://` URL.
async fn serve(fake: Arc<FakeAtom>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let router = Server::builder().add_service(AtomEngineServer::from_arc(fake));
    tokio::spawn(router.serve_with_incoming(TcpIncoming::from(listener)));
    url
}

fn atom_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .connection_mode(ConnectionMode::Grpc { port: None })
            .runtime_type(RuntimeType::Atom)
            .build(),
    )
}

fn single(worker: Arc<dyn Worker>) -> PlacementPlan {
    PlacementPlan::Single {
        worker,
        policy_name: "random",
    }
}

/// A /generate body for prompt [7, 8, 9]; `sampling` is its `sampling_params`.
fn generate_body(stream: bool, sampling: serde_json::Value) -> Arc<GenerateBody> {
    let body = serde_json::json!({
        "rid": "atom-req", "input_ids": [7, 8, 9], "stream": stream, "sampling_params": sampling,
    });
    Arc::new(serde_json::from_value(body).unwrap())
}

fn payload(stream: bool) -> GenerationPayload {
    let body = generate_body(stream, serde_json::json!({}));
    build_generate_payload("atom-req".to_string(), &body, None, vec![7, 8, 9])
}

async fn collect(plan: &PlacementPlan, stream: bool) -> Vec<Result<TokenChunk, EngineError>> {
    let handle = GrpcEngine::new().dispatch(plan, &mut payload(stream)).await;
    handle.unwrap().collect().await
}

fn chunk(token_ids: &[u32]) -> GenerateResponse {
    GenerateResponse {
        response: Some(generate_response::Response::Chunk(GenerateStreamChunk {
            token_ids: token_ids.to_vec(),
            ..Default::default()
        })),
    }
}

fn complete(output_ids: &[u32], finish_reason: &str) -> GenerateResponse {
    GenerateResponse {
        response: Some(generate_response::Response::Complete(GenerateComplete {
            output_ids: output_ids.to_vec(),
            finish_reason: finish_reason.to_string(),
            prompt_tokens: 3,
            completion_tokens: 3,
            cached_tokens: 2,
            kv_transfer_params_json: None,
        })),
    }
}

fn error(message: &str, http_status: u32) -> GenerateResponse {
    GenerateResponse {
        response: Some(generate_response::Response::Error(GenerateError {
            message: message.to_string(),
            http_status,
        })),
    }
}

/// A /generate request through atomesh's gRPC pipeline to `worker`, with the
/// mock tokenizer registered for model "m".
async fn route_generate(worker: Arc<dyn Worker>, stream: bool) -> axum::response::Response {
    let pipeline = test_mocks::pipeline_with(
        Arc::new(test_mocks::MockPdPlanner::repeat_single(worker, "random")),
        Arc::new(GrpcEngine::new()),
        metrics_labels::BACKEND_REGULAR,
    );
    let context =
        test_mocks::app_context_with_tokenizer_registry(test_mocks::tokenizer_registry_with("m"));
    let sampling = serde_json::json!({
        "temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_new_tokens": 16, "ignore_eos": true,
        "stop": ["Hello world", "<|im_end|>", "unseen"],
    });
    let body = generate_body(stream, sampling);
    pipeline
        .execute_generate(body, None, Some("m".to_string()), context)
        .await
}

#[tokio::test]
async fn a_payload_becomes_the_generate_request_atom_receives() {
    let fake = FakeAtom::new(vec![vec![complete(&[4], "eos")]]);
    let url = serve(fake.clone()).await;

    let response = route_generate(atom_worker(&url), false).await;
    assert_eq!(response.status(), StatusCode::OK);

    let stops = ["Hello world", "<|im_end|>", "unseen"].map(String::from);
    let expected = GenerateRequest {
        request_id: "atom-req".to_string(),
        prompt_token_ids: vec![7, 8, 9],
        sampling: Some(SamplingParams {
            temperature: Some(0.7),
            top_k: Some(20),
            top_p: Some(0.9),
            max_tokens: Some(16),
            ignore_eos: true,
            stop_strings: Some(StopStrings {
                value: Some(stop_strings::Value::Many(StringList {
                    items: stops.to_vec(),
                })),
            }),
            n: Some(1),
            ..Default::default()
        }),
        // The mock tokenizer's ids; "unseen" encodes to nothing and is dropped.
        stop_token_sequences: vec![TokenIds { ids: vec![1, 2] }, TokenIds { ids: vec![1002] }],
        kv_transfer_params_json: None,
        data_parallel_rank: None,
        stream: false,
        structured_output: None,
    };
    assert_eq!(*fake.requests.lock().unwrap(), vec![expected]);
}

#[test]
fn a_dp_worker_sends_its_rank_and_an_absent_max_tokens_stays_absent() {
    let worker = DPAwareWorkerBuilder::new("grpc://atom:1", 3, 4)
        .connection_mode(ConnectionMode::Grpc { port: None })
        .runtime_type(RuntimeType::Atom)
        .build();
    let request = generate_request(&payload(true), &worker, None).unwrap();
    assert_eq!(request.data_parallel_rank, Some(3));
    assert_eq!(request.sampling.unwrap().max_tokens, None);
    assert!(request.stream);
}

#[tokio::test]
async fn stop_strings_without_a_tokenizer_fail_before_the_call() {
    let fake = FakeAtom::new(vec![vec![complete(&[4], "eos")]]);
    let url = serve(fake.clone()).await;
    let mut request = payload(false);
    request.stop.stop = Some(crate::protocols::common::StringOrArray::String("x".into()));

    let result = GrpcEngine::new()
        .dispatch(&single(atom_worker(&url)), &mut request)
        .await;
    assert!(matches!(result, Err(EngineError::RequestBuildFailed(_))));
    assert!(fake.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_stream_becomes_token_chunks_in_order() {
    let fake = FakeAtom::new(vec![
        vec![chunk(&[11, 12]), chunk(&[13]), complete(&[], "stop_13")],
        vec![complete(&[11, 12, 13], "max_tokens")],
        vec![chunk(&[11])],
    ]);
    let plan = single(atom_worker(&serve(fake).await));

    let streamed = collect(&plan, true).await;
    assert!(
        matches!(&streamed[..], [
            Ok(TokenChunk::Partial { token_ids: first, .. }),
            Ok(TokenChunk::Partial { token_ids: second, .. }),
            Ok(TokenChunk::Complete {
                finish_reason: FinishReason::Stop,
                matched_stop: Some(MatchedStop::TokenId(13)),
                ..
            }),
        ] if *first == [11, 12] && *second == [13]),
        "{streamed:?}"
    );

    let whole = collect(&plan, false).await;
    let [Ok(TokenChunk::Complete {
        token_ids,
        finish_reason: FinishReason::Length,
        matched_stop: None,
        usage,
        meta,
        ..
    })] = &whole[..]
    else {
        panic!("expected one Complete, got {whole:?}");
    };
    assert_eq!(*token_ids, [11, 12, 13]);
    let counts = (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    );
    assert_eq!((counts, meta.cached_tokens), ((3, 3, 6), 2));

    // A stream that ends before its Complete fails rather than ending short.
    let cut = collect(&plan, true).await;
    assert!(
        matches!(
            &cut[..],
            [
                Ok(TokenChunk::Partial { .. }),
                Err(EngineError::DecodeIncomplete)
            ]
        ),
        "{cut:?}"
    );
}

#[tokio::test]
async fn an_error_keeps_its_http_status() {
    let fake = FakeAtom::new(vec![
        vec![error("max_tokens must be at least 1", 400)],
        vec![error("engine overloaded", 503)],
        vec![chunk(&[11]), error("engine failed", 500)],
    ]);
    let url = serve(fake).await;

    assert_eq!(
        route_generate(atom_worker(&url), false).await.status(),
        StatusCode::BAD_REQUEST
    );
    // A streamed request's error arrives before its response begins.
    assert_eq!(
        route_generate(atom_worker(&url), true).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );

    let failed = collect(&single(atom_worker(&url)), true).await;
    assert!(
        matches!(&failed[..], [
            Ok(TokenChunk::Partial { .. }),
            Err(EngineError::Http { status: StatusCode::INTERNAL_SERVER_ERROR, message }),
        ] if message == "engine failed"),
        "{failed:?}"
    );
}

#[tokio::test]
async fn dropping_the_handle_cancels_the_call() {
    let fake = Arc::new(FakeAtom {
        scripts: Mutex::new(vec![vec![chunk(&[11])]].into()),
        hold: true,
        ..Default::default()
    });
    let worker = atom_worker(&serve(fake.clone()).await);
    let mut handle = GrpcEngine::new()
        .dispatch(&single(worker), &mut payload(true))
        .await
        .unwrap();
    assert!(matches!(
        handle.next().await,
        Some(Ok(TokenChunk::Partial { .. }))
    ));
    let sender = fake.held.lock().unwrap().take().unwrap();
    assert!(!sender.is_closed());

    drop(handle);
    // The server drops the call's stream once the client cancels it; the bound only stops a hang.
    tokio::time::timeout(Duration::from_secs(60), sender.closed())
        .await
        .expect("the call was not cancelled");
}

#[tokio::test]
async fn the_client_connects_and_reads_health_and_model_and_server_info() {
    let url = serve(FakeAtom::new(Vec::new())).await;
    assert_eq!("ATOM".parse::<RuntimeType>(), Ok(RuntimeType::Atom));

    let client = GrpcClient::connect(&url, &RuntimeType::Atom.to_string())
        .await
        .unwrap();
    let health = client.health_check().await.unwrap();
    assert_eq!(
        (health.healthy, health.message.as_str()),
        (false, "engine core died")
    );

    // The response's non-empty fields, by their proto names, as for SGLang and vLLM.
    let labels = client.get_model_info().await.unwrap().to_labels();
    let expected = [
        ("model_path", "/models/m"),
        ("tokenizer_path", "/models/m-tokenizer"),
        ("served_model_name", "m"),
        ("max_context_length", "4096"),
        ("eos_token_ids", "[2,7]"),
    ];
    assert_eq!(
        labels,
        expected.map(|(k, v)| (k.to_string(), v.to_string())).into()
    );

    let GrpcClient::Atom(atom) = &client else {
        panic!("expected an ATOM client");
    };
    let info = atom.get_server_info().await.unwrap();
    assert_eq!(
        (info.protocol_version, info.server_info_json.as_str()),
        (1, r#"{"dp_size":2}"#)
    );
}
