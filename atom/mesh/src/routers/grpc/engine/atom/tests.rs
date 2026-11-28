// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
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
use super::AtomClient;
use crate::atom_engine::grpc::{
    atom_engine_server::{AtomEngine, AtomEngineServer},
    generate_response, AbortRequest, AbortResponse, Frames, GenerateComplete, GenerateError,
    GenerateRequest, GenerateResponse, GenerateStreamChunk, GetModelInfoRequest,
    GetModelInfoResponse, GetServerInfoRequest, GetServerInfoResponse, HealthCheckRequest,
    HealthCheckResponse, PrepareStructuredOutputRequest, PrepareStructuredOutputResponse,
};
use crate::atom_engine::wire::pb::{SamplingParams, TokenIds};
use crate::core::placement::PlacementPlan;
use crate::core::worker::{ConnectionMode, RuntimeType, Worker};
use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder};
use crate::observability::metrics::metrics_labels;
use crate::protocols::generate::GenerateRequest as GenerateBody;
use crate::routers::grpc::engine::{worker_client_cache::GrpcClient, GrpcEngine};
use crate::routers::prepare::{build_generate_payload, generation_payload::GenerationPayload};
use crate::routers::test_mocks;
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::{FinishReason, MatchedStop, SeqTiming, TokenChunk};

type Reply = Result<GenerateResponse, Status>;

/// Each Generate call sends the next scripted replies, then ends its stream,
/// or with `hold` keeps it open and its sender in `held`.
#[derive(Default)]
struct FakeAtom {
    scripts: Mutex<VecDeque<Vec<GenerateResponse>>>,
    hold: bool,
    requests: Mutex<Vec<GenerateRequest>>,
    held: Mutex<Option<mpsc::Sender<Reply>>>,
    /// GetServerInfo's `kv_transfer_info_json`; `{"tp_size":1}` when unset.
    kv_transfer_info: Option<String>,
    /// GetServerInfo never answers.
    stall_server_info: bool,
    /// GetServerInfo calls, each answered after 100 ms.
    server_info_reads: std::sync::atomic::AtomicUsize,
    chat_template: Option<String>,
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
            usage_on_finish: true,
            chat_template: self.chat_template.clone(),
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _: Request<GetServerInfoRequest>,
    ) -> Result<Response<GetServerInfoResponse>, Status> {
        if self.stall_server_info {
            std::future::pending::<()>().await;
        }
        self.server_info_reads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(Response::new(GetServerInfoResponse {
            protocol_version: 1,
            kv_transfer_info_json: self
                .kv_transfer_info
                .clone()
                .unwrap_or_else(|| r#"{"tp_size":1}"#.into()),
            server_info_json: r#"{"dp_size":2}"#.into(),
            ..Default::default()
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
        prefill_tokens: 0,
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
            ttft_s: 0.25,
            tpot_s: 0.125,
            latency_s: 0.5,
            ..Default::default()
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
    route_generate_with(worker, stream, None).await
}

async fn route_generate_with(
    worker: Arc<dyn Worker>,
    stream: bool,
    headers: Option<http::HeaderMap>,
) -> axum::response::Response {
    let planner = test_mocks::MockPdPlanner::repeat_single(worker, "random");
    route_planned_generate(planner, GrpcEngine::new(), stream, headers).await
}

async fn route_planned_generate(
    planner: test_mocks::MockPdPlanner,
    engine: GrpcEngine,
    stream: bool,
    headers: Option<http::HeaderMap>,
) -> axum::response::Response {
    let pipeline = test_mocks::pipeline_with(
        Arc::new(planner),
        Arc::new(engine),
        metrics_labels::BACKEND_REGULAR,
    );
    let context =
        test_mocks::app_context_with_tokenizer_registry(test_mocks::tokenizer_registry_with("m"));
    let sampling = serde_json::json!({
        "temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_new_tokens": 16, "ignore_eos": true,
        "stop": ["Hello world", "<|im_end|>", "unseen"], "stop_token_ids": [7, 42],
    });
    let body = generate_body(stream, sampling);
    pipeline
        .execute_generate(body, headers, Some("m".to_string()), context)
        .await
}

#[tokio::test]
async fn a_payload_becomes_the_generate_request_atom_receives() {
    let fake = FakeAtom::new(vec![vec![complete(&[4], "eos")]]);
    let url = serve(fake.clone()).await;

    let response = route_generate(atom_worker(&url), false).await;
    assert_eq!(response.status(), StatusCode::OK);

    let expected = GenerateRequest {
        request_id: "atom-req".to_string(),
        prompt_token_ids: vec![7, 8, 9],
        sampling: Some(SamplingParams {
            temperature: Some(0.7),
            top_k: Some(20),
            top_p: Some(0.9),
            max_tokens: Some(16),
            ignore_eos: true,
            n: Some(1),
            ..Default::default()
        }),
        // The mock tokenizer's ids; "unseen" encodes to nothing and is dropped.
        // Then each stop id, alone.
        stop_token_sequences: vec![
            TokenIds { ids: vec![1, 2] },
            TokenIds { ids: vec![1002] },
            TokenIds { ids: vec![7] },
            TokenIds { ids: vec![42] },
        ],
        kv_transfer_params_json: None,
        data_parallel_rank: None,
        stream: false,
        structured_output: None,
        ..Default::default()
    };
    assert_eq!(*fake.requests.lock().unwrap(), vec![expected]);
}

#[tokio::test]
async fn session_headers_reach_the_generate_request() {
    let fake = FakeAtom::new(vec![vec![complete(&[4], "eos")]]);
    let url = serve(fake.clone()).await;
    let mut headers = http::HeaderMap::new();
    headers.insert("x-session-id", "turns".parse().unwrap());
    headers.insert("x-dynamo-parent-session-id", "root".parse().unwrap());

    let response = route_generate_with(atom_worker(&url), false, Some(headers)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let requests = fake.requests.lock().unwrap();
    assert_eq!(requests[0].dp_session_id.as_deref(), Some("turns"));
    assert_eq!(requests[0].dp_parent_session_id.as_deref(), Some("root"));
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
async fn a_logprobs_request_fails_before_the_call() {
    let fake = FakeAtom::new(vec![vec![complete(&[4], "eos")]]);
    let url = serve(fake.clone()).await;
    let mut top = payload(false);
    top.logprob.top_logprobs_num = 2;
    let mut flag = payload(false);
    flag.logprob.return_logprob = true;

    for mut request in [top, flag] {
        let result = GrpcEngine::new()
            .dispatch(&single(atom_worker(&url)), &mut request)
            .await;
        assert!(matches!(result, Err(EngineError::RequestBuildFailed(_))));
    }
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
    let timing = SeqTiming {
        ttft: 0.25,
        tpot: 0.125,
        latency: 0.5,
    };
    assert_eq!(meta.timing, Some(timing));

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
async fn a_reply_past_4_mib_reaches_the_router() {
    let ids = vec![1 << 22; 1 << 20]; // 4-byte varints: 4 MiB and more
    let fake = Arc::new(FakeAtom {
        scripts: Mutex::new(vec![vec![complete(&ids, "max_tokens")]].into()),
        chat_template: Some("x".repeat(5 << 20)),
        ..Default::default()
    });
    let url = serve(fake).await;

    let whole = collect(&single(atom_worker(&url)), false).await;
    assert!(
        matches!(&whole[..], [Ok(TokenChunk::Complete { token_ids, .. })] if *token_ids == ids),
        "the 4 MiB Complete failed its call"
    );
    let client = AtomClient::connect(&url).await.unwrap();
    let info = client.get_model_info().await.unwrap();
    assert_eq!(info.chat_template.map(|t| t.len()), Some(5 << 20));
}

/// `complete`, answering sibling `index`.
fn sibling(output_ids: &[u32], index: u32) -> GenerateResponse {
    let mut response = complete(output_ids, "eos");
    if let Some(generate_response::Response::Complete(c)) = response.response.as_mut() {
        c.index = index;
    }
    response
}

#[tokio::test]
async fn an_unstreamed_call_yields_every_siblings_complete_by_index() {
    let fake = FakeAtom::new(vec![
        vec![sibling(&[21], 1), sibling(&[10], 0)],
        vec![sibling(&[21], 1), error("engine failed", 500)],
    ]);
    let plan = single(atom_worker(&serve(fake).await));

    let both = collect(&plan, false).await;
    let ids: Vec<_> = both
        .iter()
        .map(|item| match item {
            Ok(TokenChunk::Complete { token_ids, .. }) => token_ids.clone(),
            other => panic!("expected a Complete, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, [vec![10], vec![21]]);

    // An error after a sibling's Complete fails the call.
    let failed = GrpcEngine::new().dispatch(&plan, &mut payload(false)).await;
    assert!(
        matches!(&failed, Err(EngineError::Http { status, .. })
            if *status == StatusCode::INTERNAL_SERVER_ERROR),
        "{:?}",
        failed.err()
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

#[tokio::test(flavor = "current_thread")]
async fn a_streamed_ttft_includes_the_wait_for_the_first_token() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let metrics = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    let fake = answering_after_300_ms(vec![chunk(&[11]), complete(&[], "eos")]);
    let response = route_generate(atom_worker(&serve(fake).await), true).await;
    assert_ttft_at_least_300_ms(response, &metrics).await;
}

/// A held call that sends `replies` 300 ms after it opens.
fn answering_after_300_ms(replies: Vec<GenerateResponse>) -> Arc<FakeAtom> {
    let fake = Arc::new(FakeAtom {
        scripts: Mutex::new(vec![Vec::new()].into()),
        hold: true,
        ..Default::default()
    });
    let held = fake.clone();
    tokio::spawn(async move {
        let sender = loop {
            if let Some(sender) = held.held.lock().unwrap().take() {
                break sender;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        };
        tokio::time::sleep(Duration::from_millis(300)).await;
        for reply in replies {
            sender.send(Ok(reply)).await.unwrap();
        }
    });
    fake
}

async fn assert_ttft_at_least_300_ms(
    response: axum::response::Response,
    metrics: &metrics_exporter_prometheus::PrometheusHandle,
) {
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let rendered = metrics.render();
    let ttft: f64 = rendered
        .lines()
        .find(|line| line.starts_with("mesh_router_ttft_seconds_sum{"))
        .and_then(|line| line.rsplit_once(' ')?.1.parse().ok())
        .unwrap_or_else(|| panic!("no TTFT recorded: {rendered}"));
    assert!(
        ttft >= 0.3,
        "TTFT {ttft} s misses the wait for the first token"
    );
}

async fn assert_e2e_latency_at_least_300_ms(response: axum::response::Response) {
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let latency = body[0]["meta_info"]["e2e_latency"].as_f64().unwrap();
    assert!(
        latency >= 0.3,
        "e2e_latency {latency} s misses the generation"
    );
}

#[tokio::test]
async fn an_unstreamed_e2e_latency_includes_the_wait_for_the_complete() {
    let fake = answering_after_300_ms(vec![complete(&[4], "eos")]);
    let response = route_generate(atom_worker(&serve(fake).await), false).await;
    assert_e2e_latency_at_least_300_ms(response).await;
}

/// A pair placement: P answers at once, D 300 ms after its call opens.
async fn route_pair_generate(
    stream: bool,
    decode_replies: Vec<GenerateResponse>,
) -> axum::response::Response {
    let mut prefill_done = complete(&[], "eos");
    if let Some(generate_response::Response::Complete(c)) = prefill_done.response.as_mut() {
        c.kv_transfer_params_json = Some("{}".into());
    }
    let prefill = Arc::new(FakeAtom {
        scripts: Mutex::new(vec![vec![prefill_done]].into()),
        kv_transfer_info: Some(r#"{"tp_size":1,"kv_role":"kv_producer"}"#.into()),
        ..Default::default()
    });
    let decode = answering_after_300_ms(decode_replies);
    let planner = test_mocks::MockPdPlanner::repeat_pair(
        atom_worker(&serve(prefill).await),
        atom_worker(&serve(decode).await),
        "random",
        "random",
    );
    let pairs = super::pair::AtomPairs::new(
        Arc::new(crate::core::WorkerRegistry::new()),
        crate::config::types::AtomPdRankMappingPolicy::Idx2Idx,
    )
    .await
    .unwrap();
    let engine = GrpcEngine::with_atom_pairs(pairs);
    route_planned_generate(planner, engine, stream, None).await
}

#[tokio::test(flavor = "current_thread")]
async fn a_pairs_streamed_ttft_includes_the_wait_for_the_first_token() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let metrics = recorder.handle();
    let _guard = metrics::set_default_local_recorder(&recorder);
    let response = route_pair_generate(true, vec![chunk(&[11]), complete(&[], "eos")]).await;
    assert_ttft_at_least_300_ms(response, &metrics).await;
}

#[tokio::test]
async fn a_pairs_unstreamed_e2e_latency_includes_the_wait_for_the_complete() {
    let response = route_pair_generate(false, vec![complete(&[4], "eos")]).await;
    assert_e2e_latency_at_least_300_ms(response).await;
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
        ("usage_on_finish", "true"),
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

/// ATOM pairs through the gRPC P/D router against in-process P and D instances,
/// run as ATOM's HTTP P/D router runs them.
mod pairs {
    use axum::body::to_bytes;
    use axum::response::Response as HttpResponse;
    use serde_json::{json, Value};
    use tokio::task::JoinHandle;
    use tokio::time::timeout;

    use super::*;
    use crate::config::types::{AtomPdRankMappingPolicy, RetryConfig};
    use crate::core::{HealthConfig, WorkerRegistry, WorkerType};
    use crate::policies::RoundRobinPolicy;
    use crate::routers::grpc::engine::{atom::pair::AtomPairs, Dispatcher};
    use crate::routers::grpc::pd_router::GrpcPDRouter;
    use crate::routers::RouterTrait;

    const PREFILL: WorkerType = WorkerType::Prefill {
        bootstrap_port: None,
    };
    /// Bounds the waits that only stop a hang.
    const HANG: Duration = Duration::from_secs(60);

    fn template() -> Value {
        json!({"remote_host": "10.0.0.1", "remote_handshake_port": 6301, "block_size": 16})
    }

    /// A P instance's `kv_transfer_info`: `tp_size`, and `template` for chunked transfer.
    fn prefill_info(tp_size: u32, template: Option<Value>) -> Option<Value> {
        let mut info = json!({"tp_size": tp_size, "kv_role": "kv_producer"});
        if let Some(template) = template {
            info["chunked_transfer"] = template;
        }
        Some(info)
    }

    /// A served instance replying with `scripts`, each call held open when `hold`.
    async fn instance(
        scripts: Vec<Vec<GenerateResponse>>,
        hold: bool,
        info: Option<Value>,
    ) -> (Arc<FakeAtom>, String) {
        let fake = Arc::new(FakeAtom {
            scripts: Mutex::new(scripts.into()),
            hold,
            kv_transfer_info: info.map(|info| info.to_string()),
            ..Default::default()
        });
        let url = serve(fake.clone()).await;
        (fake, url)
    }

    fn worker(url: &str, kind: WorkerType) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(kind)
            .connection_mode(ConnectionMode::Grpc { port: None })
            .runtime_type(RuntimeType::Atom)
            .model_id("m")
            .build();
        Arc::new(worker)
    }

    /// Two url@rank workers of the instance at `url`, sharing one client to it.
    async fn ranks(url: &str, kind: WorkerType) -> Vec<Arc<dyn Worker>> {
        let client = GrpcClient::connect(url, "atom").await.unwrap();
        (0..2)
            .map(|rank| {
                let worker = DPAwareWorkerBuilder::new(url, rank, 2)
                    .worker_type(kind.clone())
                    .connection_mode(ConnectionMode::Grpc { port: None })
                    .runtime_type(RuntimeType::Atom)
                    .grpc_client(client.clone())
                    .build();
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect()
    }

    /// The gRPC P/D router over `workers`, picking P round robin.
    async fn router(workers: &[Arc<dyn Worker>]) -> GrpcPDRouter {
        let context = test_mocks::app_context_with_tokenizer_registry(
            test_mocks::tokenizer_registry_with("m"),
        );
        for worker in workers {
            context.worker_registry.register(worker.clone());
        }
        let round_robin = Arc::new(RoundRobinPolicy::new());
        context.policy_registry.set_prefill_policy(round_robin);
        GrpcPDRouter::new(&context).await.unwrap()
    }

    /// A /generate request for prompt [7, 8, 9] with `max_new_tokens` 8, under way.
    fn start(router: GrpcPDRouter, stream: bool) -> JoinHandle<HttpResponse> {
        tokio::spawn(async move {
            let body = generate_body(stream, json!({"max_new_tokens": 8}));
            router.route_generate(None, &body, Some("m")).await
        })
    }

    async fn finish(request: JoinHandle<HttpResponse>) -> HttpResponse {
        timeout(HANG, request)
            .await
            .expect("the request hung")
            .unwrap()
    }

    /// Waits for `fake`'s `n`th call, and takes a held call's sender.
    async fn calls(fake: &FakeAtom, n: usize) -> Option<mpsc::Sender<Reply>> {
        let arrived = async {
            while fake.requests.lock().unwrap().len() < n
                || (fake.hold && fake.held.lock().unwrap().is_none())
            {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            fake.held.lock().unwrap().take()
        };
        timeout(HANG, arrived).await.expect("the call never came")
    }

    /// P's Complete, with `kv` as its kv_transfer_params.
    fn prefill_done(kv: Value) -> GenerateResponse {
        let complete = GenerateComplete {
            output_ids: vec![5],
            finish_reason: "max_tokens".into(),
            kv_transfer_params_json: Some(kv.to_string()),
            ..Default::default()
        };
        GenerateResponse {
            response: Some(generate_response::Response::Complete(complete)),
        }
    }

    fn first(fake: &FakeAtom) -> GenerateRequest {
        fake.requests.lock().unwrap()[0].clone()
    }

    fn kv(request: &GenerateRequest) -> Value {
        serde_json::from_str(request.kv_transfer_params_json.as_deref().unwrap()).unwrap()
    }

    fn stream_and_max_tokens(request: &GenerateRequest) -> (bool, Option<i64>) {
        (
            request.stream,
            request.sampling.as_ref().unwrap().max_tokens,
        )
    }

    async fn output_ids(response: HttpResponse) -> Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice::<Value>(&body).unwrap()[0]["output_ids"].clone()
    }

    #[tokio::test]
    async fn relay_sends_decode_once_prefill_completes_with_its_params_enriched() {
        let params = json!({"remote_host": "10.0.0.1", "remote_block_ids": [3, 4]});
        let p_script = vec![vec![prefill_done(params)]];
        let (p, p_url) = instance(p_script, false, prefill_info(2, None)).await;
        let (d, d_url) = instance(vec![vec![]], true, None).await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let request = start(router(&[pw.clone(), dw.clone()]).await, true);

        let decode_call = calls(&d, 1).await.unwrap();
        // P's load ended at its Complete, before D was sent.
        assert_eq!((pw.load(), dw.load()), (0, 1));
        for reply in [chunk(&[21]), complete(&[], "stop_21")] {
            decode_call.send(Ok(reply)).await.unwrap();
        }
        drop(decode_call);
        let response = finish(request).await;
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!((pw.load(), dw.load()), (0, 0));

        let (prefill, decode) = (first(&p), first(&d));
        assert_eq!(stream_and_max_tokens(&prefill), (false, Some(1)));
        let prefill_params = json!({"do_remote_decode": true, "do_remote_prefill": false});
        assert_eq!(kv(&prefill), prefill_params);
        assert_eq!(stream_and_max_tokens(&decode), (true, Some(8)));
        // D has the prompt in its request, so P's enriched params carry no prompt ids.
        assert_eq!(decode.prompt_token_ids, [7, 8, 9]);
        let enriched = json!({
            "remote_host": "10.0.0.1", "remote_block_ids": [3, 4],
            "remote_dp_size": 1, "remote_tp_size": 2,
        });
        assert_eq!(kv(&decode), enriched);
    }

    #[tokio::test]
    async fn chunked_sends_both_at_once_and_answers_once_prefill_completes() {
        let (p, p_url) = instance(vec![vec![]], true, prefill_info(4, Some(template()))).await;
        let d_script = vec![vec![complete(&[21, 22], "stop_22")]];
        let (d, d_url) = instance(d_script, false, None).await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let request = start(router(&[pw.clone(), dw.clone()]).await, false);

        // D is sent and answers while P still runs.
        let prefill_call = calls(&p, 1).await.unwrap();
        calls(&d, 1).await;
        assert_eq!((pw.load(), dw.load()), (1, 1));
        prefill_call
            .send(Ok(prefill_done(json!({}))))
            .await
            .unwrap();
        assert_eq!(output_ids(finish(request).await).await, json!([21, 22]));
        assert_eq!((pw.load(), dw.load()), (0, 0));

        let (prefill, decode) = (first(&p), first(&d));
        let transfer_id = kv(&prefill)["transfer_id"].clone();
        assert!(transfer_id
            .as_str()
            .is_some_and(|id| id.starts_with("xfer-")));
        let prefill_params = json!({
            "do_remote_decode": true, "do_remote_prefill": false,
            "chunked_transfer": true, "transfer_id": transfer_id,
        });
        assert_eq!(stream_and_max_tokens(&prefill), (false, Some(1)));
        assert_eq!(kv(&prefill), prefill_params);
        // D's params are P's template for the same transfer, enriched.
        let mut decode_params = template();
        for (key, value) in [
            ("chunked_transfer", json!(true)),
            ("do_remote_prefill", json!(true)),
            ("do_remote_decode", json!(false)),
            ("transfer_id", transfer_id),
            ("remote_dp_size", json!(1)),
            ("remote_tp_size", json!(4)),
        ] {
            decode_params[key] = value;
        }
        assert_eq!(kv(&decode), decode_params);
        assert_eq!(stream_and_max_tokens(&decode), (false, Some(8)));
        assert_eq!(decode.prompt_token_ids, [7, 8, 9]);
    }

    #[tokio::test]
    async fn a_relay_prefill_failure_sends_no_decode_and_ends_as_over_http() {
        // Every call ends before P's Complete: the HTTP router's 502, after every retry.
        let (p, p_url) = instance(Vec::new(), false, prefill_info(2, None)).await;
        let (d, d_url) = instance(Vec::new(), false, None).await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let response = finish(start(router(&[pw.clone(), dw.clone()]).await, false)).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let sent = (
            p.requests.lock().unwrap().len(),
            d.requests.lock().unwrap().len(),
        );
        assert_eq!(sent, (RetryConfig::default().max_retries as usize, 0));
        assert_eq!((pw.load(), dw.load()), (0, 0));
    }

    #[tokio::test]
    async fn in_chunked_a_prefill_failure_cancels_decode_and_a_decode_refusal_cancels_prefill() {
        // P fails after D's first token, then while D still waits; then D refuses.
        for (prefill_fails, d_script) in [
            (true, vec![chunk(&[21])]),
            (true, Vec::new()),
            (false, Vec::new()),
        ] {
            let (p, p_url) = instance(vec![vec![]], true, prefill_info(2, Some(template()))).await;
            let (d, d_url) = instance(vec![d_script], true, None).await;
            let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
            let request = start(router(&[pw.clone(), dw.clone()]).await, true);

            let prefill_call = calls(&p, 1).await.unwrap();
            let decode_call = calls(&d, 1).await.unwrap();
            let (failing, other) = if prefill_fails {
                (prefill_call, decode_call)
            } else {
                (decode_call, prefill_call)
            };
            failing.send(Ok(error("refused", 400))).await.unwrap();
            assert_eq!(finish(request).await.status(), StatusCode::BAD_REQUEST);
            let cancelled = timeout(HANG, other.closed()).await;
            cancelled.expect("the other call was not cancelled");
            assert_eq!((pw.load(), dw.load()), (0, 0));
        }
    }

    #[tokio::test]
    async fn each_retry_plans_a_new_pair() {
        let script = || vec![vec![error("busy", 503)], vec![prefill_done(json!({}))]];
        let (p1, p1_url) = instance(script(), false, prefill_info(2, None)).await;
        let (p2, p2_url) = instance(script(), false, prefill_info(2, None)).await;
        let (d, d_url) = instance(vec![vec![complete(&[21], "eos")]], false, None).await;
        let workers = [
            worker(&p1_url, PREFILL),
            worker(&p2_url, PREFILL),
            worker(&d_url, WorkerType::Decode),
        ];
        let response = finish(start(router(&workers).await, false)).await;
        assert_eq!(output_ids(response).await, json!([21]));

        // One P fails, then the other, then the first answers; no D goes out before.
        let mut prefill_calls = [&p1, &p2].map(|p| p.requests.lock().unwrap().len());
        prefill_calls.sort();
        assert_eq!(
            (prefill_calls, d.requests.lock().unwrap().len()),
            ([1, 2], 1)
        );
        assert!(workers.iter().all(|w| w.load() == 0));
    }

    #[tokio::test]
    async fn a_prefill_registered_after_startup_is_read_on_its_first_pair() {
        let p_script = vec![vec![prefill_done(json!({}))]];
        let (p, p_url) = instance(p_script, false, prefill_info(2, Some(template()))).await;
        let (d, d_url) = instance(vec![vec![complete(&[21], "eos")]], false, None).await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(dw.clone());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        registry.register(pw.clone());

        let engine = GrpcEngine::with_atom_pairs(pairs);
        let plan = PlacementPlan::Pair {
            prefill: pw,
            decode: dw,
            prefill_policy: "random",
            decode_policy: "random",
            prefill_tokens: 0,
        };
        let handle = engine.dispatch(&plan, &mut payload(false)).await.unwrap();
        let chunks = timeout(HANG, handle.collect::<Vec<_>>()).await.unwrap();
        assert!(
            matches!(&chunks[..], [Ok(TokenChunk::Complete { .. })]),
            "{chunks:?}"
        );
        // Its GetServerInfo was read: chunked transfer, and its TP size for D.
        assert_eq!(kv(&first(&p))["chunked_transfer"], json!(true));
        assert_eq!(kv(&first(&d))["remote_tp_size"], json!(2));
    }

    #[tokio::test]
    async fn a_prefill_whose_server_info_never_answers_times_out_at_both_reads() {
        let fake = Arc::new(FakeAtom {
            stall_server_info: true,
            ..Default::default()
        });
        let url = serve(fake).await;
        let health = HealthConfig {
            timeout_secs: 1,
            ..Default::default()
        };
        let stalled: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(&url)
                .worker_type(PREFILL)
                .connection_mode(ConnectionMode::Grpc { port: None })
                .runtime_type(RuntimeType::Atom)
                .health_config(health)
                .build(),
        );
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(stalled.clone());
        let startup = timeout(
            HANG,
            AtomPairs::new(registry, AtomPdRankMappingPolicy::Idx2Idx),
        );
        let error = startup.await.expect("startup hung").err().unwrap();
        assert!(error.contains("timed out"), "{error}");

        let pairs = AtomPairs::new(
            Arc::new(WorkerRegistry::new()),
            AtomPdRankMappingPolicy::Idx2Idx,
        )
        .await
        .unwrap();
        let first_pair = timeout(HANG, pairs.adapter_for(&stalled));
        let error = first_pair
            .await
            .expect("the first pair hung")
            .err()
            .unwrap();
        assert!(
            matches!(error, EngineError::Http { status, .. } if status == StatusCode::GATEWAY_TIMEOUT),
            "{error}"
        );
    }

    /// Concurrent first pairs to a prefill that stops answering fail together, within
    /// one health-check timeout, rather than one read's timeout after another.
    #[tokio::test]
    async fn concurrent_first_pairs_of_a_stalled_prefill_share_one_timeout() {
        let fake = Arc::new(FakeAtom {
            stall_server_info: true,
            ..Default::default()
        });
        let url = serve(fake).await;
        let health = HealthConfig {
            timeout_secs: 1,
            ..Default::default()
        };
        let stalled: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(&url)
                .worker_type(PREFILL)
                .connection_mode(ConnectionMode::Grpc { port: None })
                .runtime_type(RuntimeType::Atom)
                .health_config(health)
                .build(),
        );
        let pairs = AtomPairs::new(
            Arc::new(WorkerRegistry::new()),
            AtomPdRankMappingPolicy::Idx2Idx,
        )
        .await
        .unwrap();
        let started = std::time::Instant::now();
        let reads = (0..4).map(|_| pairs.adapter_for(&stalled));
        let results = timeout(HANG, futures::future::join_all(reads)).await;
        let elapsed = started.elapsed();
        for result in results.unwrap() {
            let error = result.err().unwrap();
            assert!(
                matches!(&error, EngineError::Http { status, .. } if *status == StatusCode::GATEWAY_TIMEOUT),
                "{error}"
            );
        }
        assert!(elapsed < Duration::from_millis(1800), "{elapsed:?}");
    }

    #[tokio::test]
    async fn concurrent_first_pairs_of_a_prefill_read_its_info_once() {
        let (p, p_url) = instance(Vec::new(), false, prefill_info(2, None)).await;
        let pw = worker(&p_url, PREFILL);
        let pairs = AtomPairs::new(
            Arc::new(WorkerRegistry::new()),
            AtomPdRankMappingPolicy::Idx2Idx,
        )
        .await
        .unwrap();
        let reads = (0..8).map(|_| pairs.adapter_for(&pw));
        let adapters = timeout(HANG, futures::future::join_all(reads))
            .await
            .unwrap();
        assert!(adapters.iter().all(Result::is_ok));
        let reads = p
            .server_info_reads
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(reads, 1);
    }

    #[tokio::test]
    async fn first_pairs_of_a_dp_prefill_s_ranks_read_its_info_once() {
        let (p, p_url) = instance(Vec::new(), false, prefill_info(2, None)).await;
        let registry = Arc::new(WorkerRegistry::new());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        let rank_workers = ranks(&p_url, PREFILL).await;
        for worker in &rank_workers {
            registry.register(worker.clone());
        }
        let reads = rank_workers.iter().map(|w| pairs.adapter_for(w));
        let adapters = timeout(HANG, futures::future::join_all(reads))
            .await
            .unwrap();
        assert!(adapters.iter().all(Result::is_ok));
        let reads = p
            .server_info_reads
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(reads, 1);
    }

    #[tokio::test]
    async fn a_pair_of_an_atom_and_a_non_atom_worker_is_refused_before_either_call() {
        let (p, p_url) = instance(vec![vec![prefill_done(json!({}))]], false, None).await;
        let other = test_mocks::mock_http_only_worker("http://d:8000");
        assert_ne!(other.metadata().runtime_type, RuntimeType::Atom);
        let plan = PlacementPlan::Pair {
            prefill: worker(&p_url, PREFILL),
            decode: other,
            prefill_policy: "random",
            decode_policy: "random",
            prefill_tokens: 0,
        };
        let pairs = AtomPairs::new(
            Arc::new(WorkerRegistry::new()),
            AtomPdRankMappingPolicy::Idx2Idx,
        )
        .await
        .unwrap();
        let refused = GrpcEngine::with_atom_pairs(pairs)
            .dispatch(&plan, &mut payload(false))
            .await
            .err()
            .unwrap();
        assert!(
            matches!(&refused, EngineError::Http { status, message }
                if *status == StatusCode::INTERNAL_SERVER_ERROR && message.contains("mixes runtimes")),
            "{refused}"
        );
        assert!(p.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_prefill_registered_again_at_its_url_has_its_info_read_again() {
        let p_script = vec![vec![prefill_done(json!({}))]];
        let (_, p_url) = instance(p_script.clone(), false, prefill_info(2, Some(template()))).await;
        let (_, replacement_url) = instance(p_script, false, prefill_info(4, None)).await;
        let d_script = vec![vec![complete(&[21], "eos")], vec![complete(&[22], "eos")]];
        let (d, d_url) = instance(d_script, false, None).await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(pw.clone());
        registry.register(dw.clone());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        let engine = GrpcEngine::with_atom_pairs(pairs);
        let run = |prefill: Arc<dyn Worker>| {
            let plan = PlacementPlan::Pair {
                prefill,
                decode: dw.clone(),
                prefill_policy: "random",
                decode_policy: "random",
                prefill_tokens: 0,
            };
            let engine = engine.clone();
            async move {
                let handle = engine.dispatch(&plan, &mut payload(false)).await.unwrap();
                timeout(HANG, handle.collect::<Vec<_>>()).await.unwrap();
            }
        };
        run(pw).await;

        // Removed, then registered again at the same URL: another instance behind it.
        registry.remove_by_url(&p_url);
        let client = GrpcClient::connect(&replacement_url, "atom").await.unwrap();
        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(&p_url)
                .worker_type(PREFILL)
                .connection_mode(ConnectionMode::Grpc { port: None })
                .runtime_type(RuntimeType::Atom)
                .grpc_client(client)
                .build(),
        );
        registry.register(replacement.clone());
        run(replacement).await;

        let decodes = d.requests.lock().unwrap().clone();
        let tp_and_chunked = |request: &GenerateRequest| {
            let kv = kv(request);
            (kv["remote_tp_size"].clone(), kv["chunked_transfer"].clone())
        };
        assert_eq!(tp_and_chunked(&decodes[0]), (json!(2), json!(true)));
        assert_eq!(tp_and_chunked(&decodes[1]), (json!(4), Value::Null));
    }

    #[tokio::test]
    async fn a_dp_rank_replaced_before_its_sibling_reads_the_new_instance() {
        let (_, p_url) = instance(Vec::new(), false, prefill_info(2, Some(template()))).await;
        let (replacement, replacement_url) =
            instance(Vec::new(), false, prefill_info(4, None)).await;
        let registry = Arc::new(WorkerRegistry::new());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        let old = ranks(&p_url, PREFILL).await;
        for worker in old.iter().rev() {
            registry.register(worker.clone());
            pairs.adapter_for(worker).await.unwrap();
        }

        // The instance restarts with four ranks. Rank 2 is new, so it takes its
        // sibling's info; rank 0 then replaces the old rank 0, rank 1 still registered.
        let client = GrpcClient::connect(&replacement_url, "atom").await.unwrap();
        let new_rank = |rank| -> Arc<dyn Worker> {
            Arc::new(
                DPAwareWorkerBuilder::new(&p_url, rank, 4)
                    .worker_type(PREFILL)
                    .connection_mode(ConnectionMode::Grpc { port: None })
                    .runtime_type(RuntimeType::Atom)
                    .grpc_client(client.clone())
                    .build(),
            )
        };
        let (rank0, rank2) = (new_rank(0), new_rank(2));
        registry.register(rank2.clone());
        pairs.adapter_for(&rank2).await.unwrap();
        registry.remove_by_url(old[0].url());
        registry.register(rank0.clone());
        pairs.adapter_for(&rank0).await.unwrap();
        let info = pairs.adapter_for(&rank2).await.unwrap().prefill_info;
        for rank in [&rank0, &rank2] {
            assert_eq!(info.tp_sizes.get(rank.url()), Some(&4));
            assert!(!info.chunked_transfer.contains_key(rank.url()));
        }
        let reads = replacement
            .server_info_reads
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(reads, 1);
    }

    #[tokio::test]
    async fn a_rank_registered_again_after_another_prefill_s_read_reads_its_instance() {
        let (_, p_url) = instance(Vec::new(), false, prefill_info(2, Some(template()))).await;
        let (_, other_url) = instance(Vec::new(), false, prefill_info(2, None)).await;
        let (replacement, replacement_url) =
            instance(Vec::new(), false, prefill_info(4, None)).await;
        let registry = Arc::new(WorkerRegistry::new());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        let old = ranks(&p_url, PREFILL).await;
        for worker in &old {
            registry.register(worker.clone());
            pairs.adapter_for(worker).await.unwrap();
        }

        // Rank 1 is removed; another prefill's first read then prunes it.
        registry.remove_by_url(old[1].url());
        let other = worker(&other_url, PREFILL);
        registry.register(other.clone());
        pairs.adapter_for(&other).await.unwrap();
        // The instance restarts; its rank 1 registers again beside the old rank 0, which
        // read the instance.
        let client = GrpcClient::connect(&replacement_url, "atom").await.unwrap();
        let rank1: Arc<dyn Worker> = Arc::new(
            DPAwareWorkerBuilder::new(&p_url, 1, 2)
                .worker_type(PREFILL)
                .connection_mode(ConnectionMode::Grpc { port: None })
                .runtime_type(RuntimeType::Atom)
                .grpc_client(client)
                .build(),
        );
        registry.register(rank1.clone());
        let info = pairs.adapter_for(&rank1).await.unwrap().prefill_info;
        assert_eq!(info.tp_sizes.get(rank1.url()), Some(&4));
        assert!(!info.chunked_transfer.contains_key(rank1.url()));
        let reads = replacement
            .server_info_reads
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(reads, 1);
    }

    /// A read publishing while a pair finds its prefill already read cannot take that
    /// prefill's info from under it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cache_hit_takes_its_prefill_s_info_with_its_identity() {
        let (_, p_url) = instance(Vec::new(), false, prefill_info(2, None)).await;
        let pw = worker(&p_url, PREFILL);
        let registry = Arc::new(WorkerRegistry::new());
        registry.register(pw.clone());
        let pairs = AtomPairs::new(registry, AtomPdRankMappingPolicy::Idx2Idx);
        let pairs = Arc::new(pairs.await.unwrap());

        // Another prefill's read is publishing, and prunes this one.
        let hit = {
            let mut publishing = pairs.info.write().unwrap();
            let (hit_pairs, hit_worker) = (pairs.clone(), pw.clone());
            let hit = std::thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                runtime.block_on(hit_pairs.adapter_for(&hit_worker))
            });
            std::thread::sleep(Duration::from_millis(200));
            pairs.read_for.write().unwrap().clear();
            *publishing = Arc::default();
            hit
        };

        let hit = tokio::task::spawn_blocking(move || hit.join().unwrap());
        let info = hit.await.unwrap().unwrap().prefill_info;
        assert_eq!(info.tp_sizes.get(&p_url), Some(&2));
    }

    #[tokio::test]
    async fn a_removed_prefill_s_info_is_forgotten_at_the_next_read() {
        let (_, first_url) = instance(Vec::new(), false, prefill_info(2, Some(template()))).await;
        let (_, next_url) = instance(Vec::new(), false, prefill_info(4, None)).await;
        let registry = Arc::new(WorkerRegistry::new());
        let pairs = AtomPairs::new(registry.clone(), AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        let first = worker(&first_url, PREFILL);
        registry.register(first.clone());
        pairs.adapter_for(&first).await.unwrap();
        registry.remove_by_url(&first_url);

        let next = worker(&next_url, PREFILL);
        registry.register(next.clone());
        let info = pairs.adapter_for(&next).await.unwrap().prefill_info;
        assert_eq!(info.tp_sizes.keys().collect::<Vec<_>>(), [&next_url]);
        assert!(info.chunked_transfer.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_pairs_streamed_ttft_includes_the_prefill() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let metrics = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let (p, p_url) = instance(vec![vec![]], true, prefill_info(2, None)).await;
        let d_script = vec![vec![chunk(&[21]), complete(&[], "eos")]];
        let (_, d_url) = instance(d_script, false, None).await;
        let workers = [worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode)];
        let request = start(router(&workers).await, true);

        let prefill_call = calls(&p, 1).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        prefill_call
            .send(Ok(prefill_done(json!({}))))
            .await
            .unwrap();
        drop(prefill_call);
        let response = finish(request).await;
        assert_eq!(response.status(), StatusCode::OK);
        to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let rendered = metrics.render();
        let ttft: f64 = rendered
            .lines()
            .find(|line| line.starts_with("mesh_router_ttft_seconds_sum{"))
            .and_then(|line| line.rsplit_once(' ')?.1.parse().ok())
            .unwrap_or_else(|| panic!("no TTFT recorded: {rendered}"));
        assert!(ttft >= 0.3, "TTFT {ttft} s misses the prefill");
    }

    #[tokio::test]
    async fn each_rank_takes_its_instances_info_and_idx2idx_runs_prefill_on_decodes_rank() {
        let (p, p_url) = instance(vec![vec![]], true, prefill_info(4, Some(template()))).await;
        let (d, d_url) = instance(vec![vec![complete(&[21], "eos")]], false, None).await;
        let (prefill, decode) = (
            ranks(&p_url, PREFILL).await,
            ranks(&d_url, WorkerType::Decode).await,
        );
        let registry = Arc::new(WorkerRegistry::new());
        for worker in prefill.iter().chain(&decode) {
            registry.register(worker.clone());
        }
        let pairs = AtomPairs::new(registry, AtomPdRankMappingPolicy::Idx2Idx)
            .await
            .unwrap();
        // The instance's GetServerInfo gives each of its ranks the TP size and template.
        let info = pairs.adapter_for(&prefill[0]).await.unwrap().prefill_info;
        let urls = [p_url.clone(), format!("{p_url}@0"), format!("{p_url}@1")];
        assert_eq!(info.tp_sizes, urls.clone().map(|url| (url, 4)).into());
        assert_eq!(
            info.chunked_transfer,
            urls.map(|url| (url, template())).into()
        );

        let engine = GrpcEngine::with_atom_pairs(pairs);
        // An SGLang pair keeps the dual dispatch and the pipeline's load guards.
        let sglang = PlacementPlan::Pair {
            prefill: test_mocks::mock_http_only_worker("http://p:8000"),
            decode: test_mocks::mock_http_only_worker("http://d:8000"),
            prefill_policy: "random",
            decode_policy: "random",
            prefill_tokens: 0,
        };
        assert!(!engine.holds_load(&sglang));
        let dual = engine.dispatch(&sglang, &mut payload(false)).await;
        assert!(matches!(dual, Err(EngineError::ConnectionAcquireFailed(_))));

        let plan = PlacementPlan::Pair {
            prefill: prefill[0].clone(),
            decode: decode[1].clone(),
            prefill_policy: "random",
            decode_policy: "random",
            prefill_tokens: 0,
        };
        assert!(engine.holds_load(&plan));
        let run = tokio::spawn(async move {
            let handle = engine.dispatch(&plan, &mut payload(false)).await.unwrap();
            handle.collect::<Vec<_>>().await
        });
        // Idx2idx moved P to rank 1, decode's, and the dispatch holds that rank's load.
        let prefill_call = calls(&p, 1).await.unwrap();
        let loads = [&prefill[0], &prefill[1], &decode[1]].map(|w| w.load());
        assert_eq!(loads, [0, 1, 1]);
        prefill_call
            .send(Ok(prefill_done(json!({}))))
            .await
            .unwrap();
        let chunks = timeout(HANG, run)
            .await
            .expect("the dispatch hung")
            .unwrap();
        assert!(
            matches!(&chunks[..], [Ok(TokenChunk::Complete { .. })]),
            "{chunks:?}"
        );
        assert!(prefill.iter().chain(&decode).all(|w| w.load() == 0));

        let dp_ranks = (first(&p).data_parallel_rank, first(&d).data_parallel_rank);
        assert_eq!(dp_ranks, (Some(1), Some(1)));
        let params = kv(&first(&d));
        let remote = ["remote_dp_rank", "remote_dp_size", "remote_tp_size"].map(|k| &params[k]);
        assert_eq!(remote, [&json!(1), &json!(2), &json!(4)]);
    }
}
