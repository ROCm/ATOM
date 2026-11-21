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
    /// GetServerInfo's `kv_transfer_info_json`; `{"tp_size":1}` when unset.
    kv_transfer_info: Option<String>,
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
            kv_transfer_info_json: self
                .kv_transfer_info
                .clone()
                .unwrap_or_else(|| r#"{"tp_size":1}"#.into()),
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
    use crate::core::{WorkerRegistry, WorkerType};
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
        let info = pairs.adapter.prefill_info.clone();
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
        };
        assert!(!engine.holds_load(&sglang));
        let dual = engine.dispatch(&sglang, &mut payload(false)).await;
        assert!(matches!(dual, Err(EngineError::ConnectionAcquireFailed(_))));

        let plan = PlacementPlan::Pair {
            prefill: prefill[0].clone(),
            decode: decode[1].clone(),
            prefill_policy: "random",
            decode_policy: "random",
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
