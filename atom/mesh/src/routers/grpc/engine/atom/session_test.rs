//! The Session against an in-process `AtomEngine` whose Session calls the test
//! answers: per-seq_id splitting, Abort on drop, failure and reopening, Ready's version.

use std::future::Future;
use std::time::Duration;

use prost::Message as _;
use tokio::time::timeout;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tonic::transport::{server::TcpIncoming, Server};
use tonic::{Request, Response as RpcResponse};

use super::*;
use crate::atom_engine::grpc::atom_engine_server::{AtomEngine, AtomEngineServer};
use crate::atom_engine::grpc::{
    prepare_structured_output_response::Response as Prepared, AbortRequest, AbortResponse,
    GenerateRequest, GetModelInfoRequest, GetModelInfoResponse, GetServerInfoRequest,
    GetServerInfoResponse, HealthCheckRequest, HealthCheckResponse, PrepareStructuredOutputRequest,
    PrepareStructuredOutputResponse,
};
use crate::atom_engine::wire::PROTOCOL_VERSION;
use crate::core::worker::{ConnectionMode, RuntimeType};
use crate::core::DPAwareWorkerBuilder;
use crate::routers::grpc::engine::atom::stream::AtomStream;
use crate::routers::grpc::engine::worker_client_cache::GrpcClient;
use crate::routers::token_handle::token_chunk::{FinishReason, MatchedStop, TokenChunk};

/// Bounds the waits that only stop a hang.
const HANG: Duration = Duration::from_secs(60);

/// A Session call as the server sees it: the client's messages, and a sender for its own.
struct Call {
    inbound: Streaming<Frames>,
    outbound: mpsc::UnboundedSender<Result<Frames, Status>>,
}

/// Hands each Session call to the test, answers GetServerInfo with its
/// `kv_transfer_info` and PrepareStructuredOutput with `derived()`; serves no other RPC.
struct FakeAtom {
    calls: mpsc::UnboundedSender<Call>,
    kv_transfer_info: String,
    /// What PrepareStructuredOutput was asked.
    asked: Mutex<Vec<PrepareStructuredOutputRequest>>,
}

#[tonic::async_trait]
impl AtomEngine for FakeAtom {
    type GenerateStream = UnboundedReceiverStream<Output>;
    type SessionStream = UnboundedReceiverStream<Result<Frames, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<Frames>>,
    ) -> Result<RpcResponse<Self::SessionStream>, Status> {
        let (outbound, frames) = mpsc::unbounded_channel();
        let call = Call {
            inbound: request.into_inner(),
            outbound,
        };
        self.calls
            .send(call)
            .map_err(|_| Status::internal("test over"))?;
        Ok(RpcResponse::new(UnboundedReceiverStream::new(frames)))
    }

    async fn generate(
        &self,
        _: Request<GenerateRequest>,
    ) -> Result<RpcResponse<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("generate"))
    }

    async fn abort(&self, _: Request<AbortRequest>) -> Result<RpcResponse<AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }

    async fn get_model_info(
        &self,
        _: Request<GetModelInfoRequest>,
    ) -> Result<RpcResponse<GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("get_model_info"))
    }

    async fn get_server_info(
        &self,
        _: Request<GetServerInfoRequest>,
    ) -> Result<RpcResponse<GetServerInfoResponse>, Status> {
        Ok(RpcResponse::new(GetServerInfoResponse {
            protocol_version: PROTOCOL_VERSION,
            kv_transfer_info_json: self.kv_transfer_info.clone(),
            server_info_json: "{}".into(),
        }))
    }

    async fn health_check(
        &self,
        _: Request<HealthCheckRequest>,
    ) -> Result<RpcResponse<HealthCheckResponse>, Status> {
        Err(Status::unimplemented("health_check"))
    }

    async fn prepare_structured_output(
        &self,
        request: Request<PrepareStructuredOutputRequest>,
    ) -> Result<RpcResponse<PrepareStructuredOutputResponse>, Status> {
        self.asked.lock().push(request.into_inner());
        Ok(RpcResponse::new(PrepareStructuredOutputResponse {
            response: Some(Prepared::Sampling(derived())),
        }))
    }
}

/// The structured-output fields the fake's PrepareStructuredOutput derives.
fn derived() -> pb::SamplingParams {
    pb::SamplingParams {
        required_channel: Some(pb::RequiredChannel {
            after: vec![1],
            opening: vec![2],
            opens_at: 3,
            end_ids: vec![4],
        }),
        json_schema: Some(r#"{"type":"object"}"#.into()),
        structured_tool_format: Some("dsml".into()),
        structured_reasoning_end: "</think>".into(),
        ..Default::default()
    }
}

/// Serves a fake instance on a free local port, its GetServerInfo answering
/// `kv_transfer_info`: its URL, the fake and its Session calls.
async fn serve(kv_transfer_info: &str) -> (String, Arc<FakeAtom>, mpsc::UnboundedReceiver<Call>) {
    let (calls, received) = mpsc::unbounded_channel();
    let fake = Arc::new(FakeAtom {
        calls,
        kv_transfer_info: kv_transfer_info.to_string(),
        asked: Mutex::default(),
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let router = Server::builder().add_service(AtomEngineServer::from_arc(fake.clone()));
    tokio::spawn(router.serve_with_incoming(TcpIncoming::from(listener)));
    (url, fake, received)
}

/// Serves a fake instance on a free local port: two url@rank workers of it,
/// sharing one client, and its Session calls.
async fn instance() -> ([Arc<dyn Worker>; 2], mpsc::UnboundedReceiver<Call>) {
    let (url, _, received) = serve("{}").await;
    let client = GrpcClient::connect(&url, "atom").await.unwrap();
    let ranks = [0, 1].map(|rank| {
        let worker = DPAwareWorkerBuilder::new(&url, rank, 2)
            .connection_mode(ConnectionMode::Grpc { port: None })
            .runtime_type(RuntimeType::Atom)
            .grpc_client(client.clone())
            .build();
        Arc::new(worker) as Arc<dyn Worker>
    });
    (ranks, received)
}

/// Runs `submit`, which opens a Session, answering the call it opens with a
/// Ready of `version`.
async fn opening(
    calls: &mut mpsc::UnboundedReceiver<Call>,
    version: u32,
    submit: impl Future<Output = Result<SessionStream, EngineError>>,
) -> (Call, Result<SessionStream, EngineError>) {
    tokio::join!(accept(calls, version), submit)
}

/// The next Session call, answered with a Ready of `version`.
async fn accept(calls: &mut mpsc::UnboundedReceiver<Call>, version: u32) -> Call {
    let call = timeout(HANG, calls.recv()).await.expect("no Session call");
    let call = call.unwrap();
    let ready = pb::Envelope {
        protocol_version: version,
        body: Some(Body::Ready(pb::Ready::default())),
    };
    call.outbound
        .send(Ok(Frames {
            frames: vec![ready.encode_to_vec()],
        }))
        .unwrap();
    call
}

/// A request named `id`; the Session sets its seq_id and prompt reference.
fn spec(id: &str) -> pb::RequestSpec {
    pb::RequestSpec {
        external_request_id: Some(id.to_string()),
        sampling: Some(pb::SamplingParams {
            max_tokens: Some(8),
            ..Default::default()
        }),
        data_parallel_rank: Some(1),
        ..Default::default()
    }
}

/// `submit` on a Session already open.
async fn submit(
    sessions: &AtomSessions,
    worker: &Arc<dyn Worker>,
    name: &str,
    prompt: Vec<u32>,
    stream: bool,
) -> SessionStream {
    let submitted = sessions.submit(worker, spec(name), prompt, stream);
    timeout(HANG, submitted)
        .await
        .expect("the submit hung")
        .unwrap()
}

/// The client's next message on `call`.
async fn next(call: &mut Call) -> Inbound<Vec<u8>> {
    let message = timeout(HANG, call.inbound.message())
        .await
        .expect("no message");
    Inbound::decode(message.unwrap().expect("the call ended").frames).unwrap()
}

/// Reads SubmitBatches until `n` requests have come: each one's spec and
/// prompt, by its name.
async fn submitted(call: &mut Call, n: usize) -> HashMap<String, (pb::RequestSpec, Vec<u32>)> {
    let mut requests = HashMap::new();
    while requests.len() < n {
        let message = next(call).await;
        let Body::Submit(batch) = &message.body else {
            panic!("expected a SubmitBatch, got {:?}", message.body);
        };
        for spec in &batch.requests {
            let prompt = message.tokens(spec.prompt_token_ids.as_ref()).unwrap();
            let id = spec.external_request_id.clone().unwrap();
            requests.insert(id, (spec.clone(), prompt));
        }
    }
    requests
}

fn frames(body: Body, data: Vec<Vec<u8>>) -> Result<Frames, Status> {
    Ok(Frames {
        frames: wire::encode(body, data),
    })
}

/// One engine step: each entry's seq_id, its ids and its prefix-cache hit.
fn step(entries: &[(u64, Vec<i32>, i64)]) -> Result<Frames, Status> {
    let batch = pb::StreamBatch {
        seq_ids: entries.iter().map(|e| e.0).collect(),
        token_counts: entries.iter().map(|e| e.1.len() as u32).collect(),
        tokens: entries.iter().flat_map(|e| e.1.iter().copied()).collect(),
        finished: vec![false; entries.len()],
        num_cached_tokens: entries.iter().map(|e| e.2).collect(),
        ..Default::default()
    };
    frames(Body::Stream(batch), Vec::new())
}

/// A Completion of `seq_id` for `reason`, its prompt 3 tokens long.
fn completion(seq_id: u64, reason: &str) -> pb::Completion {
    pb::Completion {
        seq_id,
        num_prompt_tokens: 3,
        leave_reason: reason.to_string(),
        ..Default::default()
    }
}

/// One Completions message: each completion with every output id listed beside it.
fn done(items: Vec<(pb::Completion, Vec<u32>)>) -> Result<Frames, Status> {
    let mut data = Vec::new();
    let items = items
        .into_iter()
        .map(|(mut completion, ids)| {
            data.push(wire::token_frame(&ids));
            completion.completion_token_ids = Some(pb::BufferRef {
                frame: data.len() as u32,
                offset: 0,
                nbytes: 4 * ids.len() as u64,
            });
            completion
        })
        .collect();
    frames(Body::Completions(pb::Completions { items }), data)
}

/// Every TokenChunk of `stream`, read as a Generate stream is read.
async fn chunks(stream: SessionStream) -> Vec<Result<TokenChunk, EngineError>> {
    let read = async {
        match AtomStream::start(stream).await {
            Ok(stream) => stream.collect().await,
            Err(e) => vec![Err(e)],
        }
    };
    timeout(HANG, read).await.expect("the stream hung")
}

#[tokio::test]
async fn one_session_serves_both_ranks_and_each_request_gets_only_its_outputs_in_order() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("a"), vec![1, 2, 3], true);
    let (mut call, a) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let b = submit(&sessions, &ranks[1], "b", vec![4, 5], false).await;

    // Each request keeps its spec and prompt, under its own seq_id below 2^24.
    let sent = submitted(&mut call, 2).await;
    let (a_id, b_id) = (sent["a"].0.seq_id, sent["b"].0.seq_id);
    assert!(a_id != b_id && a_id < SEQ_IDS && b_id < SEQ_IDS);
    for (name, prompt) in [("a", vec![1, 2, 3]), ("b", vec![4, 5])] {
        let (mut received, ids) = sent[name].clone();
        (received.seq_id, received.prompt_token_ids) = (0, None);
        assert_eq!((received, ids), (spec(name), prompt));
    }

    let string = |s: &str| pb::Value {
        kind: Some(Kind::StringValue(s.to_string())),
    };
    let entry = pb::MapEntry {
        key: Some(string("remote_host")),
        value: Some(string("10.0.0.1")),
    };
    let kv = pb::Value {
        kind: Some(Kind::MapValue(pb::ValueMap {
            entries: vec![entry],
        })),
    };
    let b_done = pb::Completion {
        kv_transfer_params_output: Some(kv),
        ..completion(b_id, "max_tokens")
    };
    for message in [
        step(&[(a_id, vec![11], 2), (b_id, vec![21], 0)]),
        step(&[(b_id, vec![22], 0), (a_id, vec![12, 13], 0)]),
        done(vec![
            (b_done, vec![21, 22]),
            (completion(a_id, "stop_13"), vec![11, 12, 13]),
        ]),
    ] {
        call.outbound.send(message).unwrap();
    }

    // a streams its own ids, step by step; its Complete carries none, as Generate's does.
    let a = chunks(a.unwrap()).await;
    assert!(
        matches!(&a[..], [
            Ok(TokenChunk::Partial { token_ids: first, .. }),
            Ok(TokenChunk::Partial { token_ids: second, .. }),
            Ok(TokenChunk::Complete {
                token_ids,
                finish_reason: FinishReason::Stop,
                matched_stop: Some(MatchedStop::TokenId(13)),
                usage,
                meta,
                ..
            }),
        ] if *first == [11] && *second == [12, 13] && token_ids.is_empty()
            && (usage.prompt_tokens, usage.completion_tokens, meta.cached_tokens) == (3, 3, 2)),
        "{a:?}"
    );
    // b does not stream: one Complete with every id and P's params, as Generate answers.
    let b: Vec<_> = timeout(HANG, b.collect()).await.unwrap();
    let complete = GenerateComplete {
        output_ids: vec![21, 22],
        finish_reason: "max_tokens".to_string(),
        prompt_tokens: 3,
        completion_tokens: 2,
        cached_tokens: 0,
        kv_transfer_params_json: Some(r#"{"remote_host":"10.0.0.1"}"#.to_string()),
    };
    let expected = GenerateResponse {
        response: Some(Response::Complete(complete)),
    };
    assert!(
        matches!(&b[..], [Ok(response)] if *response == expected),
        "{b:?}"
    );
    assert!(calls.try_recv().is_err(), "a second Session call opened");
}

#[tokio::test]
async fn a_rejected_request_fails_with_the_http_status_its_code_reads_back_to() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let cases = [
        ("invalid", Code::Invalid, 400),
        ("overloaded", Code::Overloaded, 503),
        ("internal", Code::Internal, 500),
    ];
    let first = sessions.submit(&ranks[0], spec("invalid"), vec![1], true);
    let (mut call, invalid) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let mut streams = vec![invalid.unwrap()];
    for (name, ..) in &cases[1..] {
        streams.push(submit(&sessions, &ranks[0], name, vec![1], true).await);
    }
    let sent = submitted(&mut call, 3).await;
    let items = cases
        .iter()
        .map(|&(name, code, _)| pb::RejectedRequest {
            seq_id: sent[name].0.seq_id,
            reason: format!("{name} refused"),
            code: code as i32,
        })
        .collect();
    let rejected = frames(Body::Rejected(pb::Rejected { items }), Vec::new());
    call.outbound.send(rejected).unwrap();

    for (stream, (name, _, expected)) in streams.into_iter().zip(cases) {
        let failed = chunks(stream).await;
        assert!(
            matches!(&failed[..], [Err(EngineError::Http { status, message })]
                if status.as_u16() == expected && *message == format!("{name} refused")),
            "{name}: {failed:?}"
        );
    }
}

#[tokio::test]
async fn dropping_an_unfinished_request_aborts_it_and_a_finished_one_sends_nothing() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("finished"), vec![1], true);
    let (mut call, finished) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let running = submit(&sessions, &ranks[0], "running", vec![2], true).await;
    let sent = submitted(&mut call, 2).await;
    let (finished_id, running_id) = (sent["finished"].0.seq_id, sent["running"].0.seq_id);
    call.outbound
        .send(step(&[(running_id, vec![31], 0)]))
        .unwrap();
    let ended = done(vec![(completion(finished_id, "eos"), Vec::new())]);
    call.outbound.send(ended).unwrap();

    // The finished request is read to its Complete, then dropped.
    let finished = chunks(finished.unwrap()).await;
    assert!(
        matches!(&finished[..], [Ok(TokenChunk::Complete { .. })]),
        "{finished:?}"
    );
    let started = timeout(HANG, AtomStream::start(running)).await.unwrap();
    let mut running = started.unwrap();
    let first = running.next().await;
    assert!(
        matches!(&first, Some(Ok(TokenChunk::Partial { token_ids, .. })) if *token_ids == [31]),
        "{first:?}"
    );
    drop(running);

    // The next message is the running request's Abort, naming it alone.
    let message = next(&mut call).await;
    assert!(
        matches!(&message.body, Body::Abort(abort) if abort.seq_ids == [running_id]),
        "{:?}",
        message.body
    );
}

#[tokio::test]
async fn a_broken_session_fails_its_requests_and_the_next_request_reopens_it() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    // The call fails, ends, or sends a StreamBatch whose columns disagree.
    for (case, reason) in [
        (0, "engine died"),
        (1, "the ATOM Session ended"),
        (2, "StreamBatch columns disagree in length"),
    ] {
        let first = sessions.submit(&ranks[0], spec("started"), vec![1], true);
        let (mut call, started) = opening(&mut calls, PROTOCOL_VERSION, first).await;
        let waiting = submit(&sessions, &ranks[1], "waiting", vec![2], true).await;
        let started_id = submitted(&mut call, 2).await["started"].0.seq_id;
        call.outbound
            .send(step(&[(started_id, vec![11], 0)]))
            .unwrap();
        let started = timeout(HANG, AtomStream::start(started.unwrap())).await;
        let mut started = started.unwrap().unwrap();
        let chunk = started.next().await;
        assert!(matches!(chunk, Some(Ok(TokenChunk::Partial { .. }))));

        match case {
            0 => call.outbound.send(Err(Status::internal(reason))).unwrap(),
            1 => drop(call.outbound),
            _ => {
                let batch = pb::StreamBatch {
                    seq_ids: vec![started_id],
                    token_counts: vec![2],
                    tokens: vec![12],
                    num_cached_tokens: vec![0],
                    ..Default::default()
                };
                call.outbound
                    .send(frames(Body::Stream(batch), Vec::new()))
                    .unwrap();
            }
        }
        let rest: Vec<_> = timeout(HANG, started.collect()).await.unwrap();
        let waited = chunks(waiting).await;
        for failed in [&rest, &waited] {
            assert!(
                matches!(&failed[..], [Err(EngineError::Transport(status))]
                    if status.message().contains(reason)),
                "{reason}: {failed:?}"
            );
        }
    }
    // After the last failure too, the next request opens a new call and runs on it.
    let last = sessions.submit(&ranks[0], spec("after"), vec![3], false);
    let (mut call, after) = opening(&mut calls, PROTOCOL_VERSION, last).await;
    let after_id = submitted(&mut call, 1).await["after"].0.seq_id;
    let ended = done(vec![(completion(after_id, "eos"), vec![41])]);
    call.outbound.send(ended).unwrap();
    let after = chunks(after.unwrap()).await;
    assert!(
        matches!(&after[..], [Ok(TokenChunk::Complete { token_ids, .. })] if *token_ids == [41]),
        "{after:?}"
    );
}

#[tokio::test]
async fn a_ready_of_another_protocol_version_is_refused() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("refused"), vec![1], true);
    let (mut call, refused) = opening(&mut calls, PROTOCOL_VERSION + 1, first).await;
    let expected = format!("protocol version {}, expected 1", PROTOCOL_VERSION + 1);
    assert!(
        matches!(&refused, Err(EngineError::Transport(status))
            if status.message().contains(&expected)),
        "{:?}",
        refused.err()
    );
    // Nothing was sent on the refused call, and the next request opens another.
    let sent = timeout(HANG, call.inbound.message()).await.unwrap();
    assert!(!matches!(sent, Ok(Some(_))), "{sent:?}");
    let next_request = sessions.submit(&ranks[0], spec("accepted"), vec![1], true);
    let (mut call, accepted) = opening(&mut calls, PROTOCOL_VERSION, next_request).await;
    assert!(accepted.is_ok());
    assert!(submitted(&mut call, 1).await.contains_key("accepted"));
}

#[test]
fn seq_ids_stay_below_2_24_and_skip_those_open_requests_hold() {
    let mut routes = Routes {
        next_seq_id: SEQ_IDS - 2,
        ..Default::default()
    };
    for held in [SEQ_IDS - 1, 0] {
        let (tx, _) = mpsc::unbounded_channel();
        let route = Route {
            tx,
            stream: false,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
        };
        routes.open.insert(held, route);
    }
    // The second search passes the held 2^24 - 1 and 0, wrapping below 2^24.
    let ids = [routes.free_seq_id(), routes.free_seq_id()];
    assert_eq!(ids, [Some(SEQ_IDS - 2), Some(1)]);
}

/// Requests through the gRPC routers built from a config with `atom_grpc_session`:
/// a chat alone, then ATOM pairs in relay and in chunked transfer.
mod routed {
    use axum::body::to_bytes;
    use axum::response::Response as HttpResponse;
    use serde_json::json;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::app_context::AppContext;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::grpc::{
        engine::atom::request::value, pd_router::GrpcPDRouter, router::GrpcRouter,
    };
    use crate::routers::{test_mocks, RouterTrait};

    const PREFILL: WorkerType = WorkerType::Prefill {
        bootstrap_port: None,
    };

    fn worker(url: &str, kind: WorkerType) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(kind)
            .connection_mode(ConnectionMode::Grpc { port: None })
            .runtime_type(RuntimeType::Atom)
            .model_id("m")
            .build();
        Arc::new(worker)
    }

    /// `context` with `--atom-grpc-session`, `workers` registered.
    fn over_sessions(
        mut context: Arc<AppContext>,
        workers: &[&Arc<dyn Worker>],
    ) -> Arc<AppContext> {
        Arc::get_mut(&mut context)
            .unwrap()
            .router_config
            .atom_grpc_session = true;
        for worker in workers {
            context.worker_registry.register(Arc::clone(worker));
        }
        context
    }

    /// A /generate request for prompt [7, 8, 9] with `max_new_tokens` 8 through the
    /// gRPC P/D router over `workers`, under way.
    async fn generate(workers: &[&Arc<dyn Worker>], stream: bool) -> JoinHandle<HttpResponse> {
        let registry = test_mocks::tokenizer_registry_with("m");
        let context = test_mocks::app_context_with_tokenizer_registry(registry);
        let router = GrpcPDRouter::new(&over_sessions(context, workers))
            .await
            .unwrap();
        let body = json!({
            "rid": "atom-req", "input_ids": [7, 8, 9], "stream": stream,
            "sampling_params": {"max_new_tokens": 8},
        });
        let body = serde_json::from_value(body).unwrap();
        tokio::spawn(async move { router.route_generate(None, &body, Some("m")).await })
    }

    /// `request`'s status and body, once its response has ended.
    async fn respond(request: JoinHandle<HttpResponse>) -> (StatusCode, String) {
        let response = timeout(HANG, request).await.expect("the request hung");
        let response = response.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    /// The one request `call`'s next SubmitBatch carries, its prompt, and its
    /// kv_transfer_params as JSON, null when it has none.
    async fn one_request(call: &mut Call) -> (pb::RequestSpec, Vec<u32>, Json) {
        let (_, (spec, prompt)) = submitted(call, 1).await.into_iter().next().unwrap();
        let kv = spec.kv_transfer_params.as_ref().map(json).transpose();
        (spec, prompt, kv.unwrap().unwrap_or_default())
    }

    fn max_tokens(spec: &pb::RequestSpec) -> Option<i64> {
        spec.sampling.as_ref().unwrap().max_tokens
    }

    #[tokio::test]
    async fn a_chat_runs_over_the_session_after_atom_prepares_its_structured_output() {
        let (url, fake, mut calls) = serve("{}").await;
        let context = test_mocks::app_context_with_hf_tokenizer("m");
        let context = over_sessions(context, &[&worker(&url, WorkerType::Regular)]);
        let router = GrpcRouter::new(&context).await.unwrap();
        let format = json!({"type": "json_schema", "json_schema": {"name": "a", "schema": {}}});
        let chat = json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 8, "response_format": format,
        });
        let chat: ChatCompletionRequest = serde_json::from_value(chat).unwrap();
        let request = tokio::spawn(async move { router.route_chat(None, &chat, Some("m")).await });

        let mut call = accept(&mut calls, PROTOCOL_VERSION).await;
        let (spec, _, _) = one_request(&mut call).await;
        // ATOM was asked first, with the chat's fields and the request's own.
        let asked = fake.asked.lock().clone();
        let [asked] = &asked[..] else {
            panic!("{asked:?}")
        };
        let carried = asked.structured_output.as_ref().unwrap();
        let carried_format = carried.response_format_json.as_deref().unwrap();
        assert_eq!(
            serde_json::from_str::<Json>(carried_format).unwrap(),
            format
        );
        let id = spec.external_request_id.as_deref().unwrap();
        assert!(id.starts_with("chatcmpl-"), "{id}");
        assert_eq!(max_tokens(&spec), Some(8));
        // The RequestSpec's sampling is the request's, with the fields ATOM derived.
        let derived = derived();
        let expected = pb::SamplingParams {
            required_channel: derived.required_channel,
            json_schema: derived.json_schema,
            structured_tool_format: derived.structured_tool_format,
            structured_reasoning_end: derived.structured_reasoning_end,
            ..asked.sampling.clone().unwrap()
        };
        assert_eq!(spec.sampling, Some(expected));

        let ended = done(vec![(completion(spec.seq_id, "eos"), vec![11])]);
        call.outbound.send(ended).unwrap();
        let (status, body) = respond(request).await;
        let body: Json = serde_json::from_str(&body).unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn relay_runs_both_legs_over_the_session_and_decode_once_prefill_completes() {
        let (p_url, _, mut p_calls) = serve(r#"{"tp_size": 2, "kv_role": "kv_producer"}"#).await;
        let (d_url, _, mut d_calls) = serve("{}").await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let request = generate(&[&pw, &dw], true).await;

        let mut p_call = accept(&mut p_calls, PROTOCOL_VERSION).await;
        let (p_spec, p_prompt, p_kv) = one_request(&mut p_call).await;
        assert_eq!((p_prompt, max_tokens(&p_spec)), (vec![7, 8, 9], Some(1)));
        let prefill_params = json!({"do_remote_decode": true, "do_remote_prefill": false});
        assert_eq!(p_kv, prefill_params);
        // P's Completion carries its params, which D's request takes enriched.
        let params = json!({"remote_host": "10.0.0.1", "remote_block_ids": [3, 4]});
        let p_done = pb::Completion {
            kv_transfer_params_output: Some(value(&params).unwrap()),
            ..completion(p_spec.seq_id, "max_tokens")
        };
        p_call.outbound.send(done(vec![(p_done, vec![5])])).unwrap();

        let mut d_call = accept(&mut d_calls, PROTOCOL_VERSION).await;
        let (d_spec, d_prompt, d_kv) = one_request(&mut d_call).await;
        // P's load ended at its Completion, before D was sent.
        assert_eq!((pw.load(), dw.load()), (0, 1));
        assert_eq!((d_prompt, max_tokens(&d_spec)), (vec![7, 8, 9], Some(8)));
        let enriched = json!({
            "remote_host": "10.0.0.1", "remote_block_ids": [3, 4],
            "remote_dp_size": 1, "remote_tp_size": 2,
        });
        assert_eq!(d_kv, enriched);

        d_call
            .outbound
            .send(step(&[(d_spec.seq_id, vec![21], 0)]))
            .unwrap();
        let d_done = done(vec![(completion(d_spec.seq_id, "stop_21"), vec![21])]);
        d_call.outbound.send(d_done).unwrap();
        let (status, body) = respond(request).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!((pw.load(), dw.load()), (0, 0));
    }

    #[tokio::test]
    async fn chunked_runs_both_legs_over_the_session_at_once_and_answers_after_prefill() {
        let template = json!({"remote_host": "10.0.0.1", "remote_handshake_port": 6301});
        let info = json!({"tp_size": 4, "kv_role": "kv_producer", "chunked_transfer": template});
        let (p_url, _, mut p_calls) = serve(&info.to_string()).await;
        let (d_url, _, mut d_calls) = serve("{}").await;
        let (pw, dw) = (worker(&p_url, PREFILL), worker(&d_url, WorkerType::Decode));
        let request = generate(&[&pw, &dw], false).await;

        // Both requests go out before either is answered.
        let mut p_call = accept(&mut p_calls, PROTOCOL_VERSION).await;
        let mut d_call = accept(&mut d_calls, PROTOCOL_VERSION).await;
        let (p_spec, _, p_kv) = one_request(&mut p_call).await;
        let (d_spec, d_prompt, d_kv) = one_request(&mut d_call).await;
        let transfer_id = p_kv["transfer_id"].clone();
        assert!(transfer_id
            .as_str()
            .is_some_and(|id| id.starts_with("xfer-")));
        let prefill_params = json!({
            "do_remote_decode": true, "do_remote_prefill": false,
            "chunked_transfer": true, "transfer_id": transfer_id,
        });
        assert_eq!((p_kv, max_tokens(&p_spec)), (prefill_params, Some(1)));
        // D's params are P's template for the same transfer, enriched.
        let mut decode_params = template;
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
        assert_eq!((d_kv, d_prompt), (decode_params, vec![7, 8, 9]));
        assert_eq!(max_tokens(&d_spec), Some(8));

        // D completes first; the answer waits for P, whose load is still held.
        let d_done = done(vec![(completion(d_spec.seq_id, "stop_22"), vec![21, 22])]);
        d_call.outbound.send(d_done).unwrap();
        assert_eq!((pw.load(), dw.load()), (1, 1));
        let p_done = done(vec![(completion(p_spec.seq_id, "max_tokens"), vec![5])]);
        p_call.outbound.send(p_done).unwrap();
        let (status, body) = respond(request).await;
        let body: Json = serde_json::from_str(&body).unwrap();
        assert_eq!(
            (status, &body[0]["output_ids"]),
            (StatusCode::OK, &json!([21, 22]))
        );
        assert_eq!((pw.load(), dw.load()), (0, 0));
    }
}
