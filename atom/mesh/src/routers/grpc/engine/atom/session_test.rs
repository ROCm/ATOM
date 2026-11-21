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
    AbortRequest, AbortResponse, GenerateRequest, GetModelInfoRequest, GetModelInfoResponse,
    GetServerInfoRequest, GetServerInfoResponse, HealthCheckRequest, HealthCheckResponse,
    PrepareStructuredOutputRequest, PrepareStructuredOutputResponse,
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

/// Hands each Session call to the test; serves no other RPC.
struct FakeAtom {
    calls: mpsc::UnboundedSender<Call>,
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
        Err(Status::unimplemented("get_server_info"))
    }

    async fn health_check(
        &self,
        _: Request<HealthCheckRequest>,
    ) -> Result<RpcResponse<HealthCheckResponse>, Status> {
        Err(Status::unimplemented("health_check"))
    }

    async fn prepare_structured_output(
        &self,
        _: Request<PrepareStructuredOutputRequest>,
    ) -> Result<RpcResponse<PrepareStructuredOutputResponse>, Status> {
        Err(Status::unimplemented("prepare_structured_output"))
    }
}

/// Serves a fake instance on a free local port: two url@rank workers of it,
/// sharing one client, and its Session calls.
async fn instance() -> ([Arc<dyn Worker>; 2], mpsc::UnboundedReceiver<Call>) {
    let (calls, received) = mpsc::unbounded_channel();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let router = Server::builder().add_service(AtomEngineServer::new(FakeAtom { calls }));
    tokio::spawn(router.serve_with_incoming(TcpIncoming::from(listener)));
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
    let answer = async {
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
    };
    tokio::join!(answer, submit)
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
