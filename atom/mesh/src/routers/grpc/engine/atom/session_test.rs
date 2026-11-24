// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The Session against an in-process `AtomEngine` whose Session calls the test
//! answers: per-seq_id splitting, Abort on drop, failure and reopening, Ready's version.

use std::future::Future;
use std::time::{Duration, Instant};

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
use crate::core::{DPAwareWorkerBuilder, WorkerRegistry};
use crate::routers::grpc::engine::atom::stream::AtomStream;
use crate::routers::grpc::engine::worker_client_cache::GrpcClient;
use crate::routers::token_handle::token_chunk::{FinishReason, MatchedStop, SeqTiming, TokenChunk};

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
        match AtomStream::start(stream, true, Instant::now()).await {
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
        ..Default::default()
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
async fn a_request_that_does_not_stream_completes_with_the_ids_its_entries_carried() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("whole"), vec![1, 2, 3], false);
    let (mut call, whole) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let id = submitted(&mut call, 1).await["whole"].0.seq_id;
    for message in [
        step(&[(id, vec![31], 4)]),
        step(&[(id, vec![32, 33], 0)]),
        // As ATOM sends it: the item carries none of the ids the entries did.
        done(vec![(completion(id, "max_tokens"), Vec::new())]),
    ] {
        call.outbound.send(message).unwrap();
    }

    let whole: Vec<_> = timeout(HANG, whole.unwrap().collect()).await.unwrap();
    let complete = GenerateComplete {
        output_ids: vec![31, 32, 33],
        finish_reason: "max_tokens".to_string(),
        prompt_tokens: 3,
        completion_tokens: 3,
        cached_tokens: 4,
        kv_transfer_params_json: None,
        ..Default::default()
    };
    let expected = GenerateResponse {
        response: Some(Response::Complete(complete)),
    };
    assert!(
        matches!(&whole[..], [Ok(response)] if *response == expected),
        "{whole:?}"
    );
}

#[tokio::test]
async fn a_completion_reports_its_ttft_tpot_and_latency_from_its_times() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("streamed"), vec![1], true);
    let (mut call, streamed) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let whole = submit(&sessions, &ranks[0], "whole", vec![2], false).await;
    let sent = submitted(&mut call, 2).await;
    let (streamed_id, whole_id) = (sent["streamed"].0.seq_id, sent["whole"].0.seq_id);
    // Three tokens each: the first 0.25 s after arrival, the last 1 s after it.
    let timed = |seq_id| pb::Completion {
        arrive_time: 100.0,
        first_token_time: 100.25,
        leave_time: 101.25,
        ..completion(seq_id, "max_tokens")
    };
    for message in [
        step(&[(streamed_id, vec![11], 0), (whole_id, vec![21], 0)]),
        step(&[(streamed_id, vec![12, 13], 0), (whole_id, vec![22, 23], 0)]),
        done(vec![
            (timed(streamed_id), Vec::new()),
            (timed(whole_id), Vec::new()),
        ]),
    ] {
        call.outbound.send(message).unwrap();
    }

    let timing = SeqTiming {
        ttft: 0.25,
        tpot: 0.5,
        latency: 1.25,
    };
    let streamed = chunks(streamed.unwrap()).await;
    assert!(
        matches!(streamed.last(), Some(Ok(TokenChunk::Complete { meta, .. }))
            if meta.timing == Some(timing)),
        "{streamed:?}"
    );
    let whole = chunks(whole).await;
    assert!(
        matches!(&whole[..], [Ok(TokenChunk::Complete { meta, .. })] if meta.timing == Some(timing)),
        "{whole:?}"
    );
}

#[tokio::test]
async fn a_message_past_4_mib_reaches_its_request() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions::default();
    let first = sessions.submit(&ranks[0], spec("large"), vec![1], false);
    let (mut call, large) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let id = submitted(&mut call, 1).await["large"].0.seq_id;
    let blob = "x".repeat(5 << 20);
    let params = pb::Value {
        kind: Some(Kind::StringValue(blob.clone())),
    };
    let item = pb::Completion {
        kv_transfer_params_output: Some(params),
        ..completion(id, "max_tokens")
    };
    call.outbound.send(done(vec![(item, Vec::new())])).unwrap();

    let large: Vec<_> = timeout(HANG, large.unwrap().collect()).await.unwrap();
    let kv = Json::String(blob).to_string();
    assert!(
        matches!(&large[..], [Ok(GenerateResponse { response: Some(Response::Complete(complete)) })]
            if complete.kv_transfer_params_json.as_ref() == Some(&kv)),
        "the 5 MiB Completions failed the Session"
    );
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
    let started = timeout(HANG, AtomStream::start(running, true, Instant::now()))
        .await
        .unwrap();
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
        let started = timeout(
            HANG,
            AtomStream::start(started.unwrap(), true, Instant::now()),
        )
        .await;
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
    call.outbound
        .send(step(&[(after_id, vec![41], 0)]))
        .unwrap();
    let ended = done(vec![(completion(after_id, "eos"), Vec::new())]);
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

/// Waits for the client to drop `call`: it stops reading ATOM's half as well as
/// sending its own.
async fn assert_ends(mut call: Call) {
    let ended = timeout(Duration::from_secs(5), call.inbound.message()).await;
    let ended = ended.expect("the client kept sending on the call");
    assert!(!matches!(ended, Ok(Some(_))), "{ended:?}");
    let dropped = timeout(Duration::from_secs(5), call.outbound.closed()).await;
    dropped.expect("the client kept reading the call");
}

#[tokio::test]
async fn an_opening_without_ready_is_bounded_and_one_no_request_waits_for_ends() {
    let (ranks, mut calls) = instance().await;
    let sessions = AtomSessions {
        opening: Duration::from_millis(300),
        ..Default::default()
    };
    // ATOM accepts the call but never sends Ready: the request fails, the call ends.
    let first = sessions.submit(&ranks[0], spec("first"), vec![1], true);
    let (call, refused) = tokio::join!(
        async { timeout(HANG, calls.recv()).await.unwrap().unwrap() },
        timeout(Duration::from_secs(5), first)
    );
    let refused = refused.expect("the opening was not bounded");
    assert!(
        matches!(&refused, Err(EngineError::Transport(status))
            if status.code() == tonic::Code::DeadlineExceeded),
        "{:?}",
        refused.err()
    );
    assert_ends(call).await;

    // A request that gives up while its Session opens ends that call, long before
    // the bound.
    let patient = AtomSessions::default();
    let abandoned = patient.submit(&ranks[1], spec("abandoned"), vec![1], true);
    let call = tokio::select! {
        _ = abandoned => panic!("opened without Ready"),
        call = calls.recv() => call.unwrap(),
    };
    assert_ends(call).await;

    // The next request opens a new call and runs on it.
    let next = sessions.submit(&ranks[0], spec("next"), vec![1], true);
    let (mut call, opened) = opening(&mut calls, PROTOCOL_VERSION, next).await;
    assert!(opened.is_ok());
    assert!(submitted(&mut call, 1).await.contains_key("next"));
}

#[tokio::test]
async fn a_session_is_released_once_no_worker_of_its_instance_is_registered() {
    let (ranks, mut calls) = instance().await;
    let workers = Arc::new(WorkerRegistry::new());
    for rank in &ranks {
        workers.register(rank.clone());
    }
    let sessions = AtomSessions {
        sweep: Duration::ZERO,
        ..AtomSessions::releasing_removed(workers.clone())
    };
    let first = sessions.submit(&ranks[0], spec("running"), vec![1], false);
    let (mut call, running) = opening(&mut calls, PROTOCOL_VERSION, first).await;
    let running_id = submitted(&mut call, 1).await["running"].0.seq_id;

    // Both ranks are removed; the next request, to another instance, releases the Session.
    for rank in &ranks {
        workers.remove_by_url(rank.url());
    }
    let (others, mut other_calls) = instance().await;
    workers.register(others[0].clone());
    let next = sessions.submit(&others[0], spec("next"), vec![2], false);
    let (_other_call, next) = opening(&mut other_calls, PROTOCOL_VERSION, next).await;
    assert!(next.is_ok());

    // The running request keeps its Session to its end; then the client ends the call,
    // though ATOM keeps its half open.
    let ended = done(vec![(completion(running_id, "eos"), Vec::new())]);
    call.outbound.send(ended).unwrap();
    let running: Vec<_> = timeout(HANG, running.unwrap().collect()).await.unwrap();
    assert!(matches!(&running[..], [Ok(_)]), "{running:?}");
    assert_ends(call).await;
}

#[test]
fn seq_ids_stay_below_2_24_and_skip_those_open_requests_hold() {
    let mut routes = Routes {
        next_seq_id: SEQ_IDS - 2,
        ..Default::default()
    };
    for held in [SEQ_IDS - 1, 0] {
        let (tx, _) = mpsc::channel(1);
        let route = Route {
            tx,
            overflowed: false,
            stream: false,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            ids: Vec::new(),
        };
        routes.open.insert(held, route);
    }
    // The second search passes the held 2^24 - 1 and 0, wrapping below 2^24.
    let ids = [routes.free_seq_id(), routes.free_seq_id()];
    assert_eq!(ids, [Some(SEQ_IDS - 2), Some(1)]);
}

#[test]
fn a_submit_queued_after_an_abort_goes_out_after_it() {
    let submit = |id: u64| {
        let mut spec = spec("r");
        spec.seq_id = id;
        Command::Submit(Box::new(spec), vec![1])
    };
    // The Abort of seq_id 1 was queued before seq_id 1 was submitted again.
    let messages = encode(vec![submit(1), Command::Abort(1), submit(1), submit(2)]);
    let bodies: Vec<_> = messages
        .into_iter()
        .map(
            |message| match Inbound::decode(message.frames).unwrap().body {
                Body::Submit(batch) => {
                    ("submit", batch.requests.iter().map(|r| r.seq_id).collect())
                }
                Body::Abort(abort) => ("abort", abort.seq_ids),
                body => panic!("{body:?}"),
            },
        )
        .collect();
    let expected = [
        ("submit", vec![1]),
        ("abort", vec![1]),
        ("submit", vec![1, 2]),
    ];
    assert_eq!(bodies, expected);
}

#[test]
fn a_finished_stream_dropped_after_its_seq_id_is_reused_aborts_nothing() {
    let (commands, mut queued) = mpsc::unbounded_channel();
    let routes = Arc::new(Mutex::new(Routes::default()));
    let (session, _) = Session::new(routes, commands);
    // What the writer sends of the queue, taking each prompt.
    let mut sent = || {
        let queued: Vec<_> = std::iter::from_fn(|| queued.try_recv().ok()).collect();
        session.routes.lock().take_unsent(queued)
    };
    let finished = session.submit(spec("finished"), vec![1], true).unwrap();
    assert!(matches!(&sent()[..], [Command::Submit(..)]));
    let ended = done(vec![(completion(finished.seq_id, "eos"), Vec::new())]);
    let ended = Inbound::decode(ended.unwrap().frames).unwrap();
    session.routes.lock().route(&ended).unwrap();
    // The allocator has wrapped around to the finished request's seq_id.
    session.routes.lock().next_seq_id = finished.seq_id;
    let reused = session.submit(spec("reused"), vec![2], true).unwrap();
    assert_eq!(reused.seq_id, finished.seq_id);
    assert!(matches!(&sent()[..], [Command::Submit(..)]));

    drop(finished);
    assert!(sent().is_empty());
    // Dropped unfinished, the request now holding the seq_id is aborted.
    drop(reused);
    assert!(matches!(&sent()[..], [Command::Abort(0)]));
}

/// The budget bytes a request with `spec(name)` and an `ids`-long prompt holds unsent.
fn unsent_bytes(name: &str, ids: usize) -> usize {
    4 * ids + spec(name).encoded_len()
}

#[test]
fn unsent_requests_are_bounded_and_a_dropped_unsent_request_is_not_sent() {
    let (commands, mut queued) = mpsc::unbounded_channel();
    let routes = Routes {
        unsent_budget: unsent_bytes("first", 2) + unsent_bytes("secnd", 2),
        ..Default::default()
    };
    let (session, _) = Session::new(Arc::new(Mutex::new(routes)), commands);
    // ATOM reads nothing: two requests fill the budget, a third is refused.
    let first = session.submit(spec("first"), vec![1, 2], true).unwrap();
    let second = session.submit(spec("secnd"), vec![3, 4], true).unwrap();
    let refused = session.submit(spec("third"), vec![5], true);
    assert!(
        matches!(&refused, Err(EngineError::Http { status, .. }) if status.as_u16() == 503),
        "{:?}",
        refused.err()
    );

    // Dropped unsent, a request frees its prompt and seq_id and sends no Abort...
    let first_id = first.seq_id;
    drop(first);
    assert!(!session.routes.lock().open.contains_key(&first_id));
    // ...but its spec is still queued, so its bytes stay held until the writer runs.
    assert!(session.submit(spec("third"), vec![5], true).is_err());
    let waiting: Vec<_> = std::iter::from_fn(|| queued.try_recv().ok()).collect();
    let sent = session.routes.lock().take_unsent(waiting);
    let third = session.submit(spec("third"), vec![5], true).unwrap();
    let waiting: Vec<_> = std::iter::from_fn(|| queued.try_recv().ok()).collect();
    let sent: Vec<_> = sent
        .into_iter()
        .chain(session.routes.lock().take_unsent(waiting))
        .map(|command| match command {
            Command::Submit(spec, prompt) => (spec.seq_id, prompt),
            _ => panic!("only Submits were queued"),
        })
        .collect();
    // The writer sends only the requests still waiting, with their prompts.
    assert_eq!(sent, [(second.seq_id, vec![3, 4]), (third.seq_id, vec![5])]);
    assert_eq!(session.routes.lock().unsent_bytes, 0);
}

#[test]
fn a_first_request_larger_than_the_budget_is_refused() {
    let (commands, _queued) = mpsc::unbounded_channel();
    let routes = Routes {
        unsent_budget: unsent_bytes("big", 2) - 1,
        ..Default::default()
    };
    let (session, _) = Session::new(Arc::new(Mutex::new(routes)), commands);
    let refused = session.submit(spec("big"), vec![1, 2], true);
    assert!(
        matches!(&refused, Err(EngineError::Http { status, .. }) if status.as_u16() == 400),
        "{:?}",
        refused.err()
    );
    assert_eq!(session.routes.lock().unsent_bytes, 0);
    assert!(session.submit(spec("big"), vec![1], true).is_ok());
}

#[test]
fn requests_cancelled_while_the_writer_stalls_cannot_queue_without_bound() {
    let (commands, mut queued) = mpsc::unbounded_channel();
    let routes = Routes {
        unsent_budget: 2 * unsent_bytes("r", 1),
        ..Default::default()
    };
    let (session, _) = Session::new(Arc::new(Mutex::new(routes)), commands);
    // The writer stalls; each request is cancelled at once, its spec left queued.
    let cancelled = (0..10)
        .take_while(|_| session.submit(spec("r"), vec![1], true).is_ok())
        .count();
    assert_eq!(cancelled, 2);
    // Once the writer discards them, requests are taken again.
    let waiting: Vec<_> = std::iter::from_fn(|| queued.try_recv().ok()).collect();
    assert_eq!(waiting.len(), 2);
    assert!(session.routes.lock().take_unsent(waiting).is_empty());
    assert!(session.submit(spec("r"), vec![1], true).is_ok());
}

#[test]
fn a_stream_left_unread_fails_alone_and_is_aborted_once_keeping_its_seq_id() {
    let (commands, mut queued) = mpsc::unbounded_channel();
    let routes = Arc::new(Mutex::new(Routes::default()));
    let (session, _) = Session::new(routes, commands);
    let mut unread = session.submit(spec("unread"), vec![1], true).unwrap();
    let mut read = session.submit(spec("read"), vec![2], true).unwrap();
    let (unread_id, read_id) = (unread.seq_id, read.seq_id);
    let decoded = |message: Result<Frames, Status>| Inbound::decode(message.unwrap().frames);

    // Each step reaches both; only `read` is read.
    let mut aborted = Vec::new();
    for _ in 0..ROUTE_BUFFER + 10 {
        let message = decoded(step(&[(unread_id, vec![7], 0), (read_id, vec![8], 0)]));
        aborted.extend(session.routes.lock().route(&message.unwrap()).unwrap());
        assert!(matches!(read.rx.try_recv(), Ok(Ok(_))));
    }
    assert_eq!(aborted, [unread_id]);
    let outputs: Vec<_> = std::iter::from_fn(|| unread.rx.try_recv().ok()).collect();
    assert_eq!(outputs.len(), ROUTE_BUFFER);
    assert!(outputs[..ROUTE_BUFFER - 1].iter().all(Result::is_ok));
    assert!(
        matches!(outputs.last(), Some(Err(status)) if status.code() == tonic::Code::ResourceExhausted)
    );

    // Its seq_id is held until its Completion, which it is not sent; dropped, it sends
    // no second Abort.
    assert!(session.routes.lock().open.contains_key(&unread_id));
    let ended = decoded(done(vec![(completion(unread_id, "abort"), Vec::new())]));
    session.routes.lock().route(&ended.unwrap()).unwrap();
    assert!(!session.routes.lock().open.contains_key(&unread_id));
    assert!(unread.rx.try_recv().is_err());
    let _ = std::iter::from_fn(|| queued.try_recv().ok()).count();
    drop(unread);
    assert!(queued.try_recv().is_err());
}
