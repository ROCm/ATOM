//! The wire framing, the launch spec, one Core's link and the client: a
//! message decodes back with its buffers and a malformed one is refused,
//! `EngineSpec` parses a hand-written spec, a `CoreLink` handshakes with a
//! fake Core on real ipc sockets and routes input and control to it, and gives
//! up on a Core that stops before it is ready or exits, or when interrupted,
//! and `EngineClient` drives fake Cores (`test_support`) through the
//! protocol's every ending.
//!
//! The protocol across languages: the Core's messages as Python encodes them
//! (`tests/fixtures/engine_v1/core_*`, written by
//! `tests/test_engine_v1_cross_language.py`) drive the client here, and the
//! client's own messages are kept as `client_*` there for Python to decode.
//! `ATOM_UPDATE_FIXTURES=1` rewrites the `client_*` ones.

use std::{future::poll_fn, path::PathBuf, thread, time::Duration};

use prost::Message as _;
use serde_json::json;
use tempfile::TempDir;

use super::{
    client::{EngineClient, SeqEvent, SeqFailure, SeqHandle, Submission},
    link::{self, CoreLink},
    spec::{BalanceStrategy, CoreSpec, EngineSpec},
    test_support::{self, token_frame, FakeCore, WAIT},
    wire::{self, pb, Inbound, PROTOCOL_VERSION},
};

fn inbound(frames: Vec<Vec<u8>>) -> Result<Inbound, String> {
    Inbound::decode(frames.into_iter().map(zmq::Message::from).collect())
}

fn buffer(frame: u32, offset: u64, nbytes: u64) -> Option<pb::BufferRef> {
    Some(pb::BufferRef {
        frame,
        offset,
        nbytes,
    })
}

fn envelope_frame(protocol_version: u32, body: Option<pb::envelope::Body>) -> Vec<u8> {
    pb::Envelope {
        protocol_version,
        body,
    }
    .encode_to_vec()
}

#[test]
fn a_message_decodes_back_with_the_buffers_its_refs_name() {
    let floats: Vec<u8> = [0.5f64, -1.25]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let completion = pb::Completion {
        completion_token_ids: buffer(1, 4, 8),
        logprobs: buffer(2, 0, 16),
        ..Default::default()
    };
    let body = pb::envelope::Body::Completions(pb::Completions {
        items: vec![completion],
    });
    let frames = wire::encode(body, vec![wire::token_frame(&[7, 8, u32::MAX]), floats]);
    assert_eq!(frames.len(), 3);

    let message = inbound(frames).unwrap();
    let pb::envelope::Body::Completions(batch) = &message.body else {
        panic!("decoded {:?}", message.body);
    };
    let completion = &batch.items[0];
    // u32::MAX crosses as int32 -1 and comes back as itself.
    assert_eq!(
        message
            .tokens(completion.completion_token_ids.as_ref())
            .unwrap(),
        vec![8, u32::MAX]
    );
    assert_eq!(
        message.floats(completion.logprobs.as_ref()).unwrap(),
        vec![0.5, -1.25]
    );
    assert!(message.tokens(None).unwrap().is_empty());
}

#[test]
fn a_malformed_message_or_an_out_of_bounds_ref_is_refused() {
    let stopped = || Some(pb::envelope::Body::Stopped(pb::Stopped {}));
    assert!(inbound(vec![]).is_err());
    assert!(inbound(vec![envelope_frame(PROTOCOL_VERSION, None)]).is_err());
    let err = inbound(vec![envelope_frame(PROTOCOL_VERSION + 1, stopped())])
        .err()
        .unwrap();
    assert!(err.contains("protocol version"), "{err}");

    let message = inbound(vec![
        envelope_frame(PROTOCOL_VERSION, stopped()),
        vec![0; 8],
    ])
    .unwrap();
    // Frame 0 is the envelope, never a buffer.
    let err = message.tokens(buffer(0, 0, 4).as_ref()).err().unwrap();
    assert!(err.contains("envelope"), "{err}");
    assert!(message.tokens(buffer(1, 4, 8).as_ref()).is_err());
    // Within the frame, but not a whole number of elements.
    assert!(message.tokens(buffer(1, 0, 6).as_ref()).is_err());
}

fn spec_json() -> serde_json::Value {
    json!({
        "model": {
            "served_name": "m", "tokenizer_path": "/m", "default_max_tokens": 16,
            "template_efforts": ["low", "high", "max"],
        },
        "balance": {"strategy": "least_tokens", "req_equiv": 512},
        "engines": [
            {"dp_rank": 0, "pid": 10, "input": "ipc:///i0", "control": "ipc:///c0", "output": "ipc:///o0"},
            {"dp_rank": 1, "pid": 11, "input": "ipc:///i1", "control": "ipc:///c1", "output": "ipc:///o1"},
        ],
    })
}

#[test]
fn a_spec_lists_its_cores_and_refuses_cores_it_cannot_place() {
    let spec = EngineSpec::from_json(&spec_json().to_string()).unwrap();
    assert_eq!(spec.engines[1].pid, 11);
    assert_eq!(spec.model.served_name, "m");
    assert_eq!(spec.balance.strategy, BalanceStrategy::LeastTokens);
    // Fields added later default.
    assert!(!spec.balance.session_affinity);
    assert!(spec.model.reasoning_toggle.is_none());

    let mut no_cores = spec_json();
    no_cores["engines"] = json!([]);
    assert!(EngineSpec::from_json(&no_cores.to_string()).is_err());
    let mut swapped = spec_json();
    swapped["engines"].as_array_mut().unwrap().reverse();
    assert!(EngineSpec::from_json(&swapped.to_string()).is_err());
}

fn core_spec(dir: &TempDir, pid: u32) -> CoreSpec {
    let address = |name: &str| format!("ipc://{}", dir.path().join(name).display());
    CoreSpec {
        dp_rank: 0,
        pid,
        input: address("input"),
        control: address("control"),
        output: address("output"),
    }
}

/// Connect as EngineCore does (a DEALER on input and control, each announcing
/// itself with an empty frame; a PUSH on output), send `first`, then hand
/// back the next message on input and on control, or none if either does not
/// come within the receive timeout.
fn fake_core(
    spec: CoreSpec,
    first: pb::envelope::Body,
) -> thread::JoinHandle<Option<[Inbound; 2]>> {
    thread::spawn(move || {
        let ctx = zmq::Context::new();
        let dealer = |address: &str| {
            let socket = ctx.socket(zmq::DEALER).unwrap();
            // A message the link sent elsewhere fails the test, not hangs it;
            // short, since a test the link sends nothing to waits it out.
            socket.set_rcvtimeo(1000).unwrap();
            socket.connect(address).unwrap();
            socket.send(Vec::<u8>::new(), 0).unwrap();
            socket
        };
        let input = dealer(&spec.input);
        let control = dealer(&spec.control);
        let output = ctx.socket(zmq::PUSH).unwrap();
        output.connect(&spec.output).unwrap();
        output
            .send_multipart(wire::encode(first, vec![]), 0)
            .unwrap();
        let receive = |socket: &zmq::Socket| Inbound::decode(link::recv_frames(socket).ok()?).ok();
        Some([receive(&input)?, receive(&control)?])
    })
}

fn never_interrupted() -> bool {
    false
}

#[test]
fn a_link_handshakes_and_routes_input_and_control_to_their_core() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let ready = pb::envelope::Body::Ready(pb::Ready {
        max_pool_tokens: Some(4096),
    });
    let core = fake_core(spec.clone(), ready);

    let ctx = zmq::Context::new();
    let (link, _output, max_pool_tokens) =
        CoreLink::connect(&ctx, spec, &never_interrupted).unwrap();
    assert_eq!(max_pool_tokens, Some(4096));

    let tokens = wire::token_frame(&[1, 2, 3]);
    let submit = pb::SubmitBatch {
        requests: vec![pb::RequestSpec {
            prompt_token_ids: buffer(1, 0, tokens.len() as u64),
            ..Default::default()
        }],
    };
    link.send_input(wire::encode(
        pb::envelope::Body::Submit(submit),
        vec![tokens],
    ))
    .unwrap();
    let abort = pb::Abort { seq_ids: vec![7] };
    link.send_control(wire::encode(pb::envelope::Body::Abort(abort), vec![]))
        .unwrap();

    let [input, control] = core
        .join()
        .unwrap()
        .expect("the core received both messages");
    let pb::envelope::Body::Submit(batch) = &input.body else {
        panic!("input carried {:?}", input.body);
    };
    assert_eq!(
        input
            .tokens(batch.requests[0].prompt_token_ids.as_ref())
            .unwrap(),
        vec![1, 2, 3]
    );
    assert!(matches!(
        &control.body,
        pb::envelope::Body::Abort(a) if a.seq_ids == [7]
    ));
}

#[test]
fn a_link_gives_up_on_a_core_that_exits() {
    // No process has pid 0 in /proc, so the Core counts as exited.
    let dir = TempDir::new().unwrap();
    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, core_spec(&dir, 0), &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("exited"), "{err}");
}

#[test]
fn a_link_refuses_a_core_that_stops_before_it_is_ready() {
    let dir = TempDir::new().unwrap();
    let spec = core_spec(&dir, std::process::id());
    let core = fake_core(spec.clone(), pb::envelope::Body::Stopped(pb::Stopped {}));

    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, spec, &never_interrupted)
        .err()
        .unwrap();
    assert!(err.contains("stopped before it was ready"), "{err}");
    // The link sent the Core nothing.
    assert!(core.join().unwrap().is_none());
}

#[test]
fn a_link_gives_up_when_interrupted() {
    // The Core is this process, so it never counts as exited, and nothing
    // connects: only the interruption ends the wait.
    let dir = TempDir::new().unwrap();
    let ctx = zmq::Context::new();
    let err = CoreLink::connect(&ctx, core_spec(&dir, std::process::id()), &|| true)
        .err()
        .unwrap();
    assert!(err.contains("interrupted"), "{err}");
}

/// A client connected to fake Cores, one per `pools` entry.
struct Harness {
    client: EngineClient,
    cores: Vec<FakeCore>,
    _dir: TempDir,
}

impl Harness {
    fn start(pools: &[Option<u64>]) -> Self {
        let dir = TempDir::new().unwrap();
        let spec = test_support::engine_spec(
            dir.path(),
            pools.len(),
            test_support::model_spec("fake", "fake"),
        );
        let (client, cores) = test_support::connect(&spec, pools);
        Self {
            client,
            cores,
            _dir: dir,
        }
    }

    fn submit(&self, n: u32, stream: bool, logprobs: bool) -> Vec<SeqHandle> {
        let sampling = pb::SamplingParams {
            temperature: Some(1.0),
            max_tokens: Some(8),
            n: Some(n),
            logprobs: logprobs.then_some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
            ..Default::default()
        };
        self.client
            .submit(Submission {
                request_id: "req",
                prompt: &[11, 12, 13],
                sampling,
                stop_token_sequences: vec![],
                n,
                session: None,
                rank: None,
                stream,
            })
            .unwrap()
    }

    /// The Core the first submission went to, and what it received.
    fn first_submission(&self) -> (usize, Vec<pb::RequestSpec>, Vec<Vec<u8>>) {
        for (rank, core) in self.cores.iter().enumerate() {
            if let Some((specs, data)) = FakeCore::recv_on(&core.input, Duration::from_millis(500))
                .map(|(body, data)| match body {
                    pb::envelope::Body::Submit(batch) => (batch.requests, data),
                    other => panic!("expected SubmitBatch, got {other:?}"),
                })
            {
                return (rank, specs, data);
            }
        }
        panic!("no Core received the submission");
    }
}

async fn next_event(handle: &mut SeqHandle) -> SeqEvent {
    tokio::time::timeout(WAIT, poll_fn(|cx| handle.poll_event(cx)))
        .await
        .expect("an event within the timeout")
        .expect("the handle is still open")
}

#[tokio::test]
async fn ready_sets_the_pool_limit_and_a_stream_ends_on_its_finished_entry() {
    let harness = Harness::start(&[Some(1000), Some(600)]);
    assert_eq!(harness.client.max_pool_tokens(), Some(600));
    assert_eq!(harness.client.core_count(), 2);

    let mut handles = harness.submit(1, true, false);
    let (rank, specs, data) = harness.first_submission();
    let [spec] = &specs[..] else {
        panic!("expected one RequestSpec, got {specs:?}")
    };
    assert_eq!(spec.data_parallel_rank, Some(rank as i64));
    assert_eq!(spec.external_request_id.as_deref(), Some("req"));
    assert!(!spec.needs_independent_noise);
    let prompt = spec.prompt_token_ids.as_ref().unwrap();
    assert_eq!(prompt.frame, 1);
    assert_eq!(data[0], token_frame(&[11, 12, 13]));

    let core = &harness.cores[rank];
    core.stream(&[(spec.seq_id, &[7], None)]);
    core.stream(&[(spec.seq_id, &[8], Some("eos"))]);
    let handle = &mut handles[0];
    assert!(matches!(next_event(handle).await, SeqEvent::Tokens(t) if t == [7]));
    assert!(matches!(next_event(handle).await, SeqEvent::Tokens(t) if t == [8]));
    match next_event(handle).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [7, 8]);
            assert_eq!(done.leave_reason, "eos");
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 3));
            assert!(done.logprobs.is_none());
        }
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn a_logprobs_request_ends_on_its_completion() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, specs, _) = harness.first_submission();
    let seq_id = specs[0].seq_id;
    let core = &harness.cores[0];
    // Its finished stream entry does not end it: only the Completion has logprobs.
    core.stream(&[(seq_id, &[5, 6], Some("max_tokens"))]);
    let completion = pb::Completion {
        seq_id,
        completion_token_ids: Some(pb::BufferRef {
            frame: 1,
            offset: 0,
            nbytes: 8,
        }),
        logprobs: Some(pb::BufferRef {
            frame: 2,
            offset: 0,
            nbytes: 16,
        }),
        num_prompt_tokens: 3,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    let logprobs: Vec<u8> = [-0.5f64, -1.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    core.send(
        pb::envelope::Body::Completions(pb::Completions {
            items: vec![completion],
        }),
        vec![token_frame(&[5, 6]), logprobs],
    );
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [5, 6]);
            assert_eq!(done.logprobs.as_deref(), Some(&[-0.5, -1.25][..]));
            assert_eq!(done.leave_reason, "max_tokens");
        }
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn rejections_map_to_failures_by_code() {
    use pb::rejected_request::Code;
    let harness = Harness::start(&[None]);
    for (code, expect_invalid) in [(Code::Invalid, true), (Code::Overloaded, false)] {
        let mut handles = harness.submit(1, false, false);
        let (_, specs, _) = harness.first_submission();
        harness.cores[0].send(
            pb::envelope::Body::Rejected(pb::Rejected {
                items: vec![pb::RejectedRequest {
                    seq_id: specs[0].seq_id,
                    reason: "no".into(),
                    code: code as i32,
                }],
            }),
            vec![],
        );
        match next_event(&mut handles[0]).await {
            SeqEvent::Failed(SeqFailure::Invalid(reason)) if expect_invalid => {
                assert_eq!(reason, "no")
            }
            SeqEvent::Failed(SeqFailure::Unavailable(_)) if !expect_invalid => {}
            other => panic!("{code:?}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn dropping_an_unfinished_handle_aborts_its_sequence() {
    let harness = Harness::start(&[None]);
    let handles = harness.submit(1, true, false);
    let (_, specs, _) = harness.first_submission();
    drop(handles);
    match harness.cores[0].control_message() {
        pb::envelope::Body::Abort(abort) => assert_eq!(abort.seq_ids, [specs[0].seq_id]),
        other => panic!("expected Abort, got {other:?}"),
    }
}

#[tokio::test]
async fn siblings_spread_over_ranks_and_share_one_prompt_frame() {
    let harness = Harness::start(&[None, None]);
    let handles = harness.submit(4, false, false);
    assert_eq!(handles.len(), 4);
    let mut siblings = Vec::new();
    for core in &harness.cores {
        let (specs, data) = core.submitted().expect("each rank gets its share");
        assert_eq!(data.len(), 1, "one prompt frame per message");
        assert_eq!(data[0], token_frame(&[11, 12, 13]));
        for spec in specs {
            assert!(spec.needs_independent_noise);
            assert_eq!(spec.external_request_id, None);
            assert_eq!(spec.parent_request_id.as_deref(), Some("req"));
            assert_eq!(spec.prompt_token_ids.as_ref().unwrap().frame, 1);
            siblings.push(spec.sibling_index);
        }
    }
    siblings.sort();
    assert_eq!(siblings, [0, 1, 2, 3]);
}

#[tokio::test]
async fn a_core_that_stops_unasked_loses_the_client() {
    let harness = Harness::start(&[None, None]);
    let mut handles = harness.submit(1, true, false);
    let (rank, _, _) = harness.first_submission();
    assert_eq!(harness.client.lost(), None);
    harness.cores[rank].send(pb::envelope::Body::Stopped(pb::Stopped {}), vec![]);
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Unavailable(_))
    ));
    let reason = tokio::time::timeout(WAIT, harness.client.wait_lost())
        .await
        .expect("the loss is reported");
    assert_eq!(reason, format!("EngineCore DP rank {rank} stopped"));
    assert_eq!(harness.client.alive_core_count(), 1);
    // The live rank takes nothing more either: the server is exiting.
    let refused = harness.client.submit(Submission {
        request_id: "late",
        prompt: &[1],
        sampling: Default::default(),
        stop_token_sequences: vec![],
        n: 1,
        session: None,
        rank: None,
        stream: false,
    });
    assert!(matches!(refused, Err(SeqFailure::Unavailable(_))));
    let live = &harness.cores[1 - rank];
    assert!(FakeCore::recv_on(&live.input, Duration::from_millis(200)).is_none());
}

#[tokio::test]
async fn an_undecodable_output_loses_its_core() {
    // One envelope that does not decode, one whose stream columns disagree:
    // either may have held a sequence's end, so the Core is lost with them.
    let bad_outputs = [
        vec![b"not an envelope".to_vec()],
        wire::encode(
            pb::envelope::Body::Stream(pb::StreamBatch {
                seq_ids: vec![1, 2],
                token_counts: vec![1],
                tokens: vec![7],
                ..Default::default()
            }),
            vec![],
        ),
    ];
    for frames in bad_outputs {
        let harness = Harness::start(&[None, None]);
        let mut handles = harness.submit(1, true, false);
        let (rank, _, _) = harness.first_submission();
        harness.cores[rank].send_frames(frames);
        assert!(matches!(
            next_event(&mut handles[0]).await,
            SeqEvent::Failed(SeqFailure::Unavailable(_))
        ));
        let reason = tokio::time::timeout(WAIT, harness.client.wait_lost())
            .await
            .expect("the loss is reported");
        assert!(
            reason.starts_with(&format!(
                "EngineCore DP rank {rank} sent an undecodable output"
            )),
            "{reason}"
        );
        assert_eq!(harness.client.alive_core_count(), 1);
        assert_eq!(harness.client.rank_loads().0[rank], 0);
    }
}

#[tokio::test]
async fn shutdown_asks_every_core_to_stop_and_waits_for_them() {
    let harness = Harness::start(&[None, None]);
    // Answer each Shutdown with Stopped from a thread, as a Core would.
    let cores = harness.cores;
    let replies = thread::spawn(move || {
        for core in &cores {
            assert!(matches!(
                core.control_message(),
                pb::envelope::Body::Shutdown(_)
            ));
            core.send(pb::envelope::Body::Stopped(pb::Stopped {}), vec![]);
        }
        cores
    });
    harness.client.shutdown(WAIT);
    replies.join().unwrap();
    assert_eq!(harness.client.alive_core_count(), 0);
    // Stopping when asked to is not a loss.
    assert_eq!(harness.client.lost(), None);
}

#[tokio::test]
async fn an_explicit_rank_places_every_sibling_there() {
    let harness = Harness::start(&[None, None]);
    let submit = |rank| {
        harness.client.submit(Submission {
            request_id: "req",
            prompt: &[1, 2],
            sampling: pb::SamplingParams {
                temperature: Some(1.0),
                n: Some(3),
                ..Default::default()
            },
            stop_token_sequences: vec![],
            n: 3,
            session: None,
            rank: Some(rank),
            stream: false,
        })
    };
    let _handles = submit(1).unwrap();
    let (specs, _) = harness.cores[1].submitted().expect("rank 1 gets them all");
    assert_eq!(specs.len(), 3);
    assert!(FakeCore::recv_on(&harness.cores[0].input, Duration::from_millis(200)).is_none());
    assert!(matches!(submit(2), Err(SeqFailure::Invalid(_))));
}

#[tokio::test]
async fn a_logprobs_request_reports_the_prefix_hit_not_the_computed_kv() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    let (_, specs, _) = harness.first_submission();
    let core = &harness.cores[0];
    // The stream says 3 prompt tokens hit the cache ...
    core.stream(&[(specs[0].seq_id, &[5], Some("max_tokens"))]);
    // ... while the Completion counts all the prompt's KV as computed.
    let completion = pb::Completion {
        seq_id: specs[0].seq_id,
        completion_token_ids: Some(pb::BufferRef {
            frame: 1,
            offset: 0,
            nbytes: 4,
        }),
        num_prompt_tokens: 3,
        num_cached_tokens: 3_000,
        leave_reason: "max_tokens".into(),
        ..Default::default()
    };
    core.send(
        pb::envelope::Body::Completions(pb::Completions {
            items: vec![completion],
        }),
        vec![token_frame(&[5])],
    );
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => assert_eq!(done.cached_tokens, 3),
        other => panic!("expected Finished, got {other:?}"),
    }
}

#[tokio::test]
async fn an_ended_handle_dropped_later_sends_no_abort() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    let (_, specs, _) = harness.first_submission();
    let core = &harness.cores[0];
    core.stream(&[(specs[0].seq_id, &[9], Some("eos"))]);
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Finished(_)
    ));
    drop(handles);
    assert!(FakeCore::recv_on(&core.control, Duration::from_millis(200)).is_none());
    assert_eq!(harness.client.rank_loads(), (vec![0], vec![0]));
}

#[tokio::test]
async fn shutdown_fails_what_is_still_in_flight() {
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, true, false);
    let _ = harness.first_submission();
    // The Core never answers the Shutdown.
    harness.client.shutdown(Duration::from_millis(100));
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Unavailable(_))
    ));
    assert_eq!(harness.client.lost(), None);
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/engine_v1")
        .join(format!("{name}.frames"))
}

/// A message as `u32le length, bytes` per frame, as the Python test keeps it.
fn read_fixture(name: &str) -> Vec<Vec<u8>> {
    let path = fixture(name);
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut frames = Vec::new();
    let mut rest = &data[..];
    while !rest.is_empty() {
        let n = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        frames.push(rest[4..4 + n].to_vec());
        rest = &rest[4 + n..];
    }
    frames
}

/// The next whole message on `socket`, frames as sent.
fn raw_message(socket: &zmq::Socket) -> Vec<Vec<u8>> {
    assert!(
        socket.poll(zmq::POLLIN, WAIT.as_millis() as i64).unwrap() > 0,
        "no message within the timeout"
    );
    socket.recv_multipart(0).unwrap()
}

/// Fail when any of the client's `messages` drifted from its fixture, which
/// Python decodes; under `ATOM_UPDATE_FIXTURES`, rewrite them all and fail (a
/// rewrite is not a pass).
fn check_client_fixtures(messages: &[(&str, Vec<Vec<u8>>)]) {
    if std::env::var_os("ATOM_UPDATE_FIXTURES").is_some() {
        for (name, frames) in messages {
            let mut data = Vec::new();
            for frame in frames {
                data.extend_from_slice(&(frame.len() as u32).to_le_bytes());
                data.extend_from_slice(frame);
            }
            std::fs::write(fixture(name), data).unwrap();
        }
        panic!("rewrote the client_* fixtures; rerun without ATOM_UPDATE_FIXTURES");
    }
    // Arrival times are the clock's; everything else must match.
    let envelope = |frames: &[Vec<u8>]| {
        let mut envelope = pb::Envelope::decode(&frames[0][..]).unwrap();
        if let Some(pb::envelope::Body::Submit(batch)) = &mut envelope.body {
            batch.requests.iter_mut().for_each(|r| r.arrive_time = 0.0);
        }
        envelope
    };
    for (name, frames) in messages {
        let kept = read_fixture(name);
        assert!(
            envelope(frames) == envelope(&kept) && frames[1..] == kept[1..],
            "{name}.frames is stale: rerun with ATOM_UPDATE_FIXTURES=1, then \
             tests/test_engine_v1_cross_language.py against it"
        );
    }
}

#[tokio::test]
async fn the_clients_messages_match_what_python_decodes() {
    let harness = Harness::start(&[None]);
    let sampling = pb::SamplingParams {
        temperature: Some(0.7),
        top_k: Some(20),
        top_p: Some(0.9),
        max_tokens: Some(64),
        stop_strings: Some(pb::StopStrings {
            value: Some(pb::stop_strings::Value::Many(pb::StringList {
                items: vec!["\n\n".into()],
            })),
        }),
        n: Some(2),
        logprobs: Some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
        required_channel: Some(pb::RequiredChannel {
            after: vec![1],
            opening: vec![2, 3],
            opens_at: 1,
            end_ids: vec![4],
        }),
        ..Default::default()
    };
    let handles = harness
        .client
        .submit(Submission {
            request_id: "req",
            prompt: &[11, 12, 13],
            sampling,
            stop_token_sequences: vec![
                pb::TokenIds { ids: vec![5, 6] },
                pb::TokenIds { ids: vec![7] },
            ],
            n: 2,
            session: Some("sess"),
            rank: None,
            stream: false,
        })
        .unwrap();
    let core = &harness.cores[0];
    let submit = raw_message(&core.input);
    drop(handles);
    let abort = raw_message(&core.control);
    let _second_abort = raw_message(&core.control);
    harness.client.shutdown(Duration::from_millis(100));
    let shutdown = raw_message(&core.control);
    check_client_fixtures(&[
        ("client_submit", submit),
        ("client_abort", abort),
        ("client_shutdown", shutdown),
    ]);
}

#[tokio::test]
async fn python_core_messages_drive_the_client() {
    let decode = |name| {
        inbound(read_fixture(name))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .body
    };
    assert!(matches!(
        decode("core_ready"),
        pb::envelope::Body::Ready(pb::Ready {
            max_pool_tokens: Some(4096)
        })
    ));
    assert!(matches!(
        decode("core_stopped"),
        pb::envelope::Body::Stopped(_)
    ));

    // KV block ids packed, beside a value nested as deep as Python may send
    // one (`wire.MAX_VALUE_DEPTH`): still under prost's recursion limit.
    let pb::envelope::Body::Completions(deep) = decode("core_deep_completions") else {
        panic!("core_deep_completions is not a Completions");
    };
    let params = deep.items[0].kv_transfer_params_output.as_ref().unwrap();
    let Some(pb::value::Kind::MapValue(top)) = &params.kind else {
        panic!("kv_transfer_params_output is not a map");
    };
    assert_eq!(
        top.entries[0].value.as_ref().unwrap().kind,
        Some(pb::value::Kind::IntListValue(pb::IntList { items: vec![3, 1, 2] }))
    );
    let (mut value, mut maps) = (params, 0);
    while let Some(pb::value::Kind::MapValue(map)) = &value.kind {
        maps += 1;
        value = map.entries.last().unwrap().value.as_ref().unwrap();
    }
    assert_eq!(maps, 30);
    assert_eq!(value.kind, Some(pb::value::Kind::IntValue(5)));

    // Each harness's first sequence is 0, the one the fixtures name.
    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, true, false);
    harness.cores[0].send_frames(read_fixture("core_stream"));
    assert!(matches!(next_event(&mut handles[0]).await, SeqEvent::Tokens(t) if t == [7, 8]));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [7, 8]);
            assert_eq!(done.leave_reason, "eos");
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 3));
        }
        other => panic!("expected Finished, got {other:?}"),
    }

    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, true);
    harness.cores[0].send_frames(read_fixture("core_completions"));
    match next_event(&mut handles[0]).await {
        SeqEvent::Finished(done) => {
            assert_eq!(done.token_ids, [5, 6]);
            assert_eq!(done.logprobs.as_deref(), Some(&[-0.5, -1.25][..]));
            // The prefix hit comes from the stream, and this request had no
            // stream entry; the Completion's count is the KV computed.
            assert_eq!((done.prompt_tokens, done.cached_tokens), (3, 0));
            assert_eq!(done.leave_reason, "max_tokens");
        }
        other => panic!("expected Finished, got {other:?}"),
    }

    let harness = Harness::start(&[None]);
    let mut handles = harness.submit(1, false, false);
    harness.cores[0].send_frames(read_fixture("core_rejected"));
    assert!(matches!(
        next_event(&mut handles[0]).await,
        SeqEvent::Failed(SeqFailure::Invalid(reason)) if reason == "bad"
    ));
}

#[test]
fn the_spec_python_writes_parses() {
    // Written by tests/entrypoints/test_engine_spec.py (`build_engine_spec`).
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/engine_v1/engine_spec.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let spec = EngineSpec::from_json(&json).unwrap();
    let model = &spec.model;
    assert_eq!(model.served_name, "served");
    assert_eq!(model.tokenizer_path, "/models/m");
    assert_eq!(model.chat_template.as_deref(), Some("/t/chat.jinja"));
    assert_eq!(model.default_chat_template_kwargs["k"], json!(1));
    assert_eq!(
        (model.max_model_len, model.max_output_tokens),
        (Some(4096), Some(2048))
    );
    assert_eq!(model.model_card["context_window"], json!(2048));
    assert!(!model.synthetic_acceptance);
    let channel = model.required_channel.as_ref().unwrap();
    assert_eq!(
        (channel.after.as_slice(), channel.opening.as_slice()),
        (&[1][..], &[2][..])
    );
    assert_eq!(
        (channel.opens_at, channel.end_ids.as_slice()),
        (1, &[3, 7, 9][..])
    );
    let toggle = model.reasoning_toggle.as_ref().unwrap();
    assert_eq!(
        (toggle.name.as_str(), &toggle.off, &toggle.on),
        ("enable_thinking", &json!(false), &json!(true))
    );
    assert_eq!(spec.balance.strategy, BalanceStrategy::LeastRequests);
    assert_eq!(spec.engines.len(), 2);
    let core = &spec.engines[1];
    assert_eq!((core.dp_rank, core.pid), (1, 101));
    assert_eq!(core.input, "ipc:///tmp/core-1-input");
}
